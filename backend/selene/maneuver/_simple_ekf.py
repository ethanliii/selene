"""Compact STM-based extended Kalman filter for angles-only cislunar tracking.

This is the maneuver track's *reference* filter: small, deterministic and self-contained so
the detectors can be tested independently of :mod:`selene.od.ukf`.  It returns the shared
:class:`~selene.od.types.FilterRun` so the two are interchangeable downstream.

Model
-----
* Dynamics: ephemeris force model (Earth + Moon + Sun point masses, optional SRP) from
  :func:`selene.dynamics.ephemeris.propagate_ephemeris` with the 6×6 state-transition matrix Φ
  integrated alongside the state (variational equations), so
  ``P⁻ = Φ P⁺ Φᵀ + Q(Δt)``.
* Process noise: white acceleration PSD ``q`` [km²/s³],
  ``Q = q·[[Δt³/3 I, Δt²/2 I], [Δt²/2 I, Δt I]]`` (Bar-Shalom, Li & Kirubarajan 2001, §6.2.3
  "continuous white-noise acceleration" model).
* Measurement: topocentric GCRF (ra, dec) from :mod:`selene.sensors.visibility`; the innovation
  is expressed *on-sky* as ``ν = [wrap(ra_m − ra_p)·cos(dec_p), dec_m − dec_p]`` so the
  measurement noise is the isotropic ``R = σ² I₂`` that :func:`observe` applies, and the
  right-ascension wrap at 0/2π is handled explicitly (``wrap_angle``).
* Update: Joseph-form covariance update for numerical symmetry.

Linearisation caveat: over a 6-h step a 20 km / 2 m/s initial error is well inside the linear
regime of the N-body dynamics (verified by the NEES test in ``tests/test_maneuver_detection.py``),
but the EKF *will* become inconsistent for very long gaps with large uncertainty — that is
precisely the regime where the OD track's UKF/particles are preferable.
"""
from __future__ import annotations

import time
from dataclasses import replace

import numpy as np

from selene.dynamics.ephemeris import BodyCache, EphemParams, propagate_ephemeris
from selene.maneuver.types import FilterRun
from selene.sensors.visibility import Measurement, measurement_jacobian, measurement_model

__all__ = ["wrap_angle", "innovation", "process_noise", "propagate_stm", "ensure_cache", "run_ekf"]

TWO_PI = 2.0 * np.pi


def wrap_angle(a):
    """Wrap angle(s) [rad] to (−π, π]."""
    return (np.asarray(a, dtype=np.float64) + np.pi) % TWO_PI - np.pi


def innovation(meas: Measurement, x_pred: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """On-sky innovation ν (2,) [rad] and its Jacobian H (2,6) [rad/km] at the predicted state.

    ν = [wrap(ra_m − ra_p)·cos(dec_p), dec_m − dec_p]; the RA row of H is scaled by cos(dec_p)
    so both rows are in the same on-sky radian units (R = σ² I₂).
    """
    obs = np.asarray(meas.observer_pos_gcrf, dtype=np.float64)
    r = np.asarray(x_pred, dtype=np.float64)[:3]
    ra_p, dec_p = measurement_model(obs, r)
    cd = float(np.cos(dec_p))
    nu = np.array([float(wrap_angle(meas.ra_rad - ra_p)) * cd, float(meas.dec_rad - dec_p)])
    H = np.zeros((2, 6))
    H[:, :3] = measurement_jacobian(obs, r)
    H[0] *= cd
    return nu, H


def process_noise(dt_s: float, q_psd: float) -> np.ndarray:
    """Continuous white-noise-acceleration Q(Δt) for PSD ``q_psd`` [km²/s³]."""
    dt = float(dt_s)
    I3 = np.eye(3)
    Q = np.zeros((6, 6))
    Q[:3, :3] = (dt**3 / 3.0) * I3
    Q[:3, 3:] = Q[3:, :3] = (dt**2 / 2.0) * I3
    Q[3:, 3:] = dt * I3
    return q_psd * Q


def ensure_cache(params: EphemParams | None, t0_s: float, t1_s: float, pad_s: float = 86400.0) -> EphemParams:
    """Return ``params`` with a :class:`BodyCache` covering [t0, t1] (built once, reused)."""
    params = params or EphemParams()
    lo, hi = min(t0_s, t1_s), max(t0_s, t1_s)
    if params.cache is not None and params.cache.covers(lo) and params.cache.covers(hi):
        return params
    return replace(params, cache=BodyCache(lo - pad_s, hi + pad_s))


def propagate_stm(x: np.ndarray, t_from: float, t_to: float, params: EphemParams,
                  rtol: float = 1e-10, atol: float = 1e-10) -> tuple[np.ndarray, np.ndarray]:
    """Propagate GCRF state ``x`` from ``t_from`` to ``t_to`` and return (x(t_to), Φ(t_to, t_from))."""
    if t_to == t_from:
        return np.array(x, dtype=np.float64), np.eye(6)
    sol = propagate_ephemeris(x, float(t_from), float(t_to), params=params, stm=True, rtol=rtol, atol=atol)
    if not sol.success:
        raise RuntimeError(f"STM propagation failed: {sol.message}")
    y = sol.y[:, -1]
    return y[:6].copy(), y[6:].reshape(6, 6).copy()


def run_ekf(x0, P0, t0_s: float, meas: list[Measurement], params: EphemParams | None = None,
            q_psd: float = 1e-16, object_id: str = "", rtol: float = 1e-10, atol: float = 1e-10,
            update: bool = True) -> FilterRun:
    """Run the EKF from (x0, P0) at ``t0_s`` through ``meas`` (sorted by time; t ≥ t0_s).

    ``update=False`` propagates only (prediction-only run; posterior == prior) which is useful
    to visualise custody decay.  Returns a :class:`FilterRun`.
    """
    tic = time.perf_counter()
    x = np.array(x0, dtype=np.float64).ravel()[:6]
    P = np.array(P0, dtype=np.float64).reshape(6, 6)
    meas = sorted(meas, key=lambda m: m.t_s)
    n = len(meas)
    if n and meas[0].t_s < t0_s - 1e-6:
        raise ValueError("measurements must not precede the filter epoch t0_s")
    t_end = meas[-1].t_s if n else t0_s
    params = ensure_cache(params, t0_s, t_end)

    T = np.empty(n)
    X = np.empty((n, 6)); PP = np.empty((n, 6, 6))
    XP = np.empty((n, 6)); PPP = np.empty((n, 6, 6))
    NU = np.empty((n, 2)); SS = np.empty((n, 2, 2)); NIS = np.empty(n)
    t_prev = float(t0_s)
    I6 = np.eye(6)
    for k, m in enumerate(meas):
        dt = float(m.t_s) - t_prev
        x_pred, Phi = propagate_stm(x, t_prev, float(m.t_s), params, rtol, atol)
        P_pred = Phi @ P @ Phi.T + process_noise(dt, q_psd)
        P_pred = 0.5 * (P_pred + P_pred.T)
        nu, H = innovation(m, x_pred)
        R = (float(m.sigma_rad) ** 2) * np.eye(2)
        S = H @ P_pred @ H.T + R
        nis = float(nu @ np.linalg.solve(S, nu))
        if update:
            K = np.linalg.solve(S, H @ P_pred).T          # P H^T S^-1
            x = x_pred + K @ nu
            IKH = I6 - K @ H
            P = IKH @ P_pred @ IKH.T + K @ R @ K.T
            P = 0.5 * (P + P.T)
        else:
            x, P = x_pred, P_pred
        T[k] = m.t_s; X[k] = x; PP[k] = P; XP[k] = x_pred; PPP[k] = P_pred
        NU[k] = nu; SS[k] = S; NIS[k] = nis
        t_prev = float(m.t_s)
    meta = {
        "filter": "simple_ekf",
        "q_psd_km2_s3": float(q_psd),
        "t0_s": float(t0_s),
        "x0": np.asarray(x0, dtype=np.float64).ravel()[:6].tolist(),
        "P0_diag": np.diag(np.asarray(P0, dtype=np.float64).reshape(6, 6)).tolist(),
        "force_model": {"bodies": list(params.bodies), "srp": bool(params.srp), "cr_area_mass": float(params.cr_area_mass)},
        "update": bool(update),
        "elapsed_s": time.perf_counter() - tic,
    }
    return FilterRun(t_s=T, x=X, P=PP, x_pred=XP, P_pred=PPP, innov=NU, S=SS, nis=NIS,
                     meas=list(meas), object_id=object_id, meta=meta)
