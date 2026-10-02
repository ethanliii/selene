"""Unscented Kalman filter (and a sequential EKF) for angles-only cislunar tracking.

Scaled unscented transform (Julier & Uhlmann 2004; Wan & van der Merwe 2000)
--------------------------------------------------------------------------
n = 6, λ = α²(n+κ) − n, γ = √(n+λ);  χ₀ = x, χ_{±i} = x ± γ (√P)_i;
W₀ᵐ = λ/(n+λ), W₀ᶜ = W₀ᵐ + (1 − α² + β), W_iᵐ = W_iᶜ = 1/(2(n+λ)).
Defaults α = 1e-3, β = 2 (Gaussian prior), κ = 0.  With α = 1e-3 the sigma points sit 2.4e-3 σ
from the mean and W₀ᶜ ≈ −10⁶: the transform is then numerically equivalent to a central-difference
EKF, and the sigma points **must** share the integration step sequence (see
:mod:`selene.od.stacked`) or integrator noise is amplified by |W₀|.  Larger α (0.1–1) spreads the
points over the real curvature of the dynamics; both are supported through :class:`UKFConfig`.

Measurement nonlinearity and the iterated update
------------------------------------------------
Angles are linear in the state only while the prior position uncertainty δ is small against the
range ρ: the second-order term of h is ≈ ½(δ/ρ)² rad.  For an IOD-sized prior (δ ≈ 500 km) seen
from a DRO/L1 observer (ρ ≈ 75 000 km) that is ≈ 2e-5 rad ≈ 4.6″, several times the 1″ noise, so a
single EKF/UKF linearisation about the *prior* mean leaves a biased, over-confident posterior
(measured, 20-run Monte Carlo on SIM-DRO-01, 3 d, 1″, dro_obs only, P₀ = 500 km / 5 m/s: mean
NEES 120 (α = 1e-3), 12 (α = 1), 149 (EKF) against the χ² band [4.6, 7.6]).  Neither α nor the
unscented transform fixes this because the error is in the *mean*, not the covariance.  The
standard remedy is the iterated measurement update (Bell & Cathey 1993, "The iterated Kalman
filter update as a Gauss-Newton method", IEEE TAC 38(2)): Gauss-Newton on the MAP cost
``(x−x⁻)ᵀP⁻⁻¹(x−x⁻) + (z−h(x))ᵀR⁻¹(z−h(x))``,

    x_{i+1} = x⁻ + K_i [ z − h(x_i) − H_i (x⁻ − x_i) ],   K_i = P⁻H_iᵀ(H_iP⁻H_iᵀ + R)⁻¹,

with the posterior covariance evaluated at the converged iterate (Joseph form).  It is enabled by
default (``UKFConfig.iterated_update = 5``; the UKF performs its unscented update first and then
re-linearises with the analytic H of :mod:`selene.od.measurements`).  With it the same Monte
Carlo gives consistent NEES in both regimes (numbers in ``tests/test_od_ukf.py`` and the M4
report).  The stored ``innov``/``S``/``nis`` are always the *prediction* statistics (pre-update),
which is what the maneuver monitor needs.

Prediction between measurements: each sigma point is propagated with the ephemeris force model
(one stacked ``solve_ivp`` by default, ``propagation='loop'`` for the per-point alternative), then
``P⁻ = Σ W_iᶜ (χ_i − x̄)(χ_i − x̄)ᵀ + Q(Δt)``.

Process noise: continuous white acceleration of PSD ``q`` [km²/s³] mapped through the
constant-velocity kinematics (Bar-Shalom, Li & Kirubarajan 2001, §6.2.2)::

    Q(Δt) = q · [[Δt³/3 I₃, Δt²/2 I₃], [Δt²/2 I₃, Δt I₃]]

Choosing q: the synthetic truth uses the same force model as the filter (SRP with the catalog
C_R·A/m), so the consistent choice is q ≈ 0; a 40-run Monte Carlo on SIM-DRO-01 (3 days, two
space sensors, 1″) gave mean NEES 6.35 (q = 0), 6.26 (1e-18), 5.83 (1e-17) and 5.09 with the
last epochs at ≈ 3.6 (q = 1e-16, i.e. pessimistic: the added noise dominates a ~1 km posterior).
The default 1e-18 km²/s³ is therefore a numerical regulariser only.  Against real data, or a
truth with unmodelled accelerations (SRP mis-modelling at the 1e-11 km/s² level, outgassing,
small thruster activity), raise q to ≈ a²·τ (acceleration² × correlation time), e.g. 1e-16–1e-14.

Update: the sigma points are regenerated from (x⁻, P⁻) and pushed through the angles model; the
predicted mean RA is formed with wrap-around relative to χ₀, the innovation is the on-sky
residual ``[Δra·cos dec, Δdec]`` (R = σ² I₂), ``S = P_zz + R``, ``K = P_xz S⁻¹``,
``NIS = νᵀ S⁻¹ ν`` (χ²₂ under a consistent filter — stored per update for the maneuver monitor).

Maneuver-robust hook: ``inflate(ctx) -> P⁻ or None`` is called after the innovation statistics are
formed; returning a new prior covariance makes the filter recompute the update with it (the stored
``nis`` is the *pre-inflation* value, i.e. the detection statistic; the inflated prior is stored in
``P_pred``).  The maneuver module uses this to re-open the covariance after a detection.

Output: :class:`selene.od.types.FilterRun` (one row per processed measurement).
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np

from selene.dynamics.ephemeris import EphemParams, propagate_ephemeris
from selene.od.measurements import jacobian_state, predict, residual, sort_measurements, wrap_angle
from selene.od.stacked import ensure_cache, propagate_many
from selene.od.types import FilterRun

__all__ = ["UKFConfig", "ukf_run", "ekf_run", "process_noise", "sigma_points", "sqrt_psd"]


@dataclass
class UKFConfig:
    alpha: float = 1e-3
    beta: float = 2.0
    kappa: float = 0.0
    q_psd: float = 1e-18          # km²/s³; see "Choosing q" in the module docstring
    rtol: float = 1e-10
    atol: float = 1e-10
    propagation: str = "stacked"  # 'stacked' | 'loop'
    regenerate_for_update: bool = True   # False: re-use the propagated sigma points (Q not re-sampled)
    iterated_update: int = 5      # Gauss-Newton re-linearisations of the update (0 = single-pass)
    iter_tol_km: float = 1e-4     # stop iterating when the position iterate moves less than this


def process_noise(dt_s: float, q_psd: float) -> np.ndarray:
    """Continuous-white-acceleration Q(Δt) (6×6) in km², km²/s, km²/s²."""
    dt = float(dt_s)
    I3 = np.eye(3)
    Q = np.zeros((6, 6))
    Q[:3, :3] = dt ** 3 / 3.0 * I3
    Q[:3, 3:] = Q[3:, :3] = dt ** 2 / 2.0 * I3
    Q[3:, 3:] = dt * I3
    return q_psd * Q


def sqrt_psd(P: np.ndarray) -> np.ndarray:
    """Lower-triangular Cholesky factor; falls back to an eigen square root with clipped
    eigenvalues when P is not numerically positive definite."""
    P = 0.5 * (P + P.T)
    try:
        return np.linalg.cholesky(P)
    except np.linalg.LinAlgError:
        w, V = np.linalg.eigh(P)
        w = np.clip(w, 1e-18 * max(float(w.max()), 1e-30), None)
        return V * np.sqrt(w)[None, :]


def _weights(n: int, cfg: UKFConfig):
    lam = cfg.alpha ** 2 * (n + cfg.kappa) - n
    gamma = np.sqrt(n + lam)
    Wm = np.full(2 * n + 1, 1.0 / (2.0 * (n + lam)))
    Wc = Wm.copy()
    Wm[0] = lam / (n + lam)
    Wc[0] = Wm[0] + (1.0 - cfg.alpha ** 2 + cfg.beta)
    return gamma, Wm, Wc


def sigma_points(x: np.ndarray, P: np.ndarray, gamma: float) -> np.ndarray:
    """(2n+1, n) scaled sigma points."""
    n = x.size
    Lc = sqrt_psd(P)
    chi = np.empty((2 * n + 1, n))
    chi[0] = x
    chi[1:n + 1] = x[None, :] + gamma * Lc.T
    chi[n + 1:] = x[None, :] - gamma * Lc.T
    return chi


def _propagate_points(chi, t0, t1, params, cfg):
    if abs(t1 - t0) < 1e-9:
        return chi.copy()
    if cfg.propagation == "loop":
        out = np.empty_like(chi)
        for i in range(chi.shape[0]):
            sol = propagate_ephemeris(chi[i], t0, t1, params=params, rtol=cfg.rtol, atol=cfg.atol)
            if not sol.success:
                raise RuntimeError(f"sigma-point propagation failed: {sol.message}")
            out[i] = sol.y[:6, -1]
        return out
    return propagate_many(chi, t0, np.array([t1]), params, rtol=cfg.rtol, atol=cfg.atol)[0]


def _angle_stats(chi, R, Wm, Wc, sigma_rad):
    """Predicted angles of the sigma points -> (ra_mean, dec_mean, dz (2n+1,2), Pzz, S)."""
    ra, dec = predict(chi, R)
    ra_rel = ra[0] + wrap_angle(ra - ra[0])
    ra_m = float(np.sum(Wm * ra_rel))
    dec_m = float(np.sum(Wm * dec))
    dz = np.stack([wrap_angle(ra - ra_m) * np.cos(dec_m), dec - dec_m], axis=1)
    Pzz = np.einsum("i,ij,ik->jk", Wc, dz, dz)
    S = Pzz + sigma_rad ** 2 * np.eye(2)
    return ra_m, dec_m, dz, Pzz, S


def _run(meas, x0, P0, t0_s, params, cfg, object_id, inflate, kind):
    t_start = time.perf_counter()
    ms = sort_measurements(meas)
    x = np.asarray(x0, dtype=np.float64).copy()
    P = 0.5 * (np.asarray(P0, dtype=np.float64) + np.asarray(P0, dtype=np.float64).T)
    t = float(t0_s)
    if ms and ms[0].t_s < t - 1e-9:
        raise ValueError("measurements must not precede the filter epoch t0_s")
    t_hi = max([m.t_s for m in ms], default=t)
    params = ensure_cache(params, t, t_hi)
    n = 6
    gamma, Wm, Wc = _weights(n, cfg)
    N = len(ms)
    out_t = np.empty(N)
    out_x = np.empty((N, 6))
    out_P = np.empty((N, 6, 6))
    out_xp = np.empty((N, 6))
    out_Pp = np.empty((N, 6, 6))
    out_nu = np.empty((N, 2))
    out_S = np.empty((N, 2, 2))
    out_nis = np.empty(N)
    inflated = np.zeros(N, dtype=bool)
    nis_post = np.empty(N)
    n_prop = 0
    n_iter_total = 0
    for k, m in enumerate(ms):
        dt = m.t_s - t
        R = np.asarray(m.observer_pos_gcrf, dtype=np.float64)
        # ---- predict ---------------------------------------------------------
        chi_p = None
        if dt > 1e-9:
            if kind == "ukf":
                chi = sigma_points(x, P, gamma)
                chi_p = _propagate_points(chi, t, m.t_s, params, cfg)
                x_pred = Wm @ chi_p
                d = chi_p - x_pred
                P_pred = np.einsum("i,ij,ik->jk", Wc, d, d) + process_noise(dt, cfg.q_psd)
            else:
                sol = propagate_ephemeris(x, t, m.t_s, params=params, stm=True, rtol=cfg.rtol, atol=cfg.atol)
                if not sol.success:
                    raise RuntimeError(f"EKF propagation failed: {sol.message}")
                x_pred = sol.y[:6, -1]
                Phi = sol.y[6:, -1].reshape(6, 6)
                P_pred = Phi @ P @ Phi.T + process_noise(dt, cfg.q_psd)
            n_prop += 1
        else:
            x_pred, P_pred = x.copy(), P.copy()
        P_pred = 0.5 * (P_pred + P_pred.T)

        # ---- innovation statistics ----------------------------------------------
        def stats(P_use):
            if kind == "ukf":
                reuse = (not cfg.regenerate_for_update) and chi_p is not None and P_use is P_pred
                chi_u = chi_p if reuse else sigma_points(x_pred, P_use, gamma)
                ra_m, dec_m, dz, Pzz, S = _angle_stats(chi_u, R, Wm, Wc, m.sigma_rad)
                Pxz = np.einsum("i,ij,ik->jk", Wc, chi_u - x_pred, dz)
            else:
                ra_m, dec_m = predict(x_pred, R)
                H = jacobian_state(x_pred, R)
                S = H @ P_use @ H.T + m.sigma_rad ** 2 * np.eye(2)
                Pxz = P_use @ H.T
            nu = residual(m.ra_rad, m.dec_rad, ra_m, dec_m)
            nis = float(nu @ np.linalg.solve(S, nu))
            return nu, S, Pxz, nis

        nu, S, Pxz, nis = stats(P_pred)
        out_nis[k] = nis
        if inflate is not None:
            P_new = inflate({"k": k, "t_s": m.t_s, "x_pred": x_pred, "P_pred": P_pred, "innov": nu, "S": S,
                             "nis": nis, "meas": m, "dt_s": dt})
            if P_new is not None:
                P_pred = 0.5 * (np.asarray(P_new, dtype=np.float64) + np.asarray(P_new, dtype=np.float64).T)
                inflated[k] = True
                nu, S, Pxz, nis = stats(P_pred)
        nis_post[k] = nis
        # ---- update -----------------------------------------------------------------
        K = Pxz @ np.linalg.inv(S)
        x = x_pred + K @ nu
        Rm = m.sigma_rad ** 2 * np.eye(2)
        if kind == "ukf":
            P = P_pred - K @ S @ K.T
        else:
            H = jacobian_state(x_pred, R)
            IKH = np.eye(6) - K @ H
            P = IKH @ P_pred @ IKH.T + K @ Rm @ K.T
        if cfg.iterated_update > 0:
            # Gauss-Newton (IEKF) re-linearisation about the posterior iterate; see module docstring
            xi = x.copy()
            for _ in range(int(cfg.iterated_update)):
                Hi = jacobian_state(xi, R)
                ra_i, dec_i = predict(xi, R)
                nu_i = residual(m.ra_rad, m.dec_rad, ra_i, dec_i)
                Si = Hi @ P_pred @ Hi.T + Rm
                Ki = P_pred @ Hi.T @ np.linalg.inv(Si)
                x_new = x_pred + Ki @ (nu_i - Hi @ (x_pred - xi))
                moved = float(np.linalg.norm(x_new[:3] - xi[:3]))
                xi = x_new
                n_iter_total += 1
                if moved < cfg.iter_tol_km:
                    break
            x = xi
            IKH = np.eye(6) - Ki @ Hi
            P = IKH @ P_pred @ IKH.T + Ki @ Rm @ Ki.T
        P = 0.5 * (P + P.T)
        t = float(m.t_s)
        out_t[k], out_x[k], out_P[k], out_xp[k], out_Pp[k], out_nu[k], out_S[k] = t, x, P, x_pred, P_pred, nu, S
    elapsed = time.perf_counter() - t_start
    meta = {
        "filter": kind, "alpha": cfg.alpha, "beta": cfg.beta, "kappa": cfg.kappa, "q_psd_km2_s3": cfg.q_psd,
        "propagation": cfg.propagation if kind == "ukf" else "stm", "n_updates": N, "n_propagations": n_prop,
        "elapsed_s": elapsed, "t0_s": float(t0_s), "inflated": inflated.tolist(), "nis_post_inflation": nis_post.tolist(),
        "mean_nis": float(np.mean(out_nis)) if N else None,
        "iterated_update": int(cfg.iterated_update), "update_iterations_total": int(n_iter_total),
    }
    return FilterRun(out_t, out_x, out_P, out_xp, out_Pp, out_nu, out_S, out_nis, ms, object_id, meta)


def ukf_run(meas, x0, P0, t0_s: float, params: EphemParams | None = None, config: UKFConfig | None = None,
            object_id: str = "", inflate: Optional[Callable[[dict], Optional[np.ndarray]]] = None) -> FilterRun:
    """Run the UKF over ``meas`` from the prior (x0, P0) at ``t0_s``.  See module docstring."""
    return _run(meas, x0, P0, t0_s, params, config or UKFConfig(), object_id, inflate, "ukf")


def ekf_run(meas, x0, P0, t0_s: float, params: EphemParams | None = None, config: UKFConfig | None = None,
            object_id: str = "", inflate: Optional[Callable[[dict], Optional[np.ndarray]]] = None) -> FilterRun:
    """Sequential extended Kalman filter (STM linearisation, Joseph-form update); same output."""
    return _run(meas, x0, P0, t0_s, params, config or UKFConfig(), object_id, inflate, "ekf")
