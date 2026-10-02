"""Batch weighted least squares (differential correction) with a-priori information.

Estimates the GCRF state ``x₀`` at epoch ``t₀`` from angles-only measurements under the
ephemeris model.  Iteration k (Tapley, Schutz & Born, *Statistical Orbit Determination*, §4.6,
with Levenberg damping):

    y_i = ν_i(x₀)                              on-sky residual at measurement i  [rad]
    H_i = H̃_i Φ(t_i, t₀)                       (2×6) mapped to the epoch via the integrated STM
    N   = Σ H_iᵀ W_i H_i + P̄₀⁻¹,   b = Σ H_iᵀ W_i y_i + P̄₀⁻¹ (x̄₀ − x₀)
    (N + λ·diag N) δx = b,   x₀ ← x₀ + δx          (λ = 0 is Gauss-Newton)

with W_i = R_i⁻¹ = σ_i⁻² I₂.  The normal matrix is Jacobi-scaled before the solve (positions in
km and velocities in km/s differ by ~10⁵ in sensitivity).  A step is accepted when the weighted
RMS decreases, otherwise λ is increased ×10 and the step recomputed (classic Levenberg–Marquardt
control, Marquardt 1963).  Convergence: relative RMS change < ``tol`` or ‖δx‖ negligible.
Covariance: ``P₀ = N⁻¹`` (λ = 0) at the converged iterate.

All measurement epochs may lie on either side of ``t₀`` (two propagations).  Units: km, km/s,
s, rad; RMS reported in arcsec.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from selene.dynamics.ephemeris import EphemParams, propagate_ephemeris
from selene.od.measurements import ARCSEC, jacobian_state, predict, residual, sort_measurements
from selene.od.stacked import ensure_cache
from selene.od.types import jsonable

__all__ = ["BatchResult", "batch_least_squares", "propagate_with_stm_to"]


@dataclass
class BatchResult:
    t0_s: float
    x: np.ndarray                 # (6,) estimated GCRF state at t0
    P: np.ndarray                 # (6,6)
    residuals: np.ndarray         # (N,2) on-sky post-fit residuals [rad]
    rms_arcsec: float
    iterations: int
    converged: bool
    history: list = field(default_factory=list)   # per-iteration dicts
    meas: list = field(default_factory=list)
    meta: dict = field(default_factory=dict)
    elapsed_s: float = 0.0

    @property
    def sigma_pos_km(self) -> float:
        return float(np.sqrt(max(np.trace(self.P[:3, :3]), 0.0)))

    @property
    def sigma_vel_km_s(self) -> float:
        return float(np.sqrt(max(np.trace(self.P[3:, 3:]), 0.0)))

    def as_dict(self) -> dict:
        return jsonable({
            "method": "batch weighted least squares with a-priori (ephemeris STM, Levenberg damping)",
            "t0_s": self.t0_s, "state": self.x, "cov": self.P, "rms_arcsec": self.rms_arcsec,
            "iterations": self.iterations, "converged": self.converged, "n_obs": len(self.meas),
            "sigma_pos_km": self.sigma_pos_km, "sigma_vel_km_s": self.sigma_vel_km_s,
            "residuals_arcsec": self.residuals / ARCSEC, "history": self.history,
            "sensor_ids": [m.sensor_id for m in self.meas], "t_s": [m.t_s for m in self.meas],
            "elapsed_s": self.elapsed_s, **self.meta,
        })


def propagate_with_stm_to(x0: np.ndarray, t0_s: float, times: np.ndarray, params: EphemParams,
                          rtol: float = 1e-10, atol: float = 1e-10):
    """States (N,6) and STMs Φ(t_i, t0) (N,6,6) at arbitrary ``times`` (either side of t0)."""
    times_all = np.asarray(times, dtype=np.float64)
    times, inverse = np.unique(times_all, return_inverse=True)   # solve_ivp needs strictly monotone t_eval
    X = np.empty((times.size, 6))
    Phi = np.empty((times.size, 6, 6))
    at_t0 = np.abs(times - t0_s) < 1e-9
    X[at_t0] = np.asarray(x0, dtype=np.float64)[:6]
    Phi[at_t0] = np.eye(6)
    for side in ((times > t0_s) & ~at_t0, (times < t0_s) & ~at_t0):
        idx = np.where(side)[0]
        if idx.size == 0:
            continue
        ts = times[idx]
        order = np.argsort(ts) if ts[0] >= t0_s else np.argsort(-ts)
        t_sorted = ts[order]
        t_eval = np.concatenate([[t0_s], t_sorted])
        sol = propagate_ephemeris(x0, t0_s, t_eval_s=t_eval, params=params, stm=True, rtol=rtol, atol=atol)
        if not sol.success or np.asarray(sol.y).shape[1] != t_eval.size:
            raise RuntimeError(f"STM propagation failed: {sol.message}")
        Y = sol.y.T[1:]
        X[idx[order]] = Y[:, :6]
        Phi[idx[order]] = Y[:, 6:].reshape(-1, 6, 6)
    return X[inverse], Phi[inverse]


def batch_least_squares(
    meas,
    x0: np.ndarray,
    t0_s: float | None = None,
    P0: np.ndarray | None = None,
    x_apriori: np.ndarray | None = None,
    params: EphemParams | None = None,
    max_iter: int = 15,
    tol: float = 1e-4,
    lam0: float = 0.0,
    object_id: str = "",
) -> BatchResult:
    """Fit the epoch state to ``meas`` starting from ``x0`` (GCRF km, km/s) at ``t0_s`` (default:
    the first measurement epoch).  ``P0`` (6×6) with ``x_apriori`` (default ``x0``) adds the
    a-priori term; ``P0=None`` is a pure least-squares fit (needs ≥ 3 well-spread measurements).
    """
    t_start = time.perf_counter()
    ms = sort_measurements(meas)
    if len(ms) < 1:
        raise ValueError("batch_least_squares needs at least one measurement")
    times = np.array([m.t_s for m in ms])
    R = np.array([m.observer_pos_gcrf for m in ms])
    w = np.array([1.0 / m.sigma_rad ** 2 for m in ms])
    t0 = float(times[0]) if t0_s is None else float(t0_s)
    x = np.asarray(x0, dtype=np.float64).copy()
    xbar = x.copy() if x_apriori is None else np.asarray(x_apriori, dtype=np.float64).copy()
    Pinv0 = None if P0 is None else np.linalg.inv(np.asarray(P0, dtype=np.float64))
    params = ensure_cache(params, min(t0, times.min()), max(t0, times.max()))

    def evaluate(xc):
        X, Phi = propagate_with_stm_to(xc, t0, times, params)
        ra_p, dec_p = predict(X, R)
        y = residual([m.ra_rad for m in ms], [m.dec_rad for m in ms], ra_p, dec_p)  # (N,2)
        H = np.einsum("nij,njk->nik", jacobian_state(X, R), Phi)                    # (N,2,6)
        cost = float(np.sum(w[:, None] * y ** 2))
        if Pinv0 is not None:
            d = xbar - xc
            cost += float(d @ Pinv0 @ d)
        rms = float(np.sqrt(np.mean(y ** 2)))
        return y, H, cost, rms

    y, H, cost, rms = evaluate(x)
    lam = float(lam0)
    history = [{"iter": 0, "rms_arcsec": rms / ARCSEC, "cost": cost, "lambda": lam, "step_km": 0.0}]
    converged = False
    it = 0
    N = None
    for it in range(1, max_iter + 1):
        N = np.einsum("nik,n,nil->kl", H, w, H)
        b = np.einsum("nik,n,ni->k", H, w, y)
        if Pinv0 is not None:
            N = N + Pinv0
            b = b + Pinv0 @ (xbar - x)
        s = 1.0 / np.sqrt(np.maximum(np.diag(N), 1e-300))
        Ns = N * s[:, None] * s[None, :]
        bs = b * s
        accepted = False
        best_trial = np.inf
        for _ in range(12):
            try:
                dxs = np.linalg.solve(Ns + lam * np.eye(6), bs)
            except np.linalg.LinAlgError:
                dxs = np.linalg.lstsq(Ns + lam * np.eye(6), bs, rcond=None)[0]
            dx = dxs * s
            x_try = x + dx
            try:
                y_t, H_t, cost_t, rms_t = evaluate(x_try)
            except RuntimeError:
                cost_t = np.inf
            if np.isfinite(cost_t) and cost_t <= cost * (1.0 + 1e-12):
                accepted = True
                break
            best_trial = min(best_trial, cost_t)
            lam = 10.0 * lam if lam > 0 else 1e-3
        if not accepted:
            # No strictly descending step: either the iterate is already at the numerical noise floor
            # (every trial reproduces the current cost to within ``tol``), which is convergence, or
            # the damping ran out, which is a genuine failure.
            at_floor = np.isfinite(best_trial) and abs(best_trial - cost) / max(cost, 1e-300) < tol
            converged = bool(at_floor)
            history.append({"iter": it, "rms_arcsec": rms / ARCSEC, "cost": cost, "lambda": lam, "step_km": 0.0,
                            "note": "converged: cost at the noise floor (no further descent)" if at_floor
                            else "no descent step found"})
            break
        rel = abs(cost - cost_t) / max(cost, 1e-300)
        step_km = float(np.linalg.norm(dx[:3]))
        x, y, H, cost, rms_prev, rms = x_try, y_t, H_t, cost_t, rms, rms_t
        lam = lam / 10.0 if lam > 1e-12 else 0.0
        history.append({"iter": it, "rms_arcsec": rms / ARCSEC, "cost": cost, "lambda": lam, "step_km": step_km,
                        "step_m_s": float(np.linalg.norm(dx[3:]) * 1e3)})
        if rel < tol or (step_km < 1e-6 and np.linalg.norm(dx[3:]) < 1e-9):
            converged = True
            break
    # covariance at the final iterate (undamped)
    N = np.einsum("nik,n,nil->kl", H, w, H)
    if Pinv0 is not None:
        N = N + Pinv0
    s = 1.0 / np.sqrt(np.maximum(np.diag(N), 1e-300))
    Ns = N * s[:, None] * s[None, :]
    try:
        Ps = np.linalg.inv(Ns)
    except np.linalg.LinAlgError:
        Ps = np.linalg.pinv(Ns)
    P = Ps * s[:, None] * s[None, :]
    P = 0.5 * (P + P.T)
    meta = {"object_id": object_id, "a_priori": Pinv0 is not None, "tol": tol,
            "dof": int(2 * len(ms) - 6), "weighted_rms": float(np.sqrt(cost / max(2 * len(ms), 1)))}
    return BatchResult(t0, x, P, y, rms / ARCSEC, it, converged, history, ms, meta, time.perf_counter() - t_start)
