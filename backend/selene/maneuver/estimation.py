"""Impulsive-Δv estimation from angles-only measurements around a detected epoch window.

Problem
-------
Unknowns θ = (Δv ∈ ℝ³ [km/s], t_b) with t_b inside the detection window [t_lo, t_hi]
(last pre-detection observation → first flagged observation), plus optionally the pre-burn
state correction δx_pre ∈ ℝ⁶ with the filter's posterior as a Gaussian prior.  For a candidate
t_b the pre-burn state is propagated under the ephemeris model to t_b, Δv is added to the
velocity, and the post-burn arc is propagated to the post-window measurements; the whitened
on-sky residuals ``[Δra·cos(dec), Δdec]/σ`` are minimised with ``scipy.optimize.least_squares``
(trust-region reflective, box bounds on Δv).

Jacobians come from the state-transition matrices integrated with the trajectory:
``∂ν_k/∂Δv = −H_k Φ_rv(t_k, t_b)``, ``∂ν_k/∂δx_pre = −H_k Φ(t_k, t_b) Φ(t_b, t_pre)``;
the prior rows are ``L⁻¹ δx_pre`` with ``P_pre = L Lᵀ``.

t_b is found on a grid of ``n_grid`` candidates (centres of equal sub-intervals) and then
refined continuously with a bounded scalar minimisation of the optimal cost; its 1-σ comes
from the curvature of the cost (σ_t² = 2/c″ for whitened residuals, c = Σ r²).

Outputs: Δv in GCRF and in the local RTN / VNB frames of the pre-burn state (frame centre per
:class:`EstimatorConfig.center`), magnitude [m/s], direction, 1-σ covariance (formal, from
(JᵀJ)⁻¹, and scaled by the reduced χ² when residuals are worse than σ), residual RMS, and a
*labelled heuristic* classification (:func:`classify_maneuver`).
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np
from scipy.optimize import least_squares, minimize_scalar

from selene.constants import GM_EARTH, GM_MOON
from selene.dynamics import frames
from selene.dynamics.cr3bp import jacobi, lagrange_points
from selene.dynamics.ephemeris import EphemParams, get_ephemeris
from selene.maneuver._simple_ekf import ensure_cache, innovation, propagate_stm
from selene.maneuver.config import EstimatorConfig
from selene.maneuver.detection import _propagate_with_stm
from selene.maneuver.synthetic import frame_center, rtn_basis, vnb_basis

__all__ = ["DvEstimate", "estimate_impulsive_dv", "classify_maneuver"]

ARCSEC = np.pi / (180.0 * 3600.0)


@dataclass
class DvEstimate:
    t_burn_s: float
    t_burn_sigma_s: float
    dv_gcrf_kms: np.ndarray
    cov_dv_kms2: np.ndarray                # formal (3,3)
    cov_dv_scaled_kms2: np.ndarray         # formal × max(1, reduced χ²)
    magnitude_mps: float
    magnitude_sigma_mps: float
    direction_gcrf: np.ndarray
    direction_sigma_deg: float
    frame_center: str
    dv_rtn_mps: np.ndarray
    dv_vnb_mps: np.ndarray
    residual_rms_arcsec: float
    residual_rms_sigma: float
    reduced_chi2: float
    n_obs: int
    converged: bool
    window: tuple[float, float]
    grid: list[dict]
    classification: dict
    x_pre_burn: np.ndarray                 # state just before the burn (after δx_pre correction)
    x_post_burn: np.ndarray
    dx_pre_correction: Optional[np.ndarray] = None
    meta: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {
            "t_burn_s": float(self.t_burn_s), "t_burn_sigma_s": float(self.t_burn_sigma_s),
            "dv_gcrf_mps": (self.dv_gcrf_kms * 1e3).tolist(),
            "cov_dv_mps2": (self.cov_dv_kms2 * 1e6).tolist(),
            "cov_dv_scaled_mps2": (self.cov_dv_scaled_kms2 * 1e6).tolist(),
            "sigma_dv_components_mps": (np.sqrt(np.clip(np.diag(self.cov_dv_scaled_kms2), 0, None)) * 1e3).tolist(),
            "magnitude_mps": float(self.magnitude_mps), "magnitude_sigma_mps": float(self.magnitude_sigma_mps),
            "direction_gcrf": self.direction_gcrf.tolist(), "direction_sigma_deg": float(self.direction_sigma_deg),
            "frame_center": self.frame_center,
            "dv_rtn_mps": self.dv_rtn_mps.tolist(), "dv_vnb_mps": self.dv_vnb_mps.tolist(),
            "residual_rms_arcsec": float(self.residual_rms_arcsec), "residual_rms_sigma": float(self.residual_rms_sigma),
            "reduced_chi2": float(self.reduced_chi2), "n_obs": int(self.n_obs), "converged": bool(self.converged),
            "window_s": [float(self.window[0]), float(self.window[1])],
            "grid": self.grid, "classification": self.classification,
            "x_pre_burn": self.x_pre_burn.tolist(), "x_post_burn": self.x_post_burn.tolist(),
            "dx_pre_correction": None if self.dx_pre_correction is None else self.dx_pre_correction.tolist(),
            "meta": self.meta,
        }


# ---------------------------------------------------------------------------
def _solve_at_tb(t_b: float, x_pre: np.ndarray, t_pre: float, L_inv: Optional[np.ndarray], meas: Sequence,
                 params: EphemParams, cfg: EstimatorConfig, p0: Optional[np.ndarray] = None,
                 coarse: bool = False):
    """Least squares over (δx_pre?, Δv) for a fixed burn epoch; returns the scipy result plus
    the pieces needed for the covariance.  ``coarse=True`` uses the grid-stage settings
    (looser integrator tolerance, few iterations) — enough to *rank* candidate burn epochs."""
    rtol = cfg.grid_rtol if coarse else cfg.rtol
    atol = cfg.grid_rtol if coarse else cfg.atol
    max_nfev = cfg.grid_max_nfev if coarse else cfg.max_nfev
    x_b0, Phi_b = propagate_stm(x_pre, t_pre, t_b, params, rtol, atol)
    times = np.array([float(m.t_s) for m in meas])
    uniq, inv = np.unique(times, return_inverse=True)
    sig = np.array([float(m.sigma_rad) for m in meas])
    with_prior = L_inv is not None
    n_par = 9 if with_prior else 3
    B = np.zeros((6, 3)); B[3:, :] = np.eye(3)

    def unpack(p):
        if with_prior:
            dxp, dv = p[:6], p[6:]
            x_b = x_b0 + Phi_b @ dxp
        else:
            dxp, dv = None, p
            x_b = x_b0.copy()
        x_b = x_b.copy()
        x_b[3:] += dv
        return dxp, dv, x_b

    def residuals_and_jac(p):
        dxp, dv, x_b = unpack(p)
        states, phis = _propagate_with_stm(x_b, t_b, uniq, params, rtol, atol)
        r = np.empty(2 * len(meas) + (6 if with_prior else 0))
        J = np.zeros((r.size, n_par))
        for k, m in enumerate(meas):
            j = inv[k]
            nu, H = innovation(m, states[j])
            w = 1.0 / sig[k]
            r[2 * k: 2 * k + 2] = nu * w
            HPhi = H @ phis[j]                     # ∂h/∂x_b (2,6)
            if with_prior:
                J[2 * k: 2 * k + 2, :6] = -(HPhi @ Phi_b) * w
                J[2 * k: 2 * k + 2, 6:] = -(HPhi @ B) * w
            else:
                J[2 * k: 2 * k + 2, :] = -(HPhi @ B) * w
        if with_prior:
            r[-6:] = L_inv @ dxp
            J[-6:, :6] = L_inv
        return r, J

    cache = {}

    def fun(p):
        r, J = residuals_and_jac(p)
        cache["p"], cache["J"] = p.copy(), J
        return r

    def jac(p):
        if "p" in cache and np.array_equal(cache["p"], p):
            return cache["J"]
        return residuals_and_jac(p)[1]

    if p0 is None:
        p0 = np.zeros(n_par)
    lo = np.full(n_par, -np.inf); hi = np.full(n_par, np.inf)
    lo[-3:] = -cfg.dv_bound_kms; hi[-3:] = cfg.dv_bound_kms
    p0 = np.clip(p0, lo + 1e-12, hi - 1e-12)
    x_scale = np.ones(n_par)
    if with_prior:
        x_scale[:3] = 1.0; x_scale[3:6] = 1e-3
    x_scale[-3:] = 1e-3
    tol = 1e-8 if coarse else 1e-12
    res = least_squares(fun, p0, jac=jac, bounds=(lo, hi), method="trf", x_scale=x_scale,
                        max_nfev=max_nfev, xtol=tol, ftol=tol, gtol=tol)
    dxp, dv, x_b = unpack(res.x)
    return {"res": res, "dv": dv, "dxp": dxp, "x_b_post": x_b, "x_b_pre": x_b - B @ dv,
            "cost2": float(2.0 * res.cost), "n_res": 2 * len(meas)}


def estimate_impulsive_dv(x_pre, t_pre: float, meas_post: Sequence, t_window: tuple[float, float],
                          P_pre: np.ndarray | None = None, params: EphemParams | None = None,
                          cfg: EstimatorConfig | None = None) -> DvEstimate:
    """Estimate an impulsive Δv (see module docstring).

    x_pre, t_pre   pre-burn state estimate (GCRF km, km/s) and its epoch (≤ t_window[0])
    meas_post      measurements after the window (the first ``cfg.n_post_obs`` are used)
    t_window       (t_lo, t_hi) bracketing the burn epoch
    P_pre          pre-burn covariance; if given and ``cfg.include_prior`` the pre-burn state is
                   co-estimated with this prior, otherwise it is held fixed
    """
    cfg = cfg or EstimatorConfig()
    tic = time.perf_counter()
    t_lo, t_hi = float(t_window[0]), float(t_window[1])
    if not (t_pre <= t_lo < t_hi):
        raise ValueError("need t_pre <= t_window[0] < t_window[1]")
    meas = sorted([m for m in meas_post if m.t_s >= t_hi - 1e-6], key=lambda m: m.t_s)[: int(cfg.n_post_obs)]
    if len(meas) < 2:
        raise ValueError("Δv estimation needs at least 2 post-window measurements (4 equations, 3 unknowns)")
    x_pre = np.asarray(x_pre, dtype=np.float64).ravel()[:6]
    t_end = float(meas[-1].t_s)
    params = ensure_cache(params, t_pre, t_end)
    L_inv = None
    if P_pre is not None and cfg.include_prior:
        L_inv = np.linalg.inv(np.linalg.cholesky(np.asarray(P_pre, dtype=np.float64)))

    # 1) grid over t_b
    n_grid = max(1, int(cfg.n_grid))
    tb_grid = t_lo + (t_hi - t_lo) * (np.arange(n_grid) + 0.5) / n_grid
    grid = []
    best = None
    coarse = n_grid > 1
    for tb in tb_grid:
        sol = _solve_at_tb(float(tb), x_pre, t_pre, L_inv, meas, params, cfg, coarse=coarse)
        grid.append({"t_burn_s": float(tb), "cost": sol["cost2"], "dv_mps": float(np.linalg.norm(sol["dv"]) * 1e3)})
        if best is None or sol["cost2"] < best[1]["cost2"]:
            best = (float(tb), sol)
    tb_best, sol_best = best
    if coarse:  # full-precision solve at the best grid point, warm-started
        sol_best = _solve_at_tb(tb_best, x_pre, t_pre, L_inv, meas, params, cfg, p0=sol_best["res"].x)
        grid[int(np.argmin([g["cost"] for g in grid]))]["cost"] = sol_best["cost2"]
    # 2) continuous refinement between the neighbouring grid points
    if cfg.refine and n_grid > 1:
        i = int(np.argmin([g["cost"] for g in grid]))
        a = tb_grid[max(0, i - 1)] if i > 0 else t_lo
        b = tb_grid[min(n_grid - 1, i + 1)] if i < n_grid - 1 else t_hi
        p_warm = sol_best["res"].x

        def cost_of(tb):
            s = _solve_at_tb(float(tb), x_pre, t_pre, L_inv, meas, params, cfg, p0=p_warm)
            return s["cost2"]

        try:
            # t_b tolerance: 10 s or 0.2 % of the window (σ_t from the curvature is typically ≳ 1 min)
            r = minimize_scalar(cost_of, bounds=(a, b), method="bounded", options={"xatol": max(10.0, 2e-3 * (t_hi - t_lo))})
            if r.fun < sol_best["cost2"]:
                tb_best = float(r.x)
                sol_best = _solve_at_tb(tb_best, x_pre, t_pre, L_inv, meas, params, cfg, p0=p_warm)
        except Exception:  # noqa: BLE001 - keep the grid solution
            pass

    res = sol_best["res"]
    dv = np.asarray(sol_best["dv"])
    J = res.jac
    try:
        cov = np.linalg.inv(J.T @ J)
    except np.linalg.LinAlgError:
        cov = np.linalg.pinv(J.T @ J)
    cov_dv = cov[-3:, -3:]
    n_res = sol_best["n_res"]
    # Reduced χ² of the whitened problem.  With the pre-burn prior the 6 prior rows count as
    # observations and δx_pre as 6 extra parameters: dof = (n_res + 6) − 9 = n_res − 3, with the
    # prior penalty kept in the numerator (Bayesian least squares); without it dof = n_res − 3.
    dof = max(1, n_res - 3)
    red_chi2 = float(sol_best["cost2"]) / dof
    cov_dv_scaled = cov_dv * max(1.0, red_chi2)
    # t_b curvature → σ_t
    h = max(30.0, 0.02 * (t_hi - t_lo))
    sig_t = float("nan")
    try:
        c0 = sol_best["cost2"]
        cp = _solve_at_tb(min(tb_best + h, t_hi), x_pre, t_pre, L_inv, meas, params, cfg, p0=res.x)["cost2"]
        cm = _solve_at_tb(max(tb_best - h, t_lo), x_pre, t_pre, L_inv, meas, params, cfg, p0=res.x)["cost2"]
        curv = (cp - 2 * c0 + cm) / h**2
        if curv > 0:
            sig_t = float(np.sqrt(2.0 / curv))
    except Exception:  # noqa: BLE001
        pass

    mag = float(np.linalg.norm(dv))
    u = dv / mag if mag > 0 else np.array([1.0, 0.0, 0.0])
    sig_mag = float(np.sqrt(max(u @ cov_dv_scaled @ u, 0.0)))
    # direction 1-σ: RMS of the two perpendicular components over |Δv|
    perp_var = max(float(np.trace(cov_dv_scaled) - u @ cov_dv_scaled @ u), 0.0)
    sig_dir_deg = float(np.degrees(np.sqrt(perp_var / 2.0) / mag)) if mag > 0 else float("nan")

    x_b_pre, x_b_post = sol_best["x_b_pre"], sol_best["x_b_post"]
    center = frame_center(x_b_pre, tb_best, cfg.center, cfg.moon_primary_radius_km)
    Rrtn = rtn_basis(x_b_pre, tb_best, center)
    Rvnb = vnb_basis(x_b_pre, tb_best, center)
    dv_rtn = Rrtn @ dv * 1e3
    dv_vnb = Rvnb @ dv * 1e3
    r_whitened = res.fun[:n_res]
    rms_sigma = float(np.sqrt(np.mean(r_whitened**2)))
    sig_mean = float(np.mean([m.sigma_rad for m in meas]))
    cls = classify_maneuver(x_b_pre, x_b_post, tb_best, dv_rtn, center, sigma_mag_mps=sig_mag * 1e3)
    return DvEstimate(
        t_burn_s=tb_best, t_burn_sigma_s=sig_t, dv_gcrf_kms=dv, cov_dv_kms2=cov_dv, cov_dv_scaled_kms2=cov_dv_scaled,
        magnitude_mps=mag * 1e3, magnitude_sigma_mps=sig_mag * 1e3, direction_gcrf=u, direction_sigma_deg=sig_dir_deg,
        frame_center=center, dv_rtn_mps=dv_rtn, dv_vnb_mps=dv_vnb,
        residual_rms_arcsec=rms_sigma * sig_mean / ARCSEC, residual_rms_sigma=rms_sigma, reduced_chi2=red_chi2,
        n_obs=len(meas), converged=bool(res.success), window=(t_lo, t_hi), grid=grid, classification=cls,
        x_pre_burn=x_b_pre, x_post_burn=x_b_post, dx_pre_correction=sol_best["dxp"],
        meta={"elapsed_s": time.perf_counter() - tic, "nfev": int(res.nfev), "status": int(res.status),
              "with_prior": L_inv is not None, "n_grid": n_grid, "refined": bool(cfg.refine)},
    )


# ---------------------------------------------------------------------------
def classify_maneuver(x_pre: np.ndarray, x_post: np.ndarray, t_s: float, dv_rtn_mps: np.ndarray, center: str,
                      sigma_mag_mps: float = 0.0) -> dict:
    """Labelled HEURISTIC classification of an impulsive maneuver.

    Compares the Jacobi constant (instantaneous rotating frame, CR3BP C = 2U − v²) before and
    after the burn with C(L1), C(L2); the Earth- and Moon-relative two-body specific energies
    ε = v²/2 − GM/r; the dominant RTN component; and the alignment of Δv with the directions to
    Earth, the Moon, L1 and L2.  Labels are descriptive only (no intent is inferred).
    """
    x_pre = np.asarray(x_pre, dtype=np.float64); x_post = np.asarray(x_post, dtype=np.float64)
    dv_rtn = np.asarray(dv_rtn_mps, dtype=np.float64)
    mag = float(np.linalg.norm(dv_rtn))
    labels: list[str] = []
    s_pre_rot = frames.gcrf_to_rot(x_pre, float(t_s))
    s_post_rot = frames.gcrf_to_rot(x_post, float(t_s))
    C_pre, C_post = float(jacobi(s_pre_rot)), float(jacobi(s_post_rot))
    L = lagrange_points()
    C_L1 = float(jacobi(np.concatenate([L[0], np.zeros(3)])))
    C_L2 = float(jacobi(np.concatenate([L[1], np.zeros(3)])))
    eph = get_ephemeris()
    sm = eph.moon_state(float(t_s))
    def energies(x):
        r_e, v_e = np.linalg.norm(x[:3]), np.linalg.norm(x[3:])
        rm, vm = x[:3] - sm[:3], x[3:] - sm[3:]
        return 0.5 * v_e**2 - GM_EARTH / r_e, 0.5 * np.linalg.norm(vm) ** 2 - GM_MOON / np.linalg.norm(rm)
    eE_pre, eM_pre = energies(x_pre)
    eE_post, eM_post = energies(x_post)

    if mag < max(0.1, 2.0 * sigma_mag_mps):
        primary = "no significant maneuver"
        labels.append("magnitude not significant at 2σ")
    else:
        R, T, N = dv_rtn / mag
        if abs(T) >= 0.7:
            primary = "in-track raise" if T > 0 else "in-track lower"
        elif abs(N) >= 0.7:
            primary = "plane change"
        elif abs(R) >= 0.7:
            primary = "radial (outward)" if R > 0 else "radial (inward)"
        else:
            primary = "mixed-direction burn"
        labels.append(f"dominant component in {center}-centred RTN: R={R:+.2f} T={T:+.2f} N={N:+.2f}")
    dC = C_post - C_pre
    if dC < 0:
        labels.append("Jacobi constant decreased (energy added in the rotating frame)")
    elif dC > 0:
        labels.append("Jacobi constant increased (energy removed in the rotating frame)")
    if C_pre >= C_L1 > C_post:
        labels.append("L1 gateway opened (C fell below C_L1)")
    if C_pre >= C_L2 > C_post:
        labels.append("L2 gateway opened (C fell below C_L2)")
    if C_post < C_L2:
        labels.append("post-burn energy allows Earth-Moon transit through both gateways")
    elif C_post < C_L1:
        labels.append("post-burn energy allows transit through the L1 gateway only")
    else:
        labels.append("post-burn energy keeps both gateways closed (bounded realm)")
    if eM_pre >= 0 > eM_post:
        labels.append("became Moon-bound (two-body energy w.r.t. Moon turned negative)")
    elif eM_pre < 0 <= eM_post:
        labels.append("left Moon-bound energy (escape w.r.t. Moon)")
    if eE_post > eE_pre:
        labels.append("Earth-relative energy increased (outward / apogee raise)")
    elif eE_post < eE_pre:
        labels.append("Earth-relative energy decreased (inward / apogee lower)")
    # alignment with named directions (GCRF)
    dv_g = x_post[3:] - x_pre[3:]
    align = {}
    if np.linalg.norm(dv_g) > 0:
        u = dv_g / np.linalg.norm(dv_g)
        targets = {"Earth": -x_pre[:3], "Moon": sm[:3] - x_pre[:3],
                   "L1": frames.rot_pos_to_gcrf(L[0], float(t_s)) - x_pre[:3],
                   "L2": frames.rot_pos_to_gcrf(L[1], float(t_s)) - x_pre[:3]}
        align = {k: float(u @ (v / np.linalg.norm(v))) for k, v in targets.items()}
        kbest = max(align, key=align.get)
        if align[kbest] > 0.5:
            labels.append(f"Δv direction consistent with departure toward {kbest} (cos={align[kbest]:.2f})")
    return {
        "primary": primary, "labels": labels, "heuristic": True,
        "note": "Descriptive heuristic only: compares Jacobi constant / two-body energies / RTN components; does not infer intent.",
        "jacobi_pre": C_pre, "jacobi_post": C_post, "jacobi_L1": C_L1, "jacobi_L2": C_L2,
        "energy_earth_pre_km2_s2": float(eE_pre), "energy_earth_post_km2_s2": float(eE_post),
        "energy_moon_pre_km2_s2": float(eM_pre), "energy_moon_post_km2_s2": float(eM_post),
        "alignment_cos": align, "frame_center": center,
    }
