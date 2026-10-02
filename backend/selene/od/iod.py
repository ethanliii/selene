"""Angles-only initial orbit determination adapted to cislunar (non-Keplerian) dynamics.

Classical angles-only IOD (Laplace, Gauss, Gooding) assumes two-body motion so that the
ranges can be eliminated algebraically.  In cislunar space the Moon's attraction is of the same
order as the Earth's, so we keep the *structure* of Gooding's method (R. H. Gooding, "A new
procedure for the solution of the classical problem of minimal orbit determination from three
lines of sight", Celest. Mech. Dyn. Astron. 66, 1997) — the unknowns are the two end ranges —
but replace the Lambert solver by a **shooting boundary-value solve under the full ephemeris
model** (Earth + Moon + Sun point masses, optional SRP).

Method
------
Measurements sorted in time, endpoints 1 and N (3), reference epoch = the middle measurement.

1. For a range pair (ρ₁, ρ₃): ``r₁ = R₁ + ρ₁L₁``, ``r₃ = R₃ + ρ₃L₃`` (observer positions R,
   unit lines of sight L).
2. Solve the two-point BVP for v₁ such that the ephemeris propagation of (r₁, v₁) reaches r₃ at
   t₃: Newton iteration ``v₁ ← v₁ − Φ_rv(t₃,t₁)⁻¹ (r(t₃) − r₃)`` using the integrated state
   transition matrix, warm-started from the "linear + mean gravity" guess
   ``v₁ ≈ (r₃−r₁)/Δt − ½ a(r_mid) Δt`` (or from the previous solve during refinement).
   Converges in 2–4 iterations for arcs up to a few days.
3. The interior measurements (all but the endpoints) give the residual vector
   ``f(ρ₁,ρ₃) = [ν_k/σ_k]`` with ν the on-sky residual [Δra·cos dec, Δdec].
4. Solve ``min ‖f‖²`` over (ρ₁, ρ₃) with ``scipy.optimize.least_squares`` from several seeds
   (log grid over 50 000–500 000 km), using the **analytic Jacobian**::

       ∂x_k/∂ρ₁ = Φ_k [ L₁ ; −Φ₃ᵣᵥ⁻¹ Φ₃ᵣᵣ L₁ ],   ∂x_k/∂ρ₃ = Φ_k [ 0 ; Φ₃ᵣᵥ⁻¹ L₃ ],   J_k = H_k ∂x_k/∂ρ /σ_k

   which follows from differentiating the BVP constraint r(t₃; r₁, v₁) = r₃.
5. *Admissible-region* pruning (Tommei, Milani & Rossi 2007; DeMars & Jah 2013, adapted to the
   three-body setting): a seed is rejected when its implied state is unbound with respect to
   **both** the Earth (v²/2 − GM_E/r > 0) and the Moon (|v−v_M|²/2 − GM_M/|r−r_M| > 0).  Using the
   Earth-only test would wrongly reject NRHO perilune states, which are instantaneously
   hyperbolic w.r.t. the Earth while bound to the Moon.
6. Candidates (one per converged seed, de-duplicated by root) are ranked admissible-first, then
   by RMS residual.  **Three observations are a square problem** (2 residuals, 2 unknowns): every
   root has RMS ≈ 0 and *no statistic in the data can order them* — exactly as the classical
   three-observation methods (Gauss, Gooding) can return several solutions.  The result therefore
   carries ``meta["ambiguous"]`` / ``meta["quality"]`` and the full candidate list; the caller
   must discriminate with further observations (the API route runs the batch fit from every root
   and keeps the lowest post-fit cost).  Measured on the 12 h two-ground-site arc at the demo
   epoch: SIM-DRO-01 gives one root; SIM-L2-HALO-01 gives two admissible roots (the wrong one
   47 500 km from the truth, Jacobi constants 3.085 vs 3.062 — a Jacobi "cislunar band" prior
   would not separate them either, so none is used); SIM-NRHO-RELAY-01 gives two.  For the square
   case the seed grid is densified (10×10 instead of 6×6) and up to 12 basins are refined so that
   narrow basins are not missed (cost ≈ 1–2 s).  Four or more observations remove the ambiguity
   in practice (the overdetermined RMS then discriminates).

   A single-observer three-observation arc can also be *ill-conditioned* (the interior residual is
   nearly insensitive to the ranges along a valley): the optimizer then stops without converging
   and ``quality = "not_converged"`` with a correspondingly huge covariance — the honest answer.

Covariance (first order, full linear propagation)
-------------------------------------------------
With the on-sky measurement noise δz = (δz₁, …, δz_N), σ_k² I₂ each, the converged estimate is a
function x_ref(ρ̂(z), z₁, z₃).  Differentiating the normal equations of step 4,
``δρ̂ = −(JᵀJ)⁻¹Jᵀ F_z δz`` where ``F_z = ∂f/∂z`` has ``+I₂/σ_k`` for an interior measurement and
``−H_k Φ_k M_i B_i/σ_k`` for an end point (B_i = ρ_i [ê_ra ê_dec] maps on-sky angles to the end-point
position, M₁ = [I; −Φ₃ᵣᵥ⁻¹Φ₃ᵣᵣ], M₃ = [0; Φ₃ᵣᵥ⁻¹]).  Then
``δx_ref = G δρ̂ + Φ_ref (M₁B₁ δz₁ + M₃B₃ δz₃)`` and ``P_ref = D Σ_z Dᵀ``.  This keeps the correlation
between the range solution and the end-point scatter that the earlier "rough" form neglected
(which was ≈ 2× over-confident in NEES on the DRO arc).  It is still a *linear* covariance of a
three-point solution; the batch fit over all observations is the product estimate.

Units: km, s, km/s, rad (σ in rad; RMS reported in arcsec).
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import least_squares

from selene.constants import GM_EARTH, GM_MOON, R_EARTH
from selene.dynamics.ephemeris import EphemParams, get_ephemeris, propagate_ephemeris
from selene.od.measurements import (ARCSEC, jacobian_state, los_tangent_basis, los_unit, predict, residual,
                                    sort_measurements)
from selene.od.stacked import StackedModel, ensure_cache
from selene.od.types import jsonable

__all__ = ["IODCandidate", "IODResult", "iod_two_range", "solve_bvp_velocity", "admissible"]


# ---------------------------------------------------------------------------
@dataclass
class IODCandidate:
    rho1_km: float
    rho3_km: float
    t_ref_s: float
    x_ref: np.ndarray            # (6,) GCRF km, km/s at the reference (middle) epoch
    P_ref: np.ndarray            # (6,6) rough covariance
    x1: np.ndarray               # (6,) state at t1
    rms_arcsec: float
    residuals: np.ndarray        # (m,2) interior on-sky residuals [rad]
    converged: bool
    admissible: bool
    energy_earth: float          # km²/s² at t1
    energy_moon: float
    nfev: int = 0
    seed: tuple = ()
    rank: int = 0

    def as_dict(self) -> dict:
        return jsonable({
            "rho1_km": self.rho1_km, "rho3_km": self.rho3_km, "t_ref_s": self.t_ref_s, "rank": self.rank,
            "x_ref": self.x_ref, "P_ref": self.P_ref, "x1": self.x1, "rms_arcsec": self.rms_arcsec,
            "residuals_arcsec": self.residuals / ARCSEC, "converged": self.converged,
            "admissible": self.admissible, "energy_earth_km2_s2": self.energy_earth,
            "energy_moon_km2_s2": self.energy_moon, "nfev": self.nfev, "seed_km": list(self.seed),
            "sigma_pos_km": float(np.sqrt(max(np.trace(self.P_ref[:3, :3]), 0.0))),
        })


@dataclass
class IODResult:
    candidates: list[IODCandidate]
    t1_s: float
    t3_s: float
    t_ref_s: float
    n_obs: int
    n_seeds: int
    n_rejected: int
    elapsed_s: float
    meta: dict = field(default_factory=dict)

    @property
    def best(self) -> IODCandidate | None:
        return self.candidates[0] if self.candidates else None

    def as_dict(self, max_candidates: int = 5) -> dict:
        b = self.best
        return jsonable({
            "method": "two-range shooting IOD under the ephemeris model (Gooding structure, Newton/STM BVP)",
            "n_obs": self.n_obs, "t1_s": self.t1_s, "t3_s": self.t3_s, "t_ref_s": self.t_ref_s,
            "span_h": (self.t3_s - self.t1_s) / 3600.0, "n_seeds": self.n_seeds, "n_rejected": self.n_rejected,
            "n_candidates": len(self.candidates), "elapsed_s": self.elapsed_s,
            "best": b.as_dict() if b else None,
            "candidates": [c.as_dict() for c in self.candidates[:max_candidates]],
            **self.meta,
        })


# ---------------------------------------------------------------------------
def admissible(x: np.ndarray, t_s: float) -> tuple[bool, float, float]:
    """(bound to Earth or Moon?, E_earth, E_moon) for GCRF state ``x`` at ``t_s``."""
    r, v = x[:3], x[3:]
    rn = np.linalg.norm(r)
    e_e = 0.5 * v @ v - GM_EARTH / rn
    sm = get_ephemeris().moon_state(t_s)
    dr = r - sm[:3]
    dv = v - sm[3:]
    e_m = 0.5 * dv @ dv - GM_MOON / np.linalg.norm(dr)
    return bool(e_e < 0.0 or e_m < 0.0) and rn > R_EARTH, float(e_e), float(e_m)


class _Arc:
    """Propagate (r1, v1) from t1 with the STM to all measurement epochs (ascending)."""

    def __init__(self, t_eval: np.ndarray, params: EphemParams):
        self.t_eval = t_eval
        self.t_unique, self.inverse = np.unique(t_eval, return_inverse=True)  # strictly monotone for solve_ivp
        self.params = params
        self.model = StackedModel(params)

    def __call__(self, r1, v1):
        s0 = np.concatenate([r1, v1])
        sol = propagate_ephemeris(s0, float(self.t_unique[0]), t_eval_s=self.t_unique, params=self.params,
                                  stm=True, rtol=1e-10, atol=1e-10)
        if not sol.success or sol.y.shape[1] != self.t_unique.size:
            return None, None
        Y = sol.y.T[self.inverse]
        return Y[:, :6], Y[:, 6:].reshape(-1, 6, 6)


def solve_bvp_velocity(arc: _Arc, r1: np.ndarray, r3: np.ndarray, v_guess: np.ndarray | None = None,
                       tol_km: float = 1e-6, max_iter: int = 12):
    """Newton/STM shooting for v1 with r(t3) = r3.  Returns (v1, states, Phis, converged, iters)."""
    t1, t3 = float(arc.t_eval[0]), float(arc.t_eval[-1])
    dt = t3 - t1
    if v_guess is None:
        r_mid = 0.5 * (r1 + r3)
        a_mid = arc.model.accel_one(0.5 * (t1 + t3), r_mid)
        v = (r3 - r1) / dt - 0.5 * a_mid * dt
    else:
        v = np.array(v_guess, dtype=np.float64)
    best = None
    for it in range(1, max_iter + 1):
        X, Phi = arc(r1, v)
        if X is None:
            return v, None, None, False, it
        res = X[-1, :3] - r3
        nrm = float(np.linalg.norm(res))
        if best is None or nrm < best[0]:
            best = (nrm, v.copy(), X, Phi)
        if nrm < tol_km:
            return v, X, Phi, True, it
        try:
            dv = np.linalg.solve(Phi[-1, :3, 3:], res)
        except np.linalg.LinAlgError:
            return v, X, Phi, False, it
        if not np.all(np.isfinite(dv)):
            return v, X, Phi, False, it
        v = v - dv
    nrm, v, X, Phi = best
    return v, X, Phi, nrm < 1e-3, max_iter


# ---------------------------------------------------------------------------
def iod_two_range(
    meas,
    params: EphemParams | None = None,
    rho_bounds_km: tuple[float, float] = (50_000.0, 500_000.0),
    n_grid: int | None = None,
    n_refine: int | None = None,
    max_nfev: int = 40,
    hard_bounds_km: tuple[float, float] = (6_500.0, 2_000_000.0),
    keep_inadmissible: bool = False,
) -> IODResult:
    """Two-range shooting IOD over ≥ 3 angles-only measurements (see module docstring).

    ``n_grid`` / ``n_refine`` default to 6 / 4 (overdetermined) and 10 / 12 (square, 3 obs).
    ``keep_inadmissible=True`` also refines seeds that fail the admissibility test (they are still
    ranked after the admissible ones); by default they are refined only when no seed is admissible.
    """
    t_start = time.perf_counter()
    ms = sort_measurements(meas)
    n = len(ms)
    if n < 3:
        raise ValueError("IOD needs at least 3 measurements")
    t = np.array([m.t_s for m in ms])
    if t[-1] - t[0] < 60.0:
        raise ValueError("IOD needs a time span of at least 60 s between the first and last measurement")
    R = np.array([m.observer_pos_gcrf for m in ms])
    L = los_unit([m.ra_rad for m in ms], [m.dec_rad for m in ms])
    sig = np.array([m.sigma_rad for m in ms])
    i_ref = n // 2
    interior = [k for k in range(1, n - 1)]
    square = len(interior) == 1
    n_grid = (10 if square else 6) if n_grid is None else int(n_grid)
    n_refine = (12 if square else 4) if n_refine is None else int(n_refine)
    params = ensure_cache(params, t[0], t[-1])
    arc = _Arc(t, params)

    state = {"v1": None, "nfev": 0}
    e_ra, e_dec = los_tangent_basis([m.ra_rad for m in ms], [m.dec_rad for m in ms])
    B = np.stack([e_ra, e_dec], axis=-1)      # (n,3,2): ∂L_k/∂(ra·cos dec, dec)

    def _evaluate(rho, v_warm):
        """-> dict with everything needed for residuals/jacobian, or None on failure."""
        rho1, rho3 = float(rho[0]), float(rho[1])
        r1 = R[0] + rho1 * L[0]
        r3 = R[-1] + rho3 * L[-1]
        v1, X, Phi, ok, _ = solve_bvp_velocity(arc, r1, r3, v_warm)
        state["nfev"] += 1
        if X is None or not ok:
            return None
        res = np.zeros((len(interior), 2))
        J = np.zeros((len(interior), 2, 2))
        Phi3 = Phi[-1]
        try:
            Pinv = np.linalg.inv(Phi3[:3, 3:])
        except np.linalg.LinAlgError:
            return None
        A1 = np.concatenate([L[0], -Pinv @ Phi3[:3, :3] @ L[0]])     # ∂x1/∂ρ1 (6,)
        A3 = np.concatenate([np.zeros(3), Pinv @ L[-1]])             # ∂x1/∂ρ3 (6,)
        M1 = np.vstack([np.eye(3), -Pinv @ Phi3[:3, :3]])            # ∂x1/∂r1 (6,3) at fixed r3
        M3 = np.vstack([np.zeros((3, 3)), Pinv])                     # ∂x1/∂r3 (6,3)
        Fz = np.zeros((len(interior), 2, n, 2))                      # ∂f_j/∂z_k (on-sky angles)
        for j, k in enumerate(interior):
            ra_p, dec_p = predict(X[k], R[k])
            res[j] = residual(ms[k].ra_rad, ms[k].dec_rad, ra_p, dec_p) / sig[k]
            H = jacobian_state(X[k], R[k])                           # (2,6)
            G = np.stack([Phi[k] @ A1, Phi[k] @ A3], axis=1)        # (6,2)
            J[j] = -H @ G / sig[k]          # residual = measured − predicted
            Fz[j, :, k, :] = np.eye(2) / sig[k]
            Fz[j, :, 0, :] = -H @ Phi[k] @ M1 @ (rho1 * B[0]) / sig[k]      # δr1 = ρ1 B1 δz1
            Fz[j, :, n - 1, :] = -H @ Phi[k] @ M3 @ (rho3 * B[-1]) / sig[k]  # δr3 = ρ3 B3 δz3
        return {"rho": (rho1, rho3), "r1": r1, "v1": v1, "X": X, "Phi": Phi, "res": res, "J": J,
                "A1": A1, "A3": A3, "Pinv": Pinv, "M1": M1, "M3": M3, "Fz": Fz.reshape(2 * len(interior), 2 * n)}

    # ---- seeds -------------------------------------------------------------
    grid = np.geomspace(rho_bounds_km[0], rho_bounds_km[1], n_grid)
    seeds = []
    n_rejected = 0
    for a in grid:
        for b in grid:
            ev = _evaluate((a, b), None)
            if ev is None:
                continue
            adm, e_e, e_m = admissible(np.concatenate([ev["r1"], ev["v1"]]), t[0])
            rms = float(np.sqrt(np.mean(ev["res"] ** 2))) if interior else 0.0
            seeds.append((rms, adm, a, b, ev))
            if not adm:
                n_rejected += 1
    adm_seeds = [s for s in seeds if s[1]]
    bad_seeds = [s for s in seeds if not s[1]]
    adm_seeds.sort(key=lambda s: s[0])
    bad_seeds.sort(key=lambda s: s[0])
    if keep_inadmissible:
        pool = adm_seeds + bad_seeds          # admissible basins first, then the rejected ones
    else:
        pool = adm_seeds or bad_seeds         # all seeds inadmissible: degrade gracefully, flag in meta
    log_step = float(np.log(grid[1] / grid[0])) if n_grid > 1 else 1.0
    near = 0.5 * log_step                     # seeds closer than half a grid step share a basin

    # ---- refinement --------------------------------------------------------
    lo, hi = hard_bounds_km
    cands: list[IODCandidate] = []
    cache = {}

    def fun(rho):
        key = (round(rho[0], 9), round(rho[1], 9))
        if key not in cache:
            cache.clear()
            cache[key] = _evaluate(rho, state["v1"])
            if cache[key] is not None:
                state["v1"] = cache[key]["v1"]
        ev = cache[key]
        if ev is None:
            return np.full(2 * len(interior), 1e3)
        return ev["res"].ravel()

    def jac(rho):
        key = (round(rho[0], 9), round(rho[1], 9))
        ev = cache.get(key) or _evaluate(rho, state["v1"])
        if ev is None:
            return np.zeros((2 * len(interior), 2))
        return ev["J"].reshape(-1, 2)

    used = []
    for rms0, adm0, a, b, ev0 in pool:
        if len(used) >= n_refine:
            break
        if any(abs(np.log(a / ua)) < near and abs(np.log(b / ub)) < near for ua, ub in used):
            continue  # neighbouring seed: converges to the same basin
        used.append((a, b))
        state["v1"] = ev0["v1"]
        cache.clear()
        x0 = np.array([a, b])
        if interior:
            try:
                sol = least_squares(fun, x0, jac=jac, bounds=(lo, hi), x_scale=1e4, max_nfev=max_nfev,
                                    ftol=1e-10, xtol=1e-10, gtol=1e-10, method="trf")
                rho = sol.x
                nfev = int(sol.nfev)
                converged = bool(sol.success)
            except Exception:  # noqa: BLE001 - keep the seed as a (non-converged) candidate
                rho, nfev, converged = x0, 0, False
        else:
            rho, nfev, converged = x0, 1, True
        ev = _evaluate(rho, state["v1"])
        if ev is None:
            continue
        x_ref = ev["X"][i_ref]
        x1 = np.concatenate([ev["r1"], ev["v1"]])
        adm, e_e, e_m = admissible(x1, t[0])
        res = ev["res"] * sig[interior][:, None] if interior else np.zeros((0, 2))
        rms = float(np.sqrt(np.mean(res ** 2)) / ARCSEC) if interior else 0.0
        # covariance: full first-order propagation of the on-sky noise (module docstring) -------
        J = ev["J"].reshape(-1, 2)
        Phi_ref = ev["Phi"][i_ref]
        G = np.stack([Phi_ref @ ev["A1"], Phi_ref @ ev["A3"]], axis=1)   # ∂x_ref/∂ρ (6,2)
        E = np.zeros((6, 2 * n))                                          # direct end-point terms
        E[:, 0:2] = Phi_ref @ ev["M1"] @ (rho[0] * B[0])
        E[:, 2 * (n - 1):2 * n] = Phi_ref @ ev["M3"] @ (rho[1] * B[-1])
        if J.shape[0] >= 2:
            JtJ = J.T @ J
            try:
                Jp = np.linalg.solve(JtJ, J.T)
            except np.linalg.LinAlgError:
                Jp = np.linalg.pinv(JtJ) @ J.T
            D = -G @ Jp @ ev["Fz"] + E                                    # ∂x_ref/∂z (6,2n)
        else:
            D = E
        Sz = np.repeat(sig ** 2, 2)
        P_ref = (D * Sz[None, :]) @ D.T
        P_ref = 0.5 * (P_ref + P_ref.T)
        cands.append(IODCandidate(float(rho[0]), float(rho[1]), float(t[i_ref]), x_ref.copy(), P_ref, x1, rms, res,
                                  converged, adm, e_e, e_m, nfev, (float(a), float(b))))

    # ---- de-duplicate & rank -------------------------------------------------
    uniq: list[IODCandidate] = []
    for c in sorted(cands, key=lambda c: (not c.admissible, not c.converged, c.rms_arcsec)):
        if any(abs(c.rho1_km - u.rho1_km) < 0.005 * u.rho1_km and abs(c.rho3_km - u.rho3_km) < 0.005 * u.rho3_km
               for u in uniq):
            continue
        c.rank = len(uniq)
        uniq.append(c)
    elapsed = time.perf_counter() - t_start
    n_roots = sum(1 for c in uniq if c.converged and c.admissible)
    if not uniq or not uniq[0].converged:
        quality, note = "not_converged", ("no seed converged to an exact root: the geometry is ill-conditioned "
                                          "(e.g. a single observer over a short arc) — do not use as a filter seed")
    elif square and n_roots > 1:
        quality, note = "ambiguous", (f"square 3-observation problem with {n_roots} exact admissible roots; the data "
                                      "cannot order them — discriminate with further observations (batch fit from each root)")
    elif square:
        quality, note = "ok", "square 3-observation problem, one admissible root found (others may exist outside the seed grid)"
    else:
        quality, note = "ok", "overdetermined: candidates ranked by RMS residual"
    meta = {
        "all_seeds_inadmissible": bool(seeds and not any(s[1] for s in seeds)),
        "bvp_evaluations": int(state["nfev"]),
        "interior_obs": len(interior),
        "square_problem": square,
        "n_grid": n_grid, "n_refined": len(used),
        "n_roots": n_roots,
        "ambiguous": bool(square and n_roots > 1),
        "quality": quality,
        "covariance": "first-order linear propagation of the on-sky noise through the three-point solution",
        "note": note,
    }
    return IODResult(uniq, float(t[0]), float(t[-1]), float(t[i_ref]), n, len(seeds), n_rejected, elapsed, meta)
