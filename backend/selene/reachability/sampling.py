"""Reachable-set sampling under the DE440s ephemeris model.

Problem
-------
Given an estimated GCRF state ``s0`` at ``t0`` and an assumed impulsive Δv budget, sample the set
of states the object could occupy over the next 24–168 h and classify the samples against the
named regions of :mod:`selene.reachability.regions`.  The output drives sensor re-tasking
("where to point to re-acquire") and the uncertainty-balloon visualisation.

Sampling design
---------------
A sample is an impulsive burn ``Δv = m · û`` applied at burn epoch ``t_b`` to the nominal
(coasting) trajectory, followed by a ballistic coast to the horizon:

* directions ``û``: ``n_dirs`` points of a Fibonacci sphere (near-uniform on S², Swinbank &
  Purser 2006 / González 2010) expressed in GCRF axes;
* magnitudes ``m``: a **fixed absolute ladder** (:data:`LADDER_MPS` = 2, 5, 10, 20, 50, 100, 200,
  500, 1000 m/s) truncated at the budget, plus the budget itself.  Because the ladder is absolute
  (not scaled to the budget) the sample set of a smaller budget is a *subset* of that of a larger
  budget whenever both lie on the ladder, so region reachability is monotone in the budget by
  construction; the ladder also spans the physically distinct cases "station-keeping-size tweak"
  (m/s) to "departure burn" (100s of m/s);
* burn epochs ``t_b``: ``burn_epochs_h`` after ``t0`` (default now, +6, +12, +24 h) — the burn may
  have happened at any time since the last observation.

The nominal (0 Δv) trajectory is carried as an extra row of the same system.

Propagation: one stacked ODE
----------------------------
All ``N`` samples are integrated as **one** first-order system of dimension ``6N`` with
``scipy.integrate.solve_ivp`` (DOP853) and a numba right-hand side vectorised over the rows.
Body positions come from a :class:`~selene.dynamics.ephemeris.BodyCache` (one spline evaluation
per RHS call, shared by every row).  Force model per row (identical to ``ephemeris.nbody_eom``)::

    r̈ = −GM_E r/|r|³ − Σ_j GM_j [ (r − r_j)/|r − r_j|³ + r_j/|r_j|³ ] + ν P☉ C_R (A/m) (AU/|r−r☉|)² (r−r☉)/|r−r☉|

Impulsive burns are applied as velocity discontinuities: the system is integrated piecewise
between consecutive distinct burn epochs and the rows whose burn falls at a segment boundary are
kicked there.  Rows that have not yet burned simply follow the nominal.

Two deliberate numerical devices (documented, tested):

1. **Softening inside the bodies.**  The shared step size would collapse if one row fell into the
   Moon.  Inside the Earth/Moon radii the point-mass denominators are clamped,
   ``|Δ|³ → max(|Δ|², R²)^{3/2}`` (the interior field of a uniform sphere), which keeps the
   acceleration bounded.  Rows that go below the surface are *flagged as impacts and masked
   out* from the surface crossing on (see :func:`detect_terminations`); outside the bodies the
   equations are untouched, so a row is exact physics right up to its own impact and free-flying
   rows agree with per-sample propagation to the integration tolerance (measured: 0.3 m over
   72 h with ~20 m/s kicks — the shipped test; ~4 m over 168 h with 150 m/s kicks that include
   lunar fly-bys; the RHS itself matches ``ephemeris.nbody_eom`` to roundoff).
2. **Error norm.**  scipy controls the RMS of the scaled error over all ``6N`` components, which
   dilutes the worst row by up to √(6N), and the absolute tolerance binds on the velocity
   components (|v| ~ 1 km/s).  The defaults rtol = 1e-11, atol = 1e-10 (km, km/s) were chosen by
   measurement: stacked rows agree with independent per-sample DOP853 runs to 0.3 m over 72 h
   (atol = 1e-8 gives 1.6 m); DOP853's cost grows only as tol^(−1/8) so the extra accuracy is
   cheap (2000 rows × 72 h ≈ 0.2 s, × 168 h ≈ 0.8 s on a laptop).  ``max_step`` = 30 min bounds
   the step through brief lunar fly-bys.

Impact detection
----------------
Two stages (see :func:`detect_terminations`):

1. *Candidate screening on the grid.*  At every grid epoch within 20 000 km of the Moon the
   selenocentric osculating conic is evaluated; an inbound row (ṙ < 0) whose periapsis radius
   ``r_p = p/(1+e)`` is below ``R_MOON`` is a candidate impactor (Earth's tidal term is < 3 % of
   lunar gravity inside 20 000 km, so the conic periapsis is accurate to a few km).  Earth
   re-entry candidates are screened the same way inside 60 000 km with ``r_p < R_EARTH + 100 km``.
2. *Exact crossing time by event propagation.*  Each candidate row is re-integrated **alone and
   unsoftened** (same force model and tolerances) from the last valid grid epoch with a terminal
   ``solve_ivp`` event on the surface radius.  The event time is the impact epoch ``term_t_s``;
   the row is valid (and credited to every region it traverses on the way down: LLO shell,
   south-pole approach, ...) at all grid epochs *before* it and masked from the first grid epoch
   at/after it.  A candidate whose event does not fire (the conic was wrong) is not terminated.
   The impact epoch carries integrator accuracy (≪ 1 s), not 1-h grid accuracy; the candidate
   screen is only an optimisation so that a handful of rows, not all N, pay for the event run.

Terminated rows are excluded from region statistics, envelopes and sensor hints after their
termination epoch; ``first_hit_h['lunar_impact']`` is the exact impact epoch.

Outputs
-------
:class:`ReachabilitySet` holds the full grid of states (``states_gcrf``/``states_rot`` of shape
``(N, T, 6)``, ``grid_dt_h`` default 1 h), the nominal, per-region hit statistics (ray fraction,
sample fraction, earliest arrival, minimum Δv among hitting samples), per-sample first-hit
epochs, an envelope time series (centroid, covariance ellipsoid, convex-hull volume, max radius
about the nominal) and timing.  ``fraction`` is the fraction of (direction, epoch) **rays** for
which some magnitude ≤ budget reaches the region — the quantity that is monotone in budget and
horizon; ``sample_fraction`` is the plain share of samples.
"""
from __future__ import annotations

import time as _time
from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np
from numba import njit
from scipy.integrate import solve_ivp

from selene.constants import AU_KM, GM_EARTH, GM_MOON, GM_SUN, L_STAR, P_SRP_1AU, R_EARTH, R_MOON, T_STAR
from selene.dynamics import frames
from selene.dynamics.ephemeris import BodyCache, EphemParams, _cyl_shadow, get_ephemeris
from selene.reachability.regions import Region, RegionInputs, default_regions

__all__ = [
    "LADDER_MPS",
    "DEFAULT_BURN_EPOCHS_H",
    "ReachabilityConfig",
    "ReachabilitySet",
    "RegionStats",
    "fibonacci_sphere",
    "magnitude_ladder",
    "build_samples",
    "propagate_stacked",
    "propagate_with_burns",
    "detect_terminations",
    "batch_gcrf_to_rot",
    "compute_reachability",
    "reachability_for_object",
]

LADDER_MPS: tuple[float, ...] = (2.0, 5.0, 10.0, 20.0, 50.0, 100.0, 200.0, 500.0, 1000.0)
DEFAULT_BURN_EPOCHS_H: tuple[float, ...] = (0.0, 6.0, 12.0, 24.0)
_SRP_SCALE = P_SRP_1AU * 1e-3  # N/m² × m²/kg -> km/s²
HOUR_S = 3600.0


# ---------------------------------------------------------------------------
# sampling helpers
# ---------------------------------------------------------------------------
def fibonacci_sphere(n: int) -> np.ndarray:
    """(n, 3) near-uniform unit vectors (golden-angle spiral); n = 1 returns +x."""
    n = int(n)
    if n <= 0:
        return np.zeros((0, 3))
    if n == 1:
        return np.array([[1.0, 0.0, 0.0]])
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    rho = np.sqrt(np.maximum(0.0, 1.0 - z * z))
    phi = np.pi * (3.0 - np.sqrt(5.0)) * np.arange(n)
    return np.stack([rho * np.cos(phi), rho * np.sin(phi), z], axis=1)


def magnitude_ladder(dv_budget_mps: float, ladder: Sequence[float] = LADDER_MPS) -> np.ndarray:
    """Ladder levels ≤ budget plus the budget itself (ascending, unique).  Budget 0 -> [0]."""
    b = float(dv_budget_mps)
    if b < 0:
        raise ValueError("dv_budget_mps must be >= 0")
    if b == 0.0:
        return np.array([0.0])
    levels = [float(m) for m in ladder if 0.0 < m < b * (1.0 - 1e-12)]
    levels.append(b)
    return np.array(sorted(set(levels)))


@dataclass
class ReachabilityConfig:
    """Sampling and integration settings (SI-derived units: m/s, h, km)."""

    dv_budget_mps: float = 50.0
    horizon_h: float = 72.0
    n_dirs: int = 64
    burn_epochs_h: tuple[float, ...] = DEFAULT_BURN_EPOCHS_H
    magnitudes_mps: Optional[Sequence[float]] = None   # override the ladder (breaks monotonicity guarantees)
    grid_dt_h: float = 1.0
    path_dt_h: float = 6.0
    srp: bool = True
    cr_area_mass: float = 0.0        # C_R·A/m [m²/kg]; 0 disables SRP
    rtol: float = 1e-11
    atol: float = 1e-10              # km / km/s (binds on the velocity components: 1e-10 km/s over 72 h ~ 3 cm)
    max_step_s: float = 1800.0
    soften: bool = True

    def __post_init__(self):
        if not (0.0 < self.horizon_h <= 24.0 * 30):
            raise ValueError("horizon_h must be in (0, 720]")
        eps = tuple(float(e) for e in self.burn_epochs_h)
        if not eps:
            raise ValueError("burn_epochs_h must not be empty")
        if any(e < 0 for e in eps) or any(e > self.horizon_h + 1e-9 for e in eps):
            raise ValueError("burn epochs must lie within [0, horizon_h]")
        self.burn_epochs_h = tuple(sorted(set(eps)))
        self.n_dirs = max(1, int(self.n_dirs))
        if self.grid_dt_h <= 0 or self.path_dt_h <= 0:
            raise ValueError("grid_dt_h and path_dt_h must be positive")
        if not np.isfinite(self.dv_budget_mps) or self.dv_budget_mps < 0:
            raise ValueError("dv_budget_mps must be finite and >= 0")
        if self.magnitudes_mps is not None:
            try:
                m = np.array(sorted(set(float(v) for v in self.magnitudes_mps)), dtype=float)
            except (TypeError, ValueError) as err:
                raise ValueError(f"magnitudes_mps must be a list of numbers: {err}")
            if m.size == 0:
                raise ValueError("magnitudes_mps must not be empty (omit it to use the default ladder)")
            if not np.isfinite(m).all() or (m < 0).any():
                raise ValueError("magnitudes_mps must be finite and >= 0")
            if (m > self.dv_budget_mps * (1.0 + 1e-9) + 1e-9).any():
                raise ValueError(f"magnitudes_mps must not exceed dv_budget_mps = {self.dv_budget_mps:g} m/s")
            self.magnitudes_mps = tuple(float(v) for v in m)

    def magnitudes(self) -> np.ndarray:
        """Burn magnitudes [m/s] actually sampled (validated in ``__post_init__``)."""
        if self.magnitudes_mps is not None:
            return np.asarray(self.magnitudes_mps, dtype=float)
        return magnitude_ladder(self.dv_budget_mps)

    def time_grid_s(self, t0_s: float) -> np.ndarray:
        n = int(np.floor(self.horizon_h / self.grid_dt_h + 1e-9))
        g = t0_s + self.grid_dt_h * HOUR_S * np.arange(n + 1)
        t_end = t0_s + self.horizon_h * HOUR_S
        if g[-1] < t_end - 1e-6:
            g = np.append(g, t_end)
        return g


def build_samples(cfg: ReachabilityConfig) -> dict:
    """Δv sample table: ``dirs`` (M,3) unit GCRF, ``dv_mps`` (M,), ``burn_h`` (M,), plus the
    ``ray`` index (direction × epoch) and ``mag_index`` of each sample.  Row order: epoch-major,
    then magnitude, then direction."""
    dirs = fibonacci_sphere(cfg.n_dirs)
    mags = cfg.magnitudes()
    eps = np.asarray(cfg.burn_epochs_h, float)
    E, K, D = len(eps), len(mags), len(dirs)
    e_idx, m_idx, d_idx = np.meshgrid(np.arange(E), np.arange(K), np.arange(D), indexing="ij")
    e_idx, m_idx, d_idx = e_idx.ravel(), m_idx.ravel(), d_idx.ravel()
    return {
        "dirs": dirs[d_idx],
        "dv_mps": mags[m_idx],
        "burn_h": eps[e_idx],
        "dir_index": d_idx,
        "mag_index": m_idx,
        "epoch_index": e_idx,
        "ray": e_idx * D + d_idx,
        "n_rays": E * D,
        "magnitudes": mags,
        "unit_dirs": dirs,
    }


# ---------------------------------------------------------------------------
# stacked right-hand side
# ---------------------------------------------------------------------------
@njit(cache=True)
def _stacked_rhs(y, n, r_moon, r_sun, gm_e, gm_m, gm_s, soft_e2, soft_m2, srp_k, use_srp, out):
    """dy/dt for n stacked rows of (x, y, z, vx, vy, vz) [km, km/s]; Earth-centred GCRF."""
    bm2 = r_moon[0] * r_moon[0] + r_moon[1] * r_moon[1] + r_moon[2] * r_moon[2]
    bm3 = bm2 * np.sqrt(bm2)
    bs2 = r_sun[0] * r_sun[0] + r_sun[1] * r_sun[1] + r_sun[2] * r_sun[2]
    bs3 = bs2 * np.sqrt(bs2)
    im0 = gm_m * r_moon[0] / bm3 + gm_s * r_sun[0] / bs3   # indirect terms (row independent)
    im1 = gm_m * r_moon[1] / bm3 + gm_s * r_sun[1] / bs3
    im2 = gm_m * r_moon[2] / bm3 + gm_s * r_sun[2] / bs3
    zero3 = np.zeros(3)
    r = np.empty(3)
    for i in range(n):
        k = 6 * i
        x = y[k]
        yy = y[k + 1]
        z = y[k + 2]
        out[k] = y[k + 3]
        out[k + 1] = y[k + 4]
        out[k + 2] = y[k + 5]
        rn2 = x * x + yy * yy + z * z
        if rn2 < soft_e2:
            rn2 = soft_e2
        rn3 = rn2 * np.sqrt(rn2)
        ax = -gm_e * x / rn3 - im0
        ay = -gm_e * yy / rn3 - im1
        az = -gm_e * z / rn3 - im2
        # Moon direct
        d0 = x - r_moon[0]
        d1 = yy - r_moon[1]
        d2 = z - r_moon[2]
        dn2 = d0 * d0 + d1 * d1 + d2 * d2
        if dn2 < soft_m2:
            dn2 = soft_m2
        dn3 = dn2 * np.sqrt(dn2)
        ax -= gm_m * d0 / dn3
        ay -= gm_m * d1 / dn3
        az -= gm_m * d2 / dn3
        # Sun direct
        s0 = x - r_sun[0]
        s1 = yy - r_sun[1]
        s2 = z - r_sun[2]
        sn2 = s0 * s0 + s1 * s1 + s2 * s2
        sn = np.sqrt(sn2)
        sn3 = sn2 * sn
        ax -= gm_s * s0 / sn3
        ay -= gm_s * s1 / sn3
        az -= gm_s * s2 / sn3
        if use_srp:
            r[0] = x
            r[1] = yy
            r[2] = z
            nu = _cyl_shadow(r, zero3, R_EARTH, r_sun) * _cyl_shadow(r, r_moon, R_MOON, r_sun)
            if nu > 0.0:
                f = srp_k * (AU_KM / sn) ** 2 / sn
                ax += f * s0
                ay += f * s1
                az += f * s2
        out[k + 3] = ax
        out[k + 4] = ay
        out[k + 5] = az


def _make_rhs(n: int, cache: BodyCache, params: EphemParams, soften: bool):
    use_srp = bool(params.srp and params.cr_area_mass > 0.0)
    srp_k = _SRP_SCALE * float(params.cr_area_mass)
    soft_e2 = R_EARTH**2 if soften else 0.0
    soft_m2 = R_MOON**2 if soften else 0.0
    if set(params.bodies) != {"moon", "sun"}:
        raise ValueError("the stacked propagator implements the Earth+Moon+Sun model (bodies=('moon','sun'))")

    def rhs(t, y):
        r_moon = cache.position("moon", t)
        r_sun = cache.position("sun", t)
        out = np.empty_like(y)
        _stacked_rhs(y, n, r_moon, r_sun, GM_EARTH, GM_MOON, GM_SUN, soft_e2, soft_m2, srp_k, use_srp, out)
        return out

    return rhs


def propagate_stacked(
    states0: np.ndarray,
    t0_s: float,
    t_eval_s: np.ndarray,
    params: Optional[EphemParams] = None,
    rtol: float = 1e-11,
    atol: float = 1e-10,
    max_step_s: float = 1800.0,
    soften: bool = True,
    cache: Optional[BodyCache] = None,
) -> tuple[np.ndarray, dict]:
    """Integrate ``N`` GCRF rows ``states0`` (N,6) from ``t0_s`` to ascending ``t_eval_s`` (T,)
    as one 6N system.  Returns ``(states (N,T,6), info)``; ``t_eval_s[0]`` may equal ``t0_s``."""
    s = np.asarray(states0, dtype=np.float64).reshape(-1, 6)
    n = s.shape[0]
    t_eval = np.atleast_1d(np.asarray(t_eval_s, dtype=np.float64))
    params = params or EphemParams()
    t_end = float(t_eval[-1])
    if cache is None:
        cache = params.cache if (params.cache is not None and params.cache.covers(t0_s) and params.cache.covers(t_end)) \
            else BodyCache(min(t0_s, t_end), max(t0_s, t_end))
    if t_eval[0] > t0_s + 1e-9:
        t_eval_i = np.concatenate([[t0_s], t_eval])
        drop = 1
    else:
        t_eval_i, drop = t_eval, 0
    if abs(t_end - t0_s) < 1e-9:
        out = np.repeat(s[:, None, :], len(t_eval), axis=1)
        return out, {"nfev": 0, "n_steps": 0}
    rhs = _make_rhs(n, cache, params, soften)
    sol = solve_ivp(rhs, (float(t0_s), t_end), s.ravel(), method="DOP853", t_eval=t_eval_i,
                    rtol=rtol, atol=atol, max_step=max_step_s)
    if not sol.success:
        raise RuntimeError(f"stacked propagation failed: {sol.message}")
    y = sol.y[:, drop:]                      # (6N, T)
    states = y.T.reshape(len(t_eval), n, 6).transpose(1, 0, 2).copy()
    return states, {"nfev": int(sol.nfev), "n_steps_approx": int(sol.nfev // 12)}


def propagate_with_burns(
    s0_gcrf: np.ndarray,
    t0_s: float,
    dv_kms: np.ndarray,
    burn_t_s: np.ndarray,
    t_grid_s: np.ndarray,
    params: Optional[EphemParams] = None,
    rtol: float = 1e-11,
    atol: float = 1e-10,
    max_step_s: float = 1800.0,
    soften: bool = True,
) -> tuple[np.ndarray, dict]:
    """All rows start from ``s0_gcrf`` at ``t0_s``; row ``i`` receives the impulsive GCRF
    velocity increment ``dv_kms[i]`` at ``burn_t_s[i]`` (>= t0).  Returns the states of every
    row on ``t_grid_s`` (ascending, ``t_grid_s[0] == t0_s``) as ``(N, T, 6)`` plus info.

    Integrated piecewise as one stacked system between consecutive distinct burn epochs; the
    grid value at a burn epoch is the pre-burn state (same position)."""
    s0 = np.asarray(s0_gcrf, dtype=np.float64).ravel()[:6]
    dv = np.asarray(dv_kms, dtype=np.float64).reshape(-1, 3)
    tb = np.asarray(burn_t_s, dtype=np.float64).ravel()
    grid = np.asarray(t_grid_s, dtype=np.float64)
    n = dv.shape[0]
    if tb.shape[0] != n:
        raise ValueError("dv_kms and burn_t_s must have the same length")
    if abs(grid[0] - t0_s) > 1e-6 or np.any(np.diff(grid) <= 0):
        raise ValueError("t_grid_s must be ascending and start at t0_s")
    if (tb < t0_s - 1e-6).any() or (tb > grid[-1] + 1e-6).any():
        raise ValueError("burn epochs must lie within [t0, t_end]")
    params = params or EphemParams()
    cache = BodyCache(t0_s, grid[-1])
    out = np.empty((n, len(grid), 6))
    out[:, 0, :] = s0
    cur = np.repeat(s0[None, :], n, axis=0)
    t_cur = float(t0_s)
    bounds = sorted(set([float(t0_s)] + [float(v) for v in np.unique(tb)] + [float(grid[-1])]))
    nfev = 0
    for k in range(len(bounds)):
        ta = bounds[k]
        kick = np.abs(tb - ta) <= 1e-6
        if kick.any():
            cur[kick, 3:] += dv[kick]
        if k == len(bounds) - 1:
            break
        tb_seg = bounds[k + 1]
        inside = (grid > ta + 1e-6) & (grid <= tb_seg + 1e-6)
        t_eval = np.concatenate([[ta], grid[inside]])
        if t_eval[-1] < tb_seg - 1e-6:
            t_eval = np.append(t_eval, tb_seg)
        st, info = propagate_stacked(cur, ta, t_eval, params, rtol, atol, max_step_s, soften, cache)
        nfev += info["nfev"]
        idx = np.where(inside)[0]
        out[:, idx, :] = st[:, 1:1 + len(idx), :]
        cur = st[:, -1, :].copy()
        t_cur = tb_seg
    return out, {"nfev": nfev, "n_segments": len(bounds) - 1, "t_end_s": t_cur}


# ---------------------------------------------------------------------------
# frames and terminations
# ---------------------------------------------------------------------------
def batch_gcrf_to_rot(states_gcrf: np.ndarray, fr: frames.FrameInfo) -> np.ndarray:
    """(N,T,6) GCRF states -> (N,T,6) rotating-frame nd states using a :class:`FrameInfo`
    evaluated on the T grid epochs (same formulas as :func:`frames.gcrf_to_rot`)."""
    s = np.asarray(states_gcrf, dtype=np.float64)
    r_rel = s[..., :3] - fr.r_bary_gcrf[None]
    v_rel = s[..., 3:6] - fr.v_bary_gcrf[None]
    d = np.asarray(fr.d_km)[None, :, None]
    r_rot = np.einsum("tji,ntj->nti", fr.R, r_rel) / d
    w_x_r = np.cross(fr.omega_vec[None], r_rel)
    scale = (np.asarray(fr.ddot_km_s) / np.asarray(fr.d_km))[None, :, None]
    v_rot = T_STAR * (np.einsum("tji,ntj->nti", fr.R, v_rel - w_x_r) / d - r_rot * scale)
    return np.concatenate([r_rot, v_rot], axis=-1)


def _osculating_periapsis(rel_pos: np.ndarray, rel_vel: np.ndarray, gm: float) -> np.ndarray:
    """Two-body periapsis radius r_p = p/(1+e) for relative states (..., 3)."""
    r = np.linalg.norm(rel_pos, axis=-1)
    h = np.cross(rel_pos, rel_vel)
    h2 = np.sum(h * h, axis=-1)
    with np.errstate(invalid="ignore", divide="ignore"):
        p = h2 / gm
        e_vec = (np.cross(rel_vel, h) / gm) - rel_pos / r[..., None]
        e = np.linalg.norm(e_vec, axis=-1)
        rp = p / (1.0 + e)
    return np.where(np.isfinite(rp), rp, 0.0)


_REENTRY_RADIUS_KM = R_EARTH + 100.0
_LUNAR_SCREEN_KM = 20_000.0
_EARTH_SCREEN_KM = 60_000.0


def _surface_crossing_time(state: np.ndarray, t_start: float, t_stop: float, body: str, radius_km: float,
                           params: EphemParams, cache: BodyCache, rtol: float, atol: float, max_step_s: float) -> Optional[float]:
    """Exact epoch [TDB s] at which the single row ``state`` (GCRF at ``t_start``) first descends
    through ``radius_km`` about ``body`` ('moon' | 'earth'), integrating the *unsoftened* force
    model with a terminal ``solve_ivp`` event; ``None`` if it does not within ``t_stop``."""
    rhs = _make_rhs(1, cache, params, soften=False)
    if body == "moon":
        def ev(t, y):
            rm = cache.position("moon", t)
            return np.sqrt((y[0] - rm[0]) ** 2 + (y[1] - rm[1]) ** 2 + (y[2] - rm[2]) ** 2) - radius_km
    else:
        def ev(t, y):
            return np.sqrt(y[0] ** 2 + y[1] ** 2 + y[2] ** 2) - radius_km
    ev.terminal = True
    ev.direction = -1
    if ev(t_start, state) <= 0.0:           # already inside at the start epoch
        return float(t_start)
    sol = solve_ivp(rhs, (float(t_start), float(t_stop)), np.asarray(state, float).ravel()[:6], method="DOP853",
                    rtol=rtol, atol=atol, max_step=max_step_s, events=ev)
    if sol.t_events and len(sol.t_events[0]):
        return float(sol.t_events[0][0])
    return None


def _first_crossing(cand: np.ndarray, dist: np.ndarray, radius_km: float, states: np.ndarray, t_grid: np.ndarray,
                    body: str, params, cache, rtol, atol, max_step_s, window_s: float) -> Optional[float]:
    """Walk the candidate flags of one row (cand (T,) bool from the conic screen, dist (T,) km to
    the body centre) and return the exact crossing epoch found by event propagation, or None."""
    T = len(t_grid)
    j_start = 0
    while True:
        idx = np.where(cand[j_start:])[0]
        if idx.size == 0:
            return None
        j = int(idx[0]) + j_start
        j0 = j if dist[j] > radius_km else max(j - 1, 0)        # start from a state still above the surface
        t_stop = min(float(t_grid[j0]) + window_s, float(t_grid[-1]))
        t_x = _surface_crossing_time(states[j0], float(t_grid[j0]), t_stop, body, radius_km,
                                     params, cache, rtol, atol, max_step_s)
        if t_x is not None:
            return t_x
        nxt = int(np.searchsorted(t_grid, t_stop, side="right"))
        if nxt <= j or nxt >= T:
            return None
        j_start = nxt


def detect_terminations(states_gcrf: np.ndarray, moon_gcrf: np.ndarray, t_grid_s: np.ndarray,
                        params: Optional[EphemParams] = None, cache: Optional[BodyCache] = None,
                        rtol: float = 1e-11, atol: float = 1e-10, max_step_s: float = 1800.0,
                        window_h: float = 36.0) -> tuple[np.ndarray, list, np.ndarray]:
    """Per-row termination: ``(term_idx, reasons, term_t_s)``.

    * ``term_idx`` (N,) first grid index at/after which the row is invalid (``T`` if never);
    * ``reasons`` list of 'lunar_impact' | 'earth_reentry' | None;
    * ``term_t_s`` (N,) exact surface-crossing epoch [TDB s] from the event propagation (NaN if none).

    Stage 1 screens candidates with the osculating conic on the grid; stage 2 re-integrates each
    candidate row alone (unsoftened, same force model/tolerances, ``window_h`` after the first
    flag) with a terminal event on the surface radius.  See the module docstring.
    """
    s = states_gcrf
    n, T = s.shape[0], s.shape[1]
    t_grid = np.asarray(t_grid_s, float)
    term_idx = np.full(n, T, dtype=int)
    term_t = np.full(n, np.nan)
    reasons: list = [None] * n
    rel = s[..., :3] - moon_gcrf[None]          # Moon-centred position
    moon_vel = get_ephemeris().state("moon", t_grid)[:, 3:]   # DE440s Moon velocity for the selenocentric conic
    rel_v = s[..., 3:6] - moon_vel[None]
    dist_m = np.linalg.norm(rel, axis=-1)
    inbound = np.sum(rel * rel_v, axis=-1) < 0.0
    near = dist_m < _LUNAR_SCREEN_KM
    rp = np.where(near, _osculating_periapsis(rel, rel_v, GM_MOON), np.inf)
    lunar = (dist_m < R_MOON) | (near & inbound & (rp < R_MOON))
    r_e = np.linalg.norm(s[..., :3], axis=-1)
    near_e = r_e < _EARTH_SCREEN_KM
    inbound_e = np.sum(s[..., :3] * s[..., 3:6], axis=-1) < 0.0
    rp_e = np.where(near_e, _osculating_periapsis(s[..., :3], s[..., 3:6], GM_EARTH), np.inf)
    earth = (r_e < _REENTRY_RADIUS_KM) | (near_e & inbound_e & (rp_e < _REENTRY_RADIUS_KM))
    rows = np.where(lunar.any(axis=1) | earth.any(axis=1))[0]
    if rows.size == 0:
        return term_idx, reasons, term_t
    params = params or EphemParams()
    if cache is None or not (cache.covers(t_grid[0]) and cache.covers(t_grid[-1])):
        cache = BodyCache(float(t_grid[0]), float(t_grid[-1]))
    window_s = window_h * HOUR_S
    for i in rows:
        t_l = _first_crossing(lunar[i], dist_m[i], R_MOON, s[i], t_grid, "moon", params, cache, rtol, atol, max_step_s, window_s) \
            if lunar[i].any() else None
        t_e = _first_crossing(earth[i], r_e[i], _REENTRY_RADIUS_KM, s[i], t_grid, "earth", params, cache, rtol, atol, max_step_s, window_s) \
            if earth[i].any() else None
        best = None
        if t_l is not None:
            best = (t_l, "lunar_impact")
        if t_e is not None and (best is None or t_e < best[0]):
            best = (t_e, "earth_reentry")
        if best is None:
            continue
        t_x, why = best
        # the row is valid at grid epochs strictly before the crossing; invalid from the first epoch at/after it
        k = int(np.searchsorted(t_grid, t_x - 1e-6, side="left"))
        term_idx[i], reasons[i], term_t[i] = min(k, T), why, t_x
    return term_idx, reasons, term_t


# ---------------------------------------------------------------------------
# result containers
# ---------------------------------------------------------------------------
@dataclass
class RegionStats:
    key: str
    name: str
    kind: str
    fraction: float             # fraction of (direction, epoch) rays reaching the region with some Δv <= budget
    sample_fraction: float      # fraction of individual samples reaching the region
    n_hit: int
    earliest_h: Optional[float]
    min_dv_mps: Optional[float]  # 0.0 when the 0-Δv nominal already enters the region
    nominal_hits: bool
    newly_reachable: bool       # reached by some sample but not by the nominal path
    earliest_nominal_h: Optional[float] = None
    best_sample: Optional[int] = None    # index of the hitting sample with the smallest Δv (ties: earliest)

    def as_dict(self) -> dict:
        return dict(self.__dict__)


@dataclass
class ReachabilitySet:
    object_id: str
    t0_s: float
    config: ReachabilityConfig
    t_grid_s: np.ndarray                 # (T,)
    nominal_gcrf: np.ndarray             # (T,6)
    nominal_rot: np.ndarray              # (T,6)
    states_gcrf: np.ndarray              # (N,T,6)
    states_rot: np.ndarray               # (N,T,6)
    dv_dirs: np.ndarray                  # (N,3) unit GCRF
    dv_mps: np.ndarray                   # (N,)
    burn_h: np.ndarray                   # (N,)
    ray: np.ndarray                      # (N,) ray index
    n_rays: int
    term_idx: np.ndarray                 # (N,) first invalid grid index (T if never terminated)
    term_reason: list
    hits: dict                           # key -> (N,T) bool (post-termination masked)
    first_hit_h: dict                    # key -> (N,) float (nan if never)
    nominal_hits: dict                   # key -> (T,) bool
    region_stats: list                   # [RegionStats]
    regions: list                        # [Region]
    envelope: list                       # per path step dicts
    moon_gcrf: np.ndarray                # (T,3)
    frame: frames.FrameInfo
    timing: dict = field(default_factory=dict)
    meta: dict = field(default_factory=dict)
    term_t_s: Optional[np.ndarray] = None   # (N,) exact impact / re-entry epoch [TDB s] (NaN if none)

    # -- convenience ---------------------------------------------------------
    @property
    def term_t_h(self) -> np.ndarray:
        """(N,) exact termination time [h after t0] (NaN if the row is never terminated)."""
        if self.term_t_s is None:
            return np.full(self.n_samples, np.nan)
        return (self.term_t_s - self.t0_s) / HOUR_S

    @property
    def n_samples(self) -> int:
        return int(self.states_gcrf.shape[0])

    @property
    def t_h(self) -> np.ndarray:
        return (self.t_grid_s - self.t0_s) / HOUR_S

    @property
    def active(self) -> np.ndarray:
        """(N,T) bool: row still valid (before termination)."""
        T = len(self.t_grid_s)
        return np.arange(T)[None, :] < self.term_idx[:, None]

    def path_indices(self) -> np.ndarray:
        th = self.t_h
        step = self.config.path_dt_h
        idx = [i for i, h in enumerate(th) if abs(h / step - round(h / step)) < 1e-6]
        if idx[-1] != len(th) - 1:
            idx.append(len(th) - 1)
        return np.asarray(idx, dtype=int)

    def endpoints(self) -> tuple[np.ndarray, np.ndarray]:
        """(N,3) GCRF km and (N,3) rot nd positions at the last valid epoch of each row."""
        last = np.clip(self.term_idx - 1, 0, len(self.t_grid_s) - 1)
        rows = np.arange(self.n_samples)
        return self.states_gcrf[rows, last, :3], self.states_rot[rows, last, :3]

    def regions_hit_by_sample(self) -> list[list[str]]:
        keys = list(self.hits)
        out = []
        for i in range(self.n_samples):
            ks = [(self.first_hit_h[k][i], k) for k in keys if np.isfinite(self.first_hit_h[k][i])]
            out.append([k for _, k in sorted(ks)])
        return out

    def stats_by_key(self) -> dict[str, RegionStats]:
        return {s.key: s for s in self.region_stats}


# ---------------------------------------------------------------------------
# core
# ---------------------------------------------------------------------------
def _envelope(rs_states_gcrf, rs_states_rot, nominal_gcrf, active, t_h, path_idx) -> list[dict]:
    from scipy.spatial import ConvexHull, QhullError

    env = []
    for j in path_idx:
        ok = active[:, j]
        pts = rs_states_gcrf[ok, j, :3]
        pts_rot = rs_states_rot[ok, j, :3]
        n = int(ok.sum())
        entry = {"t_h": float(t_h[j]), "n_active": n, "n_terminated": int((~ok).sum())}
        if n == 0:
            env.append(entry)
            continue
        c = pts.mean(axis=0)
        entry["centroid_gcrf_km"] = c.tolist()
        entry["centroid_rot_nd"] = pts_rot.mean(axis=0).tolist()
        entry["nominal_gcrf_km"] = nominal_gcrf[j, :3].tolist()
        dev = pts - nominal_gcrf[j, :3]
        entry["max_radius_km"] = float(np.max(np.linalg.norm(dev, axis=1)))
        entry["rms_radius_km"] = float(np.sqrt(np.mean(np.sum(dev**2, axis=1))))
        if n >= 2:
            cov = np.cov(pts.T) if n > 2 else np.zeros((3, 3))
            w, V = np.linalg.eigh(cov)
            w = np.clip(w, 0.0, None)
            entry["cov_gcrf_km2"] = cov.tolist()
            entry["semi_axes_km"] = np.sqrt(w)[::-1].tolist()
            entry["axes_gcrf"] = V[:, ::-1].T.tolist()
            entry["ellipsoid_volume_km3"] = float(4.0 / 3.0 * np.pi * np.prod(np.sqrt(w)))
        if n >= 5:
            try:
                hull = ConvexHull(pts)
                entry["hull_volume_km3"] = float(hull.volume)
                entry["hull_area_km2"] = float(hull.area)
            except (QhullError, ValueError):
                entry["hull_volume_km3"] = 0.0
        env.append(entry)
    return env


def compute_reachability(
    s0_gcrf: np.ndarray,
    t0_s: float,
    cfg: ReachabilityConfig,
    object_id: str = "custom",
    regions: Optional[list[Region]] = None,
    samples: Optional[dict] = None,
) -> ReachabilitySet:
    """Full pipeline: sample Δv's, propagate the stacked system, classify, summarise.

    ``samples`` may override :func:`build_samples` (same keys) to evaluate explicit burns.
    """
    tic = _time.perf_counter()
    timing: dict = {}
    s0 = np.asarray(s0_gcrf, dtype=np.float64).ravel()[:6]
    smp = build_samples(cfg) if samples is None else samples
    dirs = np.asarray(smp["dirs"], float).reshape(-1, 3)
    dv_mps = np.asarray(smp["dv_mps"], float).ravel()
    burn_h = np.asarray(smp["burn_h"], float).ravel()
    ray = np.asarray(smp["ray"], int).ravel()
    n_rays = int(smp["n_rays"])
    N = dirs.shape[0]
    grid = cfg.time_grid_s(t0_s)
    T = len(grid)

    # --- propagate: row 0 = nominal, rows 1.. = samples ----------------------------------
    params = EphemParams(srp=cfg.srp and cfg.cr_area_mass > 0, cr_area_mass=cfg.cr_area_mass)
    dv_all = np.vstack([np.zeros((1, 3)), dirs * (dv_mps[:, None] * 1e-3)])
    tb_all = np.concatenate([[t0_s], t0_s + burn_h * HOUR_S])
    t1 = _time.perf_counter()
    st_all, pinfo = propagate_with_burns(s0, t0_s, dv_all, tb_all, grid, params, cfg.rtol, cfg.atol,
                                         cfg.max_step_s, cfg.soften)
    timing["propagate_s"] = _time.perf_counter() - t1
    nominal_gcrf = st_all[0]
    states_gcrf = st_all[1:]

    # --- frames ---------------------------------------------------------------------------
    t1 = _time.perf_counter()
    fr = frames.rotating_frame(grid)
    eph = get_ephemeris()
    moon_gcrf = eph.position("moon", grid)
    states_rot = batch_gcrf_to_rot(states_gcrf, fr)
    nominal_rot = batch_gcrf_to_rot(nominal_gcrf[None], fr)[0]
    timing["frames_s"] = _time.perf_counter() - t1

    # --- terminations ---------------------------------------------------------------------
    t1 = _time.perf_counter()
    term_cache = BodyCache(float(grid[0]), float(grid[-1]))
    term_idx, term_reason, term_t = detect_terminations(states_gcrf, moon_gcrf, grid, params, term_cache,
                                                        cfg.rtol, cfg.atol, cfg.max_step_s)
    nom_term, nom_reason, nom_term_t = detect_terminations(nominal_gcrf[None], moon_gcrf, grid, params, term_cache,
                                                           cfg.rtol, cfg.atol, cfg.max_step_s)
    active = np.arange(T)[None, :] < term_idx[:, None]
    nom_active = np.arange(T) < nom_term[0]
    timing["terminations_s"] = _time.perf_counter() - t1

    # --- classify -------------------------------------------------------------------------
    t1 = _time.perf_counter()
    regions = default_regions() if regions is None else regions
    # ω(t)·T*: rescales the T*-nondimensionalised rotating velocities to the instantaneous frame
    # rate so the CR3BP Jacobi classifier does not inherit the ±5-8 % T*·ω wobble (regions.py)
    w_tstar = np.asarray(fr.omega) * T_STAR                                  # (T,)
    inp = RegionInputs(states_rot[..., :3], states_gcrf[..., :3], states_gcrf[..., :3] - moon_gcrf[None],
                       states_rot[..., 3:6], states_gcrf[..., 3:6], omega_t_star=np.broadcast_to(w_tstar[None, :], (N, T)))
    inp_n = RegionInputs(nominal_rot[:, :3], nominal_gcrf[:, :3], nominal_gcrf[:, :3] - moon_gcrf,
                         nominal_rot[:, 3:6], nominal_gcrf[:, 3:6], omega_t_star=w_tstar)
    t_h = (grid - t0_s) / HOUR_S
    hits: dict = {}
    first_hit: dict = {}
    nominal_hits: dict = {}
    stats: list[RegionStats] = []
    for reg in regions:
        m = reg.contains(inp)
        # impacts: the terminating epoch itself counts as the hit, later epochs are masked
        if reg.key == "lunar_impact":
            m = m | ((np.arange(T)[None, :] == term_idx[:, None]) & np.array([r == "lunar_impact" for r in term_reason])[:, None])
            m &= np.arange(T)[None, :] <= term_idx[:, None]
        else:
            m &= active
        mn = reg.contains(inp_n)
        if reg.key == "lunar_impact":
            mn = mn | ((np.arange(T) == nom_term[0]) & (nom_reason[0] == "lunar_impact"))
            mn &= np.arange(T) <= nom_term[0]
        else:
            mn &= nom_active
        hits[reg.key] = m
        nominal_hits[reg.key] = mn
        any_hit = m.any(axis=1)
        fh = np.full(N, np.nan)
        if any_hit.any():
            fh[any_hit] = t_h[np.argmax(m[any_hit], axis=1)]
        if reg.key == "lunar_impact":
            # the exact surface-crossing epoch from the event propagation, not the grid epoch
            imp = np.array([r == "lunar_impact" for r in term_reason]) & np.isfinite(term_t)
            fh[imp] = (term_t[imp] - t0_s) / HOUR_S
        first_hit[reg.key] = fh
        ray_hit = np.zeros(n_rays, dtype=bool)
        if any_hit.any():
            ray_hit[np.unique(ray[any_hit])] = True
        n_hit = int(any_hit.sum())
        best = None
        min_dv = None
        earliest = None
        nom_any = bool(mn.any())
        if reg.key == "lunar_impact" and nom_reason[0] == "lunar_impact" and np.isfinite(nom_term_t[0]):
            earliest_nom = float((nom_term_t[0] - t0_s) / HOUR_S)
        else:
            earliest_nom = float(t_h[np.argmax(mn)]) if nom_any else None
        if n_hit:
            cand = np.where(any_hit)[0]
            order = np.lexsort((fh[cand], dv_mps[cand]))
            best = int(cand[order[0]])
            # the 0-Δv nominal already enters the region -> the minimum Δv is 0 by definition
            min_dv = 0.0 if nom_any else float(dv_mps[best])
            earliest = float(np.nanmin(fh))
            if nom_any and earliest_nom is not None:
                earliest = min(earliest, earliest_nom)
        elif nom_any:
            min_dv = 0.0
        stats.append(RegionStats(
            key=reg.key, name=reg.name, kind=reg.kind,
            fraction=float(ray_hit.mean()) if n_rays else 0.0,
            sample_fraction=float(any_hit.mean()) if N else 0.0,
            n_hit=n_hit, earliest_h=earliest, min_dv_mps=min_dv, nominal_hits=nom_any,
            newly_reachable=bool(n_hit > 0 and not nom_any),
            earliest_nominal_h=earliest_nom,
            best_sample=best,
        ))
    timing["classify_s"] = _time.perf_counter() - t1

    # --- envelope -------------------------------------------------------------------------
    t1 = _time.perf_counter()
    rs_tmp = ReachabilitySet(object_id, t0_s, cfg, grid, nominal_gcrf, nominal_rot, states_gcrf, states_rot,
                             dirs, dv_mps, burn_h, ray, n_rays, term_idx, term_reason, hits, first_hit,
                             nominal_hits, stats, regions, [], moon_gcrf, fr, term_t_s=term_t)
    env = _envelope(states_gcrf, states_rot, nominal_gcrf, active, t_h, rs_tmp.path_indices())
    rs_tmp.envelope = env
    timing["envelope_s"] = _time.perf_counter() - t1
    timing["total_s"] = _time.perf_counter() - tic
    timing["nfev"] = pinfo["nfev"]
    timing["n_segments"] = pinfo["n_segments"]
    rs_tmp.timing = timing
    rs_tmp.meta = {
        "n_samples": N, "n_rays": n_rays, "n_dirs": int(len(smp.get("unit_dirs", dirs))),
        "magnitudes_mps": [float(v) for v in np.asarray(smp.get("magnitudes", np.unique(dv_mps)))],
        "burn_epochs_h": [float(v) for v in np.unique(burn_h)],
        "n_terminated": int((term_idx < T).sum()),
        "n_lunar_impact": int(sum(1 for r in term_reason if r == "lunar_impact")),
        "n_earth_reentry": int(sum(1 for r in term_reason if r == "earth_reentry")),
        "nominal_terminated": None if nom_term[0] == T else {"t_h": float((nom_term_t[0] - t0_s) / HOUR_S), "reason": nom_reason[0]},
        "termination": "candidate rows screened by the osculating conic, then re-integrated alone (unsoftened) with a "
                       "terminal surface event; term_t_s is the exact crossing epoch, rows are masked from the first grid "
                       "epoch at/after it",
        "envelope_definitions": {
            "semi_axes_km": "1-sigma semi-axes of the endpoint covariance ellipsoid (sqrt of eigenvalues), descending",
            "ellipsoid_volume_km3": "volume of the 1-sigma covariance ellipsoid 4/3*pi*prod(semi_axes)",
            "hull_volume_km3": "convex hull of the active endpoints (all samples, no sigma scaling)",
            "max_radius_km": "largest distance of an active endpoint from the nominal position at that epoch",
        },
        "force_model": "DE440s Earth+Moon+Sun point masses" + (" + cannonball SRP" if params.srp else ""),
        "cr_area_mass_m2_kg": cfg.cr_area_mass,
        "integrator": f"DOP853 stacked 6N system, rtol={cfg.rtol:g}, atol={cfg.atol:g} km, max_step={cfg.max_step_s:g} s",
        "grid_dt_h": cfg.grid_dt_h,
        "disclaimer": "Reachability is a custody/awareness envelope under an ASSUMED Δv budget; it asserts no "
                      "intent and is not a maneuver prediction.",
    }
    return rs_tmp


def reachability_for_object(object_id: str, t0_s: Optional[float] = None, cfg: Optional[ReachabilityConfig] = None,
                            regions: Optional[list[Region]] = None) -> ReachabilitySet:
    """Convenience: state (and C_R·A/m) from the catalog, then :func:`compute_reachability`."""
    from selene.objects.catalog import get_catalog

    cat = get_catalog()
    e = cat.get(object_id)
    t0 = cat.epoch_s if t0_s is None else float(t0_s)
    s0 = cat.state_at(object_id, t0)
    cfg = cfg or ReachabilityConfig()
    if e.physical is not None and cfg.cr_area_mass == 0.0:
        cfg.cr_area_mass = float(e.physical.get("cr_area_mass_m2_kg", 0.0))
    rs = compute_reachability(s0, t0, cfg, object_id=object_id, regions=regions)
    rs.meta["object"] = {"id": e.id, "name": e.name, "kind": e.kind, "label": e.label, "orbit_type": e.orbit_type}
    if e.is_real:
        rs.meta["disclaimer"] += (" This is a REAL spacecraft (JPL Horizons ephemeris): the envelope is a hypothetical "
                                  "what-if for custody planning only; no maneuver is simulated or implied.")
    return rs


# ---------------------------------------------------------------------------
# minimum-Δv refinement
# ---------------------------------------------------------------------------
def refine_min_dv(rs: ReachabilitySet, region_key: str, n_bisect: int = 6, tol_mps: float = 1.0) -> Optional[dict]:
    """Bisection on the burn magnitude along the best hitting sample's direction and epoch.

    The ladder gives the minimum Δv to a region only at ladder resolution (e.g. "50 but not 20
    m/s").  This refines it between the largest non-hitting ladder level below and the hitting
    level by bisection (``n_bisect`` extra stacked propagations of a single row each, same
    horizon and tolerances).  Returns ``None`` if the region is not reached; otherwise a dict with
    the refined ``dv_mps`` (upper bound that hits), the ``dv_mps_lower`` (largest tested value
    that misses), direction, epoch and arrival time.  Note the result is a *sampled* minimum
    (fixed direction/epoch), i.e. an upper bound on the true optimum.
    """
    st = rs.stats_by_key().get(region_key)
    if st is None or (st.best_sample is None and not st.nominal_hits):
        return None
    if st.nominal_hits:
        return {"region": region_key, "dv_mps": 0.0, "dv_mps_lower": 0.0, "dir_gcrf": None, "burn_h": None,
                "arrival_h": st.earliest_nominal_h, "sample_index": None,
                "note": "reached by the 0-Δv nominal trajectory; no burn required"}
    i = st.best_sample
    u = rs.dv_dirs[i]
    burn_h = float(rs.burn_h[i])
    hi = float(rs.dv_mps[i])
    mags = np.asarray(rs.meta.get("magnitudes_mps", np.unique(rs.dv_mps)), float)
    lower = mags[mags < hi]
    lo = float(lower.max()) if lower.size else 0.0
    arrival = float(rs.first_hit_h[region_key][i])
    cfg = rs.config

    def hits(m: float) -> Optional[float]:
        smp = {"dirs": u[None, :], "dv_mps": np.array([m]), "burn_h": np.array([burn_h]), "ray": np.array([0]),
               "n_rays": 1, "magnitudes": np.array([m]), "unit_dirs": u[None, :]}
        r = compute_reachability(rs.nominal_gcrf[0], rs.t0_s, cfg, rs.object_id, rs.regions, samples=smp)
        fh = r.first_hit_h[region_key][0]
        return float(fh) if np.isfinite(fh) else None

    for _ in range(n_bisect):
        if hi - lo <= tol_mps:
            break
        mid = 0.5 * (lo + hi)
        a = hits(mid)
        if a is None:
            lo = mid
        else:
            hi, arrival = mid, a
    return {"region": region_key, "dv_mps": hi, "dv_mps_lower": lo, "dir_gcrf": u.tolist(), "burn_h": burn_h,
            "arrival_h": arrival, "sample_index": int(i),
            "note": "sampled minimum along one direction/epoch (upper bound on the true optimum)"}
