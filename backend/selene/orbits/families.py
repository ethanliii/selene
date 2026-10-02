"""Generators for the SELENE periodic-orbit families (Earth-Moon CR3BP).

Families (all nondimensional rotating-frame units, L* = 384 400 km, T* ≈ 3.7519e5 s):

* ``L1_lyapunov`` / ``L2_lyapunov`` -- planar; seeded from the linearised in-plane solution at
  the libration point and continued in x0 (natural parameter).
* ``L1_halo`` / ``L2_halo`` (branches ``N`` and ``S``) -- seeded with Richardson's (1980)
  third-order approximation, continued by pseudo-arclength in (x0, z0, vy0) until the orbit
  reaches the lunar surface; the near-rectilinear (NRHO) members are re-converged with
  multiple shooting.  The L2 southern branch is searched for the synodic-resonant 9:2, 4:1
  and 3:1 NRHOs (period = q/p of the mean synodic month; Zimovan-Spreen, Howell & Davis 2020).
* ``DRO`` -- planar distant retrograde orbits about the Moon, seeded from the two-body
  circular retrograde velocity and continued in x0.
* ``resonant_3:1`` / ``resonant_2:1`` -- planar Earth-centred p:q resonant orbits (p spacecraft
  revolutions per q lunar synodic revolutions), seeded from the Keplerian ellipse with the
  resonant semi-major axis a = ((1-μ) (q/p)²)^(1/3), perigee on the +x axis.

Initial-condition convention for library records: the perpendicular xz-plane crossing that is
*farther from the Moon* (for halos this is the apolune of the NRHO members; branch
``N``/``S`` is the sign of z there).  Planar orbits have z ≡ 0.

References (values compared in tests)
-------------------------------------
* Koon, Lo, Marsden & Ross (2011), *Dynamical Systems, the Three-Body Problem and Space Mission
  Design*, §4.2: L1 Lyapunov x0 = 0.8234, ẏ0 = 0.1263, T = 2.7430 (μ = 0.0121505).
* Richardson (1980), Celest. Mech. 22, 241-253 (third-order halo seed).
* Howell (1984), Celest. Mech. 32, 53-71 (halo families, stability index).
* Zimovan-Spreen, Howell & Davis (2020), Celest. Mech. Dyn. Astron. 132:28: NRHO regime
  bounded by stability changes at perilune radii ≈ 1 850-17 350 km (L2) and ≈ 900-19 000 km
  (L1); 9:2 synodic-resonant L2 southern NRHO, period ≈ 6.56 d, perilune ≈ 3 200 km.
* Lee (2019), NASA/TM-2019-220360 "Gateway Destination Orbit Model: A Continuous 15 Year NRHO
  Reference Trajectory": 9:2 NRHO perilune ≈ 3 366 km mean, apolune ≈ 71 000 km, period ≈ 6.56 d.
* Hénon (1969) / Broucke (1968) for DRO ("family f") and resonant families.
* JPL Three-Body Periodic Orbit Catalog (ssd-api.jpl.nasa.gov/periodic_orbits.api), see
  ``jpl_reference.py``.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import minimize_scalar

from selene.constants import L_STAR, MU, R_MOON, T_STAR
from selene.dynamics.cr3bp import jacobi, lagrange_points, propagate_cr3bp
from selene.orbits.continuation import natural_parameter, pseudo_arclength
from selene.orbits.richardson import lyapunov_linear_ic, richardson_halo_ic
from selene.orbits.shooting import (
    CLOSURE_ATOL,
    CLOSURE_RTOL,
    ShootingResult,
    multiple_shoot,
    shoot_symmetric,
    shoot_symmetric_period,
)

__all__ = [
    "FamilyMember",
    "FamilyResult",
    "orbit_geometry",
    "gen_lyapunov",
    "gen_halo",
    "gen_dro",
    "gen_resonant",
    "locate_synodic_nrhos",
    "FAMILY_SPECS",
    "generate_family",
    "SYNODIC_MONTH_DAYS",
    "NRHO_PERILUNE_BOUNDS_KM",
]

# Mean synodic month (new Moon to new Moon), days.  The 9:2 NRHO completes 9 revolutions in
# 2 synodic months (Zimovan-Spreen et al. 2020; Lee 2019).
SYNODIC_MONTH_DAYS = 29.530589
# NRHO regime bounds (perilune radius, km) from Zimovan-Spreen, Howell & Davis (2020), approx.
NRHO_PERILUNE_BOUNDS_KM = {1: (900.0, 19_000.0), 2: (1_850.0, 17_350.0)}


@dataclass
class FamilyMember:
    ic: np.ndarray
    period: float
    closure_error: float
    iterations: int
    method: str = "single_shooting"
    geometry: dict = field(default_factory=dict)
    tags: list = field(default_factory=list)
    seed: dict = field(default_factory=dict)

    @property
    def jacobi(self) -> float:
        return float(jacobi(self.ic))


@dataclass
class FamilyResult:
    family: str
    branch: str
    members: list
    params: dict
    references: list
    log: dict = field(default_factory=dict)
    elapsed_s: float = 0.0


# ---------------------------------------------------------------------------
# geometry helper
# ---------------------------------------------------------------------------
def _refine_extremum(fun, ts, vals, k: int, sign: float) -> float:
    """Minimise ``sign * fun(t)`` (continuous, from dense output) in the grid bracket around
    index ``k`` (and, at the wrap-around ends of a periodic grid, in the bracket at the other
    end too).  Returns the refined extremum of ``fun`` (never worse than the grid value)."""
    best = float(vals[k])
    n = len(ts)
    brackets = [(ts[max(k - 1, 0)], ts[min(k + 1, n - 1)])]
    if k == 0:
        brackets.append((ts[n - 2], ts[n - 1]))
    elif k == n - 1:
        brackets.append((ts[0], ts[1]))
    for lo, hi in brackets:
        res = minimize_scalar(lambda t: sign * fun(t), bounds=(float(lo), float(hi)), method="bounded",
                              options={"xatol": 1e-11})
        best = min(best, sign * float(res.fun)) if sign > 0 else max(best, sign * float(res.fun))
    return best


def orbit_geometry(ic, period: float, mu: float = MU, n: int = 600) -> dict:
    """Extent of one revolution: distances to Moon/Earth (km) and axis amplitudes (km).

    The orbit is sampled on an ``n``-point grid for the amplitudes, but the apsidal distances
    (perilune/apolune, perigee/apogee) are *refined* by a bounded 1-D minimisation on the
    integrator's dense output around the grid extremum: the perilune passage of a
    near-rectilinear halo lasts a few hours out of a week, and the grid minimum alone is biased
    high by up to ~400 km (measured for the lowest L1 halo members) -- enough to hide lunar
    impactors (R_moon = 1 737.4 km)."""
    ts = np.linspace(0.0, period, n)
    sol = propagate_cr3bp(ic, t_eval=ts, mu=mu, rtol=1e-11, atol=1e-11, dense_output=True)
    y = sol.y[:6]
    xm = 1.0 - mu
    r2 = np.sqrt((y[0] - xm) ** 2 + y[1] ** 2 + y[2] ** 2)
    r1 = np.sqrt((y[0] + mu) ** 2 + y[1] ** 2 + y[2] ** 2)

    def f_moon(t):
        p = sol.sol(t)[:3]
        return float(np.sqrt((p[0] - xm) ** 2 + p[1] ** 2 + p[2] ** 2))

    def f_earth(t):
        p = sol.sol(t)[:3]
        return float(np.sqrt((p[0] + mu) ** 2 + p[1] ** 2 + p[2] ** 2))

    perilune = _refine_extremum(f_moon, ts, r2, int(np.argmin(r2)), +1.0)
    apolune = _refine_extremum(f_moon, ts, r2, int(np.argmax(r2)), -1.0)
    perigee = _refine_extremum(f_earth, ts, r1, int(np.argmin(r1)), +1.0)
    apogee = _refine_extremum(f_earth, ts, r1, int(np.argmax(r1)), -1.0)
    return {
        "perilune_km": float(perilune * L_STAR),
        "apolune_km": float(apolune * L_STAR),
        "perigee_km": float(perigee * L_STAR),
        "apogee_km": float(apogee * L_STAR),
        "Ax_km": float(0.5 * (y[0].max() - y[0].min()) * L_STAR),
        "Ay_km": float(0.5 * (y[1].max() - y[1].min()) * L_STAR),
        "Az_km": float(np.max(np.abs(y[2])) * L_STAR),
        "amplitude_km": float(0.5 * max(y[0].max() - y[0].min(), y[1].max() - y[1].min()) * L_STAR),
        "z_max_nd": float(y[2].max()),
        "z_min_nd": float(y[2].min()),
        "x_min_nd": float(y[0].min()),
        "x_max_nd": float(y[0].max()),
    }


def _clean_symmetric(X):
    X = np.asarray(X, dtype=np.float64).copy()
    X[1] = 0.0
    X[3] = 0.0
    X[5] = 0.0
    return X


def _far_crossing(res: ShootingResult, mode: str, mu: float) -> ShootingResult:
    """Re-express a symmetric orbit at the perpendicular crossing farther from the Moon."""
    if res.half_state is None:
        return res
    xm = 1.0 - mu
    r_ic = np.hypot(res.ic[0] - xm, res.ic[2])
    r_half = np.hypot(res.half_state[0] - xm, res.half_state[2])
    if r_half <= r_ic:
        return res
    alt = shoot_symmetric(_clean_symmetric(res.half_state), mode=mode, mu=mu)
    return alt if alt.converged else res


def _member(res: ShootingResult, mu: float, seed: dict | None = None, geometry: dict | None = None) -> FamilyMember:
    geo = geometry if geometry is not None else orbit_geometry(res.ic, res.period, mu)
    return FamilyMember(
        ic=np.asarray(res.ic, dtype=np.float64).copy(),
        period=float(res.period),
        closure_error=float(res.closure_error),
        iterations=int(res.iterations),
        method=res.method,
        geometry=geo,
        seed=seed or {},
    )


# ---------------------------------------------------------------------------
# Lyapunov
# ---------------------------------------------------------------------------
def gen_lyapunov(libr: int, mu: float = MU, quick: bool = False) -> FamilyResult:
    """L1 or L2 planar Lyapunov family by natural-parameter continuation in x0."""
    t0 = time.time()
    xL = float(lagrange_points(mu)[libr - 1, 0])
    Ax0 = 0.004
    X0, T_lin, _ = lyapunov_linear_ic(libr, Ax0, mu)
    first = shoot_symmetric(X0, mode="planar", mu=mu)
    if not first.converged:
        raise RuntimeError("Lyapunov seed failed to converge")
    first = _far_crossing(first, "planar", mu)
    if libr == 1:
        # Earth side of L1; sweep inwards.  Include the Koon-Lo-Marsden-Ross (2011) member.
        x_end, step, include = (0.70, 0.012, (0.8234,)) if quick else (0.62, 0.0075, (0.8234,))
    else:
        # far side of L2; sweep outwards.
        x_end, step, include = (1.26, 0.009, ()) if quick else (1.30, 0.0055, ())

    def corrector(X, t_half):
        return shoot_symmetric(X, mode="planar", mu=mu)

    log = natural_parameter(
        first.ic, corrector, param_index=0, p_end=x_end, step=step, free_index=[4], step_max=step,
        include=include, n_max=400,
    )
    members = [_member(r, mu, seed={"x0": float(r.ic[0])}) for r in log.members]
    for m in members:
        m.tags.append("planar")
    return FamilyResult(
        family=f"L{libr}_lyapunov",
        branch="",
        members=members,
        params={
            "seed": "linearised in-plane solution, Ax=%.3f L*" % Ax0,
            "continuation": "natural parameter in x0 (far-from-Moon crossing), adaptive step",
            "x0_end": x_end,
            "step": step,
            "include_x0": list(include),
            "corrector": "symmetric single shooting, planar (free vy0)",
            "x_L": xL,
            "linear_period": T_lin,
        },
        references=[
            "Koon, Lo, Marsden & Ross (2011) §4.2: L1 Lyapunov x0=0.8234, vy0=0.1263, T=2.7430",
            "Richardson (1980) linear in-plane frequency",
        ],
        log={"stop_reason": log.stop_reason, "failures": log.failures},
        elapsed_s=time.time() - t0,
    )


# ---------------------------------------------------------------------------
# Halo
# ---------------------------------------------------------------------------
def gen_halo(libr: int, branch: str, mu: float = MU, quick: bool = False, n_patch: int = 12) -> FamilyResult:
    """L1/L2 northern or southern halo family: Richardson seed → pseudo-arclength in
    (x0, z0, vy0) → NRHO members polished with multiple shooting."""
    t0 = time.time()
    branch = branch.upper()
    Az_seeds = (0.010, 0.016)
    seeds = []
    for Az in Az_seeds:
        X0, T_est, info = richardson_halo_ic(libr, Az, branch, mu)
        r = shoot_symmetric(X0, mode="fix_z0", mu=mu)
        if not r.converged:
            raise RuntimeError(f"Richardson seed Az={Az} failed for L{libr} {branch}")
        r = _far_crossing(r, "fix_z0", mu)
        seeds.append(r)

    ds0 = 0.006
    ds_max = (0.03 if quick else 0.012) if libr == 2 else (0.03 if quick else 0.011)
    free = [0, 2, 4]
    lo_km, hi_km = NRHO_PERILUNE_BOUNDS_KM[libr]
    state = {"min_peri": np.inf, "geo": {}}

    def corrector(X_pred, v_pred, tau, ds):
        return shoot_symmetric(X_pred, mode="free3", mu=mu, arclength=(v_pred, tau, ds))

    def stop(r: ShootingResult) -> bool:
        geo = orbit_geometry(r.ic, r.period, mu)
        state["geo"][id(r)] = geo
        peri = geo["perilune_km"]
        if peri < R_MOON + 50.0:  # lunar impact: end of the usable family
            return True
        # L1 family continues past the NRHO regime into large Earth-encircling orbits;
        # stop once perilune starts growing again after having reached the NRHO regime.
        if peri < state["min_peri"]:
            state["min_peri"] = peri
        elif state["min_peri"] < 0.5 * hi_km and peri > 1.25 * state["min_peri"]:
            return True
        if r.period > 4.0 or abs(r.ic[2]) > 0.6:
            return True
        return False

    for s in seeds:
        stop(s)
    log = pseudo_arclength(seeds, corrector, free, ds=ds0, ds_max=ds_max, n_max=400, stop=stop)

    members = []
    for i, r in enumerate(log.members):
        geo = state["geo"].get(id(r)) or orbit_geometry(r.ic, r.period, mu)
        m = _member(r, mu, seed={"index": i}, geometry=geo)
        if lo_km <= geo["perilune_km"] <= hi_km:
            # sensitive near-rectilinear member: re-converge with multiple shooting
            ms = multiple_shoot(m.ic, m.period, n_patch=n_patch, mu=mu, fix={2: float(m.ic[2])})
            if ms.converged and np.isfinite(ms.closure_error):
                m.ic = ms.ic
                m.period = ms.period
                m.closure_error = ms.closure_error
                m.iterations = ms.iterations
                m.method = "multiple_shooting"
            m.tags.append("NRHO")
        members.append(m)
    return FamilyResult(
        family=f"L{libr}_halo",
        branch=branch,
        members=members,
        params={
            "seed": "Richardson (1980) third-order, Az = %s L*" % (list(Az_seeds),),
            "continuation": "pseudo-arclength in (x0, z0, vy0), adaptive step on Newton iterations",
            "ds0": ds0,
            "ds_max": ds_max,
            "termination": "perilune radius < R_moon + 50 km (perilune refined on dense output, not a grid "
                           "minimum), or perilune re-growth after the NRHO minimum (L1)",
            "nrho_perilune_bounds_km": [lo_km, hi_km],
            "nrho_corrector": f"multiple shooting, {n_patch} patch points, free period, phase y0=0",
            "ic_convention": "perpendicular xz-plane crossing farther from the Moon (apolune for NRHOs)",
        },
        references=[
            "Richardson (1980) Celest. Mech. 22:241",
            "Howell (1984) Celest. Mech. 32:53",
            "Zimovan-Spreen, Howell & Davis (2020) CMDA 132:28 (NRHO bounds, 9:2 NRHO)",
        ],
        log={"stop_reason": log.stop_reason, "failures": log.failures},
        elapsed_s=time.time() - t0,
    )


def locate_synodic_nrhos(halo: FamilyResult, mu: float = MU, ratios=((9, 2), (4, 1), (3, 1))) -> list:
    """Find members of a halo family whose period equals q/p synodic months (p revolutions per
    q synodic months) by bracketing the period along the family and solving the symmetric
    shooting problem with the period prescribed.  Returns FamilyMembers tagged ``NRHO_p:q``."""
    libr = int(halo.family[1])
    lo_km, hi_km = NRHO_PERILUNE_BOUNDS_KM[libr]
    out = []
    mem = halo.members
    for p, q in ratios:
        T_target = SYNODIC_MONTH_DAYS * 86400.0 * q / p / T_STAR
        found = None
        for a, b in zip(mem[:-1], mem[1:]):
            if (a.period - T_target) * (b.period - T_target) <= 0 and a.period != b.period:
                w = (T_target - a.period) / (b.period - a.period)
                X = _clean_symmetric(a.ic + w * (b.ic - a.ic))
                r = shoot_symmetric_period(X, T_target, mu=mu)
                if not r.converged:
                    continue
                geo = orbit_geometry(r.ic, r.period, mu)
                if not (lo_km <= geo["perilune_km"] <= hi_km):
                    continue
                ms = multiple_shoot(r.ic, r.period, n_patch=12, mu=mu, fix_period=True)
                if ms.converged and abs(ms.period - T_target) < 1e-12 and ms.closure_error < r.closure_error:
                    r = ms
                m = _member(r, mu, seed={"bracket": [float(a.period), float(b.period)]}, geometry=geo)
                m.tags += ["NRHO", f"NRHO_{p}:{q}", "synodic_resonant"]
                found = m
                break
        if found is not None:
            out.append(found)
    return out


# ---------------------------------------------------------------------------
# DRO
# ---------------------------------------------------------------------------
def gen_dro(mu: float = MU, quick: bool = False) -> FamilyResult:
    """Planar distant retrograde orbits about the Moon (Hénon's family f)."""
    t0 = time.time()
    x_start, x_end = 0.95, 0.50
    step = 0.03 if quick else 0.0125
    d = (1.0 - mu) - x_start
    # two-body circular retrograde speed about the Moon expressed in the rotating frame:
    # v_rot = v_inertial - ω × r  →  vy0 = sqrt(μ/d) + d  (clockwise, +y at x < x_Moon)
    X0 = np.array([x_start, 0.0, 0.0, 0.0, np.sqrt(mu / d) + d, 0.0])

    def corrector(X, t_half):
        return shoot_symmetric(X, mode="planar", mu=mu)

    log = natural_parameter(X0, corrector, param_index=0, p_end=x_end, step=step, free_index=[4], step_max=step, n_max=400)
    members = [_member(r, mu, seed={"x0": float(r.ic[0])}) for r in log.members]
    for m in members:
        m.tags += ["planar", "DRO"]
    return FamilyResult(
        family="DRO",
        branch="",
        members=members,
        params={
            "seed": "two-body circular retrograde velocity about the Moon at x0=%.2f" % x_start,
            "continuation": "natural parameter in x0 (decreasing), adaptive step",
            "x0_start": x_start,
            "x0_end": x_end,
            "step": step,
            "corrector": "symmetric single shooting, planar (free vy0)",
        },
        references=["Hénon (1969) A&A 1:223 (family f)", "Broucke (1968) JPL TR 32-1168"],
        log={"stop_reason": log.stop_reason, "failures": log.failures},
        elapsed_s=time.time() - t0,
    )


# ---------------------------------------------------------------------------
# Resonant
# ---------------------------------------------------------------------------
def gen_resonant(p: int, q: int, mu: float = MU, quick: bool = False) -> FamilyResult:
    """Planar p:q Earth-centred resonant family (p spacecraft revolutions per q revolutions of
    the Moon), perigee on the +x axis.  Seed: Keplerian ellipse with a = ((1-μ)(q/p)²)^(1/3)."""
    t0 = time.time()
    a = ((1.0 - mu) * (q / p) ** 2) ** (1.0 / 3.0)
    # Lower perigee bounds (x0_end): below ~0.12-0.15 L* (perigee ≈ 50 000-60 000 km, perigee
    # speed ≈ 4 L*/T*) the full-period closure check of these 2-3-perigee orbits reaches its
    # double-precision roundoff floor (~1e-10 even after polishing the IC at rtol 2.3e-14,
    # measured 1.2e-10..3.1e-10 for 3:1 at x0 = 0.10-0.11), so the families stop at a perigee
    # where every record still closes to < 1e-10 with a ~2x margin.  JPL's catalogue carries
    # the families further, down to near-Earth-collision members, which are not useful
    # cislunar reference orbits.
    if (p, q) == (3, 1):
        x_start, x_end, step = 0.30, 0.15, (0.02 if quick else 0.01)
    elif (p, q) == (2, 1):
        x_start, x_end, step = 0.50, 0.12, (0.04 if quick else 0.02)
    else:
        x_start, x_end, step = 0.9 * a, 0.3 * a, 0.02
    rp = x_start + mu
    vp = np.sqrt((1.0 - mu) * (2.0 / rp - 1.0 / a))
    X0 = np.array([x_start, 0.0, 0.0, 0.0, vp - rp, 0.0])
    T_half0 = np.pi * q  # rotating-frame period ≈ q synodic revolutions (2π q)

    def corrector(X, t_half):
        return shoot_symmetric(X, mode="planar", mu=mu, t_half=(t_half if t_half else T_half0))

    log = natural_parameter(
        X0, corrector, param_index=0, p_end=x_end, step=0.25 * step, free_index=[4], step_max=step, n_max=400,
        use_period_guess=True, max_period_jump=0.05,
    )
    # Polish: re-run the corrector with the integrator at the closure-check tolerance.  The
    # IC moves by only ~1e-13 but the full-period closure of the low-perigee members improves
    # by up to 10x (e.g. 7e-10 -> 9e-11), because the 1e-12 Newton integration error at the
    # fast perigee passages was otherwise baked into the converged vy0 / half period.
    polished = []
    for r in log.members:
        pr = shoot_symmetric(r.ic, mode="planar", mu=mu, t_half=0.5 * r.period, rtol=CLOSURE_RTOL, atol=CLOSURE_ATOL)
        if pr.converged and np.isfinite(pr.closure_error) and pr.closure_error < r.closure_error:
            pr.iterations += r.iterations
            polished.append(pr)
        else:
            polished.append(r)
    members = [_member(r, mu, seed={"x0": float(r.ic[0])}) for r in polished]
    for m in members:
        m.tags += ["planar", f"RESONANT_{p}:{q}"]
    return FamilyResult(
        family=f"resonant_{p}:{q}",
        branch="",
        members=members,
        params={
            "seed": "two-body ellipse, a=%.5f L*, perigee on +x axis" % a,
            "continuation": "natural parameter in x0 (perigee radius decreasing), adaptive step",
            "x0_start": x_start,
            "x0_end": x_end,
            "step": step,
            "corrector": "symmetric single shooting, planar, explicit half-period unknown; "
                         "final polish pass with the integrator at rtol 2.3e-14 / atol 1e-14",
            "termination": "x0 = %.2f (perigee ≈ %.0f km): below this the full-period closure check "
                           "hits its double-precision roundoff floor (~1e-10)" % (x_end, (x_end + mu) * L_STAR),
        },
        references=["JPL Three-Body Periodic Orbit Catalog (resonant family, Earth-Moon)"],
        log={"stop_reason": log.stop_reason, "failures": log.failures},
        elapsed_s=time.time() - t0,
    )


# ---------------------------------------------------------------------------
FAMILY_SPECS = {
    "L1_lyapunov": (gen_lyapunov, dict(libr=1)),
    "L2_lyapunov": (gen_lyapunov, dict(libr=2)),
    "L1_halo_N": (gen_halo, dict(libr=1, branch="N")),
    "L1_halo_S": (gen_halo, dict(libr=1, branch="S")),
    "L2_halo_N": (gen_halo, dict(libr=2, branch="N")),
    "L2_halo_S": (gen_halo, dict(libr=2, branch="S")),
    "DRO": (gen_dro, {}),
    "resonant_3:1": (gen_resonant, dict(p=3, q=1)),
    "resonant_2:1": (gen_resonant, dict(p=2, q=1)),
}


def generate_family(key: str, quick: bool = False, mu: float = MU) -> FamilyResult:
    fn, kw = FAMILY_SPECS[key]
    return fn(mu=mu, quick=quick, **kw)
