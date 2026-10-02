"""Ephemeris-based force model: Earth point mass + Moon + Sun third bodies (+ optional SRP).

Frame and units
---------------
Earth-centered GCRF/ICRF axes (the DE440s kernel is expressed in the ICRF; GCRF ≈ J2000 to
< 0.1 arcsec, adequate for display and for this MVP's estimation).  Positions in km, velocities
in km/s, accelerations in km/s².  Time is **TDB seconds past J2000.0** (``t_s``), converted to a
two-part Julian date for jplephem to retain sub-microsecond precision.

Equation of motion (PLAN §2.2)::

    r̈ = -GM_E r/|r|³ - Σ_j GM_j [ (r - r_j)/|r - r_j|³ + r_j/|r_j|³ ] + a_SRP

where the second bracketed term is the *indirect* acceleration (the Earth itself is accelerated
by body j, and this frame is non-inertial).  r_j(t) is the body position relative to the Earth
from DE440s.  Optional solar radiation pressure (cannonball)::

    a_SRP = ν · P☉ · C_R (A/m) · (1 AU / |r - r☉|)² · (r - r☉)/|r - r☉|

with P☉ = 4.56e-6 N/m² at 1 AU and ν ∈ {0, 1} a cylindrical Earth/Moon shadow factor.

Point-mass Jacobian (for the STM): ∂a_j/∂r = GM_j [ 3 Δ Δᵀ / |Δ|⁵ - I / |Δ|³ ], Δ = r - r_j;
the indirect terms and the SRP (treated as constant) do not depend on r.

References
----------
* Park, Folkner, Williams, Boggs (2021), DE440/DE441, AJ 161:105.
* Montenbruck & Gill (2000), *Satellite Orbits*, §3.2-3.4 (third-body and SRP formulation).
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field, replace
from functools import lru_cache
from pathlib import Path
from typing import Iterable

import numpy as np
from jplephem.spk import SPK
from numba import njit
from scipy.integrate import solve_ivp
from scipy.interpolate import CubicSpline

from selene.constants import (
    AU_KM,
    DAY_S,
    DE440S_PATH,
    GM_EARTH,
    GM_MOON,
    GM_SUN,
    P_SRP_1AU,
    R_EARTH,
    R_MOON,
)
from selene.time import J2000_JD

__all__ = [
    "Ephemeris",
    "get_ephemeris",
    "BodyCache",
    "EphemParams",
    "nbody_accel",
    "nbody_eom",
    "nbody_jacobian",
    "nbody_eom_stm",
    "propagate_ephemeris",
    "tdb_s_to_jd",
    "jd_to_tdb_s",
    "GM_BODIES",
]

GM_BODIES = {"earth": GM_EARTH, "moon": GM_MOON, "sun": GM_SUN}
_RADIUS = {"earth": R_EARTH, "moon": R_MOON}

# NAIF chain from the solar-system barycenter to each supported body.
_CHAIN = {
    "ssb": (),
    "emb": ((0, 3),),
    "earth": ((0, 3), (3, 399)),
    "moon": ((0, 3), (3, 301)),
    "sun": ((0, 10),),
}


def tdb_s_to_jd(t_s):
    """TDB seconds past J2000 -> TDB Julian date (single float; loses ~1 µs precision)."""
    return J2000_JD + np.asarray(t_s, dtype=np.float64) / DAY_S


def jd_to_tdb_s(jd):
    return (np.asarray(jd, dtype=np.float64) - J2000_JD) * DAY_S


# ---------------------------------------------------------------------------
# DE440s access
# ---------------------------------------------------------------------------
class Ephemeris:
    """Thin jplephem wrapper around ``data/cache/de440s.bsp`` (1849-2150 coverage).

    All methods accept a scalar or 1-D array of ``t_s`` (TDB seconds past J2000) and return
    km / km/s arrays shaped ``(3,)`` (scalar input) or ``(N, 3)``.
    Velocities come from ``compute_and_differentiate`` (km/day -> km/s).
    """

    def __init__(self, path: str | Path = DE440S_PATH):
        self.path = Path(path)
        if not self.path.exists():
            raise FileNotFoundError(
                f"DE440s kernel not found at {self.path}; run `make setup` to download it."
            )
        self.kernel = SPK.open(str(self.path))
        self._segments = {}

    def _segment(self, a: int, b: int):
        key = (a, b)
        if key not in self._segments:
            self._segments[key] = self.kernel[a, b]
        return self._segments[key]

    def _terms(self, body: str, center: str) -> list[tuple[tuple[int, int], int]]:
        body, center = body.lower(), center.lower()
        for name in (body, center):
            if name not in _CHAIN:
                raise ValueError(f"unknown body {name!r}; choose from {sorted(_CHAIN)}")
        c = Counter(_CHAIN[body])
        c.subtract(Counter(_CHAIN[center]))
        return [(seg, sgn) for seg, sgn in c.items() if sgn != 0]

    def _compute(self, body: str, center: str, t_s, velocity: bool):
        t = np.asarray(t_s, dtype=np.float64)
        scalar = t.ndim == 0
        frac = np.atleast_1d(t) / DAY_S  # two-part JD: (J2000_JD, frac)
        pos = np.zeros((3, frac.size))
        vel = np.zeros((3, frac.size)) if velocity else None
        for (a, b), sgn in self._terms(body, center):
            seg = self._segment(a, b)
            if velocity:
                p, v = seg.compute_and_differentiate(J2000_JD, frac)
                pos += sgn * p
                vel += sgn * v / DAY_S
            else:
                pos += sgn * seg.compute(J2000_JD, frac)
        pos = pos.T
        if velocity:
            vel = vel.T
        if scalar:
            pos = pos[0]
            vel = vel[0] if velocity else None
        return pos, vel

    # public ------------------------------------------------------------
    def position(self, body: str, t_s, center: str = "earth") -> np.ndarray:
        """Position of ``body`` relative to ``center`` [km]."""
        return self._compute(body, center, t_s, velocity=False)[0]

    def state(self, body: str, t_s, center: str = "earth") -> np.ndarray:
        """Position + velocity of ``body`` relative to ``center``, shape (6,) or (N, 6) [km, km/s]."""
        p, v = self._compute(body, center, t_s, velocity=True)
        return np.concatenate([p, v], axis=-1)

    def moon_state(self, t_s) -> np.ndarray:
        return self.state("moon", t_s, "earth")

    def sun_state(self, t_s) -> np.ndarray:
        return self.state("sun", t_s, "earth")

    def earth_moon_barycenter_state(self, t_s) -> np.ndarray:
        """DE440s EMB relative to Earth (uses the kernel's own Earth/Moon mass ratio)."""
        return self.state("emb", t_s, "earth")

    def body_positions(self, bodies: Iterable[str], t_s) -> np.ndarray:
        """Stack of Earth-centered positions for ``bodies`` -> (k, 3) for scalar t."""
        bodies = tuple(bodies)
        if not bodies:
            return np.zeros((0, 3))
        return np.stack([self.position(b, t_s, "earth") for b in bodies], axis=0)


@lru_cache(maxsize=1)
def get_ephemeris() -> Ephemeris:
    """Process-wide singleton (kernel memory-mapped once)."""
    return Ephemeris()


# ---------------------------------------------------------------------------
# Cached, splined body positions for hot loops
# ---------------------------------------------------------------------------
@njit(cache=True)
def _spline_eval(c, x0, dx, n, t, out):
    """Evaluate a uniform-grid scipy CubicSpline (coeffs ``c`` shape (4, n-1, 3)) at scalar t."""
    i = int(np.floor((t - x0) / dx))
    if i < 0:
        i = 0
    elif i > n - 2:
        i = n - 2
    tl = t - (x0 + i * dx)
    for k in range(3):
        out[k] = ((c[0, i, k] * tl + c[1, i, k]) * tl + c[2, i, k]) * tl + c[3, i, k]


class BodyCache:
    """Precomputed Moon and Sun Earth-centered positions on a uniform grid, cubic-spline
    interpolated (``scipy.interpolate.CubicSpline``, not-a-knot).

    Accuracy: with ``dt_s = 600`` the interpolation error is bounded by
    (5/384) h⁴ max|r⁽⁴⁾| ≈ (5/384)(600 s)⁴ ω⁴ r ≈ 2e-8 km for the Moon (ω = 2.66e-6 rad/s,
    r = 3.8e5 km) and far less for the Sun -- i.e. sub-metre, verified in ``test_ephemeris.py``
    (< 1 km required).  Building a 30-day cache costs ~10 ms.  The grid is padded by one step
    on each side; evaluation outside the padded span is clamped to the end polynomials (and
    will degrade) -- callers should size the cache to the propagation window.
    """

    def __init__(self, t0_s: float, t1_s: float, dt_s: float = 600.0, ephemeris: Ephemeris | None = None):
        eph = ephemeris or get_ephemeris()
        if t1_s < t0_s:
            t0_s, t1_s = t1_s, t0_s
        pad = 2.0 * dt_s
        n = max(4, int(np.ceil((t1_s - t0_s + 2 * pad) / dt_s)) + 1)
        self.t_grid = t0_s - pad + dt_s * np.arange(n)
        self.dt_s = float(dt_s)
        self.t0_s, self.t1_s = float(self.t_grid[0]), float(self.t_grid[-1])
        self.n = n
        self._coef = {}
        self._splines = {}
        for body in ("moon", "sun"):
            pos = eph.position(body, self.t_grid, "earth")  # (n,3)
            cs = CubicSpline(self.t_grid, pos, axis=0)
            self._splines[body] = cs
            self._coef[body] = np.ascontiguousarray(cs.c)  # (4, n-1, 3)

    def covers(self, t_s: float) -> bool:
        return self.t0_s <= t_s <= self.t1_s

    def position(self, body: str, t_s: float) -> np.ndarray:
        """Interpolated Earth-centered position [km] of 'moon' or 'sun' at scalar ``t_s``."""
        out = np.empty(3)
        _spline_eval(self._coef[body], self.t0_s, self.dt_s, self.n, float(t_s), out)
        return out

    def positions(self, body: str, t_s) -> np.ndarray:
        """Vectorised evaluation (uses the scipy spline directly)."""
        return self._splines[body](np.asarray(t_s, dtype=np.float64))

    def body_positions(self, bodies: Iterable[str], t_s: float) -> np.ndarray:
        bodies = tuple(bodies)
        out = np.empty((len(bodies), 3))
        for i, b in enumerate(bodies):
            _spline_eval(self._coef[b], self.t0_s, self.dt_s, self.n, float(t_s), out[i])
        return out


# ---------------------------------------------------------------------------
# Force model
# ---------------------------------------------------------------------------
@dataclass
class EphemParams:
    """Force-model configuration.

    srp            include cannonball solar radiation pressure
    cr_area_mass   C_R · A/m  [m²/kg]  (e.g. 1.3 × 0.01 for a small bus)
    bodies         third bodies to include, subset of ('moon', 'sun'); () gives Earth two-body
    cache          optional :class:`BodyCache`; if None, body positions are read from DE440s
                   directly (slower, ~3 segment evaluations per call)
    """

    srp: bool = False
    cr_area_mass: float = 0.0
    bodies: tuple[str, ...] = ("moon", "sun")
    cache: BodyCache | None = field(default=None, repr=False)

    def __post_init__(self):
        self.bodies = tuple(b.lower() for b in self.bodies)
        for b in self.bodies:
            if b not in ("moon", "sun"):
                raise ValueError(f"unsupported third body {b!r}")


_SRP_SCALE = P_SRP_1AU * 1e-3  # N/m² × (m²/kg) -> m/s² -> km/s²


@njit(cache=True)
def _point_mass_accel(r, r_b, gm_b, gm_e, out):
    """Earth central term + third-body direct/indirect terms.  r (3,), r_b (k,3), gm_b (k,)."""
    rn2 = r[0] * r[0] + r[1] * r[1] + r[2] * r[2]
    rn3 = rn2 * np.sqrt(rn2)
    for i in range(3):
        out[i] = -gm_e * r[i] / rn3
    for j in range(r_b.shape[0]):
        d0 = r[0] - r_b[j, 0]
        d1 = r[1] - r_b[j, 1]
        d2 = r[2] - r_b[j, 2]
        dn2 = d0 * d0 + d1 * d1 + d2 * d2
        dn3 = dn2 * np.sqrt(dn2)
        bn2 = r_b[j, 0] ** 2 + r_b[j, 1] ** 2 + r_b[j, 2] ** 2
        bn3 = bn2 * np.sqrt(bn2)
        out[0] -= gm_b[j] * (d0 / dn3 + r_b[j, 0] / bn3)
        out[1] -= gm_b[j] * (d1 / dn3 + r_b[j, 1] / bn3)
        out[2] -= gm_b[j] * (d2 / dn3 + r_b[j, 2] / bn3)


@njit(cache=True)
def _point_mass_jacobian(r, r_b, gm_b, gm_e, out):
    """∂a/∂r for the central + third-body direct terms (indirect terms are r-independent)."""
    for i in range(3):
        for j in range(3):
            out[i, j] = 0.0
    # central body: Δ = r
    rn2 = r[0] * r[0] + r[1] * r[1] + r[2] * r[2]
    rn = np.sqrt(rn2)
    rn3 = rn2 * rn
    rn5 = rn3 * rn2
    for i in range(3):
        for j in range(3):
            out[i, j] += gm_e * 3.0 * r[i] * r[j] / rn5
        out[i, i] -= gm_e / rn3
    for k in range(r_b.shape[0]):
        d = np.empty(3)
        for i in range(3):
            d[i] = r[i] - r_b[k, i]
        dn2 = d[0] * d[0] + d[1] * d[1] + d[2] * d[2]
        dn = np.sqrt(dn2)
        dn3 = dn2 * dn
        dn5 = dn3 * dn2
        for i in range(3):
            for j in range(3):
                out[i, j] += gm_b[k] * 3.0 * d[i] * d[j] / dn5
            out[i, i] -= gm_b[k] / dn3


@njit(cache=True)
def _cyl_shadow(r, r_body, radius, r_sun):
    """1.0 if sunlit, 0.0 if inside the cylindrical shadow of a body at ``r_body`` (radius km)."""
    s0 = r_sun[0] - r_body[0]
    s1 = r_sun[1] - r_body[1]
    s2 = r_sun[2] - r_body[2]
    sn = np.sqrt(s0 * s0 + s1 * s1 + s2 * s2)
    s0 /= sn
    s1 /= sn
    s2 /= sn
    p0 = r[0] - r_body[0]
    p1 = r[1] - r_body[1]
    p2 = r[2] - r_body[2]
    along = p0 * s0 + p1 * s1 + p2 * s2
    if along >= 0.0:  # on the sunward side of the body
        return 1.0
    q0 = p0 - along * s0
    q1 = p1 - along * s1
    q2 = p2 - along * s2
    perp = np.sqrt(q0 * q0 + q1 * q1 + q2 * q2)
    return 0.0 if perp < radius else 1.0


def _body_positions(t_s: float, params: EphemParams) -> np.ndarray:
    """Earth-centered positions (k,3) for params.bodies (+ Sun if SRP needs it)."""
    if params.cache is not None:
        return params.cache.body_positions(params.bodies, t_s)
    return get_ephemeris().body_positions(params.bodies, t_s)


def _sun_position(t_s: float, params: EphemParams, r_b: np.ndarray) -> np.ndarray:
    if "sun" in params.bodies:
        return r_b[params.bodies.index("sun")]
    if params.cache is not None:
        return params.cache.position("sun", t_s)
    return get_ephemeris().position("sun", t_s, "earth")


def _moon_position(t_s: float, params: EphemParams, r_b: np.ndarray) -> np.ndarray:
    if "moon" in params.bodies:
        return r_b[params.bodies.index("moon")]
    if params.cache is not None:
        return params.cache.position("moon", t_s)
    return get_ephemeris().position("moon", t_s, "earth")


def srp_accel(t_s: float, r: np.ndarray, params: EphemParams, r_b: np.ndarray | None = None) -> np.ndarray:
    """Cannonball SRP [km/s²] with cylindrical Earth and Moon shadow factors."""
    if r_b is None:
        r_b = _body_positions(t_s, params)
    r_sun = _sun_position(t_s, params, r_b)
    r_moon = _moon_position(t_s, params, r_b)
    nu = _cyl_shadow(r, np.zeros(3), R_EARTH, r_sun) * _cyl_shadow(r, r_moon, R_MOON, r_sun)
    d = r - r_sun
    dn = np.linalg.norm(d)
    return nu * _SRP_SCALE * params.cr_area_mass * (AU_KM / dn) ** 2 * d / dn


def nbody_accel(t_s: float, r: np.ndarray, params: EphemParams | None = None) -> np.ndarray:
    """Total acceleration [km/s²] at Earth-centered position ``r`` [km], TDB time ``t_s``."""
    params = params or EphemParams()
    r = np.asarray(r, dtype=np.float64)
    r_b = _body_positions(t_s, params)
    gm_b = np.array([GM_BODIES[b] for b in params.bodies], dtype=np.float64)
    out = np.empty(3)
    _point_mass_accel(r, r_b, gm_b, GM_EARTH, out)
    if params.srp and params.cr_area_mass > 0.0:
        out += srp_accel(t_s, r, params, r_b)
    return out


def nbody_eom(t_s: float, s: np.ndarray, params: EphemParams | None = None) -> np.ndarray:
    """``solve_ivp``-compatible EOM: s = (x, y, z, vx, vy, vz) [km, km/s]."""
    s = np.asarray(s, dtype=np.float64)
    out = np.empty(6)
    out[:3] = s[3:6]
    out[3:] = nbody_accel(t_s, s[:3], params)
    return out


def nbody_jacobian(t_s: float, r: np.ndarray, params: EphemParams | None = None) -> np.ndarray:
    """Analytic ∂a/∂r (3×3) of the point-mass accelerations.  SRP is treated as constant."""
    params = params or EphemParams()
    r = np.asarray(r, dtype=np.float64)
    r_b = _body_positions(t_s, params)
    gm_b = np.array([GM_BODIES[b] for b in params.bodies], dtype=np.float64)
    out = np.empty((3, 3))
    _point_mass_jacobian(r, r_b, gm_b, GM_EARTH, out)
    return out


def nbody_eom_stm(t_s: float, s42: np.ndarray, params: EphemParams | None = None) -> np.ndarray:
    """State + STM derivative (42,), STM row-major: Φ̇ = [[0, I], [G, 0]] Φ with G = ∂a/∂r."""
    params = params or EphemParams()
    s42 = np.asarray(s42, dtype=np.float64)
    out = np.empty(42)
    out[:6] = nbody_eom(t_s, s42[:6], params)
    G = nbody_jacobian(t_s, s42[:3], params)
    phi = s42[6:].reshape(6, 6)
    dphi = np.empty((6, 6))
    dphi[:3] = phi[3:]
    dphi[3:] = G @ phi[:3]
    out[6:] = dphi.ravel()
    return out


def propagate_ephemeris(
    s0: np.ndarray,
    t0_s: float,
    tf_s: float | None = None,
    t_eval_s: np.ndarray | None = None,
    params: EphemParams | None = None,
    stm: bool = False,
    rtol: float = 1e-11,
    atol: float = 1e-11,
    method: str = "DOP853",
    events=None,
    dense_output: bool = False,
    max_step: float = np.inf,
):
    """Integrate the ephemeris model from GCRF state ``s0`` [km, km/s] at TDB ``t0_s``.

    ``tf_s`` is an absolute TDB time (not a duration).  If ``params.cache`` is None a
    :class:`BodyCache` spanning the window is built automatically (≈ ms) so the integrator's
    right-hand side never touches jplephem; pass your own cache to amortise across many calls.
    ``stm=True`` integrates the 6×6 STM (identity-initialised unless ``s0`` has 42 entries).
    Returns the scipy ``OdeResult`` (``sol.y`` shape (6 or 42, N), ``sol.t`` in TDB seconds).
    """
    s0 = np.asarray(s0, dtype=np.float64).ravel()
    if t_eval_s is not None:
        t_eval_s = np.asarray(t_eval_s, dtype=np.float64)
        if tf_s is None:
            tf_s = float(t_eval_s[-1])
    if tf_s is None:
        raise ValueError("propagate_ephemeris needs tf_s or t_eval_s")
    params = params or EphemParams()
    if params.cache is None or not (params.cache.covers(t0_s) and params.cache.covers(tf_s)):
        params = replace(params, cache=BodyCache(min(t0_s, tf_s), max(t0_s, tf_s)))
    if stm:
        if s0.size == 6:
            s0 = np.concatenate([s0, np.eye(6).ravel()])
        fun = nbody_eom_stm
    else:
        s0 = s0[:6]
        fun = nbody_eom
    return solve_ivp(
        fun,
        (float(t0_s), float(tf_s)),
        s0,
        method=method,
        t_eval=t_eval_s,
        args=(params,),
        rtol=rtol,
        atol=atol,
        events=events,
        dense_output=dense_output,
        max_step=max_step,
    )
