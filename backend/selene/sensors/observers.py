"""Space-based optical observers riding cislunar periodic orbits (or GEO).

Platform orbits
---------------
``platform_orbit`` ∈ {'geo', 'l1_halo', 'l2_halo', 'dro', 'nrho', 'resonant', 'custom'}.

* ``geo`` — ideal geostationary point: fixed in the ITRS at geocentric radius ``GEO_RADIUS_KM``
  on the equator over ``geo_longitude_deg``, transformed to GCRS with the same astropy
  ITRS→GCRS chain (IAU 2006/2000A precession-nutation, ERA, polar motion) used for the ground
  sites, so a GEO observer and a ground site share one Earth-orientation model.  Velocity comes
  from the same transform (ω⊕ × r plus the tiny frame-rate terms).  :func:`geo_state_era` is an
  analytic cross-check using only the Earth Rotation Angle
  ERA = 2π(0.7790572732640 + 1.00273781191135448·Tu), Tu = JD_UT1 − 2451545 (IERS Conventions
  2010, eq. 5.15), UT1 ≈ TDB − 69.2 s; it ignores precession-nutation and therefore differs from
  the full transform by ≈ 0.13° in 2026 (the CIP offset from the J2000 pole).
* CR3BP families — the orbit comes from :mod:`selene.orbits.library` when that module is
  importable and has a matching record (looked up lazily, so this package never hard-depends
  on it); otherwise from the built-in **fallback initial conditions** below.  Library selection
  (``_LIBRARY_QUERIES``): ``orbit_ref`` may name a record id (``lib[rid]``) or a family key
  (``'L2_halo_S'``, ``'resonant_2:1'`` → mid-family member); without it the platform resolves to
  l1_halo → ``L1_halo_N`` nearest Az ≈ 30 000 km, l2_halo → ``L2_halo_S`` nearest Az ≈ 53 000 km,
  nrho → the record tagged ``NRHO_9:2``, dro → ``DRO`` nearest perilune ≈ 72 000 km,
  resonant → mid-family ``resonant_3:1``.  ``PeriodicOrbit.source`` reports ``library:<id>``.  The fallback ICs
  were obtained by a symmetric single-shooting differential corrector (Howell 1984) from
  literature seeds and are re-polished at first use (``_correct_symmetric``) so the orbit
  closes to < 1e-10 in the nondimensional CR3BP regardless of the exact ``constants.MU``.
  The CR3BP orbit is mapped to GCRF with the instantaneous rotating frame
  (:func:`selene.dynamics.frames.rot_to_gcrf`) — an approximation adequate for a sensor
  platform (see ``dynamics/propagate.py`` docstring); the ephemeris model is not used so the
  platform never drifts off its family over long coverage runs.
* ``custom`` — caller-supplied nondimensional rotating-frame IC + period (``custom_ic``,
  ``custom_period``), propagated with the CR3BP.
* ``resonant`` — only available through the orbit library (``resonant_3:1`` by default,
  ``orbit_ref='resonant_2:1'`` for the other family); there is deliberately **no** hardcoded
  fallback (no validated literature IC was available offline).

Fallback IC provenance (nondimensional Earth–Moon rotating frame, μ = 0.0121505856):

=========  ================================================================  =================
family     IC [x0, 0, z0, 0, ẏ0, 0]; period P (nd) / days                    literature check
=========  ================================================================  =================
nrho       [1.0220262, 0, -0.1821, 0, -0.1032665, 0]; 1.511173 / 6.562 d     Zimovan-Spreen, Howell, Davis (2020) & Lee (2019, NASA Gateway
           perilune radius 3 250 km, apolune 71 200 km, C = 3.0465           white paper): 9:2 synodic-resonant L2 southern NRHO, P ≈ 6.56 d,
                                                                             perilune ≈ 3 200–3 400 km.  Seed x0=1.0221, z0=-0.1821, ẏ0=-0.1032.
l2_halo    [1.1542229, 0, -0.138, 0, -0.2147520, 0]; 3.226509 / 14.01 d       L2 southern halo, Az ≈ 53 000 km (z0 = −0.138 L*), C = 3.0807,
                                                                             consistent with the Howell (1984)/Grebow (2006) L2 family
                                                                             (periods 13.8–14.8 d at this amplitude).
l1_halo    [0.8234250, 0, 0.030, 0, 0.1400329, 0]; 2.748960 / 11.94 d         L1 northern halo, Az ≈ 11 500 km, C = 3.1668; seeded from the
                                                                             KLMR (2011) planar L1 Lyapunov neighbourhood (x0 ≈ 0.8234,
                                                                             ẏ0 ≈ 0.126–0.134, P ≈ 2.74).
dro        [0.80, 0, 0, 0, 0.5263593, 0]; 3.317126 / 14.40 d                  Planar DRO 72 200 km from the Moon, C = 2.9250; the ~70 000 km
                                                                             DRO class flown by Artemis I (NASA, 2022).
=========  ================================================================  =================
"""
from __future__ import annotations

import logging
import threading
from dataclasses import asdict, dataclass, field
from typing import Optional

import numpy as np

from selene.constants import GEO_RADIUS_KM, L_STAR, T_STAR
from selene.dynamics import frames
from selene.dynamics.cr3bp import cr3bp_eom, propagate_cr3bp, y_zero_crossing_event
from selene.time import J2000_JD

__all__ = [
    "PLATFORM_ORBITS",
    "SpaceObserver",
    "PeriodicOrbit",
    "FALLBACK_ORBITS",
    "LIBRARY_QUERIES",
    "DEFAULT_OBSERVERS",
    "get_observer",
    "resolve_orbit",
    "observer_state",
    "observer_gcrf",
    "geo_state",
    "geo_state_era",
    "orbit_summary",
]

_log = logging.getLogger(__name__)

PLATFORM_ORBITS = ("geo", "l1_halo", "l2_halo", "dro", "nrho", "resonant", "custom")

#: Earth rotation rate [rad/s] (IERS 2010 nominal).
OMEGA_EARTH = 7.292115e-5
#: UT1 ≈ TDB − 69.184 s for 2017–2026 (TT−TAI 32.184 s + 37 leap seconds); see module docstring.
_TDB_MINUS_UT1_S = 69.184


@dataclass(frozen=True)
class SpaceObserver:
    """A space-based optical sensor (notional).  Angles in degrees, magnitudes in V."""

    id: str
    name: str
    platform_orbit: str
    orbit_ref: Optional[str] = None       # orbit-library record id (optional)
    limiting_mag: float = 18.0
    fov_deg: float = 2.0
    sun_exclusion_deg: float = 30.0
    earth_exclusion_deg: float = 15.0     # measured from the Earth limb
    moon_exclusion_deg: float = 5.0       # measured from the Moon limb
    slew_rate_deg_s: float = 1.0
    geo_longitude_deg: float = -100.0     # geo only (east-positive sub-satellite longitude)
    phase: float = 0.0                    # fraction of the period elapsed at ``epoch_s``
    epoch_s: float = frames.DEMO_EPOCH_TDB_S
    custom_ic: Optional[tuple] = None     # custom only: nondim rotating-frame (6,)
    custom_period: Optional[float] = None # custom only: nondim period
    notes: str = ""
    kind: str = field(default="space", init=False)

    def __post_init__(self):
        if self.platform_orbit not in PLATFORM_ORBITS:
            raise ValueError(f"platform_orbit must be one of {PLATFORM_ORBITS}, got {self.platform_orbit!r}")

    def as_dict(self) -> dict:
        d = asdict(self)
        d["kind"] = "space"
        d["spec_note"] = "ASSUMED representative specs for a notional space-based sensor"
        return d


@dataclass
class PeriodicOrbit:
    """A CR3BP periodic orbit (nondimensional rotating frame)."""

    ic: np.ndarray        # (6,)
    period: float         # nd
    family: str
    source: str           # 'library:<id>' | 'fallback' | 'custom'
    closure_error: float = float("nan")


# Fallback ICs (validated by the shooting corrector in this module; see module docstring).
FALLBACK_ORBITS: dict[str, tuple[tuple[float, ...], float]] = {
    "nrho": ((1.0220261900, 0.0, -0.1821, 0.0, -0.1032665400, 0.0), 1.5111727459),
    "l2_halo": ((1.1542228600, 0.0, -0.1380, 0.0, -0.2147520400, 0.0), 3.2265086653),
    "l1_halo": ((0.8234250200, 0.0, 0.0300, 0.0, 0.1400328800, 0.0), 2.7489596224),
    "dro": ((0.8000000000, 0.0, 0.0000, 0.0, 0.5263593100, 0.0), 3.3171258551),
}

#: Library queries per platform: (family key or None, ``OrbitLibrary.find`` kwargs).  Each entry
#: is tried in order; the first record found wins.  Criteria are nearest-in-parameter within the
#: family (see :meth:`selene.orbits.library.OrbitLibrary.find`).
_LIBRARY_QUERIES: dict[str, tuple[dict, ...]] = {
    "l1_halo": ({"family": "L1_halo_N", "Az_km": 30_000.0},),                       # P ≈ 12.0 d
    "l2_halo": ({"family": "L2_halo_S", "Az_km": 53_000.0, "exclude_located": True},),  # P ≈ 14.0 d
    "nrho": ({"tag": "NRHO_9:2"}, {"family": "L2_halo_S", "perilune_km": 3_300.0}),  # 9:2 synodic NRHO
    "dro": ({"family": "DRO", "perilune_km": 72_000.0},),                             # Artemis-I class
    "resonant": ({"family": "resonant_3:1"}, {"family": "resonant_2:1"}),            # mid-family member
}
#: Family keys that select a mid-family member when the query has no numeric criterion.
_MID_FAMILY_QUERIES = {"resonant"}
LIBRARY_QUERIES = _LIBRARY_QUERIES

DEFAULT_OBSERVERS: tuple[SpaceObserver, ...] = (
    SpaceObserver("geo_west", "GEO observer, 100 W (notional)", "geo", limiting_mag=18.0, fov_deg=3.0,
                  geo_longitude_deg=-100.0, notes="Notional 0.3 m-class GEO-hosted sensor; assumed 18 mag"),
    SpaceObserver("geo_east", "GEO observer, 60 E (notional)", "geo", limiting_mag=18.0, fov_deg=3.0,
                  geo_longitude_deg=60.0, notes="Notional 0.3 m-class GEO-hosted sensor; assumed 18 mag"),
    SpaceObserver("l1_halo_obs", "L1 halo observer (notional)", "l1_halo", limiting_mag=18.5, fov_deg=2.0,
                  notes="Notional 0.4 m-class sensor on an L1 northern halo; assumed 18.5 mag"),
    SpaceObserver("l2_halo_obs", "L2 halo observer (notional)", "l2_halo", limiting_mag=18.5, fov_deg=2.0,
                  notes="Notional 0.4 m-class sensor on an L2 southern halo; assumed 18.5 mag"),
    SpaceObserver("dro_obs", "DRO observer (notional)", "dro", limiting_mag=18.5, fov_deg=2.0,
                  notes="Notional 0.4 m-class sensor on a ~70 000 km DRO; assumed 18.5 mag"),
    SpaceObserver("nrho_obs", "NRHO observer (notional)", "nrho", limiting_mag=18.0, fov_deg=2.0, phase=0.5,
                  notes="Notional sensor hosted on a 9:2 NRHO platform; assumed 18 mag"),
)
_OBSERVERS_BY_ID = {o.id: o for o in DEFAULT_OBSERVERS}


def get_observer(observer_id: str) -> SpaceObserver:
    try:
        return _OBSERVERS_BY_ID[observer_id]
    except KeyError:
        raise KeyError(f"unknown space observer {observer_id!r}; known: {sorted(_OBSERVERS_BY_ID)}") from None


# ---------------------------------------------------------------------------
# differential corrector (symmetric periodic orbits)
# ---------------------------------------------------------------------------
def _half_period(s0: np.ndarray, direction: int, tf: float = 10.0):
    ev = y_zero_crossing_event(direction)
    ev.terminal = True
    sol = propagate_cr3bp(s0, tf=tf, stm=True, events=ev)
    if len(sol.t_events[0]) == 0:
        raise RuntimeError("no y=0 crossing found")
    y = sol.y_events[0][0]
    return float(sol.t_events[0][0]), y[:6], y[6:].reshape(6, 6)


def _correct_symmetric(ic, free: tuple[int, ...], tol: float = 1e-11, max_iter: int = 25) -> tuple[np.ndarray, float, float]:
    """Polish a symmetric CR3BP periodic orbit (x–z plane crossing at t=0 and t=P/2).

    Newton iteration on the free IC components (``free`` ⊂ {0, 2, 4}) driving ẋ(P/2) = ż(P/2) = 0
    (planar orbits: ẋ only), with the usual time-of-crossing correction term
    ∂ẋ/∂X = Φ_{ẋ,X} − f_ẋ Φ_{y,X}/ẏ (Howell 1984, eq. 10).  Returns (ic, period, max residual).
    """
    s = np.array(ic, dtype=np.float64)
    planar = abs(s[2]) < 1e-14 and abs(s[5]) < 1e-14
    rows = (3,) if planar else (3, 5)
    direction = -1 if s[4] > 0 else +1
    err = np.array([np.inf])
    T2 = 0.0
    for _ in range(max_iter):
        T2, x, Phi = _half_period(s, direction)
        f = cr3bp_eom(0.0, x)
        err = np.array([x[r] for r in rows])
        if np.max(np.abs(err)) < tol:
            break
        J = np.array([[Phi[r, c] - f[r] * Phi[1, c] / x[4] for c in free] for r in rows])
        d = np.linalg.lstsq(J, -err, rcond=None)[0]
        n = np.linalg.norm(d)
        if n > 0.02:  # trust region to stay on the family
            d *= 0.02 / n
        for j, c in enumerate(free):
            s[c] += d[j]
    return s, 2.0 * T2, float(np.max(np.abs(err)))


# ---------------------------------------------------------------------------
# orbit resolution (library -> fallback) with caching
# ---------------------------------------------------------------------------
_ORBIT_CACHE: dict[tuple, PeriodicOrbit] = {}
_SOL_CACHE: dict[tuple, object] = {}
_LOCK = threading.RLock()


def _record_ic_period(rec):
    """Extract (ic, period_nd, id) from an :class:`~selene.orbits.library.OrbitRecord` (or any
    object/mapping exposing ``ic``/``x0`` and ``period_nd``/``period``/``T``/``period_days``)."""
    def get(name):
        if isinstance(rec, dict):
            return rec.get(name)
        return getattr(rec, name, None)

    ic = get("ic")
    if ic is None:
        ic = get("x0")
    period = None
    for name in ("period_nd", "period", "T"):
        period = get(name)
        if period is not None:
            break
    if period is None and get("period_days") is not None:
        period = float(get("period_days")) * 86400.0 / T_STAR
    if ic is None or period is None:
        return None
    ic = np.asarray(ic, dtype=np.float64).ravel()
    if ic.size != 6 or not np.isfinite(float(period)) or float(period) <= 0:
        return None
    rid = get("id") or get("name") or "?"
    return ic, float(period), str(rid)


def _library_or_none():
    """The orbit library singleton, or None when the module/data is unavailable (logged once)."""
    try:
        from selene.orbits.library import get_library  # lazy: built by another track
    except Exception as e:  # pragma: no cover - module absent
        _log.debug("orbit library not importable (%r); using fallback ICs", e)
        return None
    try:
        return get_library()
    except Exception as e:  # pragma: no cover - data missing / corrupt
        _log.warning("orbit library failed to load (%r); using fallback ICs", e)
        return None


def _mid_member(lib, family_key: str):
    members = lib.members(family_key) if hasattr(lib, "members") else []
    if not members:
        found = lib.find(family=family_key, nearest=False) if hasattr(lib, "find") else []
        members = list(found or [])
    return members[len(members) // 2] if members else None


def _from_library(observer: SpaceObserver) -> PeriodicOrbit | None:
    lib = _library_or_none()
    if lib is None:
        return None
    rec = None
    try:
        if observer.orbit_ref:
            by_id = getattr(lib, "by_id", None)
            if by_id is not None and observer.orbit_ref in by_id:
                rec = by_id[observer.orbit_ref]
            else:
                rec = _mid_member(lib, observer.orbit_ref)   # family key such as 'resonant_2:1'
            if rec is None:
                _log.warning("orbit_ref %r not found in the orbit library (observer %s); trying the "
                             "platform default", observer.orbit_ref, observer.id)
        if rec is None:
            for q in _LIBRARY_QUERIES.get(observer.platform_orbit, ()):
                if observer.platform_orbit in _MID_FAMILY_QUERIES and set(q) == {"family"}:
                    rec = _mid_member(lib, q["family"])
                else:
                    rec = lib.find(**q)
                if rec is not None:
                    break
    except Exception as e:  # defensive: a library API change must not break the sensor stack
        _log.warning("orbit library lookup failed for %s (%r); using fallback ICs", observer.id, e)
        return None
    if rec is None:
        _log.info("orbit library has no record for platform %r; using fallback ICs", observer.platform_orbit)
        return None
    got = _record_ic_period(rec)
    if got is None:
        _log.warning("orbit library record %r has no usable ic/period; using fallback ICs", rec)
        return None
    ic, period, rid = got
    closure = getattr(rec, "closure_error", float("nan"))
    return PeriodicOrbit(ic, period, observer.platform_orbit, f"library:{rid}", float(closure))


def resolve_orbit(observer: SpaceObserver) -> PeriodicOrbit:
    """Periodic orbit for a CR3BP-family observer (library if available, else polished fallback)."""
    if observer.platform_orbit == "geo":
        raise ValueError("GEO observers have no CR3BP orbit")
    key = (observer.platform_orbit, observer.orbit_ref, observer.custom_ic, observer.custom_period)
    with _LOCK:
        if key in _ORBIT_CACHE:
            return _ORBIT_CACHE[key]
        if observer.platform_orbit == "custom":
            if observer.custom_ic is None or observer.custom_period is None:
                raise ValueError("custom platform requires custom_ic and custom_period")
            orb = PeriodicOrbit(np.asarray(observer.custom_ic, float), float(observer.custom_period), "custom", "custom")
        else:
            orb = _from_library(observer)
            if orb is None:
                if observer.platform_orbit not in FALLBACK_ORBITS:
                    raise ValueError(
                        f"platform {observer.platform_orbit!r} is only available through selene.orbits.library "
                        "(no validated fallback IC); the library is not importable or has no matching record"
                    )
                ic0, p0 = FALLBACK_ORBITS[observer.platform_orbit]
                free = (4,) if observer.platform_orbit == "dro" else (0, 4)
                try:
                    ic, period, resid = _correct_symmetric(ic0, free)
                    if not np.isfinite(period) or period <= 0 or abs(period - p0) > 0.05 * p0:
                        raise RuntimeError("corrector left the family")
                except Exception:
                    ic, period, resid = np.asarray(ic0, float), p0, float("nan")
                orb = PeriodicOrbit(ic, period, observer.platform_orbit, "fallback", resid)
        _ORBIT_CACHE[key] = orb
        return orb


def _orbit_solution(orb: PeriodicOrbit):
    """Dense CR3BP solution over one period (cached)."""
    key = (tuple(np.round(orb.ic, 14)), round(orb.period, 12))
    with _LOCK:
        sol = _SOL_CACHE.get(key)
        if sol is None:
            res = propagate_cr3bp(orb.ic, tf=orb.period, dense_output=True)
            if not res.success:
                raise RuntimeError(f"CR3BP propagation of observer orbit failed: {res.message}")
            sol = res.sol
            _SOL_CACHE[key] = sol
        return sol


# ---------------------------------------------------------------------------
# states
# ---------------------------------------------------------------------------
#: WGS-84 equatorial radius used by astropy's EarthLocation [km]; the geodetic height of the
#: GEO point is chosen so its *geocentric* radius is exactly GEO_RADIUS_KM.
_WGS84_A_KM = 6378.137


def geo_state(longitude_deg: float, t_s) -> np.ndarray:
    """GCRS state [km, km/s] of an ideal geostationary point at east longitude ``longitude_deg``,
    shape (N,6), via the astropy ITRS→GCRS transform (cached per time grid, see sites.py)."""
    from selene.sensors.sites import GroundSite, site_frame

    t = np.atleast_1d(np.asarray(t_s, dtype=np.float64))
    pseudo = GroundSite(f"_geo_{float(longitude_deg):+.4f}", "geo point", 0.0, float(longitude_deg),
                        (GEO_RADIUS_KM - _WGS84_A_KM) * 1e3, 0.0, 0.0, 0.0, 0.0)
    fr = site_frame(pseudo, t)
    return np.concatenate([fr.pos_km, fr.vel_km_s], axis=1)


def geo_state_era(longitude_deg: float, t_s) -> np.ndarray:
    """Analytic ERA-only GEO state (no precession-nutation); cross-check for :func:`geo_state`."""
    t = np.atleast_1d(np.asarray(t_s, dtype=np.float64))
    jd_ut1 = J2000_JD + (t - _TDB_MINUS_UT1_S) / 86400.0
    tu = jd_ut1 - J2000_JD
    era = 2.0 * np.pi * (0.7790572732640 + 1.00273781191135448 * tu)
    theta = era + np.deg2rad(longitude_deg)
    r = GEO_RADIUS_KM
    pos = np.stack([r * np.cos(theta), r * np.sin(theta), np.zeros_like(theta)], axis=1)
    vel = OMEGA_EARTH * np.stack([-pos[:, 1], pos[:, 0], np.zeros_like(theta)], axis=1)
    return np.concatenate([pos, vel], axis=1)


def observer_state(observer: SpaceObserver, t_s) -> np.ndarray:
    """Earth-centered GCRF state [km, km/s] of the observer at TDB seconds ``t_s``.

    Scalar ``t_s`` -> (6,); array (N,) -> (N,6).  Vectorised: CR3BP platforms are evaluated from
    a cached one-period dense solution at τ = ((t − epoch)/T* + phase·P) mod P.
    """
    scalar = np.ndim(t_s) == 0
    t = np.atleast_1d(np.asarray(t_s, dtype=np.float64)).ravel()
    if t.size == 0:
        return np.zeros((0, 6))
    if observer.platform_orbit == "geo":
        out = geo_state(observer.geo_longitude_deg, t)
    else:
        orb = resolve_orbit(observer)
        sol = _orbit_solution(orb)
        tau = np.mod((t - observer.epoch_s) / T_STAR + observer.phase * orb.period, orb.period)
        s_rot = sol(tau).T  # (N,6)
        out = frames.rot_to_gcrf(s_rot, t)
    return out[0] if scalar else out


def observer_gcrf(observer: SpaceObserver, t_s) -> tuple[np.ndarray, np.ndarray]:
    """(position [km], velocity [km/s]) convenience wrapper around :func:`observer_state`."""
    s = observer_state(observer, t_s)
    return s[..., :3], s[..., 3:6]


def orbit_summary(observer: SpaceObserver) -> dict:
    """Human-readable orbit facts for the API (period in days, source, closure)."""
    if observer.platform_orbit == "geo":
        return {"source": "geostationary_itrs", "period_days": 0.99726968, "radius_km": GEO_RADIUS_KM,
                "longitude_deg": observer.geo_longitude_deg}
    try:
        orb = resolve_orbit(observer)
    except ValueError as e:
        return {"source": "unavailable", "error": str(e)}
    return {
        "source": orb.source,
        "period_days": orb.period * T_STAR / 86400.0,
        "ic_rot_nd": [float(v) for v in orb.ic],
        "closure_residual": None if not np.isfinite(orb.closure_error) else float(orb.closure_error),
        "length_unit_km": L_STAR,
    }
