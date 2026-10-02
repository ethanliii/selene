"""Named high-value regions of cislunar space for reachability classification.

Purpose (awareness and traffic safety, not targeting)
----------------------------------------------------
After an unannounced burn the question for a custody analyst is *"where could it be, and which
of the places we care about could it enter before we next see it?"*.  The regions below are the
places that matter for **sensor pointing, conjunction screening and traffic deconfliction**:
transit gateways (L1/L2 necks) that every low-energy transfer must pass through, the corridor
used by crewed/relay traffic (the 9:2 NRHO), the approaches to the lunar south pole where
landers and relays operate, low lunar orbit, the GEO belt (return paths that bring an xGEO object
back into the most valuable Earth-orbit regime) and departure from the Earth-Moon system.
Reachability of a region tells the tasker *where to point to re-acquire*; it is never used to
plan an engagement.

Frames and units
----------------
Membership tests take a :class:`RegionInputs` bundle holding the same points in several frames:

* ``pos_rot``  (..., 3)  instantaneous Earth-Moon rotating frame, nondimensional (barycentric,
  length unit = instantaneous Earth-Moon distance d(t); see ``dynamics/frames.py``).  Used for
  geometry tied to the rotating-frame landmarks (L1, L2, the NRHO).  Lengths in km quoted for
  these regions are converted with the conventional L* = 384 400 km (±5.5 % because d(t) varies).
* ``vel_rot``  (..., 3)  rotating-frame nondimensional velocity (needed for the Jacobi test).
* ``pos_gcrf`` (..., 3)  Earth-centred GCRF [km]; ``vel_gcrf`` [km/s].
* ``pos_moon`` (..., 3)  Moon-centred, GCRF axes [km] (exact selenocentric distances).

Definitions (defaults; every number is a configurable parameter of :func:`default_regions`)
-------------------------------------------------------------------------------------------
``l1_gateway``        **Neck transit**, not proximity.  Three conditions, all on the time series of
                      one trajectory (see :func:`transit_mask`):

                      1. *passage*: the path enters the ball |r_rot − L1| < R_neck (default
                         0.05 nd ≈ 19 200 km);
                      2. *realm change*: the last **definite realm** before the passage and the
                         first definite realm after it differ.  Realms are defined with a
                         hysteresis margin δ (default δ = R_neck): *Earth realm* x < x_L1 − δ,
                         *lunar realm* x_L1 + δ < x < x_L2 − δ, *exterior realm* x > x_L2 + δ;
                         the bands |x − x_L| ≤ δ around the two planes are the neck zones, where
                         the realm is undetermined and the previous definite realm is kept.
                         A passage that is unresolved at the end of the horizon (the path is
                         still in the neck zone) is **not** a transit — it is reported as an
                         approach by the callers (closest approach, plane crossing depth);
                      3. *energy*: the CR3BP-equivalent Jacobi constant on the passage is below
                         C(L1) (+ a 5e-3 slack for the ±1e-3/week wander of the ephemeris
                         model's C), i.e. the neck is energetically open.  Skipped when no
                         velocities are supplied (constructed position-only paths).

                      Point-wise inputs (no time series) fall back to the proximity ball.
                      Why this strictness (all numbers measured, ``tests/test_reachability_regions.py``):
                      large distant retrograde orbits straddle L1 and L2 geometrically — the demo
                      DRO (perilune 63 700 km, C ≈ 2.95) sweeps through both 0.05 nd balls every
                      revolution and comes within ~1 000 km of the x = x_L1 plane (its CR3BP
                      reference crosses it by 5 700 km) while being a stable, non-transiting orbit.
                      A plain plane-crossing test inside the ball was therefore noise: 10-20 m/s
                      perturbations of the DRO dip ~2 000 km past the plane and leave the ball on
                      the "Earth side" without ever leaving the lunar vicinity (none reached
                      x < x_L1 − 0.05), and the first version of this classifier called 14 % of
                      the demo's reachable rays "L1 gateway transits".  The hysteresis realms make
                      a transit mean what an analyst means: the object was in the lunar realm
                      proper and is now in the Earth realm proper, having passed the L1
                      neighbourhood (Koon-Lo-Marsden-Ross ch. 2-3, transit vs non-transit orbits).
                      Caveat stated honestly: at C ≈ 2.95 (< C(L4) ≈ 2.988) there is no forbidden
                      region at all, so the "neck" at DRO energies is a geometric neighbourhood,
                      not an energetic bottleneck; a path that leaves the lunar realm far from L1
                      (|r − L1| > R_neck) is not an *L1 gateway* transit under this definition.
``l2_gateway``        Same transit test at the L2 neck (lunar realm <-> exterior realm, energy gate
                      C < C(L2)).  The L2 neck connects the lunar realm to translunar/heliocentric
                      space and hosts far-side relays and halo/NRHO traffic.
``nrho_corridor``     within ``tube_km`` (default 10 000 km) of any point of the 9:2 synodic
                      resonant L2 southern NRHO from the orbit library (``NRHO_9:2`` record).
                      This is the Gateway/relay corridor; an unidentified object entering it is
                      a conjunction-screening and traffic-safety concern.
``south_pole_approach`` Moon-centred latitude < −60° (measured from the lunar *orbit* pole,
                      i.e. the rotating-frame ẑ; the lunar spin pole differs by ≈ 6.7°, the
                      Cassini-state obliquity — a documented approximation) and selenocentric
                      distance < 20 000 km.  Approach volume for south-pole landers, relays and
                      the NRHO perilune passes.
``llo_shell``         selenocentric altitude 0 ≤ h < 5 000 km (``llo_max_alt_km``); the shell
                      occupied by orbiters (LRO, KPLO, Chandrayaan, ...) and descent/ascent arcs.
                      The inner band h < 500 km is reported separately as ``llo_inner``.
``geo_belt_return``   geocentric radius within 42 164 ± 3 000 km, **or** geocentric distance
                      < 60 000 km while moving inward (r·v < 0).  An xGEO object falling back
                      toward the GEO belt is the highest-consequence Earth-orbit safety case.
``earth_return_escape`` exterior-realm excursion: CR3BP-equivalent Jacobi constant C < C(L2)
                      (the L2 neck is open, so the exterior realm is energetically accessible)
                      **and** barycentric distance > 1.3 nd (≈ 500 000 km), i.e. the object has
                      left the Earth-Moon interior realm through the L2 neck.  This is *not* a
                      hyperbolic-escape test: most such samples remain geocentrically bound
                      (measured for the DRO demo object at 150 m/s: geocentric two-body energy
                      −0.55 … −0.41 km²/s², 0 of 59 unbound, apogee of order 1e6 km, max
                      geocentric distance ~6e5 km within the week)
                      and may return on a free-return-like arc.  What matters for awareness is
                      that they are beyond the volume Moon-pointed sensors search, so custody
                      is lost unless the network is re-tasked outward.
``lunar_impact``      selenocentric altitude < 0 (or an osculating selenocentric periapsis
                      below the surface on an inbound pass — flagged by the sampler).  A pure
                      safety region.

The Jacobi value is evaluated with the CR3BP formula on the instantaneous rotating-frame state
of the ephemeris trajectory and is used only as an energy classifier.  ``frames.gcrf_to_rot``
nondimensionalises velocity with the *constant* T*, whereas the CR3BP formula assumes unit
angular rate; the ratio ω(t)·T* wobbles by ±5-8 % over the anomalistic month, which alone moves
C by ~0.04 along a DRO (30× the genuine non-conservation).  Callers therefore pass
``RegionInputs.omega_t_star`` = ω(t)·T* and the velocity is rescaled to the instantaneous rate
before the formula is applied; with that scaling C drifts by ~1e-3 over a week on the demo DRO
(measured), the true non-conservation of the ephemeris model.

References
----------
* Koon, Lo, Marsden, Ross (2011), *Dynamical Systems, the Three-Body Problem and Space Mission
  Design*, ch. 2-3 (realms, necks, Hill's regions, Jacobi constant).
* Zimovan-Spreen, Howell, Davis (2020), CMDA 132:28 — 9:2 synodic-resonant NRHO.
* Whitley & Martinez (2016), IEEE Aerospace — staging-orbit comparison (DRO/NRHO/halo).
* Holzinger, Chow, Garretson (2021), *A Primer on Cislunar Space*, AFRL — regions of interest
  for cislunar SDA (gateways, lunar vicinity, GEO return).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np
from scipy.spatial import cKDTree

from selene.constants import GEO_RADIUS_KM, L_STAR, MU, R_MOON
from selene.dynamics.cr3bp import jacobi, lagrange_points

__all__ = [
    "RegionInputs",
    "Region",
    "default_regions",
    "classify",
    "nrho_reference_samples",
    "transit_mask",
    "realm_labels",
    "neck_passage_diagnostics",
    "GATEWAY_JACOBI_TOL",
    "jacobi_rot",
    "C_L1",
    "C_L2",
]

_L = lagrange_points(MU)
L1_ROT = _L[0]
L2_ROT = _L[1]
MOON_ROT = np.array([1.0 - MU, 0.0, 0.0])
#: Jacobi constants of the collinear points (zero velocity at L1/L2).
C_L1: float = float(jacobi(np.r_[L1_ROT, 0.0, 0.0, 0.0], MU))
C_L2: float = float(jacobi(np.r_[L2_ROT, 0.0, 0.0, 0.0], MU))


@dataclass
class RegionInputs:
    """Points in the frames needed by the membership tests; all arrays share a leading shape."""

    pos_rot: np.ndarray                 # (..., 3) nd
    pos_gcrf: np.ndarray                # (..., 3) km
    pos_moon: np.ndarray                # (..., 3) km, Moon-centred GCRF axes
    vel_rot: Optional[np.ndarray] = None    # (..., 3) nd (T*-scaled, as frames.gcrf_to_rot returns)
    vel_gcrf: Optional[np.ndarray] = None   # (..., 3) km/s
    omega_t_star: Optional[np.ndarray] = None   # (...) ω(t)·T*: instantaneous frame rate in units of 1/T* (1 if omitted)
    #: the last axis of ``shape`` is a time series of ONE trajectory per leading index ((T, 3) or (N, T, 3)),
    #: consecutive entries being consecutive epochs.  Needed by the gateway neck-transit tests; set False
    #: for unrelated points (the gateways then fall back to the proximity ball).
    series: bool = True

    @property
    def shape(self) -> tuple:
        return self.pos_rot.shape[:-1]

    @property
    def is_series(self) -> bool:
        return bool(self.series) and self.pos_rot.ndim >= 2 and self.pos_rot.shape[-2] >= 2

    @property
    def jacobi(self) -> np.ndarray:
        """CR3BP Jacobi constant with the velocity rescaled to the instantaneous frame rate."""
        if self.vel_rot is None:
            raise ValueError("Jacobi test needs vel_rot")
        v = self.vel_rot
        if self.omega_t_star is not None:
            v = v / np.asarray(self.omega_t_star, float)[..., None]
        return jacobi_rot(self.pos_rot, v)


def jacobi_rot(pos_rot, vel_rot) -> np.ndarray:
    """CR3BP Jacobi constant C = 2U − v² from rotating-frame nd position/velocity arrays."""
    s = np.concatenate([np.asarray(pos_rot, float), np.asarray(vel_rot, float)], axis=-1)
    return np.asarray(jacobi(s.reshape(-1, 6), MU)).reshape(s.shape[:-1])


@dataclass
class Region:
    key: str
    name: str
    kind: str                      # 'gateway' | 'corridor' | 'lunar' | 'earth' | 'escape' | 'safety'
    description: str
    why_it_matters: str
    params: dict
    member: Callable[[RegionInputs], np.ndarray] = field(repr=False)
    geometry: dict = field(default_factory=dict)   # UI hint: centre/radius/polyline in rot nd

    def contains(self, inputs: RegionInputs) -> np.ndarray:
        return np.asarray(self.member(inputs), dtype=bool)

    def as_dict(self) -> dict:
        return {"key": self.key, "name": self.name, "kind": self.kind, "description": self.description,
                "why_it_matters": self.why_it_matters, "params": dict(self.params), "geometry": self.geometry}


# ---------------------------------------------------------------------------
# reference geometry
# ---------------------------------------------------------------------------
_NRHO_CACHE: dict[int, np.ndarray] = {}


def nrho_reference_samples(n: int = 600) -> np.ndarray:
    """(n, 3) rotating-frame nd positions of the 9:2 synodic NRHO from the orbit library
    (falls back to the ``observers`` module fallback IC if the library has no tagged record)."""
    if n in _NRHO_CACHE:
        return _NRHO_CACHE[n]
    pts = None
    try:
        from selene.orbits.library import get_library

        lib = get_library()
        rec = lib.find(tag="NRHO_9:2")
        if rec is None:
            rec = lib.find(family="L2_halo_S", perilune_km=3300.0)
        if rec is not None:
            pts = lib.sample(rec, n)
    except Exception:  # pragma: no cover - library unavailable
        pts = None
    if pts is None:  # pragma: no cover
        from selene.dynamics.cr3bp import propagate_cr3bp
        from selene.sensors.observers import FALLBACK_ORBITS

        ic, period = FALLBACK_ORBITS["nrho"]
        sol = propagate_cr3bp(np.asarray(ic), t_eval=np.linspace(0.0, period, n))
        pts = sol.y[:3].T.copy()
    _NRHO_CACHE[n] = pts
    return pts


# ---------------------------------------------------------------------------
# membership tests
# ---------------------------------------------------------------------------
def _sphere(center: np.ndarray, radius_nd: float):
    def f(inp: RegionInputs):
        return np.linalg.norm(inp.pos_rot - center, axis=-1) < radius_nd
    return f


#: slack on the Jacobi energy gate: the CR3BP-equivalent C of an ephemeris-model trajectory wanders by ~1e-3 per
#: week (regions module docstring), so a geometric transit is not discarded for a few 1e-3 of apparent energy
GATEWAY_JACOBI_TOL: float = 5e-3


def realm_labels(x_rel: np.ndarray, margin_nd: float) -> np.ndarray:
    """Per-epoch realm label relative to one neck plane: +1 beyond ``x_rel > +margin`` (the Moon side of L1,
    the exterior side of L2), −1 beyond ``x_rel < −margin``, 0 inside the neck zone |x_rel| ≤ margin, nan
    where ``x_rel`` is nan (terminated sample)."""
    x = np.asarray(x_rel, float)
    with np.errstate(invalid="ignore"):
        lab = np.where(x > margin_nd, 1.0, np.where(x < -margin_nd, -1.0, 0.0))
    return np.where(np.isfinite(x), lab, np.nan)


def _ffill_definite(lab: np.ndarray) -> np.ndarray:
    """(M, T): the last definite (non-zero, finite) label at or before each epoch; nan if none yet."""
    M, T = lab.shape
    definite = np.isfinite(lab) & (lab != 0)
    idx = np.where(definite, np.arange(T)[None, :], -1)
    idx = np.maximum.accumulate(idx, axis=1)
    out = np.full((M, T), np.nan)
    ok = idx >= 0
    rows = np.broadcast_to(np.arange(M)[:, None], (M, T))
    out[ok] = lab[rows[ok], idx[ok]]
    return out


def transit_mask(inside: np.ndarray, x_rel: np.ndarray, margin_nd: float,
                 jacobi: Optional[np.ndarray] = None, c_open: Optional[float] = None,
                 c_tol: float = GATEWAY_JACOBI_TOL) -> np.ndarray:
    """Neck-transit epochs of ``M`` trajectories on ``T`` consecutive epochs (see the module docstring).

    ``inside`` (M, T) bool: inside the neck ball; ``x_rel`` (M, T): x − x_L in the rotating frame (nan after
    a sample terminated); ``margin_nd``: realm hysteresis margin δ; ``jacobi`` (M, T, optional): CR3BP-
    equivalent Jacobi constant, gated against ``c_open`` (+ ``c_tol``) on the passage epochs.

    A *passage* is a maximal run of consecutive inside epochs.  It is a transit when (1) the last definite
    realm (|x_rel| > δ) before the run and the first definite realm after it both exist and differ, and
    (2) the minimum C over the run is below ``c_open + c_tol`` (when ``jacobi`` is given).  Passages that
    are unresolved at either end (the trajectory starts in the neck zone, or ends there at the horizon or
    at a termination) are not transits.  Returns (M, T) bool: the epochs of transiting passages only (so
    the first True epoch is the neck entry).  Fully vectorised: runs are numbered with a cumulative sum of
    run-start marks and reduced with ``np.minimum.at``.
    """
    inside = np.asarray(inside, bool)
    x_rel = np.asarray(x_rel, float)
    M, T = inside.shape
    if T == 0 or not inside.any():
        return np.zeros_like(inside)
    lab = realm_labels(x_rel, margin_nd)
    before = _ffill_definite(lab)                                   # last definite realm at/before each epoch
    after = _ffill_definite(lab[:, ::-1])[:, ::-1]                  # first definite realm at/after each epoch
    pad = np.zeros((M, T + 1), dtype=np.int8)
    pad[:, :T] = inside
    d = np.diff(pad, axis=1, prepend=0)                             # (M, T+1): +1 run start, -1 run end (exclusive)
    r_s, k_s = np.where(d == 1)
    r_e, k_e = np.where(d == -1)                                    # same count and order as the starts
    n_runs = r_s.size
    run_id = np.cumsum((d[:, :T] == 1).ravel()).reshape(M, T)       # 1-based global run number at/after each start
    entry = np.where(k_s > 0, before[r_s, np.maximum(k_s - 1, 0)], np.nan)
    exit_ = np.where(k_e < T, after[r_s, np.minimum(k_e, T - 1)], np.nan)
    transit_run = np.isfinite(entry) & np.isfinite(exit_) & (entry != exit_)
    # the plane x = x_L must be crossed INSIDE the ball during the passage (a graze of the ball followed by a realm
    # change far from the libration point is not a transit of *this* neck)
    entry_at = np.zeros((M, T))
    entry_at[inside] = np.where(np.isfinite(entry), entry, 0.0)[run_id[inside] - 1]
    with np.errstate(invalid="ignore"):
        crossed = inside & (entry_at != 0) & (np.sign(x_rel) == -entry_at)
    crossed_run = np.zeros(n_runs, dtype=bool)
    np.logical_or.at(crossed_run, run_id[crossed] - 1, True)
    transit_run &= crossed_run
    if jacobi is not None and c_open is not None:
        C = np.asarray(jacobi, float)
        c_run = np.full(n_runs, np.inf)
        sel = inside & np.isfinite(C)
        np.minimum.at(c_run, run_id[sel] - 1, C[sel])
        transit_run &= c_run < float(c_open) + float(c_tol)
    out = np.zeros_like(inside)
    out[inside] = transit_run[run_id[inside] - 1]
    return out


def _neck_transit(center: np.ndarray, radius_nd: float, margin_nd: float, c_open: float, c_tol: float = GATEWAY_JACOBI_TOL):
    """Gateway membership: neck transit through the ball of ``radius_nd`` around ``center`` with realm margin
    ``margin_nd`` and energy gate ``c_open`` (see the module docstring); proximity-ball fallback for
    point-wise inputs."""
    x_l = float(center[0])
    ball = _sphere(center, radius_nd)

    def f(inp: RegionInputs):
        if not inp.is_series:
            return ball(inp)
        shape = inp.shape
        P = np.asarray(inp.pos_rot, float).reshape(-1, shape[-1], 3)
        with np.errstate(invalid="ignore"):
            inside = np.linalg.norm(P - center, axis=-1) < radius_nd      # nan -> False
        C = None
        if inp.vel_rot is not None:
            with np.errstate(invalid="ignore"):
                C = np.asarray(inp.jacobi, float).reshape(-1, shape[-1])
        return transit_mask(inside, P[..., 0] - x_l, margin_nd, C, c_open, c_tol).reshape(shape)
    return f


def neck_passage_diagnostics(pos_rot: np.ndarray, center: np.ndarray, radius_nd: float, margin_nd: float,
                             active: Optional[np.ndarray] = None, transit: Optional[np.ndarray] = None) -> dict:
    """What a set of trajectories does at one neck, for honest reporting when little or nothing transits.

    ``pos_rot`` (N, T, 3) rotating-frame positions, ``active`` (N, T) bool mask (default all), ``transit`` (N, T)
    bool output of the gateway region (default: none).  Returns the closest approach to the libration point over
    all active epochs (km, with sample and epoch index), how many trajectories enter the ball, cross the plane
    x = x_L inside the ball, dip past the plane inside the ball and come back to the realm they started from
    (``n_cross_and_return``), reach the far realm proper at all (``n_reach_far_realm``), reach it without a
    counted transit (``n_far_realm_other_route``) and, for those, the smallest distance to the libration point at
    which they crossed the plane toward the far realm (``other_route_min_crossing_km``); plus the deepest
    excursion past the plane inside the ball (km)."""
    P = np.asarray(pos_rot, float)
    N, T = P.shape[:2]
    act = np.ones((N, T), bool) if active is None else np.asarray(active, bool)
    tr_any = np.zeros(N, bool) if transit is None else np.asarray(transit, bool).reshape(N, T).any(axis=1)
    with np.errstate(invalid="ignore"):
        d = np.where(act, np.linalg.norm(P - center, axis=-1), np.nan)
        x_rel = np.where(act, P[..., 0] - float(center[0]), np.nan)
    inside = d < radius_nd
    lab = realm_labels(x_rel, margin_nd)
    start = _ffill_definite(lab[:, ::-1])[:, ::-1][:, 0]     # the realm each trajectory starts from: its first definite label
    far = -start                                             # the realm on the other side of the plane
    with np.errstate(invalid="ignore"):
        past = np.isfinite(x_rel) & (np.sign(x_rel) == far[:, None]) & (x_rel != 0)      # on the far side of the plane
        crossed = inside & past
        depth = np.where(crossed, -x_rel * start[:, None], 0.0)                          # positive = past the plane
        reach_far = np.isfinite(lab) & (lab == far[:, None]) & np.isfinite(start)[:, None]
    cross_any = crossed.any(axis=1)
    far_any = reach_far.any(axis=1)
    other = far_any & ~tr_any
    out = {"n": int(N), "n_enter_ball": int(inside.any(axis=1).sum()), "n_cross_plane_in_ball": int(cross_any.sum()),
           "n_cross_and_return": int((cross_any & ~far_any).sum()),
           "n_reach_far_realm": int(far_any.sum()), "n_far_realm_other_route": int(other.sum()),
           "max_depth_past_plane_km": float(np.nanmax(depth) * L_STAR) if crossed.any() else 0.0,
           "other_route_min_crossing_km": None}
    if other.any():
        # plane crossings toward the far side: epoch k with sign(x_rel[k]) == far and sign(x_rel[k-1]) == start
        prev_start = np.zeros_like(past)
        with np.errstate(invalid="ignore"):
            prev_start[:, 1:] = np.isfinite(x_rel[:, :-1]) & (np.sign(x_rel[:, :-1]) == start[:, None])
        xing = past & prev_start & other[:, None]
        dd = np.where(xing, d, np.nan)
        if np.isfinite(dd).any():
            out["other_route_min_crossing_km"] = float(np.nanmin(dd) * L_STAR)
    if np.isfinite(d).any():
        i, k = np.unravel_index(int(np.nanargmin(d)), d.shape)
        out.update(closest_km=float(d[i, k] * L_STAR), closest_sample=int(i), closest_epoch_index=int(k))
    else:
        out.update(closest_km=None, closest_sample=None, closest_epoch_index=None)
    return out


def _tube(points: np.ndarray, radius_nd: float):
    tree = cKDTree(points)

    def f(inp: RegionInputs):
        p = inp.pos_rot.reshape(-1, 3)
        d, _ = tree.query(p, k=1, distance_upper_bound=radius_nd * 1.0001)
        return (d < radius_nd).reshape(inp.shape)
    return f


def _moon_lat_alt(inp: RegionInputs):
    """Selenocentric distance [km] and latitude [rad] w.r.t. the lunar orbit pole (rotating ẑ)."""
    rm = inp.pos_rot - MOON_ROT
    rn = np.linalg.norm(rm, axis=-1)
    with np.errstate(invalid="ignore", divide="ignore"):
        lat = np.arcsin(np.clip(rm[..., 2] / np.where(rn > 0, rn, np.nan), -1.0, 1.0))
    dist_km = np.linalg.norm(inp.pos_moon, axis=-1)
    return dist_km, lat


def _south_pole(lat_deg: float, max_dist_km: float):
    lat_lim = np.deg2rad(lat_deg)

    def f(inp: RegionInputs):
        dist, lat = _moon_lat_alt(inp)
        return (lat < lat_lim) & (dist < max_dist_km)
    return f


def _alt_band(h_min_km: float, h_max_km: float):
    def f(inp: RegionInputs):
        h = np.linalg.norm(inp.pos_moon, axis=-1) - R_MOON
        return (h >= h_min_km) & (h < h_max_km)
    return f


def _geo_return(belt_half_width_km: float, inward_radius_km: float):
    def f(inp: RegionInputs):
        r = np.linalg.norm(inp.pos_gcrf, axis=-1)
        in_belt = np.abs(r - GEO_RADIUS_KM) < belt_half_width_km
        if inp.vel_gcrf is None:
            return in_belt
        inward = (r < inward_radius_km) & (np.sum(inp.pos_gcrf * inp.vel_gcrf, axis=-1) < 0.0)
        return in_belt | inward
    return f


def _escape(c_max: float, r_min_nd: float):
    def f(inp: RegionInputs):
        rb = np.linalg.norm(inp.pos_rot, axis=-1)
        outside = rb > r_min_nd
        if inp.vel_rot is None:
            return outside
        return outside & (inp.jacobi < c_max)
    return f


def _impact():
    def f(inp: RegionInputs):
        return np.linalg.norm(inp.pos_moon, axis=-1) < R_MOON
    return f


# ---------------------------------------------------------------------------
def default_regions(
    gateway_radius_nd: float = 0.05,
    nrho_tube_km: float = 10_000.0,
    realm_margin_nd: Optional[float] = None,
    gateway_jacobi_tol: float = GATEWAY_JACOBI_TOL,
    south_pole_lat_deg: float = -60.0,
    south_pole_max_dist_km: float = 20_000.0,
    llo_max_alt_km: float = 5_000.0,
    llo_inner_alt_km: float = 500.0,
    geo_half_width_km: float = 3_000.0,
    geo_inward_radius_km: float = 60_000.0,
    escape_r_min_nd: float = 1.3,
    nrho_samples: int = 600,
) -> list[Region]:
    """The default region set (see module docstring for definitions and rationale).

    ``realm_margin_nd`` is the realm hysteresis margin δ of the gateway transit test (default: the neck radius)."""
    nrho_pts = nrho_reference_samples(nrho_samples)
    tube_nd = nrho_tube_km / L_STAR
    margin = float(gateway_radius_nd if realm_margin_nd is None else realm_margin_nd)
    gw_desc = ("Neck transit: the path enters the ball of radius {r:.3f} nd (~{rk:,.0f} km) around the Earth-Moon {L} point "
               "having last been in the {a} realm proper (x {sa} x_{L} {pm} {m:.3f} nd, rotating frame) and next reaches the {b} "
               "realm proper (x {sb} x_{L} {mp} {m:.3f} nd), or the reverse, with a CR3BP-equivalent Jacobi constant below C({L}) "
               "= {c:.4f} on the passage (neck energetically open). Grazing the ball, dipping past the plane and returning to "
               "the same realm (what a large DRO and its small perturbations do), or ending the horizon inside the neck zone, "
               "does not count.")
    gw_params = lambda L, c: {"center_rot_nd": L.tolist(), "radius_nd": gateway_radius_nd, "radius_km_approx": gateway_radius_nd * L_STAR,  # noqa: E731
                              "membership": "neck_transit", "plane_x_nd": float(L[0]), "realm_margin_nd": margin,
                              "realm_margin_km_approx": margin * L_STAR, "energy_gate": {"C_open": c, "tol": gateway_jacobi_tol},
                              "unresolved_passages_count": False}
    regions = [
        Region(
            "l1_gateway", "L1 gateway", "gateway",
            gw_desc.format(r=gateway_radius_nd, rk=gateway_radius_nd * L_STAR, L="L1", a="lunar", sa=">", pm="+", m=margin,
                           b="Earth", sb="<", mp="-", c=C_L1),
            "Every low-energy transfer between the Earth realm and the lunar realm threads the L1 neck; an "
            "object that can transit it can reach the Moon-Earth corridor and the lunar vicinity.",
            gw_params(L1_ROT, C_L1),
            _neck_transit(L1_ROT, gateway_radius_nd, margin, C_L1, gateway_jacobi_tol),
            {"type": "sphere", "center_rot_nd": L1_ROT.tolist(), "radius_nd": gateway_radius_nd, "membership": "neck_transit"},
        ),
        Region(
            "l2_gateway", "L2 gateway", "gateway",
            gw_desc.format(r=gateway_radius_nd, rk=gateway_radius_nd * L_STAR, L="L2", a="lunar", sa="<", pm="-", m=margin,
                           b="exterior", sb=">", mp="+", c=C_L2),
            "The L2 neck is the doorway to translunar space and the home of far-side relays and halo/NRHO "
            "traffic; it is also the hardest region to observe from Earth (behind the Moon, lunar glare).",
            gw_params(L2_ROT, C_L2),
            _neck_transit(L2_ROT, gateway_radius_nd, margin, C_L2, gateway_jacobi_tol),
            {"type": "sphere", "center_rot_nd": L2_ROT.tolist(), "radius_nd": gateway_radius_nd, "membership": "neck_transit"},
        ),
        Region(
            "nrho_corridor", "NRHO corridor (9:2)", "corridor",
            f"Tube of radius {nrho_tube_km:,.0f} km around the 9:2 synodic-resonant L2 southern NRHO "
            f"(orbit-library record, {nrho_samples} sample points; rotating frame).",
            "The 9:2 NRHO is the staging/relay corridor (Gateway class); an unidentified object entering the "
            "tube is a conjunction-screening and traffic-safety concern for the notional allied relay.",
            {"tube_km": nrho_tube_km, "tube_nd": tube_nd, "n_samples": nrho_samples},
            _tube(nrho_pts, tube_nd),
            {"type": "tube", "polyline_rot_nd": nrho_pts[:: max(1, nrho_samples // 150)].tolist(), "radius_nd": tube_nd},
        ),
        Region(
            "south_pole_approach", "Lunar south-pole approach", "lunar",
            f"Moon-centred cone: latitude < {south_pole_lat_deg:.0f}° (w.r.t. the lunar orbit pole) and "
            f"selenocentric distance < {south_pole_max_dist_km:,.0f} km.",
            "South-pole landers, relays and the NRHO perilune passes all live here; it is the busiest "
            "lunar approach volume of the coming decade and sits in the Earth-facing/far-side limb geometry "
            "that ground sensors struggle with.",
            {"lat_max_deg": south_pole_lat_deg, "max_dist_km": south_pole_max_dist_km,
             "note": "latitude measured from the Earth-Moon orbit pole, not the lunar spin pole (6.7 deg apart)"},
            _south_pole(south_pole_lat_deg, south_pole_max_dist_km),
            {"type": "cone", "apex_rot_nd": MOON_ROT.tolist(), "axis": [0, 0, -1],
             "half_angle_deg": 90.0 + south_pole_lat_deg, "length_nd": south_pole_max_dist_km / L_STAR},
        ),
        Region(
            "llo_shell", "Low lunar orbit shell", "lunar",
            f"Selenocentric altitude 0 <= h < {llo_max_alt_km:,.0f} km.",
            "The shell occupied by science orbiters and descent/ascent arcs; entry means a close lunar "
            "approach and a conjunction-screening obligation.",
            {"alt_min_km": 0.0, "alt_max_km": llo_max_alt_km},
            _alt_band(0.0, llo_max_alt_km),
            {"type": "shell", "center_rot_nd": MOON_ROT.tolist(), "r_inner_nd": R_MOON / L_STAR,
             "r_outer_nd": (R_MOON + llo_max_alt_km) / L_STAR},
        ),
        Region(
            "llo_inner", "Low lunar orbit (inner, < 500 km)", "lunar",
            f"Selenocentric altitude 0 <= h < {llo_inner_alt_km:,.0f} km.",
            "Where most active lunar orbiters fly; the tightest conjunction-screening volume.",
            {"alt_min_km": 0.0, "alt_max_km": llo_inner_alt_km},
            _alt_band(0.0, llo_inner_alt_km),
            {"type": "shell", "center_rot_nd": MOON_ROT.tolist(), "r_inner_nd": R_MOON / L_STAR,
             "r_outer_nd": (R_MOON + llo_inner_alt_km) / L_STAR},
        ),
        Region(
            "geo_belt_return", "GEO belt return", "earth",
            f"Geocentric radius within {GEO_RADIUS_KM:,.0f} ± {geo_half_width_km:,.0f} km, or geocentric distance "
            f"< {geo_inward_radius_km:,.0f} km while moving inward (r·v < 0).",
            "An xGEO object dropping back toward the GEO belt is the highest-consequence Earth-orbit safety "
            "case: it crosses the most valuable orbital regime at km/s relative speeds and is hard to track "
            "until it is close.",
            {"geo_radius_km": GEO_RADIUS_KM, "half_width_km": geo_half_width_km, "inward_radius_km": geo_inward_radius_km},
            _geo_return(geo_half_width_km, geo_inward_radius_km),
            {"type": "shell_earth", "r_inner_km": GEO_RADIUS_KM - geo_half_width_km,
             "r_outer_km": GEO_RADIUS_KM + geo_half_width_km, "inward_radius_km": geo_inward_radius_km},
        ),
        Region(
            "earth_return_escape", "Exterior-realm excursion (beyond the L2 neck)", "escape",
            f"CR3BP-equivalent Jacobi constant C < C(L2) = {C_L2:.4f} (L2 neck energetically open) and barycentric "
            f"distance > {escape_r_min_nd:.2f} nd (~{escape_r_min_nd * L_STAR:,.0f} km): the object has left the "
            "Earth-Moon interior realm through the L2 neck. Not a hyperbolic-escape test: such objects are usually "
            "still geocentrically bound (apogee of order 1e6 km) and may return on a free-return-like arc.",
            "Beyond ~500 000 km the object is outside the volume Moon-pointed sensors search; whether it later "
            "escapes or swings back, custody is lost unless the network is re-tasked outward.",
            {"C_max": C_L2, "r_min_nd": escape_r_min_nd, "C_L1": C_L1, "C_L2": C_L2},
            _escape(C_L2, escape_r_min_nd),
            {"type": "outside_sphere", "center_rot_nd": [0.0, 0.0, 0.0], "radius_nd": escape_r_min_nd},
        ),
        Region(
            "lunar_impact", "Lunar impact", "safety",
            "Selenocentric altitude below the surface (the sampler also flags inbound passes whose osculating "
            "selenocentric periapsis is below the surface).",
            "Pure safety: an impacting trajectory ends custody and may endanger surface assets.",
            {"radius_km": R_MOON},
            _impact(),
            {"type": "sphere", "center_rot_nd": MOON_ROT.tolist(), "radius_nd": R_MOON / L_STAR},
        ),
    ]
    return regions


def classify(inputs: RegionInputs, regions: Optional[list[Region]] = None) -> dict[str, np.ndarray]:
    """``{region.key: bool array of inputs.shape}``."""
    regions = default_regions() if regions is None else regions
    return {r.key: r.contains(inputs) for r in regions}
