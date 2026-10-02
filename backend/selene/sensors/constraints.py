"""Geometric observing constraints for ground sites and space-based optical observers.

Every function broadcasts over leading axes: positions are ``(..., 3)`` arrays in Earth-centered
GCRF [km]; a typical coverage call passes observers as ``(T, 1, 3)`` and target cells as
``(T, C, 3)``.  Public constraint checks return boolean arrays where ``True`` means the
constraint is **violated**; the ``*_reasons`` helpers OR the corresponding bits from
:mod:`selene.sensors.reasons` into an integer mask.

Definitions
-----------
* Angular separation: ``atan2(|u×v|, u·v)`` (stable for small and near-π angles).
* Sun exclusion (space): separation(target, Sun) < ``sun_excl``.
* Earth exclusion (space): separation(target, Earth centre) − Earth angular radius < ``earth_excl``
  (angle from the *limb*).  A target projected onto the Earth disc is therefore always flagged,
  whatever the exclusion angle: it is occulted if it lies beyond the tangent point, and seen
  against the sunlit/airglow Earth background if in front (unobservable for a passive optical
  sensor either way).
* Moon exclusion: separation(target, Moon centre) − Moon angular radius < ``moon_excl``.
* Eclipse: a cylindrical umbra of radius R_body along the anti-Sun direction; a target inside
  (behind the body as seen from the Sun, within one body radius of the axis) is unlit and
  invisible to a passive optical sensor.  The penumbra and the finite Sun are neglected (the
  cylinder slightly over-estimates the Earth's umbra; see Montenbruck & Gill §3.4).
* Ground elevation: elevation = asin( (target − site)·ẑ_geodetic / |target − site| ).
* Daylight: Sun elevation at the site > ``sun_elev_max`` (default −12°, astronomical twilight;
  −18° would be fully dark but is unnecessarily conservative for bright-ish targets).
* Lunar glare (ground): Moon–target separation must exceed
  ``θ_excl = θ_min + (θ_max − θ_min) · f_illum`` with defaults 3° → 15°, where ``f_illum`` is the
  Moon's illuminated fraction (``(1 + cos α)/2``, α = Sun–Moon–Earth phase angle).  **This is a
  modelling assumption** standing in for scattered moonlight / sky-background loss near a bright
  Moon; real sites would fold it into a sky-brightness-dependent limiting magnitude.
"""
from __future__ import annotations

import numpy as np

from selene.constants import R_EARTH, R_MOON
from selene.sensors.reasons import (
    REASON_DAYLIGHT,
    REASON_EARTH_EXCLUSION,
    REASON_IN_SHADOW,
    REASON_LOW_ELEVATION,
    REASON_MOON_EXCLUSION,
    REASON_OUT_OF_FOV,
    REASON_SUN_EXCLUSION,
)

__all__ = [
    "angular_separation",
    "angular_radius",
    "sun_exclusion",
    "earth_exclusion",
    "moon_exclusion",
    "in_body_shadow",
    "in_earth_shadow",
    "in_moon_shadow",
    "elevation_angle",
    "sun_elevation",
    "illuminated_fraction",
    "lunar_glare_threshold",
    "lunar_glare_exclusion",
    "out_of_fov",
    "space_reasons",
    "ground_reasons",
    "GLARE_THETA_MIN_DEG",
    "GLARE_THETA_MAX_DEG",
]

GLARE_THETA_MIN_DEG = 3.0    # new Moon: only the Moon's own disc + a small halo is lost (assumption)
GLARE_THETA_MAX_DEG = 15.0   # full Moon: scattered moonlight kills faint detections out to ~15° (assumption)


def _dot(a, b):
    return np.sum(a * b, axis=-1)


def _norm(a):
    return np.linalg.norm(a, axis=-1)


def angular_separation(u, v) -> np.ndarray:
    """Angle [rad] between direction vectors ``u`` and ``v`` (``(...,3)``, broadcastable)."""
    u = np.asarray(u, dtype=np.float64)
    v = np.asarray(v, dtype=np.float64)
    return np.arctan2(_norm(np.cross(u, v)), _dot(u, v))


def angular_radius(radius_km: float, distance_km) -> np.ndarray:
    """Apparent angular radius [rad] of a sphere; π/2 if the observer is inside it."""
    d = np.asarray(distance_km, dtype=np.float64)
    return np.arcsin(np.clip(radius_km / np.maximum(d, 1e-9), 0.0, 1.0))


# ---------------------------------------------------------------------------
# space-observer exclusions
# ---------------------------------------------------------------------------
def sun_exclusion(obs_pos, tgt_pos, sun_pos, excl_rad) -> np.ndarray:
    """True where the target lies within ``excl_rad`` of the Sun as seen from the observer."""
    los = np.asarray(tgt_pos, dtype=np.float64) - obs_pos
    return angular_separation(los, np.asarray(sun_pos, dtype=np.float64) - obs_pos) < excl_rad


def _body_exclusion(obs_pos, tgt_pos, body_pos, body_radius_km, excl_rad) -> np.ndarray:
    obs_pos = np.asarray(obs_pos, dtype=np.float64)
    los = np.asarray(tgt_pos, dtype=np.float64) - obs_pos
    to_body = np.asarray(body_pos, dtype=np.float64) - obs_pos
    d_body = _norm(to_body)
    sep = angular_separation(los, to_body)
    rho = angular_radius(body_radius_km, d_body)
    too_close = (sep - rho) < excl_rad
    # Occultation: inside the disc *and* beyond the tangent point along the LOS.
    behind = (sep < rho) & (_norm(los) > np.sqrt(np.maximum(d_body**2 - body_radius_km**2, 0.0)))
    return too_close | behind


def earth_exclusion(obs_pos, tgt_pos, excl_rad) -> np.ndarray:
    """True where the target is within ``excl_rad`` of the Earth's limb (or behind the Earth).
    Earth is at the GCRF origin."""
    return _body_exclusion(obs_pos, tgt_pos, np.zeros(3), R_EARTH, excl_rad)


def moon_exclusion(obs_pos, tgt_pos, moon_pos, excl_rad) -> np.ndarray:
    """True where the target is within ``excl_rad`` of the Moon's limb (or behind the Moon)."""
    return _body_exclusion(obs_pos, tgt_pos, moon_pos, R_MOON, excl_rad)


# ---------------------------------------------------------------------------
# eclipse (cylindrical umbra)
# ---------------------------------------------------------------------------
def in_body_shadow(tgt_pos, body_pos, body_radius_km, sun_pos) -> np.ndarray:
    """True where ``tgt_pos`` is inside the cylindrical shadow of a body of radius ``body_radius_km``.

    Shadow axis: from the body centre along the anti-Sun direction.  Inside if the along-axis
    coordinate is positive (behind the body) and the perpendicular distance < radius.
    """
    rel = np.asarray(tgt_pos, dtype=np.float64) - body_pos
    s_hat = np.asarray(sun_pos, dtype=np.float64) - body_pos
    s_hat = s_hat / _norm(s_hat)[..., None]
    along = _dot(rel, s_hat)          # > 0 toward the Sun
    perp2 = _dot(rel, rel) - along**2
    return (along < 0.0) & (perp2 < body_radius_km**2)


def in_earth_shadow(tgt_pos, sun_pos) -> np.ndarray:
    return in_body_shadow(tgt_pos, np.zeros(3), R_EARTH, sun_pos)


def in_moon_shadow(tgt_pos, moon_pos, sun_pos) -> np.ndarray:
    return in_body_shadow(tgt_pos, moon_pos, R_MOON, sun_pos)


# ---------------------------------------------------------------------------
# ground-site geometry
# ---------------------------------------------------------------------------
def elevation_angle(site_pos, zenith, tgt_pos) -> np.ndarray:
    """Elevation [rad] of ``tgt_pos`` above the site's geodetic horizon (no refraction)."""
    los = np.asarray(tgt_pos, dtype=np.float64) - site_pos
    los = los / _norm(los)[..., None]
    return np.arcsin(np.clip(_dot(los, zenith), -1.0, 1.0))


def sun_elevation(site_pos, zenith, sun_pos) -> np.ndarray:
    """Elevation [rad] of the Sun's centre at the site (geometric, no refraction)."""
    return elevation_angle(site_pos, zenith, sun_pos)


def illuminated_fraction(sun_pos, moon_pos, viewer_pos=None) -> np.ndarray:
    """Illuminated fraction of the Moon's disc as seen from ``viewer_pos`` (default: Earth centre):
    k = (1 + cos α)/2 with α the Sun–Moon–viewer angle at the Moon (Meeus, *Astronomical
    Algorithms*, ch. 48)."""
    moon_pos = np.asarray(moon_pos, dtype=np.float64)
    viewer = np.zeros(3) if viewer_pos is None else np.asarray(viewer_pos, dtype=np.float64)
    alpha = angular_separation(np.asarray(sun_pos, dtype=np.float64) - moon_pos, viewer - moon_pos)
    return 0.5 * (1.0 + np.cos(alpha))


def lunar_glare_threshold(illum_frac, theta_min_deg=GLARE_THETA_MIN_DEG, theta_max_deg=GLARE_THETA_MAX_DEG) -> np.ndarray:
    """Required Moon–target separation [rad] as a function of illuminated fraction (assumption)."""
    k = np.clip(np.asarray(illum_frac, dtype=np.float64), 0.0, 1.0)
    return np.deg2rad(theta_min_deg + (theta_max_deg - theta_min_deg) * k)


def lunar_glare_exclusion(site_pos, tgt_pos, moon_pos, sun_pos,
                          theta_min_deg=GLARE_THETA_MIN_DEG, theta_max_deg=GLARE_THETA_MAX_DEG) -> np.ndarray:
    """True where a ground target is inside the phase-dependent lunar-glare exclusion zone
    (or behind the Moon's disc)."""
    site_pos = np.asarray(site_pos, dtype=np.float64)
    moon_pos = np.asarray(moon_pos, dtype=np.float64)
    k = illuminated_fraction(sun_pos, moon_pos, site_pos)
    thr = lunar_glare_threshold(k, theta_min_deg, theta_max_deg)
    los = np.asarray(tgt_pos, dtype=np.float64) - site_pos
    to_moon = moon_pos - site_pos
    sep = angular_separation(los, to_moon)
    occulted = (sep < angular_radius(R_MOON, _norm(to_moon))) & (_norm(los) > _norm(to_moon))
    return (sep < thr) | occulted


def out_of_fov(obs_pos, tgt_pos, boresight, fov_deg) -> np.ndarray:
    """True where the target is farther than half the FOV from the ``boresight`` direction."""
    los = np.asarray(tgt_pos, dtype=np.float64) - obs_pos
    return angular_separation(los, boresight) > np.deg2rad(0.5 * float(fov_deg))


# ---------------------------------------------------------------------------
# reason-mask assemblers
# ---------------------------------------------------------------------------
def space_reasons(obs_pos, tgt_pos, sun_pos, moon_pos, *, sun_excl_deg, earth_excl_deg, moon_excl_deg,
                  boresight=None, fov_deg=None) -> np.ndarray:
    """Geometric reason bitmask for a space-based observer (photometry is added by visibility.py)."""
    obs_pos = np.asarray(obs_pos, dtype=np.float64)
    tgt_pos = np.asarray(tgt_pos, dtype=np.float64)
    sun_pos = np.asarray(sun_pos, dtype=np.float64)
    moon_pos = np.asarray(moon_pos, dtype=np.float64)
    shape = np.broadcast(obs_pos[..., 0], tgt_pos[..., 0], sun_pos[..., 0], moon_pos[..., 0]).shape
    mask = np.zeros(shape, dtype=np.int64)
    mask |= np.where(sun_exclusion(obs_pos, tgt_pos, sun_pos, np.deg2rad(sun_excl_deg)), REASON_SUN_EXCLUSION, 0)
    mask |= np.where(earth_exclusion(obs_pos, tgt_pos, np.deg2rad(earth_excl_deg)), REASON_EARTH_EXCLUSION, 0)
    mask |= np.where(moon_exclusion(obs_pos, tgt_pos, moon_pos, np.deg2rad(moon_excl_deg)), REASON_MOON_EXCLUSION, 0)
    shadow = in_earth_shadow(tgt_pos, sun_pos) | in_moon_shadow(tgt_pos, moon_pos, sun_pos)
    mask |= np.where(shadow, REASON_IN_SHADOW, 0)
    if boresight is not None and fov_deg is not None:
        mask |= np.where(out_of_fov(obs_pos, tgt_pos, boresight, fov_deg), REASON_OUT_OF_FOV, 0)
    return mask


def ground_reasons(site_pos, zenith, tgt_pos, sun_pos, moon_pos, *, min_elevation_deg, sun_elev_max_deg,
                   glare_min_deg=GLARE_THETA_MIN_DEG, glare_max_deg=GLARE_THETA_MAX_DEG,
                   boresight=None, fov_deg=None) -> np.ndarray:
    """Geometric reason bitmask for a ground optical site."""
    site_pos = np.asarray(site_pos, dtype=np.float64)
    zenith = np.asarray(zenith, dtype=np.float64)
    tgt_pos = np.asarray(tgt_pos, dtype=np.float64)
    sun_pos = np.asarray(sun_pos, dtype=np.float64)
    moon_pos = np.asarray(moon_pos, dtype=np.float64)
    shape = np.broadcast(site_pos[..., 0], tgt_pos[..., 0], sun_pos[..., 0], moon_pos[..., 0]).shape
    mask = np.zeros(shape, dtype=np.int64)
    day = sun_elevation(site_pos, zenith, sun_pos) > np.deg2rad(sun_elev_max_deg)
    mask |= np.where(np.broadcast_to(day, shape), REASON_DAYLIGHT, 0)
    low = elevation_angle(site_pos, zenith, tgt_pos) < np.deg2rad(min_elevation_deg)
    mask |= np.where(low, REASON_LOW_ELEVATION, 0)
    mask |= np.where(lunar_glare_exclusion(site_pos, tgt_pos, moon_pos, sun_pos, glare_min_deg, glare_max_deg),
                     REASON_MOON_EXCLUSION, 0)
    shadow = in_earth_shadow(tgt_pos, sun_pos) | in_moon_shadow(tgt_pos, moon_pos, sun_pos)
    mask |= np.where(shadow, REASON_IN_SHADOW, 0)
    if boresight is not None and fov_deg is not None:
        mask |= np.where(out_of_fov(site_pos, tgt_pos, boresight, fov_deg), REASON_OUT_OF_FOV, 0)
    return mask
