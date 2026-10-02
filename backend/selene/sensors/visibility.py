"""Per-(sensor, target, time) visibility, and angles-only measurement generation.

A *sensor* is either a :class:`~selene.sensors.sites.GroundSite` or a
:class:`~selene.sensors.observers.SpaceObserver`.  Visibility combines the geometric reason
mask from :mod:`selene.sensors.constraints` with the photometric test from
:mod:`selene.sensors.photometry`; a target is visible when the mask is ``0``.

Measurement model
-----------------
Topocentric right ascension / declination in GCRF of the line of sight ρ = r_target − r_observer::

    ra = atan2(ρ_y, ρ_x),  dec = asin(ρ_z / |ρ|)

with the analytic Jacobian w.r.t. target position (for batch LS / UKF linearisation)::

    ∂ra/∂ρ  = ( −ρ_y, ρ_x, 0 ) / (ρ_x² + ρ_y²)
    ∂dec/∂ρ = ( −ρ_x ρ_z, −ρ_y ρ_z, ρ_x² + ρ_y² ) / ( |ρ|² · sqrt(ρ_x² + ρ_y²) )

Noise: ``dec += σ n₁`` and ``ra += σ n₂ / cos(dec)`` so that the *on-sky* error is isotropic with
standard deviation σ (``sigma_arcsec``, default 1″, typical for a 1 m-class astrometric
telescope against the Gaia reference frame).

**Light-time correction is not applied** (documented simplification): at 4×10⁵ km the one-way
light time is 1.3 s; for a 1 km/s target this displaces the apparent position by ~1.3 km,
≈ 0.7″ — comparable to the assumed noise.  OD code that needs it can evaluate the target state
at ``t − |ρ|/c`` and re-call :func:`measurement_model`; the ``light_time_s`` field of each
measurement reports the delay so this is a one-liner downstream.

**Stellar aberration is not applied either**: directions are geometric GCRF (BCRS-parallel axes,
no observer-velocity aberration).  Star-calibrated astrometry of a real detection yields
*apparent* places; the annual term is up to ~20.5″ for a ground site (Earth at 30 km/s) and the
diurnal term ~0.3″, i.e. real-data ingest must reduce observations to geometric (astrometric)
directions before they are compared with this model.  Synthetic measurements generated here are
self-consistent with the model, so the OD/maneuver tracks are unaffected.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Union

import numpy as np

from selene.dynamics.ephemeris import get_ephemeris
from selene.sensors import constraints as C
from selene.sensors.observers import SpaceObserver, observer_state
from selene.sensors.photometry import detectable, phase_angle, visual_magnitude
from selene.sensors.reasons import REASON_TOO_FAINT, reason_names
from selene.sensors.sites import GroundSite, site_frame

__all__ = [
    "Sensor",
    "VisibilityResult",
    "Measurement",
    "sensor_geometry",
    "evaluate",
    "visibility",
    "measurement_model",
    "measurement_jacobian",
    "observe",
    "C_LIGHT_KM_S",
]

Sensor = Union[GroundSite, SpaceObserver]
C_LIGHT_KM_S = 299_792.458
ARCSEC = np.pi / (180.0 * 3600.0)


@dataclass
class VisibilityResult:
    sensor_id: str
    t_s: np.ndarray          # (N,)
    visible: np.ndarray      # (N,) bool
    magnitude: np.ndarray    # (N,) predicted V magnitude
    reasons: np.ndarray      # (N,) int bitmask (0 = visible)
    range_km: np.ndarray     # (N,)
    phase_deg: np.ndarray    # (N,)

    def reason_strings(self) -> list[list[str]]:
        return [reason_names(int(m)) for m in self.reasons]

    def as_dict(self) -> dict:
        return {
            "sensor_id": self.sensor_id,
            "t_s": self.t_s.tolist(),
            "visible": self.visible.astype(bool).tolist(),
            "magnitude": [None if not np.isfinite(m) else float(m) for m in self.magnitude],
            "reasons": self.reasons.astype(int).tolist(),
            "reason_names": self.reason_strings(),
            "range_km": self.range_km.tolist(),
            "phase_deg": self.phase_deg.tolist(),
            "fraction_visible": float(np.mean(self.visible)) if self.visible.size else 0.0,
        }


@dataclass
class Measurement:
    """One angles-only observation (GCRF topocentric RA/Dec, radians)."""

    t_s: float
    sensor_id: str
    ra_rad: float
    dec_rad: float
    sigma_rad: float
    observer_pos_gcrf: np.ndarray    # (3,) km
    observer_vel_gcrf: np.ndarray    # (3,) km/s
    visible: bool = True
    magnitude: float = float("nan")
    reasons: int = 0
    light_time_s: float = 0.0
    truth: dict = field(default_factory=dict)   # noiseless (ra, dec) for diagnostics

    def as_dict(self) -> dict:
        return {
            "t_s": self.t_s, "sensor_id": self.sensor_id, "ra_rad": self.ra_rad, "dec_rad": self.dec_rad,
            "sigma_rad": self.sigma_rad, "observer_pos_gcrf": self.observer_pos_gcrf.tolist(),
            "observer_vel_gcrf": self.observer_vel_gcrf.tolist(), "visible": bool(self.visible),
            "magnitude": None if not np.isfinite(self.magnitude) else float(self.magnitude),
            "reasons": int(self.reasons), "reason_names": reason_names(self.reasons),
            "light_time_s": self.light_time_s,
        }


# ---------------------------------------------------------------------------
# geometry helpers
# ---------------------------------------------------------------------------
def sensor_geometry(sensor: Sensor, t_s) -> tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    """(position (N,3) km, velocity (N,3) km/s, zenith (N,3) or None) in GCRF for array ``t_s``."""
    t = np.atleast_1d(np.asarray(t_s, dtype=np.float64)).ravel()
    if isinstance(sensor, GroundSite):
        fr = site_frame(sensor, t)
        return fr.pos_km, fr.vel_km_s, fr.zenith
    if isinstance(sensor, SpaceObserver):
        s = observer_state(sensor, t)
        return s[:, :3], s[:, 3:6], None
    raise TypeError(f"unsupported sensor type {type(sensor).__name__}")


def evaluate(sensor: Sensor, obs_pos, zenith, tgt_pos, sun_pos, moon_pos, radius_m: float, albedo: float,
             margin_mag: float = 0.0, boresight=None):
    """Broadcasting core: returns (reasons, magnitude, range_km, phase_rad) over the common shape.

    ``obs_pos``/``zenith``/``sun_pos``/``moon_pos`` are typically ``(T,1,3)`` and ``tgt_pos``
    ``(T,C,3)``; all are ``(...,3)`` and broadcast together.
    """
    tgt_pos = np.asarray(tgt_pos, dtype=np.float64)
    obs_pos = np.asarray(obs_pos, dtype=np.float64)
    if isinstance(sensor, GroundSite):
        mask = C.ground_reasons(obs_pos, zenith, tgt_pos, sun_pos, moon_pos,
                                min_elevation_deg=sensor.min_elevation_deg,
                                sun_elev_max_deg=sensor.sun_elev_max_deg,
                                boresight=boresight, fov_deg=sensor.fov_deg if boresight is not None else None)
    else:
        mask = C.space_reasons(obs_pos, tgt_pos, sun_pos, moon_pos,
                               sun_excl_deg=sensor.sun_exclusion_deg,
                               earth_excl_deg=sensor.earth_exclusion_deg,
                               moon_excl_deg=sensor.moon_exclusion_deg,
                               boresight=boresight, fov_deg=sensor.fov_deg if boresight is not None else None)
    rng = np.linalg.norm(tgt_pos - obs_pos, axis=-1)
    phi = phase_angle(sun_pos, obs_pos, tgt_pos)
    mag = visual_magnitude(radius_m, albedo, phi, rng)
    faint = ~detectable(mag, sensor.limiting_mag, margin_mag)
    mask = mask | np.where(faint, REASON_TOO_FAINT, 0)
    return mask, mag, rng, phi


def visibility(sensor: Sensor, target_pos_gcrf, t_s, target_radius_m: float = 1.0, albedo: float = 0.2,
               margin_mag: float = 0.0) -> VisibilityResult:
    """Visibility of a target track ``target_pos_gcrf`` (N,3) [km] at TDB seconds ``t_s`` (N,)."""
    t = np.atleast_1d(np.asarray(t_s, dtype=np.float64)).ravel()
    tgt = np.asarray(target_pos_gcrf, dtype=np.float64).reshape(-1, 3)
    if t.size == 0:
        z = np.zeros(0)
        return VisibilityResult(sensor.id, t, z.astype(bool), z, z.astype(np.int64), z, z)
    if tgt.shape[0] != t.size:
        if tgt.shape[0] == 1:
            tgt = np.broadcast_to(tgt, (t.size, 3))
        else:
            raise ValueError("target_pos_gcrf and t_s must have the same length")
    obs_pos, _, zen = sensor_geometry(sensor, t)
    eph = get_ephemeris()
    sun = eph.position("sun", t).reshape(-1, 3)
    moon = eph.position("moon", t).reshape(-1, 3)
    mask, mag, rng, phi = evaluate(sensor, obs_pos, zen, tgt, sun, moon, target_radius_m, albedo, margin_mag)
    return VisibilityResult(sensor.id, t, mask == 0, mag, mask.astype(np.int64), rng, np.rad2deg(phi))


# ---------------------------------------------------------------------------
# measurement model
# ---------------------------------------------------------------------------
def measurement_model(observer_pos, target_pos) -> tuple[np.ndarray, np.ndarray]:
    """Topocentric GCRF (ra, dec) [rad] of ``target_pos`` seen from ``observer_pos`` (broadcasting)."""
    rho = np.asarray(target_pos, dtype=np.float64) - np.asarray(observer_pos, dtype=np.float64)
    r = np.linalg.norm(rho, axis=-1)
    ra = np.mod(np.arctan2(rho[..., 1], rho[..., 0]), 2.0 * np.pi)
    dec = np.arcsin(np.clip(rho[..., 2] / r, -1.0, 1.0))
    return ra, dec


def measurement_jacobian(observer_pos, target_pos) -> np.ndarray:
    """∂(ra, dec)/∂(target position) — shape ``(..., 2, 3)`` [rad/km].

    At the polar singularity (line of sight along ±z, ρ_x = ρ_y = 0) right ascension is undefined;
    the RA row and the dec x/y entries are returned as **0** (not NaN) so a filter never ingests
    NaNs — callers should treat such a measurement's RA as uninformative.
    """
    rho = np.asarray(target_pos, dtype=np.float64) - np.asarray(observer_pos, dtype=np.float64)
    x, y, z = rho[..., 0], rho[..., 1], rho[..., 2]
    rxy2 = x * x + y * y
    r2 = rxy2 + z * z
    rxy = np.sqrt(rxy2)
    H = np.zeros(rho.shape[:-1] + (2, 3))
    with np.errstate(divide="ignore", invalid="ignore"):
        H[..., 0, 0] = -y / rxy2
        H[..., 0, 1] = x / rxy2
        H[..., 1, 0] = -x * z / (r2 * rxy)
        H[..., 1, 1] = -y * z / (r2 * rxy)
        H[..., 1, 2] = rxy / r2
    return np.where(np.isfinite(H), H, 0.0)


def observe(sensor: Sensor, target_state, t_s: float, sigma_arcsec: float = 1.0, rng=None,
            target_radius_m: float = 1.0, albedo: float = 0.2, require_visible: bool = True) -> Optional[Measurement]:
    """Generate one noisy RA/Dec measurement of ``target_state`` (6,) [km, km/s GCRF] at ``t_s``.

    Returns ``None`` if the target is not visible and ``require_visible`` is set; otherwise the
    measurement carries ``visible=False`` and the reason mask.
    """
    rng = np.random.default_rng() if rng is None else rng
    t = float(t_s)
    s = np.asarray(target_state, dtype=np.float64).ravel()
    tgt = s[:3]
    obs_pos, obs_vel, zen = sensor_geometry(sensor, t)
    obs_pos, obs_vel = obs_pos[0], obs_vel[0]
    zen = None if zen is None else zen[0]
    eph = get_ephemeris()
    sun = eph.position("sun", t)
    moon = eph.position("moon", t)
    mask, mag, dist, _ = evaluate(sensor, obs_pos, zen, tgt, sun, moon, target_radius_m, albedo)
    visible = bool(mask == 0)
    if require_visible and not visible:
        return None
    ra, dec = measurement_model(obs_pos, tgt)
    sigma = float(sigma_arcsec) * ARCSEC
    n1, n2 = rng.standard_normal(2)
    dec_m = float(dec) + sigma * n1
    ra_m = float(np.mod(ra + sigma * n2 / np.cos(dec_m), 2.0 * np.pi))
    return Measurement(
        t_s=t, sensor_id=sensor.id, ra_rad=ra_m, dec_rad=dec_m, sigma_rad=sigma,
        observer_pos_gcrf=np.array(obs_pos), observer_vel_gcrf=np.array(obs_vel),
        visible=visible, magnitude=float(mag), reasons=int(mask), light_time_s=float(dist) / C_LIGHT_KM_S,
        truth={"ra_rad": float(ra), "dec_rad": float(dec)},
    )
