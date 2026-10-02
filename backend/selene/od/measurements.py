"""Angles-only measurement model for the OD track (thin layer over :mod:`selene.sensors.visibility`).

Conventions
-----------
* State ``x = (r, v)`` Earth-centred GCRF [km, km/s]; observer position [km] in the same frame.
* ``h(x; R) = (ra, dec)`` topocentric GCRF angles of the line of sight ρ = r − R (radians).
* **Residual / innovation** is expressed *on the sky*::

      ν = [ wrap(ra_meas − ra_pred) · cos(dec_pred),  dec_meas − dec_pred ]      [rad]

  so that a measurement with isotropic on-sky noise σ has covariance ``R = σ² I₂`` and the RA
  wrap-around (0/2π) is handled once, here.  Every Jacobian in this module is the Jacobian of
  that scaled residual, i.e. the RA row of the sensor Jacobian is multiplied by cos(dec).
* ``H`` (2×6) = [ cos(dec)·∂ra/∂r ; ∂dec/∂r | 0₃ ] — angles do not depend on velocity directly
  (light-time and aberration are not modelled, see the sensors module).

Simulation
----------
:func:`simulate_observations` generates a synthetic measurement set from the catalog truth (or
any trajectory) for a list of sensors on a time grid, applying the full visibility model
(exclusion zones, eclipse, photometric limit, daylight, elevation) and recording *why* each
dropped epoch was unobservable — the record of blind spots is part of the product, not noise.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Callable, Iterable, Sequence, Union

import numpy as np

from selene.dynamics.ephemeris import get_ephemeris
from selene.sensors.observers import DEFAULT_OBSERVERS, SpaceObserver, get_observer
from selene.sensors.reasons import reason_names
from selene.sensors.sites import DEFAULT_SITES, GroundSite, get_site
from selene.sensors.visibility import (
    ARCSEC,
    C_LIGHT_KM_S,
    Measurement,
    Sensor,
    evaluate,
    measurement_jacobian,
    measurement_model,
    sensor_geometry,
)

__all__ = [
    "ARCSEC",
    "wrap_angle",
    "los_unit",
    "los_tangent_basis",
    "predict",
    "residual",
    "jacobian_state",
    "noise_cov",
    "resolve_sensors",
    "all_sensors",
    "ObservationSet",
    "simulate_observations",
    "sort_measurements",
    "TruthLike",
]

TruthLike = Union[str, Callable[[np.ndarray], np.ndarray], object]


# ---------------------------------------------------------------------------
# angles
# ---------------------------------------------------------------------------
def wrap_angle(a):
    """Wrap angle(s) to (−π, π]."""
    a = np.asarray(a, dtype=np.float64)
    return -((-a + np.pi) % (2.0 * np.pi) - np.pi)


def los_unit(ra, dec) -> np.ndarray:
    """Unit line-of-sight vector(s) (…,3) from (ra, dec) [rad]."""
    ra = np.asarray(ra, dtype=np.float64)
    dec = np.asarray(dec, dtype=np.float64)
    cd = np.cos(dec)
    return np.stack([cd * np.cos(ra), cd * np.sin(ra), np.sin(dec)], axis=-1)


def los_tangent_basis(ra, dec) -> tuple[np.ndarray, np.ndarray]:
    """On-sky unit tangent vectors (ê_ra, ê_dec) at (ra, dec): ∂L/∂ra / cos(dec) and ∂L/∂dec."""
    ra = np.asarray(ra, dtype=np.float64)
    dec = np.asarray(dec, dtype=np.float64)
    e_ra = np.stack([-np.sin(ra), np.cos(ra), np.zeros_like(ra)], axis=-1)
    e_dec = np.stack([-np.sin(dec) * np.cos(ra), -np.sin(dec) * np.sin(ra), np.cos(dec)], axis=-1)
    return e_ra, e_dec


# ---------------------------------------------------------------------------
# model, residual, Jacobian
# ---------------------------------------------------------------------------
def predict(x, observer_pos) -> tuple[np.ndarray, np.ndarray]:
    """h(x; R): (ra, dec) [rad] of state(s) ``x`` (…,6) seen from ``observer_pos`` (…,3)."""
    x = np.asarray(x, dtype=np.float64)
    return measurement_model(observer_pos, x[..., :3])


def residual(ra_meas, dec_meas, ra_pred, dec_pred) -> np.ndarray:
    """On-sky residual (…,2) = [wrap(Δra)·cos(dec_pred), Δdec] in radians."""
    dra = wrap_angle(np.asarray(ra_meas) - np.asarray(ra_pred))
    ddec = np.asarray(dec_meas, dtype=np.float64) - np.asarray(dec_pred, dtype=np.float64)
    return np.stack([dra * np.cos(dec_pred), ddec], axis=-1)


def jacobian_state(x, observer_pos) -> np.ndarray:
    """Analytic Jacobian (…,2,6) of the *scaled* residual w.r.t. the state (rad/km, zeros for v)."""
    x = np.asarray(x, dtype=np.float64)
    r = x[..., :3]
    Hr = measurement_jacobian(observer_pos, r)          # (...,2,3): rows d(ra), d(dec)
    _, dec = measurement_model(observer_pos, r)
    Hr = Hr.copy()
    Hr[..., 0, :] *= np.cos(dec)[..., None]
    H = np.zeros(x.shape[:-1] + (2, 6))
    H[..., :, :3] = Hr
    return H


def noise_cov(m: Measurement) -> np.ndarray:
    """Measurement-noise covariance of the scaled residual: σ² I₂."""
    return float(m.sigma_rad) ** 2 * np.eye(2)


# ---------------------------------------------------------------------------
# sensors
# ---------------------------------------------------------------------------
def all_sensors() -> list[Sensor]:
    return list(DEFAULT_SITES) + list(DEFAULT_OBSERVERS)


def resolve_sensors(spec: Union[str, Sequence[Union[str, Sensor]], None]) -> list[Sensor]:
    """'all' / None -> every default sensor; 'ground' / 'space'; else a list of ids or objects."""
    if spec is None or (isinstance(spec, str) and spec.lower() == "all"):
        return all_sensors()
    if isinstance(spec, str):
        if spec.lower() == "ground":
            return list(DEFAULT_SITES)
        if spec.lower() == "space":
            return list(DEFAULT_OBSERVERS)
        spec = [spec]
    out: list[Sensor] = []
    for s in spec:
        if isinstance(s, (GroundSite, SpaceObserver)):
            out.append(s)
            continue
        sid = str(s)
        try:
            out.append(get_site(sid))
        except KeyError:
            out.append(get_observer(sid))  # raises KeyError with the known ids
    return out


# ---------------------------------------------------------------------------
# truth resolution
# ---------------------------------------------------------------------------
def _resolve_truth(truth: TruthLike) -> tuple[Callable[[np.ndarray], np.ndarray], dict]:
    """-> (callable t_s -> (N,6) GCRF, physical dict {radius_m, albedo, cr_area_mass, object_id})."""
    if isinstance(truth, str):
        from selene.objects.catalog import get_catalog

        cat = get_catalog()
        e = cat.get(truth)
        phys = {"object_id": truth, "radius_m": 1.0, "albedo": 0.2, "cr_area_mass": 0.0}
        if e.physical:
            phys.update(radius_m=float(e.physical["radius_m"]), albedo=float(e.physical["albedo"]),
                        cr_area_mass=float(e.physical.get("cr_area_mass_m2_kg", 0.0)))
        return (lambda t: cat.state_at(truth, t)), phys
    if hasattr(truth, "at"):
        return (lambda t: np.asarray(truth.at(t), dtype=np.float64)), {"object_id": getattr(truth, "object_id", "")}
    if callable(truth):
        return truth, {"object_id": ""}
    raise TypeError("truth must be an object id, a Trajectory/TruthArcs, or a callable")


# ---------------------------------------------------------------------------
# simulation
# ---------------------------------------------------------------------------
class ObservationSet(list):
    """A ``list[Measurement]`` (time-ordered) that also remembers the epochs that were dropped
    and why (``dropped``: list of dicts), the sensors tried, and the simulation settings."""

    def __init__(self, meas: Iterable[Measurement] = (), dropped: list | None = None, meta: dict | None = None):
        super().__init__(meas)
        self.dropped: list[dict] = dropped or []
        self.meta: dict = meta or {}

    @property
    def n_candidates(self) -> int:
        return len(self) + len(self.dropped)

    def t_s(self) -> np.ndarray:
        return np.array([m.t_s for m in self], dtype=np.float64)

    def by_sensor(self) -> dict[str, int]:
        return dict(Counter(m.sensor_id for m in self))

    def dropped_reasons(self) -> dict[str, int]:
        """Counts of each reason bit over the dropped epochs (an epoch may carry several)."""
        c: Counter = Counter()
        for d in self.dropped:
            for name in d["reason_names"]:
                c[name] += 1
        return dict(c)

    def summary(self) -> dict:
        return {
            "n_measurements": len(self),
            "n_candidates": self.n_candidates,
            "fraction_observed": (len(self) / self.n_candidates) if self.n_candidates else 0.0,
            "by_sensor": self.by_sensor(),
            "dropped_by_reason": self.dropped_reasons(),
            "dropped_by_sensor": dict(Counter(d["sensor_id"] for d in self.dropped)),
            "span_h": float((self[-1].t_s - self[0].t_s) / 3600.0) if len(self) > 1 else 0.0,
            **{k: v for k, v in self.meta.items() if k != "truth"},
        }

    def as_dict(self, include_dropped: bool = False) -> dict:
        d = {"measurements": [m.as_dict() for m in self], "summary": self.summary()}
        if include_dropped:
            d["dropped"] = self.dropped
        return d


def sort_measurements(meas: Iterable[Measurement]) -> list[Measurement]:
    return sorted(meas, key=lambda m: (m.t_s, m.sensor_id))


def simulate_observations(
    truth: TruthLike,
    sensors,
    t_grid,
    sigma_arcsec: float = 1.0,
    rng=None,
    respect_visibility: bool = True,
    radius_m: float | None = None,
    albedo: float | None = None,
    margin_mag: float = 0.0,
) -> ObservationSet:
    """Synthetic RA/Dec measurements of ``truth`` from ``sensors`` at the epochs ``t_grid`` (TDB s).

    ``truth``: a catalog object id (uses its truth arc and physical size/albedo), a Trajectory /
    TruthArcs (``.at``), or a callable ``t -> (N,6)``.  ``sensors``: anything :func:`resolve_sensors`
    accepts.  Noise: isotropic on-sky Gaussian, ``sigma_arcsec`` (1σ), exactly as
    :func:`selene.sensors.visibility.observe` (dec += σn₁, ra += σn₂/cos dec).  With
    ``respect_visibility`` every epoch failing the visibility model is dropped and recorded in
    ``ObservationSet.dropped`` with its reason names; with it off every epoch yields a measurement
    whose ``visible``/``reasons`` fields still report the model's verdict.
    """
    rng = np.random.default_rng(0) if rng is None else rng
    truth_fn, phys = _resolve_truth(truth)
    radius_m = float(phys.get("radius_m", 1.0)) if radius_m is None else float(radius_m)
    albedo = float(phys.get("albedo", 0.2)) if albedo is None else float(albedo)
    sensors = resolve_sensors(sensors)
    t = np.atleast_1d(np.asarray(t_grid, dtype=np.float64)).ravel()
    sigma = float(sigma_arcsec) * ARCSEC
    meas: list[Measurement] = []
    dropped: list[dict] = []
    if t.size == 0 or not sensors:
        return ObservationSet(meas, dropped, {"sigma_arcsec": sigma_arcsec, **phys})
    X = np.asarray(truth_fn(t), dtype=np.float64).reshape(t.size, 6)
    tgt = X[:, :3]
    eph = get_ephemeris()
    sun = eph.position("sun", t).reshape(-1, 3)
    moon = eph.position("moon", t).reshape(-1, 3)
    for sen in sensors:
        obs_pos, obs_vel, zen = sensor_geometry(sen, t)
        mask, mag, rng_km, _ = evaluate(sen, obs_pos, zen, tgt, sun, moon, radius_m, albedo, margin_mag)
        ra, dec = measurement_model(obs_pos, tgt)
        noise = rng.standard_normal((t.size, 2))  # one draw per epoch per sensor, visible or not -> reproducible
        for k in range(t.size):
            visible = bool(mask[k] == 0)
            if respect_visibility and not visible:
                dropped.append({"t_s": float(t[k]), "sensor_id": sen.id, "reasons": int(mask[k]),
                                "reason_names": reason_names(int(mask[k])),
                                "magnitude": float(mag[k]) if np.isfinite(mag[k]) else None})
                continue
            dec_m = float(dec[k]) + sigma * noise[k, 0]
            ra_m = float(np.mod(ra[k] + sigma * noise[k, 1] / np.cos(dec_m), 2.0 * np.pi))
            meas.append(Measurement(
                t_s=float(t[k]), sensor_id=sen.id, ra_rad=ra_m, dec_rad=dec_m, sigma_rad=sigma,
                observer_pos_gcrf=np.array(obs_pos[k]), observer_vel_gcrf=np.array(obs_vel[k]),
                visible=visible, magnitude=float(mag[k]), reasons=int(mask[k]),
                light_time_s=float(rng_km[k]) / C_LIGHT_KM_S,
                truth={"ra_rad": float(ra[k]), "dec_rad": float(dec[k]), "state": X[k].tolist()},
            ))
    meta = {"sigma_arcsec": float(sigma_arcsec), "respect_visibility": bool(respect_visibility),
            "sensors": [s.id for s in sensors], "n_epochs": int(t.size), "radius_m": radius_m, "albedo": albedo,
            **{k: v for k, v in phys.items() if k in ("object_id", "cr_area_mass")}}
    return ObservationSet(sort_measurements(meas), sorted(dropped, key=lambda d: (d["t_s"], d["sensor_id"])), meta)
