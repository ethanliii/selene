"""Bridge from a reachable set to sensor tasking: *where to point to re-acquire*.

For every future time step of the reachable set (the path steps, default every 6 h) and every
sensor, the active (non-terminated) samples are treated as equally likely hypotheses of the
object's position and run through the ordinary visibility model
(:func:`selene.sensors.visibility.evaluate`: Sun/Moon/Earth exclusion, eclipse, elevation and
daylight for ground sites, limiting magnitude with the object's size/albedo).  Reported per
(step, sensor):

* ``visible_fraction``   share of active samples the sensor could see with *some* pointing
                         (an upper bound for a mosaic / search pattern);
* ``pointing``           the centroid direction of the visible samples (unit vector in GCRF and
                         RA/Dec in degrees), i.e. the single best boresight;
* ``spread_deg``         angular radius of the visible set about that boresight (p50 / p90 / max);
* ``fov_capture_fraction`` share of *all* active samples that are visible **and** inside the
                         sensor's field of view when pointed at the centroid (single exposure);
* ``n_fields_p90``       rough number of FOV tiles needed to cover the p90 spread,
                         ``max(1, ceil((2·θ90 / fov)²))``;
* ``range_km`` (min/median/max) and median predicted magnitude of the visible samples.

The best sensor per step is the one with the largest ``fov_capture_fraction`` (single-field
re-acquisition), with ``visible_fraction`` as the tie-breaker; the ranking by ``visible_fraction``
alone is also returned (which sensor could cover the whole cloud with a search).  Across the
horizon, per-sensor ``first_opportunity_h`` (first step with visible_fraction ≥ 0.5) and
``peak_visible_fraction`` summarise the re-acquisition windows.

Assumptions: uniform weights over samples (no prior on burn size — the ladder already spans
small to large burns); no slew/time-budget modelling here (that is the scheduler's job in
``selene.tasking``); pointing centroid computed from unit LOS vectors (fine for spreads ≲ 60°).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np

from selene.dynamics.ephemeris import get_ephemeris
from selene.sensors.observers import DEFAULT_OBSERVERS, SpaceObserver
from selene.sensors.sites import DEFAULT_SITES, GroundSite
from selene.sensors.visibility import Sensor, evaluate, sensor_geometry

__all__ = ["SensorHint", "StepHint", "sensor_hints", "default_sensor_list", "resolve_sensors"]


def default_sensor_list() -> list[Sensor]:
    return list(DEFAULT_SITES) + list(DEFAULT_OBSERVERS)


def resolve_sensors(ids: Optional[Sequence] = None) -> list[Sensor]:
    """Sensor ids (ground site or space observer ids), Sensor objects, or None for the full default set."""
    if ids is None:
        return default_sensor_list()
    by_id = {s.id: s for s in default_sensor_list()}
    out: list[Sensor] = []
    for item in ids:
        if isinstance(item, (GroundSite, SpaceObserver)):
            out.append(item)
        elif item in by_id:
            out.append(by_id[item])
        else:
            raise KeyError(f"unknown sensor {item!r}; known: {sorted(by_id)}")
    return out


@dataclass
class SensorHint:
    sensor_id: str
    kind: str
    visible_fraction: float
    fov_capture_fraction: float
    n_visible: int
    pointing_gcrf: Optional[list]      # unit vector
    ra_deg: Optional[float]
    dec_deg: Optional[float]
    spread_p50_deg: Optional[float]
    spread_p90_deg: Optional[float]
    spread_max_deg: Optional[float]
    n_fields_p90: Optional[int]
    fov_deg: float
    range_min_km: Optional[float]
    range_median_km: Optional[float]
    range_max_km: Optional[float]
    magnitude_median: Optional[float]
    dominant_block_reason: Optional[str]

    def as_dict(self) -> dict:
        return dict(self.__dict__)


@dataclass
class StepHint:
    t_h: float
    n_active: int                      # active (non-terminated) samples at this step — same as envelope[].n_active
    sensors: list                      # [SensorHint] sorted by fov_capture_fraction desc
    best_single_field: Optional[str]
    best_any_pointing: Optional[str]
    n_evaluated: int = 0               # active samples actually run through the visibility model (<= n_active when thinned)

    def as_dict(self) -> dict:
        return {"t_h": self.t_h, "n_active": self.n_active, "n_evaluated": self.n_evaluated,
                "best_single_field": self.best_single_field, "best_any_pointing": self.best_any_pointing,
                "sensors": [s.as_dict() for s in self.sensors]}


def _reason_name(mask: np.ndarray) -> Optional[str]:
    from selene.sensors.reasons import REASON_NAMES, primary_reason

    blocked = mask[mask != 0]
    if blocked.size == 0:
        return None
    pr = primary_reason(blocked)
    vals, counts = np.unique(pr, return_counts=True)
    return REASON_NAMES.get(int(vals[np.argmax(counts)]), None)


def sensor_hints(rs, sensors: Optional[Sequence] = None, radius_m: float = 1.0, albedo: float = 0.2,
                 step_indices: Optional[np.ndarray] = None, max_samples: int = 4000, min_fraction: float = 0.5) -> dict:
    """Compute :class:`StepHint` for each path step of a :class:`~selene.reachability.sampling.ReachabilitySet`.

    Returns ``{"steps": [StepHint...], "summary": {sensor_id: {...}}, "best_overall": sensor_id | None}``.
    """
    sensors = resolve_sensors(sensors) if (sensors is None or not all(isinstance(s, (GroundSite, SpaceObserver)) for s in sensors)) else list(sensors)
    idx = rs.path_indices() if step_indices is None else np.asarray(step_indices, dtype=int)
    t = rs.t_grid_s[idx]
    t_h = rs.t_h[idx]
    eph = get_ephemeris()
    sun = eph.position("sun", t).reshape(-1, 3)
    moon = eph.position("moon", t).reshape(-1, 3)
    active = rs.active[:, idx]                       # (N, P)
    N = rs.n_samples
    sub = np.arange(N)
    if N > max_samples:                                 # deterministic thinning for very large sets
        sub = np.linspace(0, N - 1, max_samples).astype(int)
    geoms = {s.id: sensor_geometry(s, t) for s in sensors}
    steps: list[StepHint] = []
    summary = {s.id: {"sensor_id": s.id, "kind": s.kind, "peak_visible_fraction": 0.0, "peak_fov_capture_fraction": 0.0,
                      "first_opportunity_h": None, "n_steps_visible": 0} for s in sensors}
    for p, j in enumerate(idx):
        ok = active[sub, p]
        rows = sub[ok]
        n_act = int(rows.size)                          # evaluated subset (fractions are relative to it)
        n_active_true = int(active[:, p].sum())         # all active samples (matches the envelope)
        hints: list[SensorHint] = []
        if n_act:
            tgt = rs.states_gcrf[rows, j, :3]
            for s in sensors:
                pos, _, zen = geoms[s.id]
                obs = pos[p]
                z = None if zen is None else zen[p]
                mask, mag, rng, _ = evaluate(s, obs, z, tgt, sun[p], moon[p], radius_m, albedo)
                vis = mask == 0
                nv = int(vis.sum())
                vf = nv / n_act
                if nv:
                    los = tgt[vis] - obs
                    u = los / np.linalg.norm(los, axis=1, keepdims=True)
                    c = u.mean(axis=0)
                    c /= np.linalg.norm(c)
                    ang = np.degrees(np.arccos(np.clip(u @ c, -1.0, 1.0)))
                    half = 0.5 * float(s.fov_deg)
                    cap = int((ang <= half).sum()) / n_act
                    p50, p90, mx = (float(np.percentile(ang, 50)), float(np.percentile(ang, 90)), float(ang.max()))
                    nf = max(1, int(math.ceil((2.0 * p90 / float(s.fov_deg)) ** 2)))
                    ra = float(np.degrees(np.mod(np.arctan2(c[1], c[0]), 2 * np.pi)))
                    dec = float(np.degrees(np.arcsin(np.clip(c[2], -1, 1))))
                    r_vis = rng[vis]
                    m_vis = mag[vis]
                    hints.append(SensorHint(s.id, s.kind, vf, cap, nv, c.tolist(), ra, dec, p50, p90, mx, nf, float(s.fov_deg),
                                            float(r_vis.min()), float(np.median(r_vis)), float(r_vis.max()),
                                            float(np.median(m_vis[np.isfinite(m_vis)])) if np.isfinite(m_vis).any() else None,
                                            _reason_name(mask)))
                else:
                    hints.append(SensorHint(s.id, s.kind, 0.0, 0.0, 0, None, None, None, None, None, None, None,
                                            float(s.fov_deg), None, None, None, None, _reason_name(mask)))
                sm = summary[s.id]
                sm["peak_visible_fraction"] = max(sm["peak_visible_fraction"], vf)
                sm["peak_fov_capture_fraction"] = max(sm["peak_fov_capture_fraction"], hints[-1].fov_capture_fraction)
                if vf > 0:
                    sm["n_steps_visible"] += 1
                if vf >= min_fraction and sm["first_opportunity_h"] is None:
                    sm["first_opportunity_h"] = float(t_h[p])
        hints.sort(key=lambda h: (h.fov_capture_fraction, h.visible_fraction), reverse=True)
        best_single = hints[0].sensor_id if hints and hints[0].fov_capture_fraction > 0 else None
        best_any = max(hints, key=lambda h: h.visible_fraction).sensor_id if hints and max(h.visible_fraction for h in hints) > 0 else None
        steps.append(StepHint(float(t_h[p]), n_active_true, hints, best_single, best_any, n_evaluated=n_act))
    best_overall = None
    if summary:
        cand = max(summary.values(), key=lambda d: (d["peak_fov_capture_fraction"], d["peak_visible_fraction"]))
        if cand["peak_visible_fraction"] > 0:
            best_overall = cand["sensor_id"]
    return {"steps": steps, "summary": summary, "best_overall": best_overall,
            "note": "uniform weights over active samples; fov_capture_fraction = single exposure at the centroid "
                    "boresight; visible_fraction = any pointing (search/mosaic upper bound)"}
