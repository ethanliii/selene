"""Coverage heatmap of the cislunar volume: where can the sensor network see a reference object?

Grid
----
Cells are fixed in the **Earth–Moon rotating frame** (nondimensional, L* = 384 400 km) so the
Moon, L1 and L2 stay put while the Sun sweeps around once per synodic month.  Default: a 2-D
slice ``z = 0`` with x ∈ [−1.6, 1.6] (64 cells) × y ∈ [−1.4, 1.4] (56 cells); optional coarse 3-D
grid 24×20×9 with z ∈ [−0.4, 0.4].  Each cell centre is mapped to GCRF at every time step with
the instantaneous rotating frame (``frames.rotating_frame``), i.e. the same transform used for
display.

Reference object: diffuse sphere, radius 1 m, albedo 0.2 (configurable) — a "small bus"
class target.  A cell is *covered* at time t if **any** sensor in the network has a zero reason
mask for a target at that cell centre (geometry + photometry).  Pointing/FOV is not applied
here (coverage asks "could a sensor be tasked to see it", the tasker applies FOV/slew).

Outputs
-------
* ``coverage``        (C,)   fraction of time steps the cell is covered by ≥ 1 sensor
* ``per_time_pct``    (T,)   percentage of cells covered at each time step
* ``reason_dominant`` (C,)   the reason bit that most often explains why the cell was *not*
  covered.  Per (t, cell) the "closest-to-seeing" sensor is chosen — the one whose mask has the
  fewest set bits (ties broken toward target-specific reasons: Moon/Sun/Earth exclusion,
  shadow, faintness, then low elevation, then daylight) — so a near-Moon cell shows
  ``moon_exclusion`` rather than ``daylight`` merely because half the globe is in daytime.
* ``reason_fraction`` (C, 8) fraction of time steps each reason was the chosen explanation.

Performance: sensor states are computed once per time step and broadcast over cells; the
default 2-D run (9 sites, 168 h hourly) takes ~1–3 s on a laptop.
"""
from __future__ import annotations

import time as _time
from dataclasses import dataclass, field
from typing import Optional, Sequence, Union

import numpy as np

from selene.dynamics import frames
from selene.dynamics.ephemeris import get_ephemeris
from selene.sensors.observers import DEFAULT_OBSERVERS, SpaceObserver, get_observer
from selene.sensors.reasons import ALL_REASONS, REASON_NAMES, REASON_PRIORITY, popcount, primary_reason
from selene.sensors.sites import DEFAULT_SITES, GroundSite, get_site
from selene.sensors.visibility import Sensor, evaluate, sensor_geometry

__all__ = [
    "CoverageGrid",
    "CoverageResult",
    "NETWORK_PRESETS",
    "make_grid",
    "resolve_network",
    "compute_coverage",
    "DEFAULT_RADIUS_M",
    "DEFAULT_ALBEDO",
]

DEFAULT_RADIUS_M = 1.0
DEFAULT_ALBEDO = 0.2

_ALL_SITE_IDS = tuple(s.id for s in DEFAULT_SITES)

#: preset -> (ground site ids, space observer ids)
NETWORK_PRESETS: dict[str, tuple[tuple[str, ...], tuple[str, ...]]] = {
    "ground_only": (_ALL_SITE_IDS, ()),
    "ground_plus_geo": (_ALL_SITE_IDS, ("geo_west", "geo_east")),
    "ground_plus_l1_halo": (_ALL_SITE_IDS, ("l1_halo_obs",)),
    "ground_plus_l2_halo": (_ALL_SITE_IDS, ("l2_halo_obs",)),
    "ground_plus_dro": (_ALL_SITE_IDS, ("dro_obs",)),
    "ground_plus_nrho": (_ALL_SITE_IDS, ("nrho_obs",)),
    "space_only": ((), ("geo_west", "geo_east", "l2_halo_obs", "dro_obs")),
    "full": (_ALL_SITE_IDS, ("geo_west", "geo_east", "l1_halo_obs", "l2_halo_obs", "dro_obs")),
}


@dataclass
class CoverageGrid:
    kind: str                     # '2d' | '3d'
    xs: np.ndarray
    ys: np.ndarray
    zs: Optional[np.ndarray]
    points_rot: np.ndarray        # (C,3) nondimensional rotating-frame cell centres

    @property
    def shape(self) -> tuple:
        return (len(self.xs), len(self.ys)) if self.zs is None else (len(self.xs), len(self.ys), len(self.zs))

    def as_dict(self) -> dict:
        d = {"kind": self.kind, "xs": self.xs.tolist(), "ys": self.ys.tolist(), "shape": list(self.shape),
             "frame": "earth_moon_rotating_nd", "length_unit_km": 384_400.0, "order": "C (x slowest, z fastest)"}
        if self.zs is not None:
            d["zs"] = self.zs.tolist()
        return d


def make_grid(kind: str = "2d", nx: Optional[int] = None, ny: Optional[int] = None, nz: int = 9,
              xlim=(-1.6, 1.6), ylim=(-1.4, 1.4), zlim=(-0.4, 0.4)) -> CoverageGrid:
    """Cell-centre grid (``np.linspace`` endpoints inclusive) in the rotating frame.
    Defaults: 2-D 64×56; 3-D 24×20×9."""
    if nx is None:
        nx = 64 if kind == "2d" else 24
    if ny is None:
        ny = 56 if kind == "2d" else 20
    xs = np.linspace(xlim[0], xlim[1], int(nx))
    ys = np.linspace(ylim[0], ylim[1], int(ny))
    if kind == "2d":
        X, Y = np.meshgrid(xs, ys, indexing="ij")
        pts = np.stack([X.ravel(), Y.ravel(), np.zeros(X.size)], axis=1)
        return CoverageGrid("2d", xs, ys, None, pts)
    if kind == "3d":
        zs = np.linspace(zlim[0], zlim[1], int(nz))
        X, Y, Z = np.meshgrid(xs, ys, zs, indexing="ij")
        pts = np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=1)
        return CoverageGrid("3d", xs, ys, zs, pts)
    raise ValueError("grid kind must be '2d' or '3d'")


NetworkSpec = Union[str, dict, Sequence[Sensor]]


def resolve_network(spec: NetworkSpec) -> list[Sensor]:
    """Preset name, ``{'ground_ids': [...], 'space': [ids or SpaceObserver dicts]}``, or sensor list."""
    if isinstance(spec, str):
        if spec not in NETWORK_PRESETS:
            raise KeyError(f"unknown network preset {spec!r}; known: {sorted(NETWORK_PRESETS)}")
        g, s = NETWORK_PRESETS[spec]
        return [get_site(i) for i in g] + [get_observer(i) for i in s]
    if isinstance(spec, dict):
        sensors: list[Sensor] = [get_site(i) for i in spec.get("ground_ids", ())]
        for item in spec.get("space", ()):
            if isinstance(item, str):
                sensors.append(get_observer(item))
            elif isinstance(item, SpaceObserver):
                sensors.append(item)
            elif isinstance(item, dict):
                kw = {k: v for k, v in item.items() if k not in ("kind", "spec_note")}
                if "custom_ic" in kw and kw["custom_ic"] is not None:
                    kw["custom_ic"] = tuple(float(v) for v in kw["custom_ic"])
                sensors.append(SpaceObserver(**kw))
            else:
                raise TypeError(f"unsupported space sensor spec {item!r}")
        return sensors
    return list(spec)


@dataclass
class CoverageResult:
    grid: CoverageGrid
    t_s: np.ndarray
    coverage: np.ndarray            # (C,)
    per_time_pct: np.ndarray        # (T,)
    reason_dominant: np.ndarray     # (C,) int bit (0 if always covered)
    reason_fraction: np.ndarray     # (C, 8)
    sensor_ids: list[str]
    n_sensors_visible: np.ndarray   # (C,) mean number of sensors that can see the cell
    meta: dict = field(default_factory=dict)

    @property
    def mean_coverage_pct(self) -> float:
        return float(100.0 * np.mean(self.coverage))

    def coverage_near(self, point_rot, radius_nd: float) -> float:
        """Mean coverage fraction of cells within ``radius_nd`` of ``point_rot`` (nd)."""
        d = np.linalg.norm(self.grid.points_rot - np.asarray(point_rot, float), axis=1)
        sel = d <= radius_nd
        return float(np.mean(self.coverage[sel])) if np.any(sel) else float("nan")

    def as_dict(self) -> dict:
        return {
            "grid": self.grid.as_dict(),
            "coverage": self.coverage.round(4).tolist(),
            "reason_dominant": self.reason_dominant.astype(int).tolist(),
            "reason_dominant_name": [REASON_NAMES.get(int(b), "covered") for b in self.reason_dominant],
            "reason_fraction": {REASON_NAMES[b]: self.reason_fraction[:, i].round(4).tolist()
                                for i, b in enumerate(ALL_REASONS)},
            "n_sensors_visible": self.n_sensors_visible.round(3).tolist(),
            "per_time": [{"t": float(t), "pct": float(p)} for t, p in zip(self.t_s, self.per_time_pct)],
            "meta": {**self.meta, "mean_coverage_pct": self.mean_coverage_pct, "sensor_ids": self.sensor_ids,
                     "reason_bits": {REASON_NAMES[b]: int(b) for b in ALL_REASONS}},
        }


_PRIORITY_RANK = {bit: i for i, bit in enumerate(REASON_PRIORITY)}


def _mask_score(mask: np.ndarray) -> np.ndarray:
    """Lower = closer to visible.  popcount first, then the priority of the primary reason."""
    prim = primary_reason(mask)
    rank = np.zeros_like(mask)
    for bit, r in _PRIORITY_RANK.items():
        rank = np.where(prim == bit, r, rank)
    return popcount(mask).astype(np.int64) * 16 + rank


def compute_coverage(t0_s: float, t1_s: float, n_t: int = 168, network: NetworkSpec = "ground_only",
                     grid: Union[str, CoverageGrid] = "2d", radius_m: float = DEFAULT_RADIUS_M,
                     albedo: float = DEFAULT_ALBEDO, margin_mag: float = 0.0) -> CoverageResult:
    """Run the coverage analysis over ``n_t`` evenly spaced TDB epochs in [t0_s, t1_s]."""
    tic = _time.perf_counter()
    g = make_grid(grid) if isinstance(grid, str) else grid
    sensors = resolve_network(network)
    t = np.linspace(float(t0_s), float(t1_s), int(n_t))
    T, Cn = t.size, g.points_rot.shape[0]

    # cell centres -> GCRF, (T, C, 3)
    fr = frames.rotating_frame(t)
    tgt = np.einsum("tij,cj->tci", fr.R, g.points_rot) * fr.d_km[:, None, None] + fr.r_bary_gcrf[:, None, :]

    eph = get_ephemeris()
    sun = eph.position("sun", t).reshape(T, 1, 3)
    moon = eph.position("moon", t).reshape(T, 1, 3)

    covered = np.zeros((T, Cn), dtype=bool)
    n_vis = np.zeros((T, Cn), dtype=np.int16)
    best_mask = np.zeros((T, Cn), dtype=np.int64)
    best_score = np.full((T, Cn), np.iinfo(np.int64).max, dtype=np.int64)
    timings = {}
    for s in sensors:
        t_s0 = _time.perf_counter()
        pos, _, zen = sensor_geometry(s, t)
        pos = pos.reshape(T, 1, 3)
        zen = None if zen is None else zen.reshape(T, 1, 3)
        mask, _, _, _ = evaluate(s, pos, zen, tgt, sun, moon, radius_m, albedo, margin_mag)
        vis = mask == 0
        covered |= vis
        n_vis += vis
        score = _mask_score(mask)
        better = score < best_score
        best_score = np.where(better, score, best_score)
        best_mask = np.where(better, mask, best_mask)
        timings[s.id] = round(_time.perf_counter() - t_s0, 4)

    coverage = covered.mean(axis=0)
    per_time_pct = 100.0 * covered.mean(axis=1)
    chosen = np.where(covered, 0, primary_reason(best_mask))        # (T, C)
    reason_fraction = np.stack([(chosen == b).mean(axis=0) for b in ALL_REASONS], axis=1)  # (C, 8)
    dominant_idx = reason_fraction.argmax(axis=1)
    reason_dominant = np.where(coverage >= 1.0, 0, np.array(ALL_REASONS)[dominant_idx])

    meta = {
        "t0_s": float(t[0]), "t1_s": float(t[-1]), "n_t": int(T), "n_cells": int(Cn),
        "radius_m": float(radius_m), "albedo": float(albedo), "margin_mag": float(margin_mag),
        "network": network if isinstance(network, str) else "custom",
        "elapsed_s": round(_time.perf_counter() - tic, 3), "per_sensor_s": timings,
        "reference_object": "diffuse sphere (Lambertian), SIMULATED reference target",
        "note": "Coverage = any sensor has zero geometric+photometric violations; FOV/slew not applied.",
    }
    return CoverageResult(g, t, coverage, per_time_pct, reason_dominant, reason_fraction,
                          [s.id for s in sensors], n_vis.mean(axis=0), meta)
