"""Convenience builders: object / sensor selection, slot grid, and the full :class:`Scenario`.

Defaults (the API's default request): the 8 xGEO custody objects below, the 9-sensor mixed
network ``'default'`` (5 ground sites spread in longitude + GEO, L1-halo, L2-halo and DRO
observers), 48 h from the demo epoch, 20-minute slots.

Why the default is not ground-only: the demo epoch (2026-03-01) is two days before full Moon,
and under the project's lunar-glare model (3° → 15° exclusion with illuminated fraction,
``sensors/constraints.py``) every one of the eight objects -- all within 3–11° of the Moon as
seen from Earth (illuminated fraction 0.93 → 14° exclusion) -- is unobservable from every ground site for the whole 48 h.  Scanning start
days across the 14-day catalog window, the ground network sees these objects at most ~2 % of
slot nodes.  That is the market problem, and ``preset='ground_only'`` reproduces it honestly;
but a default scenario in which no policy can observe anything would not exercise the tasker.
"""
from __future__ import annotations

import time
from typing import Optional, Sequence

import numpy as np

from selene.dynamics.frames import DEMO_EPOCH_TDB_S
from selene.sensors.coverage import NETWORK_PRESETS
from selene.sensors.observers import DEFAULT_OBSERVERS, SpaceObserver
from selene.sensors.sites import DEFAULT_SITES
from selene.sensors.visibility import Sensor
from selene.tasking.greedy import Scenario
from selene.tasking.information import DEFAULT_Q_PSD, build_tracks, build_visibility_table

__all__ = ["DEFAULT_OBJECT_IDS", "DEFAULT_SLOT_MIN", "DEFAULT_HORIZON_H", "TASKING_PRESETS", "resolve_objects",
           "resolve_sensors", "slot_grid", "make_scenario"]

#: the default custody set: eight SIMULATED xGEO objects (DROs, NRHOs, halos, Lyapunov)
DEFAULT_OBJECT_IDS: tuple[str, ...] = (
    "SIM-DRO-01", "SIM-DRO-02", "SIM-NRHO-RELAY-01", "SIM-NRHO-02",
    "SIM-L1-HALO-01", "SIM-L2-HALO-01", "SIM-L2-HALO-LOG-01", "SIM-L2-LYAP-01",
)
DEFAULT_SLOT_MIN = 20.0
DEFAULT_HORIZON_H = 48.0

_ALL_SENSORS: dict[str, Sensor] = {s.id: s for s in DEFAULT_SITES}
_ALL_SENSORS.update({o.id: o for o in DEFAULT_OBSERVERS})

#: tasking presets: the coverage-module presets plus the 9-sensor mixed default
TASKING_PRESETS: dict[str, tuple[tuple[str, ...], tuple[str, ...]]] = dict(NETWORK_PRESETS)
TASKING_PRESETS["mixed_9"] = (("haleakala", "mt_lemmon", "teide", "sutherland", "siding_spring"),
                              ("geo_west", "l1_halo_obs", "l2_halo_obs", "dro_obs"))
TASKING_PRESETS["default"] = TASKING_PRESETS["mixed_9"]


def resolve_objects(spec: str | Sequence[str] | None, catalog=None) -> list[str]:
    """'default' | 'all_simulated' | 'all' | explicit id list -> validated object ids."""
    from selene.objects.catalog import get_catalog

    cat = catalog or get_catalog()
    if spec is None or spec == "default":
        ids = list(DEFAULT_OBJECT_IDS)
    elif spec == "all_simulated":
        ids = [e.id for e in cat.objects("simulated")]
    elif spec == "all":
        ids = cat.ids()
    elif isinstance(spec, str):
        ids = [spec]
    else:
        ids = list(spec)
    ids = list(dict.fromkeys(ids))          # drop duplicates, keep order (duplicates would double-count metrics)
    unknown = [i for i in ids if i not in cat]
    if unknown:
        raise KeyError(f"unknown object ids {unknown}; known: {cat.ids()}")
    if not ids:
        raise ValueError("no objects selected")
    return ids


def resolve_sensors(sensor_ids: Optional[Sequence[str]] = None, preset: Optional[str] = None,
                    extra: Sequence[Sensor | dict] = ()) -> list[Sensor]:
    """Explicit ids and/or a :data:`TASKING_PRESETS` name ('default' = 'mixed_9'), plus ad-hoc
    :class:`SpaceObserver` specs (dicts) for architecture studies."""
    out: list[Sensor] = []
    seen: set[str] = set()
    if preset:
        if preset not in TASKING_PRESETS:
            raise KeyError(f"unknown sensor preset {preset!r}; known: {sorted(TASKING_PRESETS)}")
        g, s = TASKING_PRESETS[preset]
        for i in list(g) + list(s):
            out.append(_ALL_SENSORS[i]); seen.add(i)
    for i in sensor_ids or ():
        if i in seen:
            continue
        if i not in _ALL_SENSORS:
            raise KeyError(f"unknown sensor id {i!r}; known: {sorted(_ALL_SENSORS)}")
        out.append(_ALL_SENSORS[i]); seen.add(i)
    for item in extra:
        if isinstance(item, dict):
            kw = {k: v for k, v in item.items() if k not in ("kind", "spec_note", "orbit")}
            if kw.get("custom_ic") is not None:
                kw["custom_ic"] = tuple(float(v) for v in kw["custom_ic"])
            item = SpaceObserver(**kw)
        if item.id in seen:
            continue
        out.append(item); seen.add(item.id)
    if not out and sensor_ids is None and preset is None:
        g, s = TASKING_PRESETS["default"]
        out = [_ALL_SENSORS[i] for i in list(g) + list(s)]
    if not out:
        raise ValueError("no sensors selected")
    return out


def slot_grid(t0_s: float, t1_s: float, slot_min: float) -> np.ndarray:
    """Node times t0, t0+Δ, ..., ≤ t1 (at least two nodes)."""
    dt = float(slot_min) * 60.0
    if dt <= 0:
        raise ValueError("slot_min must be positive")
    n = int(np.floor((t1_s - t0_s) / dt + 1e-9))
    if n < 1:
        raise ValueError("window shorter than one slot")
    return t0_s + dt * np.arange(n + 1)


def make_scenario(
    object_ids: str | Sequence[str] | None = None,
    sensors: Sequence[Sensor] | None = None,
    *,
    sensor_ids: Optional[Sequence[str]] = None,
    preset: Optional[str] = None,
    extra_sensors: Sequence[Sensor | dict] = (),
    t0_s: float = DEMO_EPOCH_TDB_S,
    horizon_h: float = DEFAULT_HORIZON_H,
    t1_s: Optional[float] = None,
    slot_min: float = DEFAULT_SLOT_MIN,
    q_psd: float = DEFAULT_Q_PSD,
    initial_sigma_km: float = 10.0,
    initial_sigma_vel_kms: float = 1e-4,
    sigma_arcsec: float | dict[str, float] = 1.0,
    priorities: Optional[dict[str, float]] = None,
    gain_kind: str = "trace",
    acquisition: str = "fov",
    n_obs_per_slot: int = 1,
    search_tiles: int = 1,
    slew_fraction: float = 1.0,
    tslo_weight: float = 0.0,
    margin_mag: float = 0.0,
) -> Scenario:
    """Build tracks + visibility table and wrap them in a :class:`Scenario` (timings in ``meta``)."""
    t_a = time.perf_counter()
    ids = resolve_objects(object_ids)
    sens = list(sensors) if sensors is not None else resolve_sensors(sensor_ids, preset, extra_sensors)
    t1 = float(t1_s) if t1_s is not None else float(t0_s) + float(horizon_h) * 3600.0
    nodes = slot_grid(float(t0_s), t1, slot_min)
    tracks = build_tracks(ids, nodes, q_psd=q_psd, sigma0_pos_km=initial_sigma_km, sigma0_vel_kms=initial_sigma_vel_kms,
                          priorities=priorities)
    t_b = time.perf_counter()
    table = build_visibility_table(sens, tracks, sigma_arcsec=sigma_arcsec, margin_mag=margin_mag)
    t_c = time.perf_counter()
    scn = Scenario(tracks=tracks, table=table, slot_s=float(slot_min) * 60.0, gain_kind=gain_kind,  # type: ignore[arg-type]
                   acquisition=acquisition, n_obs_per_slot=int(n_obs_per_slot),  # type: ignore[arg-type]
                   search_tiles=int(max(1, search_tiles)), slew_fraction=float(slew_fraction), tslo_weight=float(tslo_weight))
    scn.timing = {"tracks_s": t_b - t_a, "visibility_s": t_c - t_b, "total_s": t_c - t_a}
    scn.config = {
        "object_ids": ids, "sensor_ids": [s.id for s in sens], "t0_s": float(t0_s), "t1_s": float(nodes[-1]),
        "slot_min": float(slot_min), "n_slots": int(nodes.size - 1), "q_psd_km2_s3": q_psd,
        "initial_sigma_km": initial_sigma_km, "initial_sigma_vel_kms": initial_sigma_vel_kms,
        "sigma_arcsec": sigma_arcsec, "gain_kind": gain_kind, "acquisition": acquisition,
        "n_obs_per_slot": n_obs_per_slot, "search_tiles": int(max(1, search_tiles)), "slew_fraction": slew_fraction,
        "tslo_weight": tslo_weight,
    }
    return scn
