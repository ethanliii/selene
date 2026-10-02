"""GET /api/sensors and GET /api/sensors/{id}/visibility."""
from __future__ import annotations

from typing import Optional

import numpy as np
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel

from selene.dynamics.frames import DEMO_EPOCH_UTC
from selene.sensors.observers import DEFAULT_OBSERVERS, orbit_summary
from selene.sensors.reasons import ALL_REASONS, REASON_NAMES
from selene.sensors.sites import DEFAULT_SITES, SPEC_DISCLAIMER
from selene.sensors.visibility import visibility
from selene.time import seconds_since_j2000_tdb

router = APIRouter(prefix="/sensors", tags=["sensors"])

_SENSORS = {s.id: s for s in DEFAULT_SITES}
_SENSORS.update({o.id: o for o in DEFAULT_OBSERVERS})


class SensorListResponse(BaseModel):
    ground: list[dict]
    space: list[dict]
    note: str
    reason_bits: dict[str, int]


class VisibilityResponse(BaseModel):
    sensor_id: str
    object_id: str
    t0: str
    t1: str
    n: int
    fraction_visible: float
    t_s: list[float]
    visible: list[bool]
    magnitude: list[Optional[float]]
    reasons: list[int]
    reason_names: list[list[str]]
    range_km: list[float]
    phase_deg: list[float]


@router.get("", response_model=SensorListResponse)
def list_sensors():
    space = []
    for o in DEFAULT_OBSERVERS:
        d = o.as_dict()
        d["orbit"] = orbit_summary(o)
        space.append(d)
    return SensorListResponse(
        ground=[s.as_dict() for s in DEFAULT_SITES],
        space=space,
        note=SPEC_DISCLAIMER + " Space observers are notional platforms on CR3BP periodic orbits.",
        reason_bits={REASON_NAMES[b]: int(b) for b in ALL_REASONS},
    )


def _object_track(object_id: str, t_s: np.ndarray) -> np.ndarray:
    """(N,3) GCRF positions of a catalog object, or raise 404 with a clear message."""
    try:
        from selene.objects.catalog import get_catalog  # built by another track; optional here
    except Exception:
        raise HTTPException(404, detail="object catalog (selene.objects.catalog) is not available yet; "
                                        "visibility by object_id requires it")
    try:
        cat = get_catalog()
        try:
            st = np.asarray(cat.state_at(object_id, t_s), dtype=float)
            if st.ndim != 2 or st.shape[0] != t_s.size:
                raise ValueError("not vectorised")
        except Exception:
            st = np.stack([np.asarray(cat.state_at(object_id, float(ti)), dtype=float).ravel() for ti in t_s])
    except HTTPException:
        raise
    except KeyError:
        raise HTTPException(404, detail=f"unknown object_id {object_id!r}")
    except Exception as e:  # noqa: BLE001
        raise HTTPException(404, detail=f"could not evaluate object {object_id!r}: {e}")
    return st[:, :3]


@router.get("/{sensor_id}/visibility", response_model=VisibilityResponse)
def sensor_visibility(
    sensor_id: str,
    object_id: str = Query(..., description="catalog object id"),
    t0: str = Query(DEMO_EPOCH_UTC, description="UTC ISO start"),
    t1: Optional[str] = Query(None, description="UTC ISO end (default t0 + 7 days)"),
    n: int = Query(169, ge=2, le=5000),
    radius_m: float = Query(1.0, gt=0),
    albedo: float = Query(0.2, gt=0, le=1),
):
    sensor = _SENSORS.get(sensor_id)
    if sensor is None:
        raise HTTPException(404, detail=f"unknown sensor {sensor_id!r}; known: {sorted(_SENSORS)}")
    t0_s = float(seconds_since_j2000_tdb(t0))
    t1_s = float(seconds_since_j2000_tdb(t1)) if t1 else t0_s + 7 * 86400.0
    t_s = np.linspace(t0_s, t1_s, int(n))
    track = _object_track(object_id, t_s)
    res = visibility(sensor, track, t_s, radius_m, albedo).as_dict()
    return VisibilityResponse(sensor_id=sensor_id, object_id=object_id, t0=t0, t1=t1 or "", n=int(n), **res)
