"""POST /api/coverage — cislunar coverage heatmap for a sensor network."""
from __future__ import annotations

from typing import Any, Literal, Optional, Union

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from selene.dynamics.frames import DEMO_EPOCH_UTC
from selene.sensors.coverage import NETWORK_PRESETS, compute_coverage
from selene.time import seconds_since_j2000_tdb

router = APIRouter(prefix="/coverage", tags=["coverage"])


class CustomNetwork(BaseModel):
    ground_ids: list[str] = Field(default_factory=list)
    space: list[Union[str, dict[str, Any]]] = Field(default_factory=list,
                                                   description="observer ids or SpaceObserver kwargs")


class CoverageRequest(BaseModel):
    t0: str = Field(DEMO_EPOCH_UTC, description="UTC ISO start")
    t1: Optional[str] = Field(None, description="UTC ISO end; default t0 + 7 days")
    n_t: int = Field(168, ge=2, le=720,
                     description="time steps; runtime scales with n_t x sensors (default 168 h hourly, 9 sites "
                                 "+ 1 observer ~1.5 s; 720 steps with the 'full' preset ~9 s, computed synchronously)")
    network: Union[str, CustomNetwork] = "ground_only"
    grid: Literal["2d", "3d"] = "2d"
    radius_m: float = Field(1.0, gt=0)
    albedo: float = Field(0.2, gt=0, le=1)
    margin_mag: float = Field(0.0, ge=0)


class PerTime(BaseModel):
    t: float
    pct: float


class CoverageResponse(BaseModel):
    grid: dict
    coverage: list[float]
    reason_dominant: list[int]
    reason_dominant_name: list[str]
    reason_fraction: dict[str, list[float]]
    n_sensors_visible: list[float]
    per_time: list[PerTime]
    meta: dict


@router.get("/presets")
def presets():
    return {name: {"ground_ids": list(g), "space": list(s)} for name, (g, s) in NETWORK_PRESETS.items()}


@router.post("", response_model=CoverageResponse)
def coverage(req: CoverageRequest):
    t0_s = float(seconds_since_j2000_tdb(req.t0))
    t1_s = float(seconds_since_j2000_tdb(req.t1)) if req.t1 else t0_s + 7 * 86400.0
    if t1_s <= t0_s:
        raise HTTPException(400, detail="t1 must be after t0")
    if req.grid == "3d" and req.n_t > 400:
        raise HTTPException(400, detail="3d grid limited to n_t <= 400 to bound runtime")
    net = req.network if isinstance(req.network, str) else req.network.model_dump()
    try:
        res = compute_coverage(t0_s, t1_s, req.n_t, net, req.grid, req.radius_m, req.albedo, req.margin_mag)
    except KeyError as e:
        raise HTTPException(400, detail=str(e))
    except ValueError as e:
        raise HTTPException(400, detail=str(e))
    d = res.as_dict()
    d["meta"]["t0_utc"] = req.t0
    d["meta"]["t1_utc"] = req.t1
    return CoverageResponse(**d)
