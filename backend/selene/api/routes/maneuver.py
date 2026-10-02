"""POST /api/maneuver/detect — synthetic maneuver-detection scenario on a SIMULATED catalog object.

Runs truth (+ optional SIMULATED injected burn) → angles-only measurements from the chosen
sensors → sequential filter (``selene.od.ukf`` when importable, else the compact EKF; the
response says which) → filter-health gate → NIS / windowed-NIS / gap re-fit / CUSUM / NEES
tests → impulsive Δv estimate when a maneuver is declared.

Real (JPL Horizons) objects are refused with 400: their cached ephemeris is an interpolated
navigation product, so residual tests against it would report model mismatch as a "maneuver"
of a real spacecraft, which the project never does.

``status`` in the response distinguishes ``no_observations`` (blind), ``insufficient_updates``,
``filter_not_converged`` / ``filter_inconsistent`` (no verdict possible), ``quiet`` and
``maneuver_declared``; ``declared`` is True only for the last.
"""
from __future__ import annotations

from typing import Any, Literal, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, ConfigDict, Field

from selene.api.routes.catalog import err_detail, parse_utc, round_floats
from selene.dynamics.frames import DEMO_EPOCH_UTC
from selene.maneuver.config import DetectorConfig, EstimatorConfig, FilterConfig
from selene.maneuver.pipeline import run_scenario
from selene.maneuver.synthetic import DIRECTION_NAMES

router = APIRouter(prefix="/maneuver", tags=["maneuver"])

MAX_SPAN_DAYS = 30.0
#: budget on epochs x max_obs_per_epoch (sequential filter updates, ~2-3 ms each plus the truth
#: propagation): 2 000 keeps the route under ~5 s; 29 d / 30 min / 6 per epoch was 8 352 updates and 19 s
MAX_OBSERVATIONS = 2000


class InjectedBurn(BaseModel):
    model_config = ConfigDict(extra="forbid")
    t_burn_utc: str
    dv_mps: Optional[list[float]] = Field(None, min_length=3, max_length=3, description="GCRF components [m/s]")
    magnitude_mps: Optional[float] = Field(None, ge=0.0, le=2000.0)
    direction: Optional[str] = Field(None, description=f"one of {DIRECTION_NAMES}")


class ManeuverDetectRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    object_id: str = "SIM-DRO-01"
    t0: str = DEMO_EPOCH_UTC
    t1: Optional[str] = Field(None, description="UTC ISO end (default t0 + 7 days)")
    sensors: Optional[list[str]] = Field(None, description="sensor ids (default: all ground sites then all space observers)")
    cadence_min: float = Field(360.0, ge=10.0, le=1440.0)
    sigma_arcsec: float = Field(1.0, gt=0.0, le=60.0)
    alpha: float = Field(0.01, gt=0.0, lt=0.5)
    window: int = Field(5, ge=1, le=50)
    gap_hours_for_refit: float = Field(12.0, ge=1.0, le=240.0)
    injected: Optional[InjectedBurn] = None
    seed: int = 0
    estimate: bool = True
    filter: Literal["auto", "ukf", "ekf"] = "auto"
    max_obs_per_epoch: int = Field(1, ge=1, le=6)
    sigma_pos0_km: Optional[float] = Field(None, gt=0.0, description="initial 1-σ position [km]; default: regime-dependent (20 km cislunar, 2 km Moon-bound)")
    sigma_vel0_mps: Optional[float] = Field(None, gt=0.0, description="initial 1-σ velocity [m/s]; default: regime-dependent (2 m/s cislunar, 0.1 m/s Moon-bound)")
    q_psd_km2_s3: float = Field(1e-18, ge=0.0)


class ManeuverDetectResponse(BaseModel):
    model_config = ConfigDict(extra="allow")
    object_id: str
    kind: str
    label: str
    filter: dict[str, Any]
    detections: list[dict[str, Any]]
    declared: bool
    status: str
    status_note: str = ""
    dv_estimate: Optional[dict[str, Any]] = None
    truth: Optional[dict[str, Any]] = None
    timing: dict[str, Any]
    summary: dict[str, Any]


@router.post("/detect", response_model=ManeuverDetectResponse)
def maneuver_detect(req: ManeuverDetectRequest):
    t0_s = parse_utc(req.t0, "t0")
    t1_s = parse_utc(req.t1, "t1") if req.t1 else t0_s + 7 * 86400.0
    if t1_s <= t0_s:
        raise HTTPException(400, detail="t1 must be after t0")
    if (t1_s - t0_s) > MAX_SPAN_DAYS * 86400.0 + 60.0:   # 60 s tolerance: the cap is meant in UTC, t_s is TDB
        raise HTTPException(400, detail=f"span must be <= {MAX_SPAN_DAYS:.0f} days")
    cadence_s = req.cadence_min * 60.0
    n_epochs = int((t1_s - t0_s) / cadence_s) + 1
    if n_epochs * req.max_obs_per_epoch > MAX_OBSERVATIONS:
        raise HTTPException(400, detail=f"{n_epochs} epochs x {req.max_obs_per_epoch} observations per epoch exceeds the per-request "
                                        f"budget of {MAX_OBSERVATIONS} filter updates; increase cadence_min, shorten the window or lower "
                                        f"max_obs_per_epoch")
    burn = None
    if req.injected is not None:
        tb = parse_utc(req.injected.t_burn_utc, "t_burn_utc")
        if not (t0_s < tb < t1_s):
            raise HTTPException(400, detail="t_burn_utc must lie strictly inside (t0, t1)")
        if req.injected.dv_mps is None and (req.injected.magnitude_mps is None or req.injected.direction is None):
            raise HTTPException(400, detail="injected needs dv_mps or (magnitude_mps and direction)")
        if req.injected.direction is not None and req.injected.direction.lower() not in DIRECTION_NAMES:
            raise HTTPException(400, detail=f"direction must be one of {DIRECTION_NAMES}")
        burn = {"t_s": tb, "dv_mps": req.injected.dv_mps, "magnitude_mps": req.injected.magnitude_mps,
                "direction": req.injected.direction}
    det_cfg = DetectorConfig(alpha=req.alpha, window=req.window, gap_hours_for_refit=req.gap_hours_for_refit)
    filt_cfg = FilterConfig(sigma_pos0_km=req.sigma_pos0_km, sigma_vel0_mps=req.sigma_vel0_mps, q_psd_km2_s3=req.q_psd_km2_s3)
    try:
        res = run_scenario(req.object_id, t0_s, t1_s, req.sensors, cadence_s, req.sigma_arcsec, burn, req.seed,
                           det_cfg, EstimatorConfig(), filt_cfg, req.filter, req.estimate, req.max_obs_per_epoch)
    except KeyError as e:
        raise HTTPException(404, detail=err_detail(e))
    except ValueError as e:
        raise HTTPException(400, detail=err_detail(e))
    except RuntimeError as e:
        raise HTTPException(500, detail=str(e))
    payload = round_floats(res.as_dict(), 8)
    return ManeuverDetectResponse(**payload)
