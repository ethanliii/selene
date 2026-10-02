"""POST /api/tasking/schedule — sensor tasking for custody (greedy / MILP / baselines / compare).

Also ``GET /api/tasking/presets`` (sensor presets, default objects) for the UI.  Pydantic models
are defined here (this route owns its contract; ``schemas.TaskingRequest`` is the older stub).
"""
from __future__ import annotations

import time
from typing import Any, Literal, Optional, Union

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from selene.api.routes.catalog import parse_utc, round_floats, tdb_s_to_utc_iso
from selene.dynamics.frames import DEMO_EPOCH_UTC
from selene.tasking.greedy import ScheduleResult, run_greedy
from selene.tasking.information import DEFAULT_Q_PSD
from selene.tasking.metrics import DEFAULT_CUSTODY_THRESHOLD_KM, comparison_row, custody_metrics, null_custody_pct, run_random, run_round_robin
from selene.tasking.optimize import HAVE_MILP, run_milp
from selene.tasking.scenario import DEFAULT_HORIZON_H, DEFAULT_OBJECT_IDS, DEFAULT_SLOT_MIN, TASKING_PRESETS, make_scenario

router = APIRouter(prefix="/tasking", tags=["tasking"])

Method = Literal["greedy", "milp", "random", "round_robin", "compare"]

ASSUMPTIONS = [
    "Linear covariance analysis: measurement noise is never sampled, so the estimate mean stays on the "
    "reference (truth) trajectory and all visibility/geometry is evaluated there.",
    "Prior covariance is isotropic diag(sigma_pos^2 I3, sigma_vel^2 I3); real post-fit covariances are along-track elongated.",
    "Process noise is continuous white-noise acceleration with PSD q_psd [km^2/s^3] (default 1e-18 ~ 30 % of SRP "
    "on a medium bus; raise to 1e-12..1e-10 for objects that may be thrusting).",
    "One angles-only (RA/Dec) tracklet per assigned slot with the sensor's astrometric sigma (default 1 arcsec). "
    "Acquisition ('fov', default): the expected posterior covariance is the Bernoulli mixture p*P_det + (1-p)*P_prior with "
    "p the probability that the object falls inside the field of view (search_tiles fields around the prediction), so "
    "custody that is lost stays lost under pointed observations until a search or a wider field is used; 'irf' (R/p) "
    "is the optimistic information-reduction-factor alternative, 'none' assumes certain detection.",
    "Slew: a sensor may re-point by slew_rate * slew_fraction * (time since its previous pointing); no settle time, "
    "no slew-only action.",
    "Lunar-glare exclusion for ground sites follows sensors/constraints.py (3 deg at new Moon to 15 deg at full Moon); "
    "at the demo epoch (2 days before full Moon) it blinds every ground site to every xGEO object in the default set.",
    "Custody = RSS position sigma below custody_threshold_km at a slot node (post-update), reported as a time fraction; "
    "an object whose prior is already inside the threshold stays in custody without any observation for as long as the "
    "propagated sigma holds (overall.custody_pct_null is the zero-observation reference; on the default request it is 100 %, "
    "so custody_pct does not discriminate policies there -- summed_trace_mean_km2 and mean_tslo_h do).",
    "MILP objective values are a surrogate (rank-weighted gains frozen at each window start) and are reported next to the "
    "realised metrics; the run is bounded by time_budget_s, after which remaining slots fall back to greedy (budget_exhausted).",
]


class TaskingScheduleRequest(BaseModel):
    object_ids: Union[str, list[str]] = Field("default", description="'default' (8 xGEO objects), 'all_simulated', 'all', or ids")
    sensor_ids: Optional[list[str]] = Field(None, description="explicit sensor ids (ground sites / space observers)")
    preset: Optional[str] = Field(None, description=f"sensor preset; one of {sorted(TASKING_PRESETS)} (default 'default' = mixed_9)")
    extra_sensors: list[dict[str, Any]] = Field(default_factory=list, description="ad-hoc SpaceObserver kwargs (architecture studies)")
    t0: str = Field(DEMO_EPOCH_UTC, description="UTC ISO start")
    t1: Optional[str] = Field(None, description="UTC ISO end (default t0 + 48 h)")
    slot_min: float = Field(DEFAULT_SLOT_MIN, gt=0.5, le=720)
    method: Method = "greedy"
    custody_threshold_km: float = Field(DEFAULT_CUSTODY_THRESHOLD_KM, gt=0)
    initial_sigma_km: float = Field(10.0, gt=0)
    initial_sigma_vel_kms: float = Field(1e-4, gt=0)
    q_psd: float = Field(DEFAULT_Q_PSD, ge=0, description="acceleration PSD [km^2/s^3]")
    sigma_arcsec: float = Field(1.0, gt=0)
    gain_kind: Literal["logdet", "trace", "maxeig"] = "trace"
    acquisition: Literal["fov", "irf", "none"] = "fov"
    n_obs_per_slot: int = Field(1, ge=1, le=100)
    search_tiles: int = Field(1, ge=1, le=100, description="fields mosaicked around the prediction (acquisition probability)")
    slew_fraction: float = Field(1.0, gt=0, le=1.0)
    tslo_weight: float = Field(0.0, ge=0)
    priorities: dict[str, float] = Field(default_factory=dict)
    horizon_slots: int = Field(4, ge=1, le=12, description="MILP look-ahead window (slots)")
    revisit_weight: float = Field(0.05, ge=0)
    time_budget_s: float = Field(20.0, gt=0, le=300, description="wall-clock budget for the MILP path; remaining slots fall back to greedy")
    seed: int = 0
    max_slots: int = Field(1000, ge=1, le=5000, description="guard on n_slots (runtime)")
    include_series: bool = True


class TaskingScheduleResponse(BaseModel):
    method: str
    config: dict[str, Any]
    schedule: list[dict[str, Any]]
    per_object: list[dict[str, Any]]
    sensors: list[dict[str, Any]]
    overall: dict[str, Any]
    comparison: list[dict[str, Any]]
    visibility_fraction: dict[str, dict[str, float]]
    timing: dict[str, float]
    epochs_utc: list[str]
    assumptions: list[str]
    disclaimer: str


DISCLAIMER = ("All objects of kind 'simulated' are SIMULATED notional spacecraft; sensor specifications are assumed "
              "representative values, not those of real instruments. Metrics are reported as computed, including poor custody.")


@router.get("/presets")
def presets():
    return {
        "sensor_presets": {k: {"ground_ids": list(g), "space": list(s)} for k, (g, s) in TASKING_PRESETS.items()},
        "default_object_ids": list(DEFAULT_OBJECT_IDS),
        "methods": ["greedy", "milp", "random", "round_robin", "compare"],
        "milp_available": HAVE_MILP,
        "defaults": {"slot_min": DEFAULT_SLOT_MIN, "horizon_h": DEFAULT_HORIZON_H, "custody_threshold_km": DEFAULT_CUSTODY_THRESHOLD_KM,
                     "q_psd": DEFAULT_Q_PSD, "initial_sigma_km": 10.0},
        "assumptions": ASSUMPTIONS,
    }


def _run(method: str, scn, req: TaskingScheduleRequest, budget_s: float) -> ScheduleResult:
    if method == "greedy":
        return run_greedy(scn)
    if method == "milp":
        return run_milp(scn, horizon=req.horizon_slots, revisit_weight=req.revisit_weight, time_budget_s=budget_s)
    if method == "random":
        return run_random(scn, req.seed)
    if method == "round_robin":
        return run_round_robin(scn)
    raise HTTPException(400, detail=f"unknown method {method!r}")


@router.post("/schedule", response_model=TaskingScheduleResponse)
def schedule(req: TaskingScheduleRequest):
    t_start = time.perf_counter()
    t0_s = parse_utc(req.t0, "t0")
    t1_s = parse_utc(req.t1, "t1") if req.t1 else t0_s + DEFAULT_HORIZON_H * 3600.0
    if t1_s <= t0_s:
        raise HTTPException(400, detail="t1 must be after t0")
    n_slots = int((t1_s - t0_s) / (req.slot_min * 60.0))
    if n_slots < 1:
        raise HTTPException(400, detail="window shorter than one slot")
    if n_slots > req.max_slots:
        raise HTTPException(400, detail=f"{n_slots} slots exceeds max_slots={req.max_slots}; increase slot_min or shorten the window")
    try:
        scn = make_scenario(
            req.object_ids, sensor_ids=req.sensor_ids, preset=req.preset, extra_sensors=req.extra_sensors,
            t0_s=t0_s, t1_s=t1_s, slot_min=req.slot_min, q_psd=req.q_psd, initial_sigma_km=req.initial_sigma_km,
            initial_sigma_vel_kms=req.initial_sigma_vel_kms, sigma_arcsec=req.sigma_arcsec, priorities=req.priorities,
            gain_kind=req.gain_kind, acquisition=req.acquisition, n_obs_per_slot=req.n_obs_per_slot,
            search_tiles=req.search_tiles, slew_fraction=req.slew_fraction, tslo_weight=req.tslo_weight,
        )
    except (KeyError, ValueError, TypeError) as e:
        raise HTTPException(400, detail=str(e))
    except RuntimeError as e:
        raise HTTPException(500, detail=f"scenario build failed: {e}")
    timing = dict(scn.timing)
    comparison: list[dict] = []
    # MILP wall-clock budget: what is left of time_budget_s after the scenario build (and, in compare mode, the fast policies)
    def _budget() -> float:
        return max(1.0, req.time_budget_s - (time.perf_counter() - t_start))

    if req.method == "compare":
        results = {}
        for m in ("greedy", "random", "round_robin", "milp"):
            t = time.perf_counter()
            results[m] = _run(m, scn, req, _budget())
            timing[f"{m}_s"] = time.perf_counter() - t
        comparison = [comparison_row(results[m], req.custody_threshold_km) for m in ("greedy", "milp", "random", "round_robin")]
        primary = results["greedy"]
        method = "compare(primary=greedy)"
    else:
        t = time.perf_counter()
        primary = _run(req.method, scn, req, _budget())
        timing[f"{req.method}_s"] = time.perf_counter() - t
        method = primary.method
    m = custody_metrics(primary, req.custody_threshold_km)
    per_object = m["per_object"]
    overall = m["overall"]
    overall["custody_pct_null"] = null_custody_pct(scn, req.custody_threshold_km)
    for key in ("budget_exhausted", "n_budget_exhausted", "n_greedy_kept", "n_filled_idle", "time_budget_s"):
        if key in primary.meta:
            overall[key] = primary.meta[key]
    for row in comparison:
        row["custody_pct_null"] = overall["custody_pct_null"]
    if not req.include_series:
        for o in per_object:
            for k in ("sigma_series_km", "tslo_series_h", "in_custody_series"):
                o.pop(k, None)
        overall.pop("summed_trace_series_km2", None)
    if "objective_windows" in primary.meta:
        overall["objective_windows"] = primary.meta["objective_windows"]
    frac = scn.table.fraction_visible()
    vis = {sid: {oid: float(frac[s, j]) for j, oid in enumerate(scn.table.object_ids)} for s, sid in enumerate(scn.table.sensor_ids)}
    timing["total_s"] = time.perf_counter() - t_start
    return TaskingScheduleResponse(
        method=method,
        config={**scn.config, "custody_threshold_km": req.custody_threshold_km, "method": req.method,
                "horizon_slots": req.horizon_slots, "time_budget_s": req.time_budget_s, "t0_utc": tdb_s_to_utc_iso(t0_s)[0],
                "t1_utc": tdb_s_to_utc_iso(scn.table.t_nodes[-1])[0]},  # not rounded: q_psd ~ 1e-18
        schedule=[round_floats(a.as_dict(), 4) for a in primary.assignments],
        per_object=round_floats(per_object, 4),
        sensors=round_floats(m["sensors"], 4),
        overall=round_floats(overall, 4),
        comparison=round_floats(comparison, 4),
        visibility_fraction=round_floats(vis, 4),
        timing=round_floats(timing, 4),
        epochs_utc=tdb_s_to_utc_iso(scn.table.t_nodes),
        assumptions=ASSUMPTIONS,
        disclaimer=DISCLAIMER,
    )
