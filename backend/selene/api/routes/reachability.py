"""POST /api/reachability — Δv-sampled reachable set, region hits, envelope and sensor hints.

Also ``GET /api/reachability/regions`` (region definitions + display geometry) and
``GET /api/reachability/ladder`` (the magnitude ladder for a budget).  See
:mod:`selene.reachability.sampling` for the method and :mod:`selene.reachability.regions` for
the region definitions.
"""
from __future__ import annotations

import time as _time
from typing import Any, Optional

import numpy as np
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, ConfigDict, Field

from selene.api.routes.catalog import err_detail, parse_utc, round_floats, tdb_s_to_utc_iso
from selene.objects.catalog import get_catalog
from selene.reachability.regions import default_regions
from selene.reachability.sampling import (
    LADDER_MPS,
    ReachabilityConfig,
    compute_reachability,
    magnitude_ladder,
    refine_min_dv,
)
from selene.reachability.tasking_hint import resolve_sensors, sensor_hints

router = APIRouter(prefix="/reachability", tags=["reachability"])

DISCLAIMER = (
    "Reachability is a custody/awareness envelope under an ASSUMED impulsive delta-v budget: it says where an "
    "object COULD be (and which high-value regions it could enter) if it had maneuvered, so sensors can be pointed "
    "to re-acquire it. It asserts no intent, is not a maneuver prediction, and is never used for engagement planning. "
    "SIMULATED objects and events are notional."
)

#: n_samples -> n_dirs conversion constant: the full magnitude ladder has 9 levels (+ the budget
#: itself), so n_dirs = n_samples / (n_epochs * 8) yields ~n_samples at budgets near the top of the
#: ladder and fewer at small budgets (the direction set is kept budget-independent so results nest).
_LADDER_LEVELS_FOR_SIZING = 8
#: Hard server-side cap on the ACTUAL sample count n_dirs x n_magnitudes x n_epochs (one stacked
#: 6N ODE; ~20 000 rows x 168 h is ~15-40 s of propagation on a laptop and ~1.5 MB without paths).
MAX_TOTAL_SAMPLES = 20_000
#: With ``include_paths`` every sample carries (horizon_h / path_dt_h + 1) points in two frames at
#: ~60 bytes per point: 20 000 x 169 points was a 121 MB body.  ``path_dt_h`` is clamped UP so the
#: total stays below this (~9 MB; reported in ``config.caps_applied``); endpoints are always complete.
MAX_PATH_POINTS = 150_000


class ReachabilityRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    object_id: str
    t0: Optional[str] = Field(None, description="UTC ISO epoch of the estimated state; default: catalog epoch")
    dv_budget_mps: float = Field(50.0, ge=0.0, le=3000.0)
    horizon_h: float = Field(168.0, ge=24.0, le=168.0,
                             description="default 168 h: from a DRO the gateways/NRHO corridor only open up on the "
                                         "week scale (at 72 h and 50 m/s nothing new is reachable)")
    n_samples: int = Field(2000, ge=16, le=MAX_TOTAL_SAMPLES,
                           description="target sample count; converted to a budget-independent number of "
                                       "Fibonacci directions n_dirs = n_samples / (n_epochs * 8); the actual count "
                                       f"n_dirs x n_magnitudes x n_epochs is capped at {MAX_TOTAL_SAMPLES}")
    n_dirs: Optional[int] = Field(None, ge=1, le=4000, description="overrides n_samples (400 if the total would exceed the cap)")
    burn_epochs_h: list[float] = Field(default_factory=lambda: [0.0, 6.0, 12.0, 24.0])
    magnitudes_mps: Optional[list[float]] = Field(None, description="explicit magnitude levels in (0, dv_budget_mps] "
                                                                    "(default: absolute ladder <= budget)")
    seed: Optional[int] = Field(None, description="accepted for client compatibility; sampling is deterministic "
                                                  "(Fibonacci sphere x fixed ladder), so it has no effect")
    include_paths: bool = False
    path_dt_h: float = Field(6.0, gt=0.5, le=24.0)
    sensors: Optional[list[str]] = Field(None, description="sensor ids for the tasking hints; default: all notional sensors")
    include_hints: bool = True
    refine: bool = Field(True, description="bisect the minimum delta-v for every reached region (a few extra single-row runs)")
    gateway_radius_nd: float = Field(0.05, gt=0.0, le=0.2)
    nrho_tube_km: float = Field(10_000.0, gt=0.0, le=50_000.0)


class RegionOut(BaseModel):
    model_config = ConfigDict(extra="allow")
    key: str
    name: str
    kind: str
    fraction: float
    sample_fraction: float
    n_hit: int
    earliest_h: Optional[float] = None
    min_dv_mps: Optional[float] = None
    nominal_hits: bool
    newly_reachable: bool


class ReachabilityResponse(BaseModel):
    model_config = ConfigDict(extra="allow")
    object: dict[str, Any]
    t0_utc: str
    t0_tdb_s: float
    config: dict[str, Any]
    nominal: dict[str, Any]
    samples: dict[str, Any]
    regions: list[RegionOut]
    envelope: list[dict[str, Any]]
    sensor_hints: dict[str, Any]
    timing: dict[str, Any]
    meta: dict[str, Any]
    disclaimer: str


@router.get("/regions")
def regions(gateway_radius_nd: float = Query(0.05, gt=0.0, le=0.2), nrho_tube_km: float = Query(10_000.0, gt=0.0, le=50_000.0)):
    regs = default_regions(gateway_radius_nd=gateway_radius_nd, nrho_tube_km=nrho_tube_km)
    return {"regions": [round_floats(r.as_dict(), 6) for r in regs],
            "frame": "Earth-Moon rotating frame, nondimensional (L* = 384 400 km) unless a field says km",
            "disclaimer": DISCLAIMER}


@router.get("/ladder")
def ladder(dv_budget_mps: float = Query(50.0, ge=0.0, le=3000.0)):
    return {"ladder_mps": list(LADDER_MPS), "magnitudes_mps": magnitude_ladder(dv_budget_mps).tolist()}


@router.post("", response_model=ReachabilityResponse)
def reachability(req: ReachabilityRequest):
    tic = _time.perf_counter()
    cat = get_catalog()
    if req.object_id not in cat:
        raise HTTPException(404, detail=f"unknown object {req.object_id!r}; known ids: {cat.ids()}")
    e = cat.get(req.object_id)
    t0_s = parse_utc(req.t0, "t0") if req.t0 else cat.epoch_s
    try:
        s0 = cat.state_at(req.object_id, t0_s)
    except ValueError as err:
        raise HTTPException(400, detail=str(err))
    eps = sorted(set(float(v) for v in req.burn_epochs_h))
    if not eps or eps[0] < 0 or eps[-1] > req.horizon_h:
        raise HTTPException(400, detail="burn_epochs_h must lie within [0, horizon_h]")
    n_dirs = req.n_dirs if req.n_dirs is not None else max(8, req.n_samples // (len(eps) * _LADDER_LEVELS_FOR_SIZING))
    cr_am = float(e.physical.get("cr_area_mass_m2_kg", 0.0)) if e.physical else 0.0
    try:
        cfg = ReachabilityConfig(dv_budget_mps=req.dv_budget_mps, horizon_h=req.horizon_h, n_dirs=n_dirs,
                                 burn_epochs_h=tuple(eps), magnitudes_mps=req.magnitudes_mps, path_dt_h=req.path_dt_h,
                                 cr_area_mass=cr_am)
        sensors = resolve_sensors(req.sensors) if req.sensors is not None else None
    except (ValueError, KeyError, TypeError) as err:
        raise HTTPException(400, detail=err_detail(err))
    # hard cost guard on the ACTUAL sample count (n_dirs x n_magnitudes x n_epochs)
    per_dir = len(cfg.magnitudes()) * len(cfg.burn_epochs_h)
    n_total = cfg.n_dirs * per_dir
    caps: list[str] = []
    if n_total > MAX_TOTAL_SAMPLES:
        if req.n_dirs is not None:
            raise HTTPException(400, detail=f"n_dirs={req.n_dirs} x {per_dir} (magnitudes x epochs) = {n_total} samples "
                                            f"exceeds the server cap of {MAX_TOTAL_SAMPLES}; lower n_dirs, the budget "
                                            f"ladder or the number of burn epochs")
        cfg.n_dirs = max(1, MAX_TOTAL_SAMPLES // per_dir)
        caps.append(f"n_dirs -> {cfg.n_dirs} ({n_total} samples > {MAX_TOTAL_SAMPLES})")
    if req.include_paths:
        n_act = cfg.n_dirs * per_dir
        n_pts = n_act * (int(np.floor(cfg.horizon_h / cfg.path_dt_h + 1e-9)) + 1)
        if n_pts > MAX_PATH_POINTS:
            steps_ok = max(1, MAX_PATH_POINTS // n_act - 1)
            new_dt = max(cfg.grid_dt_h, np.ceil(cfg.horizon_h / steps_ok / cfg.grid_dt_h) * cfg.grid_dt_h)
            caps.append(f"path_dt_h {cfg.path_dt_h:g} -> {new_dt:g} h ({n_pts} path points > {MAX_PATH_POINTS})")
            cfg.path_dt_h = float(new_dt)
    regs = default_regions(gateway_radius_nd=req.gateway_radius_nd, nrho_tube_km=req.nrho_tube_km)
    try:
        rs = compute_reachability(s0, t0_s, cfg, object_id=req.object_id, regions=regs)
    except (ValueError, TypeError) as err:
        raise HTTPException(400, detail=f"invalid reachability request: {err}")
    except RuntimeError as err:
        raise HTTPException(500, detail=f"propagation failed: {err}")
    rs.meta["object"] = {"id": e.id, "name": e.name, "kind": e.kind, "label": e.label, "orbit_type": e.orbit_type,
                         "actor": e.actor, "is_real": e.is_real}

    # --- refinement of the minimum Δv per reached region --------------------------------
    refined: dict[str, dict] = {}
    if req.refine:
        t1 = _time.perf_counter()
        for st in rs.region_stats:
            if st.n_hit and st.min_dv_mps and st.min_dv_mps > 0:
                r = refine_min_dv(rs, st.key)
                if r:
                    refined[st.key] = r
        rs.timing["refine_s"] = _time.perf_counter() - t1

    # --- sensor hints -----------------------------------------------------------------
    hints: dict[str, Any] = {"steps": [], "summary": {}, "best_overall": None, "note": "hints disabled"}
    if req.include_hints:
        t1 = _time.perf_counter()
        radius_m = float(e.physical.get("radius_m", 1.0)) if e.physical else 1.0
        albedo = float(e.physical.get("albedo", 0.2)) if e.physical else 0.2
        h = sensor_hints(rs, sensors, radius_m=radius_m, albedo=albedo)
        hints = {"steps": [s.as_dict() for s in h["steps"]], "summary": h["summary"], "best_overall": h["best_overall"],
                 "note": h["note"], "photometry": {"radius_m": radius_m, "albedo": albedo}}
        rs.timing["hints_s"] = _time.perf_counter() - t1

    # --- payload ------------------------------------------------------------------------
    th = rs.t_h
    pidx = rs.path_indices()
    end_g, end_r = rs.endpoints()
    hit_lists = rs.regions_hit_by_sample()
    regions_out = []
    for st in rs.region_stats:
        d = st.as_dict()
        d["refined_min_dv"] = refined.get(st.key)
        reg = next(r for r in rs.regions if r.key == st.key)
        d["description"] = reg.description
        d["why_it_matters"] = reg.why_it_matters
        regions_out.append(d)
    samples: dict[str, Any] = {
        "n": rs.n_samples,
        "dv_dir_gcrf": round_floats(rs.dv_dirs, 5),
        "dv_mps": round_floats(rs.dv_mps, 3),
        "burn_h": round_floats(rs.burn_h, 3),
        "ray": rs.ray.tolist(),
        "end_gcrf": round_floats(end_g, 1),
        "end_rot": round_floats(end_r, 5),
        "end_t_h": round_floats(th[np.clip(rs.term_idx - 1, 0, len(th) - 1)], 3),
        "terminated": [r is not None for r in rs.term_reason],
        "termination_reason": rs.term_reason,
        "termination_t_h": round_floats(np.where(np.isfinite(rs.term_t_h), rs.term_t_h, np.nan), 3),
        "hit_regions": hit_lists,
        "hit_region": [h[0] if h else None for h in hit_lists],          # first region entered (spec name)
        "primary_region": [h[0] if h else None for h in hit_lists],      # alias kept for existing clients
        "first_hit_h": {k: round_floats(v, 2) for k, v in rs.first_hit_h.items()},
    }
    if req.include_paths:
        act = rs.active[:, pidx]
        pr = rs.states_rot[:, pidx, :3].copy()
        pg = rs.states_gcrf[:, pidx, :3].copy()
        pr[~act] = np.nan
        pg[~act] = np.nan
        samples["paths"] = {"t_h": round_floats(th[pidx], 3), "rot": round_floats(pr, 5), "gcrf": round_floats(pg, 1),
                            "note": "NaN (null) after a sample's termination (lunar impact / Earth re-entry)"}
    nominal = {
        "t_h": round_floats(th[pidx], 3),
        "gcrf": round_floats(rs.nominal_gcrf[pidx, :3], 1),
        "rot": round_floats(rs.nominal_rot[pidx, :3], 5),
        "state_t0_gcrf": round_floats(rs.nominal_gcrf[0], 6),
        "hit_regions": [k for k, v in rs.nominal_hits.items() if v.any()],
        "terminated": rs.meta.get("nominal_terminated"),
    }
    rs.timing["route_total_s"] = _time.perf_counter() - tic
    cfg_out = {"dv_budget_mps": cfg.dv_budget_mps, "horizon_h": cfg.horizon_h, "n_dirs": cfg.n_dirs,
               "burn_epochs_h": list(cfg.burn_epochs_h), "magnitudes_mps": rs.meta["magnitudes_mps"],
               "grid_dt_h": cfg.grid_dt_h, "path_dt_h": cfg.path_dt_h, "srp": bool(cfg.srp and cfg.cr_area_mass > 0),
               "cr_area_mass_m2_kg": cfg.cr_area_mass, "rtol": cfg.rtol, "atol_km": cfg.atol,
               "gateway_radius_nd": req.gateway_radius_nd, "nrho_tube_km": req.nrho_tube_km,
               "n_samples_requested": req.n_samples, "n_samples_actual": rs.n_samples,
               "n_dirs_clamped": bool(n_total > MAX_TOTAL_SAMPLES), "max_total_samples": MAX_TOTAL_SAMPLES,
               "max_path_points": MAX_PATH_POINTS, "caps_applied": caps}
    return ReachabilityResponse(
        object=rs.meta["object"],
        t0_utc=tdb_s_to_utc_iso(t0_s)[0],
        t0_tdb_s=float(t0_s),
        config=cfg_out,
        nominal=nominal,
        samples=samples,
        regions=[RegionOut(**round_floats(d, 6)) for d in regions_out],
        envelope=round_floats(rs.envelope, 4),
        sensor_hints=round_floats(hints, 4),
        timing=round_floats(rs.timing, 4),
        meta=round_floats({k: v for k, v in rs.meta.items() if k != "object"}, 6),
        disclaimer=DISCLAIMER + (" NOTE: this object is REAL (JPL Horizons); the envelope is a hypothetical what-if "
                                 "for custody planning only; no maneuver is simulated or implied." if e.is_real else ""),
    )
