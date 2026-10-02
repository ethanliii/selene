"""POST /api/architecture/evaluate and GET /api/architecture/presets — the architecture trade studio.

The route owns its contract (``schemas.ArchitectureRequest`` is the older stub and is not used).
Two request dialects are accepted and normalised to one:

* the canonical one documented in ``ArchitectureEvaluateRequest`` (``ground``, ``sensors[].platform``
  in ``PLATFORMS``, ``slew_rate_deg_s``), and
* the studio page's dialect (``frontend/src/studio/types.ts``: ``ground_network``,
  ``sensors[].orbit`` in {GEO, L1_halo, L2_halo, DRO, resonant_3_1}, ``slew_rate_dps``,
  ``target_radius_m`` / ``target_albedo``).  Unknown keys are still rejected (``extra='forbid'``
  after alias translation).

The response carries the canonical score fields **and** the studio page's names
(``revisit_h``, ``detect_latency_h_mean``, ``detect_latency_h_p95``, ``n_mc``, ``n_sensors``,
``undetected_pct``, ``never_observed_pct``) plus a top-level ``method`` string, so the page renders
live results without a browser-side mock.

Runtime is bounded by a cap on ``horizon_days`` (the cached truth window), on the number of
architectures and sensors, and by a *work-units budget* (architectures × draws × horizon slots):
``n_mc`` is **clamped** to the budget (and to ``MAX_N_MC``) rather than rejected, because the studio
page asks for 100 draws by default; every clamp is reported in ``meta.caps_applied`` and in the
``method`` string.  Draws run on a persistent 4-process spawn pool by default (``workers`` /
``executor``; results are bit-identical to the sequential path, which is the fallback).  Results
are deterministic for a seed.
"""
from __future__ import annotations

import time
from typing import Any, Literal, Optional, Union

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from selene.api.routes.catalog import round_floats, tdb_s_to_utc_iso
from selene.architecture.candidates import (
    PLATFORMS,
    PLATFORM_INFO,
    PRESET_ARCHITECTURES,
    Architecture,
    SensorSpec,
    limiting_mag_from_aperture,
)
from selene.architecture.montecarlo import METHOD_NOTES, EvalConfig, ManeuverModel, evaluate
from selene.objects.catalog import TRUTH_WINDOW_S
from selene.sensors.sites import DEFAULT_SITES, SPEC_DISCLAIMER
from selene.tasking.information import DEFAULT_Q_PSD

router = APIRouter(prefix="/architecture", tags=["architecture"])

Platform = Literal["geo", "l1_halo", "l2_halo_S", "dro", "nrho_9_2", "resonant_3_1"]

#: studio-page orbit ids (frontend/src/studio/types.ts CandidateOrbit) -> canonical platform ids
ORBIT_ALIASES: dict[str, str] = {
    "GEO": "geo", "L1_halo": "l1_halo", "L2_halo": "l2_halo_S", "L2_halo_S": "l2_halo_S", "DRO": "dro",
    "NRHO": "nrho_9_2", "nrho": "nrho_9_2", "resonant_3_1": "resonant_3_1", "resonant": "resonant_3_1",
}

MAX_N_MC = 16
MAX_HORIZON_DAYS = 7.0
MAX_ARCHITECTURES = 8
MAX_SPACE_SENSORS = 8
#: budget on architectures × draws × slots.  Measured sequential cost ≈ 2.2 ms per unit on a laptop
#: (4 presets × 8 draws × 144 slots = 4608 units ≈ 10 s; 4 × 9 × 504 = 18144 units ≈ 34-40 s), so
#: 18 000 keeps every request under ~45 s even on the sequential fallback and lets the 4 presets
#: run 9 draws at the 7-day cap; with the default 4-process pool the same requests take 2.3 s
#: (4608 units, warm) and 11 s (18 144 units, warm); a cold pool adds ≈ 2-3 s once per server.
MAX_WORK_UNITS = 18_000
TRUTH_WINDOW_DAYS = TRUTH_WINDOW_S[1] / 86400.0

DISCLAIMER = ("All objects are SIMULATED notional spacecraft and every injected maneuver is attributed to a 'notional actor'; "
              "sensor specifications are ASSUMED representative values, not those of real instruments; cost figures are "
              "proxies (sensor counts, summed aperture), not monetary estimates. Metrics are reported as computed, "
              "including poor custody and undetected maneuvers.")


def _translate_sensor(v: Any) -> Any:
    """Studio dialect -> canonical keys (``orbit`` -> ``platform``, ``slew_rate_dps`` -> ``slew_rate_deg_s``)."""
    if not isinstance(v, dict):
        return v
    d = dict(v)
    if "orbit" in d:
        if "platform" in d:
            raise ValueError("give either 'platform' or its alias 'orbit', not both")
        orbit = d.pop("orbit")
        if orbit in PLATFORMS:
            d["platform"] = orbit
        elif isinstance(orbit, str) and orbit in ORBIT_ALIASES:
            d["platform"] = ORBIT_ALIASES[orbit]
        else:
            raise ValueError(f"unknown orbit {orbit!r}; known: {sorted(ORBIT_ALIASES)} (or a platform id {sorted(PLATFORMS)})")
    if "slew_rate_dps" in d:
        if "slew_rate_deg_s" in d:
            raise ValueError("give either 'slew_rate_deg_s' or its alias 'slew_rate_dps', not both")
        d["slew_rate_deg_s"] = d.pop("slew_rate_dps")
    d.pop("id", None)   # the page's client-side sensor id, meaningless to the backend
    return d


class SensorIn(BaseModel):
    model_config = ConfigDict(extra="forbid")

    platform: Platform
    aperture_m: Optional[float] = Field(None, gt=0, le=5.0)
    limiting_mag: Optional[float] = Field(None, ge=10.0, le=25.0)
    fov_deg: float = Field(2.0, gt=0, le=60.0)
    slew_rate_deg_s: float = Field(1.0, gt=0, le=90.0)
    lon_deg: float = Field(-100.0, ge=-180.0, le=360.0, description="GEO only: east-positive sub-satellite longitude")
    phase: float = Field(0.0, ge=0.0, lt=1.0, description="fraction of the platform orbit period elapsed at the demo epoch")
    label: str = ""

    @model_validator(mode="before")
    @classmethod
    def _aliases(cls, v: Any) -> Any:
        return _translate_sensor(v)


class ArchitectureIn(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = Field(..., min_length=1, max_length=80)
    ground: bool = Field(True, description="include the notional 9-site ground optical network (alias: ground_network)")
    sensors: list[SensorIn] = Field(default_factory=list, max_length=MAX_SPACE_SENSORS)
    notes: str = ""

    @model_validator(mode="before")
    @classmethod
    def _aliases(cls, v: Any) -> Any:
        if isinstance(v, dict) and "ground_network" in v:
            if "ground" in v:
                raise ValueError("give either 'ground' or its alias 'ground_network', not both")
            v = dict(v)
            v["ground"] = v.pop("ground_network")
        if isinstance(v, dict):
            v = dict(v)
            v.pop("id", None)
            v.pop("slot", None)   # page-side colour slot
        return v

    @field_validator("name")
    @classmethod
    def _strip(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("name must not be blank")
        return v


class ManeuverModelIn(BaseModel):
    model_config = ConfigDict(extra="forbid")

    rate_per_object_per_day: float = Field(0.5, ge=0.0, le=5.0)
    dv_mps_range: list[float] = Field([1.0, 20.0], min_length=2, max_length=2)
    burn_window_fraction: float = Field(0.75, gt=0.0, le=1.0)
    alpha: float = Field(0.01, gt=0.0, lt=1.0, description="per-update NIS false-alarm probability")
    miss_p_acq_min: float = Field(0.95, ge=0.0, le=1.0)

    @field_validator("dv_mps_range")
    @classmethod
    def _range(cls, v: list[float]) -> list[float]:
        if not (0.0 <= v[0] <= v[1] <= 1000.0):
            raise ValueError("dv_mps_range must satisfy 0 <= lo <= hi <= 1000")
        return v


class ArchitectureEvaluateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    architectures: list[ArchitectureIn] = Field(default_factory=list, max_length=MAX_ARCHITECTURES,
                                                description="empty -> the four presets")
    n_mc: int = Field(8, ge=1, description=f"draws; clamped to {MAX_N_MC} and to the work-units budget (reported in meta.caps_applied)")
    horizon_days: float = Field(2.0, gt=0.0, le=MAX_HORIZON_DAYS)
    seed: int = Field(0, ge=0)
    custody_threshold_km: float = Field(100.0, gt=0.0, le=10_000.0)
    object_set: Union[str, list[str]] = Field("all_simulated", description="'all_simulated' or SIMULATED object ids")
    maneuver_model: ManeuverModelIn = Field(default_factory=ManeuverModelIn)
    slot_min: float = Field(20.0, ge=5.0, le=240.0)
    initial_sigma_km: float = Field(25.0, gt=0.0, le=1e5)
    initial_sigma_vel_kms: float = Field(5e-4, gt=0.0, le=1.0)
    q_psd: float = Field(DEFAULT_Q_PSD, ge=0.0)
    sigma_arcsec: float = Field(1.0, gt=0.0, le=60.0)
    phasing_span_days: Optional[float] = Field(None, ge=0.0, le=TRUTH_WINDOW_DAYS,
                                               description=f"t0 offset span; default = {TRUTH_WINDOW_DAYS:.0f} d - horizon; "
                                                           f"span + horizon must stay <= {TRUTH_WINDOW_DAYS:.0f} d")
    target_radius_m: Optional[float] = Field(None, gt=0.0, le=50.0,
                                             description="reference target population: overrides every object's catalog radius")
    target_albedo: Optional[float] = Field(None, gt=0.0, le=1.0, description="reference target population: overrides catalog albedo")
    workers: int = Field(4, ge=1, le=4, description="parallel draws; 'process' uses a persistent spawn pool, bit-identical results")
    executor: Literal["process", "thread"] = "process"
    include_draws: bool = False

    @model_validator(mode="after")
    def _window(self):
        if self.phasing_span_days is not None and self.phasing_span_days + self.horizon_days > TRUTH_WINDOW_DAYS + 1e-9:
            raise ValueError(f"phasing_span_days + horizon_days = {self.phasing_span_days + self.horizon_days:.2f} d exceeds the "
                             f"{TRUTH_WINDOW_DAYS:.0f}-day cached truth window (the truth integrator would run on a clamped "
                             f"Moon/Sun spline); reduce one of them or omit phasing_span_days")
        return self


class ArchitectureScoreOut(BaseModel):
    model_config = ConfigDict(extra="allow")

    name: str
    coverage_pct: float
    custody_pct: float
    custody_pct_null: float
    revisit_mean_h: float
    revisit_p95_h: Optional[float] = None
    revisit_p95_censored_h: Optional[float] = None
    revisit_censored_objects_pct: float
    detection_latency_mean_h: Optional[float] = None
    detection_latency_p95_h: Optional[float] = None
    detection_latency_censored_mean_h: Optional[float] = None
    detections_pct: Optional[float] = None
    n_burns: int
    n_detected: int
    n_space_sensors: int
    total_space_aperture_m: float
    notes: str
    # --- studio-page dialect (frontend/src/studio/types.ts ArchitectureScore); always numeric ---
    revisit_h: float
    detect_latency_h_mean: float
    detect_latency_h_p95: float
    n_mc: int
    n_sensors: int
    undetected_pct: float
    never_observed_pct: float


class ArchitectureEvaluateResponse(BaseModel):
    scores: list[ArchitectureScoreOut]
    per_object: dict[str, dict[str, Any]]
    meta: dict[str, Any]
    disclaimer: str
    method: str
    mock: bool = False


def _preset_payload(a: Architecture) -> dict:
    return a.as_dict()


def max_n_mc_for(n_architectures: int, horizon_days: float, slot_min: float) -> int:
    """Largest ``n_mc`` the work-units budget and ``MAX_N_MC`` allow for this request shape (>= 1)."""
    n_slots = max(1, int(float(horizon_days) * 86400.0 / (float(slot_min) * 60.0)))
    return max(1, min(MAX_N_MC, MAX_WORK_UNITS // max(1, n_architectures * n_slots)))


@router.get("/presets")
def presets():
    n_presets = len(PRESET_ARCHITECTURES)
    return {
        "presets": [_preset_payload(a) for a in PRESET_ARCHITECTURES],
        "platforms": [{"id": pid, "platform_orbit": PLATFORMS[pid], **PLATFORM_INFO[pid],
                       "aliases": sorted(k for k, v in ORBIT_ALIASES.items() if v == pid)} for pid in PLATFORMS],
        "ground_network": [{"id": s.id, "name": s.name, "aperture_m": s.aperture_m, "limiting_mag": s.limiting_mag,
                            "lat_deg": s.lat_deg, "lon_deg": s.lon_deg} for s in DEFAULT_SITES],
        "limiting_mag_rule": {"formula": "m_lim = 18.5 + 5 log10(D / 0.5 m), clipped to [14, 23]",
                              "examples": {f"{d} m": round(limiting_mag_from_aperture(d), 2) for d in (0.2, 0.3, 0.5, 1.0, 1.5)}},
        "defaults": {"n_mc": 8, "horizon_days": 2.0, "custody_threshold_km": 100.0, "slot_min": 20.0,
                     "initial_sigma_km": 25.0, "initial_sigma_vel_kms": 5e-4, "maneuver_model": ManeuverModel().as_dict()},
        "limits": {
            "n_mc": MAX_N_MC, "horizon_days": MAX_HORIZON_DAYS, "architectures": MAX_ARCHITECTURES,
            "space_sensors_per_architecture": MAX_SPACE_SENSORS, "work_units": MAX_WORK_UNITS,
            "truth_window_days": TRUTH_WINDOW_DAYS,
            "rule": "n_mc is clamped (not rejected) to min(n_mc limit, work_units // (architectures x slots)), "
                    "slots = horizon_days x 1440 / slot_min; clamps are reported in meta.caps_applied",
            "max_n_mc_examples": {
                f"{n_presets} presets, 2 d, 20-min slots": max_n_mc_for(n_presets, 2.0, 20.0),
                f"{n_presets} presets, 7 d, 20-min slots": max_n_mc_for(n_presets, 7.0, 20.0),
                "1 architecture, 7 d, 20-min slots": max_n_mc_for(1, 7.0, 20.0),
                f"{MAX_ARCHITECTURES} architectures, 7 d, 20-min slots": max_n_mc_for(MAX_ARCHITECTURES, 7.0, 20.0),
            },
        },
        "request_aliases": {"ground_network": "ground", "sensors[].orbit": "sensors[].platform (GEO, L1_halo, L2_halo, DRO, NRHO, resonant_3_1)",
                            "sensors[].slew_rate_dps": "sensors[].slew_rate_deg_s",
                            "target_radius_m / target_albedo": "reference target population overriding catalog physical parameters"},
        "metric_definitions": {
            "coverage_pct": "fraction of (object, slot node) pairs visible to at least one sensor (geometry + photometry)",
            "custody_pct": "greedy-tasker time fraction with RSS position sigma below custody_threshold_km, from a common stale prior "
                           "on quiet trajectories; custody_pct_null = the same with zero observations",
            "revisit_mean_h": "mean over objects of the mean interval between successive scheduled observations; objects observed "
                              "fewer than twice are censored at the horizon length (alias revisit_h)",
            "revisit_p95_h": "95th percentile of the observed intervals pooled over objects and draws",
            "revisit_p95_censored_h": "95th percentile over (object, draw) cells of the per-object mean interval, censored cells at "
                                      "the horizon (saturates at the horizon when >= 5 % of cells are never revisited; read with "
                                      "revisit_censored_objects_pct)",
            "detection_latency_mean_h": "time from an injected burn to the first scheduled post-burn observation that detects it "
                                        "(surrogate NIS test, see meta.method_notes); over detected burns only. With the default "
                                        "1-20 m/s burns and 1 arcsec noise the test is saturated: this is time-to-first-re-observation, "
                                        "not detector sensitivity",
            "detection_latency_censored_mean_h": "as above, with undetected burns counted at the remaining horizon (lower bound); "
                                                 "alias detect_latency_h_mean (always numeric)",
            "detections_pct": "fraction of injected burns detected before the end of the horizon (undetected_pct = 100 - this)",
            "false_alarm_rate_per_obs": "NIS exceedance rate on quiet observations; equals alpha by construction (bookkeeping, not "
                                        "a calibration against the UKF)",
            "replay_sigma_max_rel_err": "self-check: largest relative mismatch between the surrogate's replayed covariance and the "
                                        "tasker's sigma series (should be ~0)",
            "never_observed_pct": "fraction of (object, draw) cells with zero scheduled observations",
        },
        "method_notes": METHOD_NOTES,
        "spec_disclaimer": SPEC_DISCLAIMER,
        "disclaimer": DISCLAIMER,
    }


def _to_architecture(a: ArchitectureIn) -> Architecture:
    specs = [SensorSpec(platform=s.platform, aperture_m=s.aperture_m, limiting_mag=s.limiting_mag, fov_deg=s.fov_deg,
                        slew_rate_deg_s=s.slew_rate_deg_s, lon_deg=s.lon_deg, phase=s.phase, label=s.label) for s in a.sensors]
    return Architecture(name=a.name, sensors=specs, ground=a.ground, notes=a.notes)


def _studio_fields(s: dict, n_mc: int, horizon_h: float, per_object: dict) -> dict:
    """The studio page's score names, always numeric (None -> documented fallbacks)."""
    lat_c = s.get("detection_latency_censored_mean_h")
    lat_c95 = s.get("detection_latency_censored_p95_h")
    det = s.get("detections_pct")
    never = [1.0 if po.get("n_obs_mean", 0.0) == 0.0 else 0.0 for po in per_object.values()]
    return {
        "revisit_h": float(s["revisit_mean_h"]),
        # censored latency: defined for every architecture (undetected burns at the remaining horizon); no burns -> horizon
        "detect_latency_h_mean": float(lat_c) if lat_c is not None else float(horizon_h),
        "detect_latency_h_p95": float(lat_c95) if lat_c95 is not None else float(horizon_h),
        "n_mc": int(n_mc),
        "n_sensors": int(s["n_space_sensors"]) + int(s.get("n_ground_sites", 0)),
        "undetected_pct": float(100.0 - det) if det is not None else 0.0,
        "never_observed_pct": 100.0 * float(sum(never) / len(never)) if never else 0.0,
    }


@router.post("/evaluate", response_model=ArchitectureEvaluateResponse)
def evaluate_architectures(req: ArchitectureEvaluateRequest):
    t_start = time.perf_counter()
    archs = [_to_architecture(a) for a in req.architectures] if req.architectures else \
        [Architecture(a.name, list(a.sensors), a.ground, a.notes) for a in PRESET_ARCHITECTURES]
    names = [a.name for a in archs]
    if len(set(names)) != len(names):
        raise HTTPException(400, detail=f"architecture names must be unique: {names}")
    for a in archs:
        if not a.ground and not a.sensors:
            raise HTTPException(400, detail=f"architecture {a.name!r} has neither the ground network nor any space sensor")
    caps: list[str] = []
    n_mc_max = max_n_mc_for(len(archs), req.horizon_days, req.slot_min)
    n_mc = req.n_mc
    if n_mc > n_mc_max:
        n_slots = int(req.horizon_days * 86400.0 / (req.slot_min * 60.0))
        caps.append(f"n_mc {req.n_mc} -> {n_mc_max} (limit {MAX_N_MC}; work-units budget {MAX_WORK_UNITS} / "
                    f"({len(archs)} architectures x {n_slots} slots))")
        n_mc = n_mc_max
    try:
        cfg = EvalConfig(horizon_days=req.horizon_days, slot_min=req.slot_min, custody_threshold_km=req.custody_threshold_km,
                         initial_sigma_km=req.initial_sigma_km, initial_sigma_vel_kms=req.initial_sigma_vel_kms,
                         q_psd=req.q_psd, sigma_arcsec=req.sigma_arcsec, phasing_span_days=req.phasing_span_days,
                         target_radius_m=req.target_radius_m, target_albedo=req.target_albedo)
        model = ManeuverModel(rate_per_object_per_day=req.maneuver_model.rate_per_object_per_day,
                              dv_mps_range=tuple(req.maneuver_model.dv_mps_range),
                              burn_window_fraction=req.maneuver_model.burn_window_fraction,
                              alpha=req.maneuver_model.alpha, miss_p_acq_min=req.maneuver_model.miss_p_acq_min)
        res = evaluate(archs, n_mc=n_mc, seed=req.seed, object_set=req.object_set, maneuver_model=model, config=cfg,
                       workers=req.workers, executor=req.executor)
    except (KeyError, ValueError, TypeError) as e:
        raise HTTPException(400, detail=str(e))
    except RuntimeError as e:
        raise HTTPException(500, detail=f"evaluation failed: {e}")
    payload = res.as_dict(include_draws=req.include_draws)
    meta = payload["meta"]
    meta["timing"]["route_total_s"] = time.perf_counter() - t_start
    for d in meta["draws"]:
        d["t0_utc"] = tdb_s_to_utc_iso(d["t0_tdb_s"])[0]
    meta["custody_threshold_km"] = req.custody_threshold_km
    meta["n_mc_requested"] = req.n_mc
    meta["caps_applied"] = caps
    meta["disclaimer"] = DISCLAIMER
    horizon_h = req.horizon_days * 24.0
    method = (f"SELENE Monte Carlo architecture evaluation: {n_mc} draws x {len(archs)} architectures x {len(res.object_ids)} "
              f"SIMULATED objects, {req.horizon_days:g} d horizon, {req.slot_min:g}-min slots; greedy information-theoretic "
              f"tasking, linear covariance custody, surrogate NIS detection (see meta.method_notes)."
              + (f" Caps applied: {'; '.join(caps)}." if caps else ""))
    scores = []
    for s in payload["scores"]:
        s = round_floats(s, 4)
        s.update(_studio_fields(s, n_mc, horizon_h, payload["per_object"].get(s["name"], {})))
        scores.append(ArchitectureScoreOut(**s))
    return ArchitectureEvaluateResponse(
        scores=scores,
        per_object=round_floats(payload["per_object"], 4),
        meta={**round_floats({k: v for k, v in meta.items() if k != "config"}, 4), "config": meta["config"]},  # q_psd ~ 1e-18
        disclaimer=DISCLAIMER,
        method=method,
        mock=False,
    )
