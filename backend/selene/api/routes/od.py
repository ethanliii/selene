"""POST /api/od/run — synthetic end-to-end orbit determination on a catalog object.

Pipeline (deterministic for a given ``seed``):

1. simulate angles-only observations of the object's truth arc from the requested sensors on a
   ``cadence_min`` grid between ``t0`` and ``t1`` (full visibility model; dropped epochs and
   their reasons are returned — blind spots are part of the answer);
2. ``iod``   two-range shooting IOD on the first ``iod_max_obs`` observations (acquisition from scratch);
3. ``batch`` weighted least squares over all observations, started from the IOD solution
   (or from a simulated prior if the IOD is unavailable);
4. ``ukf``   unscented Kalman filter from a prior (``ukf_prior``: 'simulated' = truth + N(0, P₀)
   draw, i.e. an object already in custody; 'iod' = the IOD candidate with its rough covariance);
5. ``particles`` sample the final posterior and propagate the cloud without further observations
   (custody decay), exported in GCRF km and rotating-frame nd positions.

``truth_error_km`` reports the honest filter error against the simulated truth per update.
Defaults are tuned for the UI (24 h, 2-h cadence, 1000 particles over 72 h: ≈ 1–5 s).

Numbers in the payload are rounded to 9 *significant* digits (not fixed decimals: velocity
covariance entries are O(1e-10) km²/s² and must survive the round trip intact).
"""
from __future__ import annotations

import time
from typing import Literal, Optional, Union

import numpy as np
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from selene.api.routes.catalog import err_detail, parse_utc, tdb_s_to_utc_iso
from selene.dynamics.ephemeris import EphemParams
from selene.dynamics.frames import DEMO_EPOCH_UTC
from selene.objects.catalog import get_catalog
from selene.od.batch import batch_least_squares
from selene.od.iod import iod_two_range
from selene.od.measurements import resolve_sensors, simulate_observations
from selene.od.particles import propagate_cloud
from selene.od.realism import nees
from selene.od.types import jsonable
from selene.od.ukf import UKFConfig, ekf_run, ukf_run

router = APIRouter(prefix="/od", tags=["od"])

#: Per-request budgets (every field bound is individually sane, their product was not: a 14-day /
#: 5-min / 5000-particle / 0.5-h-step / 2000-exported request took 36 s and returned 87 MB).
#: visibility evaluations = epochs x sensors (each ~0.3 ms; 20 000 ~ 6 s worst case)
MAX_EPOCH_SENSOR_EVALS = 20_000
#: exported particle positions = frames x max_export (two frames of 3 floats each ~ 60 B -> ~7 MB)
MAX_PARTICLE_POINTS = 120_000
#: particle output frames (t_grid_h / step_h + 1); step_h is clamped up, never rejected
MAX_PARTICLE_FRAMES = 400
#: sequential filter updates (~15 ms each with the iterated update): 600 ~ 9 s worst case
MAX_OBS = 600


def round_sig(x, digits: int = 9):
    """Recursively round floats to ``digits`` significant figures (NaN/inf -> None).

    Fixed-decimal rounding (``round(x, 6)``) would zero every velocity-covariance entry
    (~1e-10 km²/s²) and quantise angles to 0.2″; significant-figure rounding keeps covariances
    non-singular while still trimming the payload.
    """
    if isinstance(x, dict):
        return {k: round_sig(v, digits) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [round_sig(v, digits) for v in x]
    if isinstance(x, np.ndarray):
        return round_sig(x.tolist(), digits)
    if isinstance(x, (float, np.floating)):
        xf = float(x)
        if not np.isfinite(xf):
            return None
        return 0.0 if xf == 0.0 else float(f"{xf:.{digits}g}")
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.bool_,)):
        return bool(x)
    return x


class ParticleOpts(BaseModel):
    n: int = Field(1000, ge=10, le=5000, description="number of particles")
    t_grid_h: float = Field(72.0, gt=0, le=14 * 24, description="horizon after the last update [h]")
    step_h: float = Field(6.0, gt=0, description="output step [h]")
    max_export: int = Field(400, ge=10, le=2000, description="particles exported per frame")


class OdRequest(BaseModel):
    object_id: str = "SIM-DRO-01"
    t0: str = Field(DEMO_EPOCH_UTC, description="UTC ISO start of the observation window")
    t1: Optional[str] = Field(None, description="UTC ISO end (default t0 + 24 h)")
    sensors: Union[list[str], str] = Field("all", description="sensor ids, or 'all' | 'ground' | 'space'")
    cadence_min: float = Field(120.0, ge=5.0, le=24 * 60.0)
    sigma_arcsec: float = Field(1.0, gt=0.0, le=60.0)
    method: Literal["ukf", "batch", "iod", "all"] = "all"
    filter: Literal["ukf", "ekf"] = "ukf"
    q_psd: float = Field(1e-18, ge=0.0, description="process-noise acceleration PSD [km²/s³] (matched model: ~0; "
                                                   "raise to 1e-16..1e-14 for unmodelled accelerations)")
    alpha: float = Field(1e-3, gt=0.0, le=1.0)
    iterated_update: int = Field(5, ge=0, le=20, description="Gauss-Newton re-linearisations per measurement update "
                                                            "(0 = single-pass EKF/UKF; needed for IOD-sized priors)")
    prior_sigma_pos_km: float = Field(100.0, gt=0.0)
    prior_sigma_vel_m_s: float = Field(1.0, gt=0.0)
    ukf_prior: Literal["simulated", "iod"] = "simulated"
    iod_max_obs: int = Field(5, ge=3, le=12)
    max_obs: int = Field(200, ge=1, le=MAX_OBS, description="observations kept (evenly thinned) for the filters")
    respect_visibility: bool = True
    particles: Optional[ParticleOpts] = Field(default_factory=ParticleOpts)
    seed: int = 0


PRESETS = {
    "ui_quick": {"t1": None, "cadence_min": 120, "sensors": "all", "method": "all",
                 "particles": {"n": 1000, "t_grid_h": 72, "step_h": 6}, "note": "~2-5 s"},
    "custody_week": {"cadence_min": 180, "sensors": "space", "method": "ukf",
                     "particles": {"n": 2000, "t_grid_h": 168, "step_h": 6}, "note": "7-day custody decay after a 24 h arc"},
    "ground_only": {"cadence_min": 60, "sensors": "ground", "method": "all",
                    "note": "shows the lunar-glare blind spot: often zero usable observations"},
}


@router.get("/presets")
def presets():
    """Presets, request defaults and the supported observation window per SIMULATED object (``t0``/``t1`` must lie
    inside ``supported_window_utc[object_id]``: the catalog epoch ± 120 days of deterministic truth extension;
    requests outside it are 400, see :func:`od_run`)."""
    cat = get_catalog()
    sims = [e.id for e in cat.objects("simulated")]
    return {"presets": PRESETS, "defaults": OdRequest().model_dump(),
            "supported_window_utc": {oid: cat.truth_window_utc(oid) for oid in sims},
            "supported_window_note": "synthetic truth is served for the catalog epoch +/- 120 days (cached 15-day window plus "
                                     "deterministic extension); t0 and t1 of a run must lie inside it, and the window itself is "
                                     "at most 14 days long"}


def _prior(truth_state, req: OdRequest, rng):
    P0 = np.diag([req.prior_sigma_pos_km ** 2] * 3 + [(req.prior_sigma_vel_m_s * 1e-3) ** 2] * 3)
    x0 = truth_state + np.linalg.cholesky(P0) @ rng.standard_normal(6)
    return x0, P0


@router.post("/run")
def od_run(req: OdRequest):
    timing: dict[str, float] = {}
    t_all = time.perf_counter()
    cat = get_catalog()
    if req.object_id not in cat:
        raise HTTPException(404, detail=f"unknown object {req.object_id!r}; known ids: {cat.ids()}")
    entry = cat.get(req.object_id)
    if entry.notional is None:
        raise HTTPException(400, detail="OD runs are only available for SIMULATED objects (they have a synthetic truth); "
                                        "real Horizons objects never get synthetic observations or events")
    t0_s = parse_utc(req.t0, "t0")
    t1_s = parse_utc(req.t1, "t1") if req.t1 else t0_s + 86400.0
    if t1_s <= t0_s:
        raise HTTPException(400, detail="t1 must be after t0")
    if t1_s - t0_s > 14 * 86400.0 + 60.0:   # 60 s tolerance: the cap is meant in UTC, t_s is TDB
        raise HTTPException(400, detail="window longer than 14 days is not supported by this endpoint")
    w_lo, w_hi = cat.truth_window_s(req.object_id)
    if t0_s < w_lo or t1_s > w_hi:
        lo_utc, hi_utc = cat.truth_window_utc(req.object_id)
        raise HTTPException(400, detail=f"window outside the supported truth window of {req.object_id}: {lo_utc} .. {hi_utc} "
                                        f"(catalog epoch {cat.epoch_utc[:19]}Z +/- 120 days; see /api/od/presets supported_window_utc)")
    try:
        sensors = resolve_sensors(req.sensors)
    except KeyError as e:
        raise HTTPException(400, detail=err_detail(e))
    truth = cat.truth(req.object_id)
    params = EphemParams(srp=True, cr_area_mass=truth.params.cr_area_mass)
    rng = np.random.default_rng(int(req.seed))
    caps: list[str] = []

    # --- per-request budget (see the constants at the top) ---------------------------
    grid = np.arange(t0_s, t1_s + 1e-6, req.cadence_min * 60.0)
    n_evals = grid.size * len(sensors)
    if n_evals > MAX_EPOCH_SENSOR_EVALS:
        raise HTTPException(400, detail=f"{grid.size} epochs x {len(sensors)} sensors = {n_evals} visibility evaluations exceeds the "
                                        f"per-request budget of {MAX_EPOCH_SENSOR_EVALS}; increase cadence_min, shorten the window or "
                                        f"pick fewer sensors")
    if req.particles is not None:
        po = req.particles
        n_frames = int(np.floor(po.t_grid_h / po.step_h + 1e-9)) + 1
        if n_frames > MAX_PARTICLE_FRAMES:
            step = po.t_grid_h / (MAX_PARTICLE_FRAMES - 1)
            caps.append(f"particles.step_h {po.step_h:g} -> {step:.4g} h ({n_frames} frames > {MAX_PARTICLE_FRAMES})")
            po.step_h = step
            n_frames = MAX_PARTICLE_FRAMES
        if n_frames * po.max_export > MAX_PARTICLE_POINTS:
            new_export = max(10, MAX_PARTICLE_POINTS // n_frames)
            caps.append(f"particles.max_export {po.max_export} -> {new_export} ({n_frames} frames x {po.max_export} > "
                        f"{MAX_PARTICLE_POINTS} exported positions)")
            po.max_export = new_export

    # 1. observations ------------------------------------------------------------
    t = time.perf_counter()
    try:
        obs = simulate_observations(req.object_id, sensors, grid, req.sigma_arcsec, rng, req.respect_visibility)
    except ValueError as e:   # truth extension limit or a sensor-model domain error: a client error, not a 500
        raise HTTPException(400, detail=err_detail(e))
    if len(obs) > req.max_obs:
        keep = np.linspace(0, len(obs) - 1, req.max_obs).astype(int)
        obs_list = [obs[i] for i in keep]
    else:
        obs_list = list(obs)
    timing["simulate_s"] = time.perf_counter() - t
    t0_iso, t1_iso = tdb_s_to_utc_iso(t0_s)[0], tdb_s_to_utc_iso(t1_s)[0]
    out: dict = {
        "object_id": req.object_id, "label": entry.label, "t0": t0_iso, "t1": t1_iso, "t0_utc": t0_iso, "t1_utc": t1_iso,
        "sensors": [s.id for s in sensors], "method": req.method, "seed": req.seed, "caps_applied": caps,
        "limits": {"epoch_sensor_evals": MAX_EPOCH_SENSOR_EVALS, "particle_points": MAX_PARTICLE_POINTS,
                   "particle_frames": MAX_PARTICLE_FRAMES, "max_obs": MAX_OBS},
        "observations": {"n_used": len(obs_list), **obs.summary(),
                         "dropped": obs.dropped[:500], "n_dropped": len(obs.dropped),
                         "measurements": [m.as_dict() for m in obs_list]},
        "force_model": {"filter": "DE440s Earth+Moon+Sun point masses + cannonball SRP (object C_R·A/m assumed known)",
                        "truth": "identical force model; measurement noise isotropic on-sky Gaussian"},
        "iod": None, "batch": None, "ukf": None, "particles": None, "truth_error_km": None,
        "notes": [],
    }
    if not obs_list:
        out["notes"].append("No usable observations: every candidate epoch failed the visibility model "
                            "(see observations.dropped_by_reason). Custody cannot be established with these sensors "
                            "in this window — this is the cislunar blind-spot problem, reported honestly.")
        timing["total_s"] = time.perf_counter() - t_all
        out["timing"] = timing
        return round_sig(jsonable(out))

    want = {"iod", "batch", "ukf"} if req.method == "all" else {req.method}
    truth_fn = truth.at

    # 2. IOD -----------------------------------------------------------------------
    iod_best = None
    iod_seeds: list = []          # converged admissible roots, best first (several when ambiguous)
    if "iod" in want or ("batch" in want and len(obs_list) >= 3) or req.ukf_prior == "iod":
        if len(obs_list) >= 3:
            t = time.perf_counter()
            # spread the IOD observations over the first 24 h (or the whole arc if shorter)
            span_end = obs_list[0].t_s + min(24 * 3600.0, obs_list[-1].t_s - obs_list[0].t_s)
            pool = [m for m in obs_list if m.t_s <= span_end + 1e-6]
            if len(pool) < 3:
                pool = obs_list[:3]
            k = min(req.iod_max_obs, len(pool))
            sel = [pool[i] for i in sorted(set(np.linspace(0, len(pool) - 1, k).astype(int)))]
            try:
                iod = iod_two_range(sel, params)
                d = iod.as_dict()
                if iod.best is not None:
                    xt = truth_fn(iod.t_ref_s)
                    d["best"]["truth_error_pos_km"] = float(np.linalg.norm(iod.best.x_ref[:3] - xt[:3]))
                    d["best"]["truth_error_vel_m_s"] = float(np.linalg.norm(iod.best.x_ref[3:] - xt[3:]) * 1e3)
                    d["best"]["truth_range_error_pct"] = float(
                        100 * abs(np.linalg.norm(iod.best.x_ref[:3] - sel[len(sel) // 2].observer_pos_gcrf)
                                  - np.linalg.norm(xt[:3] - sel[len(sel) // 2].observer_pos_gcrf))
                        / np.linalg.norm(xt[:3] - sel[len(sel) // 2].observer_pos_gcrf))
                    d["best"]["epoch"] = tdb_s_to_utc_iso(iod.t_ref_s)[0]
                    d["best"]["state"] = d["best"]["x_ref"]
                    d["best"]["pos_err_km"] = d["best"]["truth_error_pos_km"]
                    iod_seeds = [c for c in iod.candidates if c.converged and c.admissible][:3]
                    iod_best = iod_seeds[0] if iod_seeds else None
                    for c, cd in zip(iod.candidates, d["candidates"]):
                        xt_c = truth_fn(c.t_ref_s)
                        cd["truth_error_pos_km"] = float(np.linalg.norm(c.x_ref[:3] - xt_c[:3]))
                d["obs_used"] = [m.as_dict() for m in sel]
                d["usable_as_seed"] = iod_best is not None
                if iod.meta.get("quality") == "ambiguous":
                    out["notes"].append(f"IOD ambiguous: {iod.meta['note']}; the batch fit is started from each root "
                                        "and the lowest post-fit cost is kept")
                elif iod.meta.get("quality") == "not_converged":
                    out["notes"].append(f"IOD not converged: {iod.meta['note']}; a simulated prior is used instead")
                if "iod" in want:
                    out["iod"] = d
            except Exception as e:  # noqa: BLE001 - report instead of failing the whole run
                out["notes"].append(f"IOD failed: {e}")
            timing["iod_s"] = time.perf_counter() - t
        else:
            out["notes"].append("IOD skipped: fewer than 3 observations")

    # 3. batch -----------------------------------------------------------------------
    batch = None
    if "batch" in want:
        t = time.perf_counter()
        overdetermined = 2 * len(obs_list) >= 12
        starts = []   # (x_init, t_init, P_init, label)
        if iod_seeds:
            for c in iod_seeds:   # every exact root when the IOD is ambiguous; the post-fit cost discriminates
                starts.append((c.x_ref.copy(), c.t_ref_s, None if overdetermined else c.P_ref * 4.0, f"iod root {c.rank}"))
        else:
            x_init, P_init = _prior(truth_fn(obs_list[0].t_s), req, rng)
            starts.append((x_init, obs_list[0].t_s, None if overdetermined else P_init,
                           "simulated prior (truth + N(0,P0) draw)"))
        tried = []
        for x_init, t_init, P_init, src in starts:
            try:
                b = batch_least_squares(obs_list, x_init, t_init, P0=P_init, params=params, object_id=req.object_id)
            except Exception as e:  # noqa: BLE001
                out["notes"].append(f"batch LS from {src} failed: {e}")
                continue
            tried.append((b.meta["weighted_rms"] if b.converged else np.inf, b.meta["weighted_rms"], src, b))
        if tried:
            tried.sort(key=lambda r: (r[0], r[1]))
            _, _, src, batch = tried[0]
            d = batch.as_dict()
            xt = truth_fn(batch.t0_s)
            e = batch.x - xt
            d.update(epoch=tdb_s_to_utc_iso(batch.t0_s)[0], initial_guess_source=src,
                     truth_error_pos_km=float(np.linalg.norm(e[:3])), truth_error_vel_m_s=float(np.linalg.norm(e[3:]) * 1e3),
                     truth_nees=float(e @ np.linalg.solve(batch.P, e)),
                     starts_tried=[{"source": s_, "weighted_rms": float(w), "converged": bool(np.isfinite(c))}
                                   for c, w, s_, _ in tried])
            if len(tried) > 1:
                out["notes"].append(f"batch LS tried {len(tried)} starts; kept '{src}' (lowest post-fit weighted RMS)")
            out["batch"] = d
        timing["batch_s"] = time.perf_counter() - t

    # 4. sequential filter -------------------------------------------------------------
    run = None
    if "ukf" in want:
        t = time.perf_counter()
        if req.ukf_prior == "iod" and (batch is not None and batch.converged and
                                       batch.meta.get("object_id") == req.object_id and len(iod_seeds) > 1):
            # ambiguous IOD: the batch fit over all observations picked the root; start the filter there
            # at the IOD reference epoch is not available, so use the batch epoch and only later observations
            x0, P0, tf0, src = batch.x.copy(), batch.P * 4.0, batch.t0_s, "batch fit from the selected IOD root (covariance ×4)"
            meas_f = [m for m in obs_list if m.t_s > tf0 + 1e-6]
        elif req.ukf_prior == "iod" and iod_best is not None:
            x0, P0, tf0, src = iod_best.x_ref.copy(), iod_best.P_ref * 4.0, iod_best.t_ref_s, "iod (covariance ×4)"
            meas_f = [m for m in obs_list if m.t_s > tf0 + 1e-6]
        else:
            tf0 = obs_list[0].t_s
            x0, P0 = _prior(truth_fn(tf0), req, rng)
            src, meas_f = "simulated prior (truth + N(0,P0) draw)", obs_list
        cfg = UKFConfig(alpha=req.alpha, q_psd=req.q_psd, iterated_update=req.iterated_update)
        runner = ukf_run if req.filter == "ukf" else ekf_run
        try:
            if meas_f:
                run = runner(meas_f, x0, P0, tf0, params=params, config=cfg, object_id=req.object_id)
                e = run.truth_error(truth_fn)
                d = run.to_dict(include_pred=True, include_meas=False)
                d["prior_source"] = src
                d["prior"] = {"epoch": tdb_s_to_utc_iso(tf0)[0], "state": x0.tolist(), "cov": P0.tolist()}
                out["ukf"] = d
                out["truth_error_km"] = np.linalg.norm(e[:, :3], axis=1).tolist()
                out["truth_error_vel_m_s"] = (np.linalg.norm(e[:, 3:], axis=1) * 1e3).tolist()
                out["nees"] = nees(e, run.P).tolist()
                out["nees_bounds_95_single_epoch"] = [1.237, 14.449]  # χ²₆ 2.5 % / 97.5 % quantiles
            else:
                out["notes"].append("filter skipped: no observations after the prior epoch")
        except Exception as e:  # noqa: BLE001
            out["notes"].append(f"{req.filter} failed: {e}")
        timing["filter_s"] = time.perf_counter() - t

    # 5. particles -------------------------------------------------------------------------
    if req.particles is not None:
        t = time.perf_counter()
        if run is not None:
            tp, xp, Pp, src = run.final_state() + ("ukf posterior",)
        elif batch is not None and batch.converged:
            tp, xp, Pp, src = batch.t0_s, batch.x, batch.P, "batch posterior"
        else:
            tp = obs_list[-1].t_s
            xp, Pp = _prior(truth_fn(tp), req, rng)
            src = "simulated prior"
        po = req.particles
        tg = tp + np.arange(0.0, po.t_grid_h * 3600.0 + 1e-6, po.step_h * 3600.0)
        try:
            cloud = propagate_cloud(xp, Pp, tp, tg, po.n, np.random.default_rng(req.seed + 7), params,
                                    object_id=req.object_id)
            d = cloud.to_dict(max_particles=po.max_export)
            d["source"] = src
            # truth position along the same grid (for the UI to show where the object really is)
            d["truth_positions_km"] = truth_fn(tg)[:, :3].tolist()
            m = d["metrics"]
            d["custody"] = {
                "sigma_pos_km_start": m["sigma_pos_km"][0], "sigma_pos_km_end": m["sigma_pos_km"][-1],
                "growth_factor": (m["sigma_pos_km"][-1] / m["sigma_pos_km"][0]) if m["sigma_pos_km"][0] else None,
                "hours_to_1000km": next((h for h, s in zip(m["hours"], m["sigma_pos_km"]) if s > 1000.0), None),
                "note": "σ_pos = sqrt(trace of the sample position covariance); no observations after the start epoch",
            }
            out["particles"] = d
        except Exception as e:  # noqa: BLE001
            out["notes"].append(f"particle propagation failed: {e}")
        timing["particles_s"] = time.perf_counter() - t

    timing["total_s"] = time.perf_counter() - t_all
    out["timing"] = timing
    return round_sig(jsonable(out))
