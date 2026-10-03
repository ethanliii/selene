# SELENE API reference

The backend is a FastAPI app (`backend/selene/api/`). Every route lives under `/api`. With `make dev` it runs on `http://127.0.0.1:8000`, and the interactive OpenAPI docs are at `http://127.0.0.1:8000/docs`. The web app reaches it through the Vite proxy on :5173, or same-origin when served by `make demo`.

Routers are registered lazily from `ROUTE_MODULES` in `backend/selene/api/routes/__init__.py`, so a missing module never breaks the app. `GET /api/health` reports which routers loaded (`routers_loaded`) and lists every path (`routes`).

## Conventions

- **Units and frames.** Positions in km, velocities in km/s, Earth-centred GCRF unless a field name says otherwise: `rot` or `nd` means the nondimensional Earth-Moon rotating frame, `moon` means Moon-centred with GCRF axes. Delta-v is in m/s, horizons in hours. Floats are rounded to 9 significant figures.
- **Time.** Every ISO timestamp the API emits ends in `Z` (`2026-03-01T00:00:00.000Z`), because browsers parse a bare `YYYY-MM-DDTHH:MM:SS` as local time. Requests accept ISO UTC with or without `Z`. Absolute-time arrays are named `epochs_utc` (ISO) and `tdb_s` (TDB seconds past J2000). Routes that historically used `t_s` (sensors, OD, maneuver, tasking) keep it as an alias of `tdb_s`, and `/api/coverage` keeps `per_time[].t` next to `tdb_s` and `utc`. Request windows are echoed back normalised as `t0_utc` and `t1_utc`, including defaults. The demo bundle is the one place with relative times: `t` (same as `t_rel_s`) is seconds since `meta.t0_utc`, documented in `meta.time_fields`.
- **Default epoch.** The catalog and most routes default to 2026-03-01T00:00:00Z, inside the DE440s and Horizons cache span.
- **SIMULATED labels.** Every response that involves notional objects carries a `label` or `disclaimer`. Real (JPL Horizons) objects are tracked but never the subject of a maneuver claim.

## Errors

| Situation | Response |
|---|---|
| Unknown `/api/...` path | JSON 404 (never the SPA shell), for example `{"detail": "unknown API route GET /api/nope; see /docs or GET /api/health -> routes"}`. Unknown methods on unknown paths are 404 too. |
| Trailing slash | `/api/x/` redirects to `/api/x`. |
| Bad epoch, reversed window, epoch outside DE440s | 400, for example `{"detail": "t0 '2200-01-01' is outside the DE440s ephemeris coverage 1849-12-25 .. 2150-01-21"}`. |
| Unknown object or sensor id | 404 or 400 with an unquoted `detail`. |
| Unknown request key or invalid ad-hoc sensor spec | 422 with field-level detail (models use `extra='forbid'`), for example `{"type": "extra_forbidden", "loc": ["body", "preset"], ...}`. |
| Maneuver detection on a real Horizons object | 400 explaining that maneuver detection runs only on SIMULATED objects. |
| Missing JPL kernel or orbit library | `/api/health` reports `status: degraded`; routes that need the kernel answer 503 with a download hint. |
| Request too large | Clamped with the clamp listed under `caps_applied`, or rejected with 400 naming the rule (see budgets below). |

## Request budgets

Every route is synchronous. Shapes that are individually in bounds but jointly unbounded are capped:

- OD: `epochs × sensors ≤ 20,000`, `max_obs ≤ 600`, particle export `frames ≤ 400` and `frames × max_export ≤ 120,000` (clamped).
- Reachability: `samples ≤ 20,000`; with `include_paths`, `samples × path steps ≤ 150,000` (`path_dt_h` is clamped up).
- Architecture: `architectures × slots ≤ 6,000` per draw, otherwise the request is rejected with HTTP 400. `n_mc` is clamped to the work-unit and per-worker budgets, and any clamp is listed in `meta.caps_applied`.
- Maneuver: `epochs × max_obs_per_epoch ≤ 2,000`, span ≤ 30 days (+60 s UTC/TDB tolerance).
- Tasking: `time_budget_s ≤ 30`.
- Orbit families: `members × n_samples ≤ 200,000`.

## Routes

Latencies were measured with FastAPI's `TestClient` on the build laptop (Apple Silicon, warm numba cache, default parameters unless noted). Treat them as orders of magnitude.

| Method | Path | Purpose | Typical latency |
|---|---|---|---|
| GET | `/api/health` | Status (`ok` or `degraded`), version, `offline` flag, which cached inputs are present, loaded routers, every route, CORS origins. | 0.06 s |
| GET | `/api/catalog/objects` | Catalog envelope: epoch, counts, disclaimer, 11 SIMULATED and 7 Horizons objects. `?kind=simulated\|horizons` filters. | 1.4 s first call (builds truth arcs), 0.01 s after |
| GET | `/api/catalog/objects/{id}` | One object: orbit type, actor, role, physical assumptions, state at the catalog epoch. | < 0.01 s |
| GET | `/api/catalog/objects/{id}/trajectory` | Truth or Horizons track over `[t0, t1]` (default 7 days), `n ≤ 5000`, `frame = gcrf_km \| rot_nd \| moon_km`. | < 0.01 s |
| GET | `/api/orbits/families` | Periodic-orbit families thinned to `max_members` (NRHO-tagged members always kept) with `n_samples` rotating-frame points each; `family=` filters by key or prefix. | 0.1 to 0.4 s |
| GET | `/api/orbits/records/{id}` | One library record (initial condition, period, Jacobi constant, stability, geometry, tags) plus samples; `gcrf_epoch=` also maps it to GCRF. | < 0.01 s |
| GET | `/api/ephemeris/bodies` | DE440s Earth, Moon and Sun positions on a UTC grid (default hourly over 14 days) plus, per epoch, the rotating-frame basis `R`, `d_km`, `r_bary_km` and L1 to L5, so the UI's frame toggle is exact. | 0.01 s |
| GET | `/api/sensors` | The notional network: 9 ground sites and 6 space observers with orbit summaries, reason bits and the specification disclaimer. | < 0.01 s |
| GET | `/api/sensors/{id}/visibility` | Per-epoch visibility of a catalog object (`object_id=` required) from one sensor, with reason bits, magnitude and range. | 0.01 to 0.3 s |
| GET | `/api/coverage/presets` | The 8 network presets: `ground_only`, `ground_plus_geo`, `ground_plus_l1_halo`, `ground_plus_l2_halo`, `ground_plus_dro`, `ground_plus_nrho`, `space_only`, `full`. | < 0.01 s |
| POST | `/api/coverage` | Coverage heatmap of the cislunar volume for a network over `[t0, t1]`, with the dominant blind-spot reason per cell. | about 1 to 2 s (2-D, 168 steps); 2.2 s for `full` in 3-D |
| GET | `/api/od/presets` | Sensor sets and defaults for the OD panel (`ui_quick`, `custody_week`, `ground_only`). | < 0.01 s |
| POST | `/api/od/run` | Synthetic end-to-end orbit determination on a catalog object: simulate RA/Dec observations (with the epochs that were blocked and why), IOD, batch least squares, UKF or EKF, and a particle cloud with no further observations. Reports honest `truth_error_km`, `caps_applied` and `limits`. | 1.2 s |
| POST | `/api/maneuver/detect` | Truth (plus an optional SIMULATED injected burn), measurements, filter, health gate, NIS / windowed NIS / gap re-fit / CUSUM / NEES tests, then a delta-v estimate. `status` is one of `no_observations`, `insufficient_updates`, `filter_not_converged`, `filter_inconsistent`, `quiet`, `maneuver_declared`. Horizons objects are refused (400). | 1.1 s |
| GET | `/api/reachability/regions` | Region definitions and display geometry (`gateway_radius_nd` and `nrho_tube_km` are configurable). | 0.01 s |
| GET | `/api/reachability/ladder` | The delta-v magnitude ladder for a budget. | < 0.01 s |
| POST | `/api/reachability` | Delta-v-sampled reachable set over 24 to 168 h: per-region hit fraction, earliest arrival and minimum delta-v (gateways are neck transits, see [TECHNICAL.md](TECHNICAL.md#reachability)), envelope over time, per-sensor pointing hints, `config.caps_applied`. | 1.3 s (about 1,500 samples); 4.4 s at the 20,000-sample cap |
| GET | `/api/tasking/presets` | Sensor presets (`mixed_9` = `default`, plus the coverage presets), default object list, methods. | < 0.01 s |
| POST | `/api/tasking/schedule` | Sensor tasking over a slot grid. `method` is `greedy`, `milp`, `random`, `round_robin` or `compare`; `gain_kind` is `trace`, `logdet` or `maxeig`; `acquisition` is `fov`, `irf` or `none`; `extra_sensors` takes validated ad-hoc space observers. Returns the schedule, custody %, mean time since last observation, covariance-trace series and timings. | 0.35 s greedy; 2.9 s MILP; 4.4 s `compare` |
| GET | `/api/architecture/presets` | Candidate platforms, the four preset architectures, limits and budget rule, metric definitions, method notes. | < 0.01 s |
| POST | `/api/architecture/evaluate` | Monte Carlo scoring of up to 8 architectures (coverage, custody, revisit, detection latency including censored variants) with common random numbers on a 4-process pool. An empty `architectures` list runs the four presets. | 1.7 to 4 s for the presets (`n_mc=4`, 3 days); the first call also starts the worker pool |
| GET | `/api/demo/scenario` | The committed six-day story: `{meta, frames (145 hourly), events (33), metrics, brief}`, served from disk and cached in memory. | 0.2 s |
| GET | `/api/demo/scenario/meta` | The bundle's `meta` and `metrics` without the frames or events (plus `n_frames`, `n_events`, `bundle_bytes`). | < 0.01 s |
| GET | `/api/demo/brief` | The analyst brief as plain text (Markdown). | < 0.01 s |
| POST | `/api/demo/rebuild?fast=true` | Rebuilds the bundle synchronously. `fast=true` uses reduced samples; `fast=false` is the full build. Returns 403 unless `SELENE_ALLOW_REBUILD=1` or `SELENE_DEMO_SCENARIO` is set, so a live demo cannot be overwritten by accident. | about 6 s warm for the full build |

## Examples

Responses below were produced by calling the routes through `TestClient` and then trimmed by hand: long arrays show their first items followed by `…`, and nested objects keep only the fields worth reading. Values are real outputs at the default epoch, 2026-03-01T00:00:00Z.

### Health

```http
GET /api/health
```

```json
{
  "status": "ok",
  "version": "0.1.0",
  "offline": true,
  "ephemeris": "de440s.bsp",
  "data": {
    "present": {"de440s_kernel": true, "gm_de440": true, "orbit_library": true,
                "horizons_cache": true, "demo_bundle": true},
    "missing": [],
    "required_missing": []
  },
  "routers_loaded": ["catalog", "orbits", "…"],
  "routes": ["/api/architecture/evaluate", "/api/architecture/presets", "…"],
  "cors_origins": ["http://127.0.0.1:5173", "http://localhost:5173", "…"]
}
```

### Coverage

```http
POST /api/coverage
{"network": "ground_plus_dro"}
```

Other fields: `t0`, `t1` (default t0 + 7 days), `n_t` (default 168), `grid` (`2d` or `3d`), `radius_m`, `albedo`, `margin_mag`. `network` may also be `{"ground_ids": [...], "space": [...]}`, where `space` holds observer ids or ad-hoc specs (`id` and `platform_orbit` required).

```json
{
  "grid": {"kind": "2d", "shape": [64, 56], "frame": "earth_moon_rotating_nd",
           "length_unit_km": 384400.0, "xs": [-1.6, -1.549, "…"], "ys": [-1.4, -1.349, "…"]},
  "coverage": [0.0, 0.0, "… 3584 cells"],
  "reason_dominant_name": ["too_faint", "too_faint", "…"],
  "reason_fraction": {"sun_exclusion": [0.3095, "…"], "moon_exclusion": [0.0238, "…"], "too_faint": [0.6667, "…"], "…": "…"},
  "per_time": [{"tdb_s": 825595269.185, "utc": "2026-03-01T00:00:00.000Z", "pct": 42.2}, "… 168 steps"],
  "meta": {
    "network": "ground_plus_dro",
    "n_cells": 3584,
    "mean_coverage_pct": 40.71,
    "reference_object": "diffuse sphere (Lambertian), SIMULATED reference object",
    "note": "Coverage = any sensor has zero geometric+photometric violations; FOV/slew not applied.",
    "reason_bits": {"daylight": 1, "low_elevation": 2, "sun_exclusion": 4, "moon_exclusion": 8,
                    "earth_exclusion": 16, "in_shadow": 32, "too_faint": 64, "out_of_fov": 128},
    "elapsed_s": 1.222,
    "t0_utc": "2026-03-01T00:00:00.000Z",
    "t1_utc": "2026-03-08T00:00:00.000Z"
  }
}
```

### Orbit determination

```http
POST /api/od/run
{"object_id": "SIM-DRO-01", "sensors": "all", "method": "all"}
```

Other fields include `t1` (default t0 + 24 h), `cadence_min` (120), `sigma_arcsec` (1.0), `filter` (`ukf` or `ekf`), `q_psd` (1e-18), `iterated_update` (5), `prior_sigma_pos_km`, `prior_sigma_vel_m_s`, `ukf_prior` (`simulated` or `iod`), `max_obs`, `respect_visibility` and `particles` (`n`, `t_grid_h`, `step_h`, `max_export`).

```json
{
  "object_id": "SIM-DRO-01",
  "label": "SIMULATED",
  "t0_utc": "2026-03-01T00:00:00.000Z",
  "t1_utc": "2026-03-02T00:00:00.000Z",
  "caps_applied": [],
  "observations": {
    "n_used": 75,
    "n_candidates": 195,
    "by_sensor": {"dro_obs": 13, "geo_east": 11, "geo_west": 12, "l1_halo_obs": 13, "l2_halo_obs": 13, "nrho_obs": 13},
    "dropped_by_reason": {"moon_exclusion": 117, "daylight": 68, "low_elevation": 76, "earth_exclusion": 3}
  },
  "force_model": {"filter": "DE440s Earth+Moon+Sun point masses + cannonball SRP (object C_R·A/m assumed known)"},
  "iod": {"method": "two-range shooting IOD under the ephemeris model (Gooding structure, Newton/STM BVP)",
          "n_seeds": 36, "n_rejected": 26, "quality": "ok",
          "best": {"rho1_km": 75698.6, "rho3_km": 77606.1, "rms_arcsec": 0.984, "…": "…"}},
  "batch": {"converged": true, "iterations": 2, "rms_arcsec": 1.015, "sigma_pos_km": 0.127,
            "truth_error_pos_km": 0.0695, "truth_nees": 4.92},
  "ukf": {"n": 75, "sigma_pos_km": ["…"], "nis": ["…"], "prior_source": "simulated prior (truth + N(0,P0) draw)"},
  "particles": {"n_particles": 1000, "n_exported": 400, "source": "ukf posterior",
                "metrics": {"hours": [0.0, 6.0, "…"], "sigma_pos_km": [0.239, 0.332, "…"]},
                "frames": [{"epoch": "2026-03-02T00:00:00.000Z", "positions_km": ["…"], "positions_rot": ["…"]}, "…"]},
  "truth_error_km": [68.35, 0.757, "…"],
  "timing": {"simulate_s": 0.307, "iod_s": 0.387, "batch_s": 0.028, "filter_s": 0.014, "particles_s": 0.055, "total_s": 1.213}
}
```

Ground sites contribute nothing in this window: the object is inside the lunar glare cone, which is what `dropped_by_reason` shows.

### Maneuver detection

```http
POST /api/maneuver/detect
{"object_id": "SIM-DRO-01",
 "injected": {"t_burn_utc": "2026-03-03T00:00:00Z", "magnitude_mps": 5.0, "direction": "prograde"}}
```

`injected` takes either `dv_mps` (three GCRF components) or `magnitude_mps` with a named `direction`: `prograde`, `retrograde`, `radial_out`, `radial_in`, `normal`, `anti_normal`, `toward_moon`, `toward_earth`, `toward_l1`, `toward_l2`. Other fields: `t1` (default t0 + 7 days), `sensors`, `cadence_min` (360), `alpha` (0.01), `window` (5), `gap_hours_for_refit` (12), `filter` (`auto`, `ukf`, `ekf`), `estimate`, `seed`.

```json
{
  "object_id": "SIM-DRO-01",
  "label": "SIMULATED",
  "status": "maneuver_declared",
  "declared": true,
  "status_note": "Track established; at least one test exceeded its family-wise threshold after the baseline.",
  "filter": {"name": "ukf", "n_updates": 26, "nis": [0.561, 2.003, "…"], "sigma_pos_km": [47.7, 83.2, "…"]},
  "detections": [{"t_s": "…", "test": "nis", "statistic": "…", "threshold": "…", "p_value": "…"}, "… 66 rows in all"],
  "summary": {
    "alpha": 0.01,
    "threshold_nis": 9.21,
    "threshold_nis_familywise": 16.48,
    "counts": {"nis": 15, "nis_window": 17, "gap_refit": 0, "cusum": 15, "nees": 19},
    "filter_health": {"baseline_established": true, "nees_consistent": true, "…": "…"}
  },
  "dv_estimate": {
    "magnitude_mps": 5.062, "magnitude_sigma_mps": 0.061,
    "direction_sigma_deg": 0.87,
    "dv_rtn_mps": [-2.22, 4.55, -0.03], "frame_center": "moon",
    "t_burn_sigma_s": 507.7, "residual_rms_arcsec": 0.669
  },
  "truth": {
    "magnitude_mps": 5.0, "t_burn_utc": "2026-03-03T00:00:00.000Z",
    "note": "SIMULATED burn injected into a notional object's truth (notional actor)",
    "estimate_error": {"magnitude_error_pct": 1.25, "direction_error_deg": 1.29, "t_burn_error_s": 47.4},
    "detection_latency_s": 21600.0,
    "detected_after_n_post_burn_obs": 1,
    "premature_declaration": false
  },
  "timing": {"total_s": 1.086}
}
```

With the default 6-hour cadence the burn is caught on the first observation after it, so the latency equals the cadence.

### Reachability

```http
POST /api/reachability
{"object_id": "SIM-DRO-01", "dv_budget_mps": 100, "horizon_h": 168}
```

Other fields: `t0` (default: catalog epoch), `n_samples` (2000; converted to Fibonacci directions), `n_dirs`, `burn_epochs_h` (default `[0, 6, 12, 24]`), `magnitudes_mps`, `include_paths`, `path_dt_h`, `sensors`, `include_hints`, `refine`, `gateway_radius_nd` (0.05), `nrho_tube_km` (10,000). `seed` is accepted but has no effect, because sampling is deterministic.

```json
{
  "object": {"id": "SIM-DRO-01", "label": "SIMULATED", "orbit_type": "DRO", "actor": "notional actor", "is_real": false},
  "t0_utc": "2026-03-01T00:00:00.000Z",
  "config": {"dv_budget_mps": 100.0, "horizon_h": 168.0, "n_dirs": 62,
             "burn_epochs_h": [0.0, 6.0, 12.0, 24.0], "magnitudes_mps": [2.0, 5.0, 10.0, 20.0, 50.0, 100.0],
             "gateway_radius_nd": 0.05, "nrho_tube_km": 10000.0},
  "regions": [
    {"key": "l1_gateway",    "fraction": 0.0,    "n_hit": 0,  "earliest_h": null, "min_dv_mps": null, "newly_reachable": false},
    {"key": "l2_gateway",    "fraction": 0.0202, "n_hit": 5,  "earliest_h": 90.0, "min_dv_mps": 50.0, "newly_reachable": true},
    {"key": "nrho_corridor", "fraction": 0.2056, "n_hit": 54, "earliest_h": 89.0, "min_dv_mps": 50.0, "newly_reachable": true},
    {"key": "llo_shell",     "fraction": 0.0403, "n_hit": 10, "earliest_h": 91.0, "min_dv_mps": 100.0, "newly_reachable": true},
    "… 5 more regions"
  ],
  "samples": {"n": 1488, "dv_mps": [2.0, "…"], "end_rot": [[0.9517, -0.1331, 0.0001], "…"], "hit_region": [null, "…"]},
  "envelope": [{"t_h": 0.0, "n_active": 1488, "centroid_rot_nd": "…", "max_radius_km": "…"}, "… 29 steps in all"],
  "sensor_hints": {"best_overall": "geo_west", "summary": {"haleakala": "…", "…": "…"}},
  "timing": {"propagate_s": 0.52, "refine_s": 0.38, "hints_s": 0.19, "route_total_s": 1.26},
  "disclaimer": "Reachability is a custody/awareness envelope under an ASSUMED impulsive delta-v budget: it says where an object COULD be … It asserts no intent, is not a maneuver prediction, and is never used for engagement planning. SIMULATED objects and events are notional."
}
```

`fraction` is the share of sampled rays (direction and burn epoch) that enter the region; `n_hit` counts samples. `newly_reachable` is false when the unperturbed orbit already visits the region.

### Sensor tasking

```http
POST /api/tasking/schedule
{"method": "greedy"}
```

Defaults: the 8 default xGEO objects, the `mixed_9` sensor preset, 48 h from the demo epoch, 20-minute slots, `gain_kind` `trace`, `acquisition` `fov`, custody threshold 100 km, initial σ 10 km and 0.1 m/s.

```json
{
  "method": "greedy",
  "config": {"object_ids": ["SIM-DRO-01", "SIM-DRO-02", "…"],
             "sensor_ids": ["haleakala", "mt_lemmon", "teide", "sutherland", "siding_spring",
                            "geo_west", "l1_halo_obs", "l2_halo_obs", "dro_obs"],
             "slot_min": 20.0, "n_slots": 144, "gain_kind": "trace", "acquisition": "fov"},
  "schedule": [{"slot": 0, "sensor_id": "geo_west", "object_id": "SIM-L1-HALO-01", "gain": 194.9, "p_acq": 1.0,
                "magnitude": 17.48, "range_km": 334168.6, "sigma_before_km": 17.32, "sigma_after_km": 10.25}, "… 562 more"],
  "per_object": [{"id": "SIM-DRO-01", "custody_pct": "…", "n_obs": "…"}, "…"],
  "overall": {"custody_pct": 100.0, "n_objects": 8, "n_sensors": 9, "horizon_h": 48.0,
              "n_observations": 563, "mean_tslo_h": 0.39, "max_tslo_h": 4.0},
  "assumptions": ["Linear covariance analysis: measurement noise is never sampled, …", "…"],
  "timing": {"greedy_s": 0.225, "total_s": 0.348}
}
```

With the default 10 km prior every object stays in custody even without observations; use a staler prior (`initial_sigma_km`, `initial_sigma_vel_kms`) or fewer sensors to see the policies separate. `"method": "compare"` runs greedy, MILP, random and round-robin and fills `comparison`.

### Architecture Monte Carlo

```http
POST /api/architecture/evaluate
{"n_mc": 4, "horizon_days": 3, "seed": 0}
```

An empty `architectures` list runs the four presets. A custom architecture looks like `{"name": "Ground + DRO", "ground": true, "sensors": [{"platform": "dro", "aperture_m": 0.5, "fov_deg": 2.0}]}`, where `platform` is one of `geo`, `l1_halo`, `l2_halo_S`, `dro`, `nrho_9_2`, `resonant_3_1`. Other fields: `custody_threshold_km`, `object_set`, `maneuver_model` (`rate_per_object_per_day` 0.5, `dv_mps_range` [1, 20], `alpha` 0.01), `slot_min` (20), `initial_sigma_km`, `q_psd`, `include_draws`.

```json
{
  "scores": [
    {"name": "Ground only", "coverage_pct": 13.97, "custody_pct": 53.71, "custody_pct_null": 34.58,
     "revisit_mean_h": 44.67, "detections_pct": 34.29, "detection_latency_mean_h": 12.22,
     "custody_pct_p05": 39.06, "custody_pct_p95": 70.43, "…": "about 60 more fields"},
    {"name": "Ground + 2 GEO", "coverage_pct": 37.61, "custody_pct": 74.39, "revisit_mean_h": 20.54,
     "detections_pct": 71.43, "detection_latency_mean_h": 8.19, "…": "…"},
    "… Ground + L2 halo, Ground + DRO + L1 halo"
  ],
  "per_object": {"Ground only": {"SIM-DRO-01": {"coverage_pct": 14.93, "custody_pct": 69.70, "n_burns": 4, "n_detected": 2}, "…": "…"}},
  "meta": {
    "n_mc": 4, "seed": 0,
    "maneuver_model": {"rate_per_object_per_day": 0.5, "dv_mps_range": [1.0, 20.0], "alpha": 0.01,
                       "direction": "isotropic (notional actor; no intent assumed)"},
    "timing": {"workers": 4, "executor": "process x4", "mean_draw_s": 3.23, "route_total_s": 3.93}
  },
  "disclaimer": "…",
  "method": "…"
}
```

### Demo scenario

```http
GET /api/demo/scenario
```

```json
{
  "meta": {"title": "Unannounced DRO departure — custody loss and recovery (SIMULATED)",
           "t0_utc": "2026-02-23T00:00:00Z", "t1_utc": "2026-03-01T00:00:00Z",
           "n_frames": 145, "playback_speed": 4320.0, "protagonist_id": "SIM-DRO-01", "relay_id": "SIM-NRHO-RELAY-01",
           "epoch_choice_rationale": "…", "burn_rationale": "…", "assumptions": ["…"], "disclaimer": "…"},
  "frames": [{"t": 0.0, "t_utc": "2026-02-23T00:00:00Z", "objects": ["…"], "clouds": ["…"], "sensors": ["…"],
              "observations": ["…"], "ground_blind_reason": "…", "moon_illum": 0.33, "moon_sep_deg": 6.2}, "… 145 frames in all"],
  "events": [{"t_utc": "2026-02-25T10:00:00Z", "kind": "maneuver_detected", "severity": "…", "text": "MANEUVER DETECTED on SIM-DRO-01: …", "object_id": "SIM-DRO-01", "data": {}}, "… 33 events in all"],
  "metrics": {"detection_latency_h": 1.5, "max_sigma_km": 1052.7, "regained_after_h": 1.0,
              "dv_est_mps": 30.022, "dv_est_err_pct": 0.07, "dir_err_deg": 0.112, "custody_pct": 93.1, "…": "…"},
  "brief": "# Analyst brief — Unannounced DRO departure …"
}
```

The title and brief strings are reproduced as the bundle stores them.
