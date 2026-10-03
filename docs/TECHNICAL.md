# SELENE technical notes

This document holds the detail behind the [README](../README.md): the equations, the numerical methods, how each engine was validated, the full list of assumptions and limitations, data sources and the literature. The API is documented separately in [API.md](API.md).

Units are SI throughout, with km and km/s for positions and velocities, m/s for delta-v and hours for time horizons unless a field says otherwise.

## Contents

- [Setup details](#setup-details)
- [Code architecture](#code-architecture)
- [CR3BP in the Earth-Moon rotating frame](#cr3bp-in-the-earth-moon-rotating-frame)
- [Ephemeris force model](#ephemeris-force-model)
- [Instantaneous rotating frame](#instantaneous-rotating-frame)
- [Periodic-orbit library](#periodic-orbit-library)
- [Sensor models and coverage](#sensor-models-and-coverage)
- [Orbit determination](#orbit-determination)
- [Maneuver detection and delta-v estimation](#maneuver-detection-and-delta-v-estimation)
- [Reachability](#reachability)
- [Sensor tasking](#sensor-tasking)
- [Architecture Monte Carlo](#architecture-monte-carlo)
- [Demo scenario and analyst brief](#demo-scenario-and-analyst-brief)
- [Validation and test results](#validation-and-test-results)
- [Assumptions and limitations](#assumptions-and-limitations)
- [Data sources and licences](#data-sources-and-licences)
- [References](#references)

## Setup details

**Toolchain.** Python 3.11+ (the repo's venv was built with 3.12.4) and Node 20+ (the repo uses Node 24.21 from `.tools/node`). The venv lives in `.venv/` and Node in `.tools/node/bin`, both git-ignored. The Makefile prepends both to `PATH`, so nothing is installed globally. `backend/pyproject.toml` carries compatible-release bounds on every dependency and `backend/requirements.lock` is the exact freeze the test suite was validated with.

**Ephemeris kernels.** `make setup` runs the `data` target, which downloads two NAIF files with `curl` when they are missing (about 33 MB, once):

```
data/cache/de440s.bsp      https://naif.jpl.nasa.gov/pub/naif/generic_kernels/spk/planets/de440s.bsp   (~31 MB)
data/cache/gm_de440.tpc    https://naif.jpl.nasa.gov/pub/naif/generic_kernels/pck/gm_de440.tpc
```

If `gm_de440.tpc` is missing, `constants.py` falls back to identical literal GM values. If `de440s.bsp` or the orbit library (`data/orbits/orbits.parquet`) is missing, the API still starts and `GET /api/health` reports `status: "degraded"` with the missing input under `data.required_missing`. Every route that needs the kernel then answers 503 with the download hint (never a bare 500). Epochs outside the kernel coverage (the route message gives 1849-12-25 to 2150-01-21) are rejected with 400 naming the span.

**Make targets** (checked against the `Makefile`):

| Target | What it does |
|---|---|
| `make setup` | `setup-py` (creates `.venv` with `python3 -m venv` if absent, installs `backend/requirements.lock`, then `pip install -e backend[dev]`), `setup-js` (`npm install` in `frontend/`) and `data` (kernel download). |
| `make data` | Downloads `de440s.bsp` and `gm_de440.tpc` into `data/cache/` if absent. |
| `make dev` | uvicorn on `http://127.0.0.1:8000` and Vite on `http://127.0.0.1:5173` (Vite proxies `/api` and `/openapi.json` to :8000). Ctrl-C stops both. |
| `make api` / `make web` | Either half alone (`api` adds `--reload`). |
| `make test` | `pytest -q -n auto` in `backend/` (45 test files; the `slow` marker is registered in `pyproject.toml`). |
| `make test-fast` | `pytest -q -x -m "not slow"`. |
| `make build` | `npm run build` (`tsc --noEmit && vite build`) and copies `frontend/dist` to `backend/selene/api/static/`. |
| `make demo` | `build`, then a single uvicorn process on :8000 serving both the API and the built UI (SPA fallback to `index.html`). |
| `make precompute` | Regenerates committed caches: `python -m selene.orbits.library --rebuild` (orbit library, about 5 s on 8 cores), `python -m selene.objects.horizons --refresh` (needs network; `|| true`), `python -m selene.scenario.demo --rebuild` (full demo bundle, about 6 s warm; `--fast` for a reduced build, `--out` for a scratch path). |
| `make clean` | Removes the pytest cache, the numba cache and `frontend/dist`. |
| `make screenshots` | Regenerates the README images. Runs `build`, starts a throwaway uvicorn on :8765 (`SHOT_PORT`), drives headless Chrome through `scripts/screenshots.json` with `scripts/capture-screenshots.mjs` (raw 2x PNGs in `.tools/screenshots/`), then `scripts/build-readme-images.py` writes `docs/images/*.jpg`, the panel crops and `demo.gif`. Installs Pillow into `.venv` if missing (dev tooling only); `CHROME_PATH` overrides the browser. About 4 minutes. |
| `docker compose up` | Multi-stage build: `node:24-slim` builds the UI, `python:3.12-slim` installs `requirements.lock` and runs uvicorn on :8000. The image copies `data/orbits`, `data/horizons` and `data/demo` and fetches the two JPL kernels at build time, so a plain `docker run` works offline afterwards. `.dockerignore` keeps `.venv`, `.tools`, `node_modules`, `data/cache` and build output out of the context. Not verified on the build machine (no Docker there). |

**Offline operation.** After `make setup` the product runs with no network: DE440s and the GM file (`data/cache/`), the precomputed periodic-orbit library (`data/orbits/`, 500 KB, 325 orbits), compact JPL Horizons samples for 25 spacecraft (`data/horizons/`, 4.4 MB), the JPL periodic-orbit catalog subset used for validation (`data/orbits/jpl_reference.json`) and the precomputed demo bundle (`data/demo/scenario.json`, 3.7 MB, 145 hourly frames, 33 events and the analyst brief; committed). `GET /api/health` reports `offline: true` only when every one of these inputs is present (`data.present` lists them). astropy's IERS auto-download is disabled (`sensors/sites.py`), so site transforms use the bundled tables.

The frontend is mock-first: every API call records whether it was served `LIVE` or `MOCK`, and the banner prints that verbatim, so mock data cannot pass itself off as backend physics. The analysis panels (OD, maneuver, reachability, tasking) and the architecture studio are live-only and show the backend's error text instead of a placeholder. Details are in [frontend/README.md](../frontend/README.md).

**Environment variables.**

- `SELENE_CORS_ORIGINS`: comma-separated allowed origins. The default is the local Vite and uvicorn origins only; the dev server proxies `/api` and the built UI is same-origin, so no cross-origin access is needed. `*` opens it.
- `SELENE_DEMO_SCENARIO`: serve or rebuild the demo bundle from another path.
- `SELENE_ALLOW_REBUILD=1`: enables `POST /api/demo/rebuild`, which otherwise answers 403 so a stray call cannot overwrite the committed bundle during a live demo.
- `NUMBA_CACHE_DIR`: where numba caches compiled kernels (the Makefile sets `.numba_cache/`).

Deep links: `/ops?demo=1` auto-plays the story, `&t=<seconds>` jumps there and pauses, `&frame=inertial` switches the view, `&view=overview|earth|moon|l1|l2` flies the camera to a preset, `&layers=exclusion,reach,...` switches scene layers on, `&tab=object|events|brief` picks the dock tab and `&select=<object id>` selects an object.

## Code architecture

```
selene/
├── backend/selene/
│   ├── constants.py        GM from gm_de440.tpc (literal fallbacks), CR3BP μ, L*, T*, V*, radii, SRP constant
│   ├── time.py             UTC <-> TDB (astropy), TDB seconds past J2000 (the internal time scale)
│   ├── dynamics/           cr3bp (numba EOM, STM, Jacobi, L1-L5, events) · ephemeris (DE440s loader, BodyCache,
│   │                       Earth+Moon+Sun point masses + indirect terms + cannonball SRP, analytic Jacobian/STM)
│   │                       · frames (instantaneous rotating <-> GCRF <-> Moon-centred, exact velocity transform)
│   │                       · propagate (frame-aware Trajectory wrapper, DOP853 default, Radau option)
│   ├── orbits/             richardson (Lindstedt-Poincaré / Richardson 3rd-order seeds) · shooting (symmetric single
│   │                       shooting, multiple shooting, closure error) · continuation (natural-parameter,
│   │                       pseudo-arclength) · stability (monodromy, ν index) · families (9 family generators,
│   │                       synodic-resonant NRHO location) · library (parquet/json catalogue) · jpl_reference
│   ├── objects/            notional (11 SIMULATED spacecraft on library orbits + a Keplerian ELFO) · horizons
│   │                       (JPL Horizons client, offline cache, Hermite interpolation, data hygiene) · catalog
│   │                       (unified truth arcs under the DE440s model, one-time epoch-velocity fit for unstable orbits)
│   ├── sensors/            sites (9 notional ground telescopes, astropy ITRS->GCRS) · observers (6 space observers
│   │                       on GEO / L1 halo / L2 halo / DRO / NRHO) · photometry (Lambertian sphere) · constraints
│   │                       (Sun/Moon/Earth exclusion, eclipse, elevation, daylight, lunar glare, FOV) · reasons
│   │                       (bitmask) · visibility (RA/Dec measurement model + Jacobian) · coverage (heatmap grid)
│   ├── od/                 measurements (on-sky residual convention, synthetic observation sets) · iod (two-range
│   │                       shooting IOD) · batch (WLS with Levenberg damping) · ukf (stacked-sigma-point UKF with
│   │                       iterated update, EKF) · stacked (one solve_ivp for N trajectories) · particles (cloud
│   │                       propagation, non-Gaussianity metrics) · realism (NEES/NIS Monte Carlo)
│   ├── maneuver/           detection (NIS, windowed NIS, gap re-fit Mahalanobis, NEES monitor, CUSUM, health gate)
│   │                       · estimation (impulsive Δv + burn epoch, RTN/VNB, heuristic classification) · synthetic
│   │                       (truth-with-burn, named burn directions) · pipeline (end-to-end runner) · _simple_ekf
│   ├── reachability/       sampling (Fibonacci sphere × magnitude ladder × burn epochs, stacked propagation,
│   │                       termination events) · regions (9 named high-value regions) · tasking_hint (where to point)
│   ├── tasking/            information (linear-covariance engine, STM chains, angles update, acquisition probability,
│   │                       gains) · greedy (slot engine + greedy policy) · optimize (receding-horizon MILP via
│   │                       scipy HiGHS, local search) · metrics (custody %, TSLO, random/round-robin/null baselines)
│   │                       · scenario (object/sensor presets, slot grid)
│   ├── architecture/       candidates (6 platform orbits, 4 preset architectures, aperture -> limiting magnitude) ·
│   │                       montecarlo (common-random-number draws, coverage %, greedy-tasker custody %, revisit,
│   │                       surrogate-NIS detection latency, 4-process pool)
│   ├── scenario/           demo (the 2-minute story computed end to end by the engines, hourly bundle) · brief
│   │                       (template analyst brief with glossary, no LLM)
│   └── api/                app (FastAPI factory, CORS, 503/400 error shaping, SPA mount that never shadows /api) ·
│                           observer_spec (validated ad-hoc space-observer model) · schemas · routes/*
├── backend/tests/          45 pytest files, one per module/route (+ test_api_contract.py for the API conventions)
├── backend/requirements.lock   exact dependency freeze the suite was validated with
├── frontend/src/
│   ├── pages/              OpsPage (/ops, default) · ArchitecturePage (/architecture) · CoveragePage (/coverage)
│   ├── scene/              SceneRoot, Bodies (true-scale day/night Earth & Moon, L-points), OrbitFamilies, Objects
│   │                       (+trails, labels, declutter), ParticleCloud, Cones (FOV + Sun/Moon/Earth exclusion),
│   │                       GroundSites, Reachable, AnalysisOverlays, ObservationFlashes, CameraRig, frameBus
│   ├── panels/             TopBar, Timeline, EventsFeed, ObjectPanel (+SigmaSparkline), ScenarioMetrics,
│   │                       AnalystBrief, Toasts, CameraPresets, Dock, LiveStatus, analysis/ (OD, Maneuver,
│   │                       Reachability, Tasking panels)
│   ├── studio/             architecture and coverage studio: CoverageHeatmap, SystemPlot, model, mock data
│   ├── demo/               scenario driver, frame interpolation, live trajectory tracks, mock scenario/network
│   ├── lib/                browser CR3BP (offline shapes), ephem (rotating basis, GMST), tracks, groundVis
│   └── api/ · store/       typed client with a per-endpoint LIVE/MOCK registry · zustand stores
├── data/
│   ├── cache/              de440s.bsp, gm_de440.tpc, raw Horizons responses   (git-ignored)
│   ├── orbits/             orbits.parquet, families.json, jpl_reference.json, README.md   (committed)
│   ├── horizons/           <id>.json × 25 + index.json   (committed)
│   └── demo/               scenario.json (3.7 MB) + scenario_meta.json, the committed full-fidelity demo bundle
├── docs/                   TECHNICAL.md (this file), API.md, images/ (README screenshots and demo.gif)
├── scripts/                README screenshot capture: screenshots.json, capture-screenshots.mjs, build-readme-images.py
├── Makefile · Dockerfile · docker-compose.yml · .dockerignore
└── PLAN.md · PROGRESS.md · DECISIONS.md · PITCH.md · README.md · FINAL_REPORT.md
```

Frontend stack: Vite 6, React 18.3, TypeScript 5.9, react-three-fiber 8 with drei 9 and three 0.170, zustand 5, Recharts 2, react-router 6. Backend: FastAPI, numpy, scipy, numba, jplephem, skyfield, astropy, pyarrow and pandas, pydantic v2.

## CR3BP in the Earth-Moon rotating frame

Nondimensional units come from the DE440 gravitational parameters (`constants.py`; GM_E = 398 600.435507, GM_M = 4 902.800118 km³/s²):

```
μ  = GM_M / (GM_E + GM_M) = 0.0121505844
L* = 384 400 km            T* = sqrt(L*³ / (GM_E + GM_M)) ≈ 375 190 s
V* = L*/T* ≈ 1.0245 km/s   synodic period = 2π T* ≈ 27.28 d
```

The Earth sits at (−μ, 0, 0) and the Moon at (1−μ, 0, 0); x̂ points from Earth to Moon and ẑ along the orbital angular momentum. With the pseudo-potential U = ½(x²+y²) + (1−μ)/r₁ + μ/r₂:

```
ẍ − 2ẏ = ∂U/∂x,   ÿ + 2ẋ = ∂U/∂y,   z̈ = ∂U/∂z
Jacobi constant  C = 2U − (ẋ² + ẏ² + ż²)                (conserved; the primary accuracy diagnostic)
STM              Φ̇ = A(t) Φ,  A = [[0, I], [U_xx, 2Ω]],  Ω = [[0,1,0],[−1,0,0],[0,0,0]]
```

Lagrange points come from the collinear quintic and the equilateral solutions: L1 = 0.836915, L2 = 1.155682, L3 = −1.005063 (matched to 1e-5 in `tests/test_cr3bp.py`). The kernels are numba-compiled; integration uses `solve_ivp` with DOP853 (Radau selectable).

## Ephemeris force model

Earth-centred GCRF, km and s. DE440s positions come from jplephem (two-part Julian date, TDB). The model is an Earth point mass plus Moon and Sun third-body terms, including the indirect acceleration (the Earth itself is accelerated by body *j*), plus optional cannonball solar radiation pressure:

```
r̈ = −GM_E r/|r|³ − Σ_{j∈{Moon,Sun}} GM_j [ (r − r_j)/|r − r_j|³ + r_j/|r_j|³ ] + a_SRP
a_SRP = ν · P☉ · C_R (A/m) · (1 AU/|r − r☉|)² · (r − r☉)/|r − r☉|,   P☉ = 4.56e-6 N/m², ν ∈ {0,1}
```

ν is a cylindrical Earth/Moon shadow factor (penumbra neglected). The variational equations use the analytic point-mass Jacobian ∂a_j/∂r = GM_j[3ΔΔᵀ/|Δ|⁵ − I/|Δ|³]; SRP is treated as constant for the STM. A `BodyCache` spline of Moon and Sun positions (under 1 m error, `tests/test_ephemeris.py`) feeds the numba right-hand side, so many trajectories can be stacked in one `solve_ivp` call (`od/stacked.py`, `reachability/sampling.py`). GCRF is treated as J2000 (difference under 0.1″), which is adequate for display and for this MVP's estimation.

Internal time is TDB seconds past J2000 (jplephem's native scale); UTC appears only at the API boundary.

## Instantaneous rotating frame

The frame is rebuilt at each instant from the DE440s Earth-to-Moon state (r, v): x̂ = r/|r|, ẑ = r×v/|r×v|, ŷ = ẑ×x̂, d(t) = |r|, ḋ = r·v/d. The origin is the barycentre r_b = μ r, so the Moon sits exactly at (1−μ, 0, 0). Lengths are scaled by the instantaneous d(t) and velocities by d(t)/T*. The angular velocity ω = ω_z ẑ + ω_x x̂ includes the out-of-plane precession term, derived kinematically from DE440s by central differences, which makes the velocity transform the exact time derivative of the position transform:

```
r_rot = Rᵀ (r − r_b) / d
v_rot = T* · [ Rᵀ ((v − v_b) − ω × (r − r_b)) / d  −  r_rot · ḋ/d ]
```

Round-trip relative error is under 1e-9; the velocity-transform residual is asserted under 1e-8 nd (6e-11 measured, DECISIONS.md). Because d(t) and ω(t) oscillate by about ±10% with the lunar eccentricity, a CR3BP solution expressed in GCRF through this frame is not a solution of any single force model. It is used for visualisation, seeding and sensor-platform motion, never as truth (`dynamics/propagate.py`).

The UI's frame toggle is exact when the backend is up: `/api/ephemeris/bodies` serves the basis `R`, `d_km` and `r_bary_km` per epoch, and the inertial view applies `r_gcrf = r_bary + d · R · r_rot`. The displayed Moon matches DE440s to under 1 m.

## Periodic-orbit library

- **Symmetric single shooting** (Howell 1984; Pavlak 2013). From X₀ = (x₀, 0, z₀, 0, ẏ₀, 0), integrate to the next y = 0 crossing (terminal event) and Newton-drive ẋ = ż = 0 there, using the event-corrected sensitivity dX(t_c)/dv = Φ[:,v] − f(X(t_c)) Φ[1,v]/ẏ(t_c). Modes: `fix_x0`, `fix_z0`, `planar`, `free3`/`free2` (with a pseudo-arclength row), and a fixed-`t_half` variant for multi-loop resonant orbits.
- **Multiple shooting** (Pavlak §2.4; Marchand, Howell & Wilson 2007). N patch states plus the period, with continuity, periodicity and phase constraints and a minimum-norm Newton step. Used for near-rectilinear members, whose perilune of about 3,000 km makes single shooting ill-conditioned.
- **Seeds.** The linearised Lyapunov solution, and Richardson's (1980) third-order halo expansion (`orbits/richardson.py`, coefficients eqs. 10-24).
- **Continuation** (`orbits/continuation.py`). Natural parameter with adaptive step, and Keller pseudo-arclength (v_pred = v_k + ds·τ with the constraint (v − v_pred)·τ = 0), which follows the halo family through its fold into the NRHO regime.
- **Stability.** Monodromy M = Φ(T,0) and ν = ½(|λ_max| + 1/|λ_max|) (Howell 1984; Zimovan-Spreen et al. 2020; the same quantity as the JPL catalog), plus Broucke per-pair indices.
- **Families** (325 records, `data/orbits/README.md`): `L1_lyapunov` (35), `L2_lyapunov` (31), `L1_halo` N/S (42+42), `L2_halo` N/S (36+39), `DRO` (46), `resonant_3:1` (19), `resonant_2:1` (35). Tags: `NRHO`, `NRHO_9:2`, `NRHO_4:1`, `NRHO_3:1` (L2 southern members whose period equals q/p synodic months, root-found along the family), `stable`, `planar`. Halo families stop before lunar impact (lowest perilune 1,961 km for L1 and 1,901 km for L2). Perilune and apolune are refined by bounded 1-D minimisation on dense output; grid sampling had overstated NRHO perilune by up to 400 km and hidden four sub-surface halo members. Library IC convention: the perpendicular xz-crossing farther from the Moon, with north/south set by the sign of z there.
- **The 9:2 NRHO** (`L2_halo_S_nrho_9_2`): T = 6.5624 d (2/9 of a synodic month to 1e-6 d), perilune 3,249 km, apolune 71,222 km, ν = 1.32. Literature: 6.56 d and about 3,200 to 3,366 km perilune.
- **Validation against JPL's Three-Body Periodic Orbit Catalog** (`orbits/jpl_reference.py`, `tests/test_orbits_jpl.py`, accessed 2026-10-02). JPL uses μ = 1.215058560962404e-2 against SELENE's 1.2150584395829193e-2 (Δμ = 1.2e-9). JPL normalises with 389 703.26 km and 382 981.29 s, so dimensionless quantities are comparable but km/day values are not. Perturbed JPL initial conditions re-converge with our corrector to |ΔT| < 1e-6, |ΔIC| < 1e-7 and closure < 1e-9 (|ΔT| ≤ 3e-12 measured). Every library record lies within 2e-4 of the catalog's (Jacobi, period) poly-line (6.1e-7 against the full catalog, mean ≤ 1e-7), and stability indices agree within 1%.

## Sensor models and coverage

**Photometry** (`sensors/photometry.py`). A Lambertian sphere of radius ρ and surface albedo *a* at range R and phase angle φ, with the phase function normalised per unit projected area π ρ²:

```
p(φ) = (2/(3π²)) [ (π − φ) cos φ + sin φ ],   p(0) = 2/(3π)
m    = m☉ − 2.5 log10( a · π ρ² · p(φ) / R² ),   m☉ = −26.74
```

The π factor is the cross-section (a π p(0) = 2a/3, the geometric albedo of a Lambertian sphere). It was confirmed by direct surface integration in the tests; an earlier version without it was 1.24 mag too faint. Hand check: ρ = 1 m, a = 0.2, R = 400 000 km, φ = 0 gives m = 18.46. An object is detectable if m ≤ m_lim − margin.

**Constraints** (`sensors/constraints.py`, all broadcasting and returning reason bitmasks): Sun exclusion (space sensors); Earth and Moon exclusion measured from the limb, so a target projected on the disc is always flagged; cylindrical umbra eclipse; ground elevation ≥ el_min; daylight (Sun elevation > −12°); field of view; and a phase-dependent lunar-glare zone for ground sites, θ_excl = 3° + (15° − 3°)·f_illum. The glare zone is a stated modelling assumption that stands in for a sky-brightness model. Reason bits: daylight 1, low_elevation 2, sun_exclusion 4, moon_exclusion 8, earth_exclusion 16, in_shadow 32, too_faint 64, out_of_fov 128.

**Sites and observers.** Nine notional ground telescopes at public observatory coordinates (Haleakala, Socorro, Mt. Lemmon, Cerro Tololo, Teide, Sutherland, Siding Spring, Diego Garcia, Ascension), 0.5 to 1.5 m class with assumed limiting magnitudes from 18.3 to 20.5, placed with astropy's full ITRS-to-GCRS chain. Six space observers (`geo_west`, `geo_east`, `l1_halo_obs`, `l2_halo_obs`, `dro_obs`, `nrho_obs`; limiting magnitude 18.0 to 18.5, field of view 2 to 3°) ride library orbits mapped through the instantaneous frame. GEO uses the same Earth-orientation chain as the sites.

**Coverage grid** (`sensors/coverage.py`). Cells are fixed in the rotating frame (default z = 0 slice, 64 × 56 over x ∈ [−1.6, 1.6], y ∈ [−1.4, 1.4]; optional 24 × 20 × 9 3-D grid) and mapped to GCRF each hour. A cell is covered if any sensor has a zero reason mask for a 1 m, albedo-0.2 reference sphere. Outputs are the per-cell coverage fraction, the per-time coverage percentage and the dominant blind-spot reason, chosen from the sensor closest to seeing, so near-Moon cells report `moon_exclusion` rather than `daylight`. FOV and slew are not applied in coverage. Measured over the week from 2026-03-01 (default grid, 168 hourly steps): `ground_only` 35.3%, `ground_plus_dro` 40.7%, `full` (ground plus GEO ×2, L1 halo, L2 halo and DRO observers) 44.1%; the dominant blind reason in all three is `too_faint`. Each run takes about 1 to 2 s on the build laptop.

**Measurements** (`sensors/visibility.py`, `od/measurements.py`). Topocentric GCRF RA/Dec with an analytic Jacobian. Residuals are expressed on the sky, ν = [wrap(Δra)·cos dec, Δdec], so R = σ² I₂ (default σ = 1″). Each measurement also carries `light_time_s`.

## Orbit determination

- **IOD** (`od/iod.py`). Gooding's two-range structure with the Lambert solver replaced by STM shooting under the full ephemeris model. For a pair of ranges (ρ₁, ρ₃), solve v₁ such that r(t₃) = r₃ by Newton on Φ_rv, then minimise the interior on-sky residuals over (ρ₁, ρ₃) with `least_squares` from a log grid of seeds (50 000 to 500 000 km) and the analytic Jacobian ∂x_k/∂ρ. Admissible-region pruning rejects seeds that are unbound with respect to both Earth and Moon; an Earth-only test would reject NRHO perilune states.
- **Batch weighted least squares** (`od/batch.py`; Tapley, Schutz & Born §4.6 with Levenberg damping). N = Σ H_iᵀW_iH_i + P̄₀⁻¹ with H_i = H̃_i Φ(t_i, t₀), a Jacobi-scaled normal matrix, and a step accepted when the weighted RMS falls. P₀ = N⁻¹ at convergence.
- **UKF** (`od/ukf.py`). Scaled unscented transform (α = 1e-3, β = 2, κ = 0 by default) with all 13 sigma points propagated in one stacked ODE, so integrator error is common-mode. With W₀ᶜ ≈ −10⁶, uncorrelated errors of 1e-6 km would otherwise become km-level covariance noise; stacking is also 16× faster. Process noise is continuous white acceleration, Q(Δt) = q·[[Δt³/3 I, Δt²/2 I],[Δt²/2 I, Δt I]], with default q = 1e-18 km²/s³ chosen by a 40-run NEES study (1e-16 is pessimistic). The iterated Gauss-Newton measurement update (Bell & Cathey 1993) is on by default because angles are nonlinear for IOD-sized priors: δ/ρ ≈ 500 km / 75 000 km gives a 4.6″ second-order term against 1″ noise. A sequential EKF with the same output is provided.
- **Particle clouds** (`od/particles.py`). N Cholesky samples of the posterior propagated through the stacked ephemeris model. Reports σ_pos, principal axes, the 1σ volume, extent, the STM-propagated Gaussian for comparison, and non-Gaussianity metrics (skew and kurtosis along the major axis, curvature of the "banana"). Exported in GCRF km and rotating-frame nd for the 3-D scene. 2,000 particles over 7 days take 0.07 s; per-particle mismatch against an individual 1e-12 propagation is at most 0.19 km.
- **Covariance realism** (`od/realism.py`; Bar-Shalom §5.4). NEES ε = eᵀP⁻¹e ~ χ²₆ and NIS d = νᵀS⁻¹ν ~ χ²₂, with Monte Carlo run-averaged statistics tested against [χ²_{Mn}(α/2), χ²_{Mn}(1−α/2)]/M. Verdicts are `consistent`, `over-confident` (the dangerous failure for custody) or pessimistic.

## Maneuver detection and delta-v estimation

Tests run on a `FilterRun` (`maneuver/detection.py`), each with a χ² p-value:

1. Per-update NIS against χ²₂(1−α), so the per-update false-alarm rate is exactly α under the null hypothesis.
2. Windowed NIS: the sum of the last W values against χ²_{2W}.
3. Gap re-fit Mahalanobis. After an observation gap longer than `gap_hours_for_refit` and 2.5× the median cadence (a sensor blinded by glare or daylight), a short-arc batch fit of the first k post-gap observations is compared with the filter's through-gap prediction in state space: d² = (x̂_fit − x̂_pred)ᵀ(P_fit + P_pred)⁻¹(x̂_fit − x̂_pred) ~ χ²₆. This catches burns that a covariance-inflated single innovation would absorb.
4. A NEES monitor against truth (simulation only).
5. One-sided Page CUSUM on (NIS − m − k), with the threshold h tuned by Monte Carlo for a target ARL₀.

A filter-health gate requires a consistent baseline of updates before any test runs; a filter that never converges yields `filter_not_converged`, not a detection. Run-level declarations use family-wise (Šidák) thresholds. Real Horizons objects are never flagged (their cached ephemerides have km-level interpolation error against the point-mass truth), and the maneuver route refuses them with 400.

**Delta-v estimation** (`maneuver/estimation.py`). The unknowns are Δv ∈ ℝ³ and the burn epoch t_b, with t_b in the window between the last quiet and the first flagged observation, plus an optional pre-burn state correction with the filter posterior as prior. Whitened on-sky residuals are minimised by trust-region `least_squares` with STM Jacobians ∂ν_k/∂Δv = −H_k Φ_rv(t_k, t_b). t_b comes from a grid followed by a bounded scalar refinement, and σ_t from the cost curvature. Outputs are Δv in GCRF, RTN and VNB (Earth- or Moon-centred), a formal 1σ (scaled by the reduced χ² when residuals exceed σ) and a labelled heuristic classification (Jacobi change, two-body energies, RTN components; it never infers intent).

## Reachability

**Sampling** (`reachability/sampling.py`). Impulsive burns Δv = m·û at epoch t_b on the nominal coasting trajectory, followed by a ballistic coast. û comes from a Fibonacci sphere; m comes from a fixed absolute ladder (2, 5, 10, 20, 50, 100, 200, 500, 1000 m/s, truncated at the budget, plus the budget itself), so smaller-budget sample sets are subsets of larger ones and region reachability is monotone by construction. The default burn epochs are t_b ∈ {0, +6, +12, +24 h}. All N samples integrate as one 6N-dimensional DOP853 system with the numba ephemeris right-hand side; burns are velocity discontinuities at segment boundaries. Lunar or Earth impact and escape terminate rows by event (candidate rows are screened by the osculating conic and then re-integrated alone with a terminal surface event). 2,000 samples over 72 h take 0.2 s. `refine_min_dv` bisects the minimum magnitude that reaches each region; it is a sampled minimum along one direction and epoch, so an upper bound on the true optimum.

**Regions** (`reachability/regions.py`). Nine named regions. Every number is a configurable parameter of `default_regions`.

- `l1_gateway`: a neck transit, not proximity. Three conditions on the time series of one trajectory (`transit_mask`):
  1. *Passage*: the path enters the ball |r_rot − L1| < R_neck (default 0.05 nd, about 19,200 km) and crosses the plane x = x_L1 while inside it. Leaving the ball and changing realm far from L1 does not count.
  2. *Realm change*: the last definite realm before the passage and the first definite realm after it differ. Realms use a hysteresis margin δ (default δ = R_neck): Earth realm x < x_L1 − δ; lunar realm x_L1 + δ < x < x_L2 − δ; exterior realm x > x_L2 + δ. Inside the bands |x − x_L| ≤ δ the realm is undetermined and the previous definite realm is kept. A passage still unresolved at the end of the horizon is not a transit; it is reported as an approach (closest approach, plane-crossing depth).
  3. *Energy*: the CR3BP-equivalent Jacobi constant on the passage is below C(L1), plus a 5e-3 slack for the roughly ±1e-3 per week wander of C in the ephemeris model. The test is skipped when no velocities are supplied (position-only paths).

  Point-wise inputs with no time series fall back to the proximity ball.
- `l2_gateway`: the same transit test at the L2 neck (lunar realm to exterior realm, energy gate C < C(L2)).
- `nrho_corridor`: within `tube_km` (default 10,000 km) of any point of the library's 9:2 NRHO (`NRHO_9:2` record), the Gateway and relay corridor.
- `south_pole_approach`: Moon-centred latitude below −60° (measured from the lunar orbit pole, the rotating-frame ẑ; the spin pole differs by about 6.7°, a documented approximation) and selenocentric distance under 20,000 km.
- `llo_shell`: selenocentric altitude 0 ≤ h < 5,000 km. The inner band h < 500 km is reported separately as `llo_inner`.
- `geo_belt_return`: geocentric radius within 42,164 ± 3,000 km, or geocentric distance under 60,000 km while moving inward (r·v < 0).
- `earth_return_escape`: an exterior-realm excursion, C < C(L2) and barycentric distance over 1.3 nd (about 500,000 km). This is not a hyperbolic-escape test: for the demo DRO object at 150 m/s, 0 of 59 such samples were geocentrically unbound (two-body energy −0.55 to −0.41 km²/s²). What matters for awareness is that these objects leave the volume Moon-pointed sensors search.
- `lunar_impact`: selenocentric altitude below zero, or an osculating periapsis below the surface on an inbound pass. A pure safety region.

**Why the gateway definition is strict.** Large DROs straddle L1 and L2 geometrically. The demo DRO (perilune 63,700 km, C ≈ 2.95) sweeps through both 0.05 nd balls every revolution and comes within about 1,000 km of the x = x_L1 plane (its CR3BP reference crosses it by 5,700 km), while being a stable, non-transiting orbit. A proximity sphere flagged the quiet orbit as "entering the gateway" twice a month. A plain plane-crossing test inside the ball was also noise: 10 to 20 m/s perturbations of the DRO dip about 2,000 km past the plane and leave the ball on the "Earth side" without leaving the lunar vicinity, and the first version of the classifier called 14% of the demo's reachable rays "L1 gateway transits". The hysteresis realms make a transit mean what an analyst means: the object was in the lunar realm proper and is now in the Earth realm proper, having passed near L1 (Koon, Lo, Marsden & Ross, ch. 2-3). One caveat: at C ≈ 2.95, below C(L4) ≈ 2.988, there is no forbidden region at all, so at DRO energies the "neck" is a geometric neighbourhood rather than an energetic bottleneck, and a path that leaves the lunar realm far from L1 is not counted as an L1 gateway transit.

**Jacobi scaling.** The Jacobi value is computed with the CR3BP formula on the instantaneous rotating-frame state and used only as an energy classifier. `frames.gcrf_to_rot` nondimensionalises velocity with the constant T*, whereas the formula assumes unit angular rate; ω(t)·T* wobbles by ±5 to 8% over the anomalistic month, which alone moves C by about 0.04 along a DRO. Callers pass `omega_t_star` and the velocity is rescaled before the formula is applied; C then drifts by about 1e-3 over a week on the demo DRO, the genuine non-conservation of the ephemeris model.

**Pointing hints** (`reachability/tasking_hint.py`). The active samples run through the visibility model per sensor and per step to produce the best boresight, the p50 and p90 angular spread, the single-field capture fraction and the tile count. This is the bridge to tasking.

Example (`POST /api/reachability`, SIM-DRO-01 at 2026-03-01, 100 m/s budget, 168 h, 1,488 samples): L2 gateway 2.0% of rays (earliest +90 h, from 50 m/s), NRHO corridor 20.6% (+89 h, from 50 m/s), LLO shell 4.0% (from 100 m/s), exterior-realm excursion 4.4% (from 100 m/s); no L1 transit, no south-pole approach, no GEO return.

## Sensor tasking

A linear-covariance engine handles many objects at once (`tasking/information.py`). Each object's 6 × 6 P is propagated between 20-minute slot nodes with the ephemeris STM chain plus process noise and updated by the linearised angles-only measurement in Joseph form. The acquisition probability p = 1 − exp(−n_tiles·θ_f²/(2σ_θ²)) (the Rayleigh CDF of the on-sky prediction error against the FOV half-angle) gives the expected posterior P⁺ = p·P⁺_det + (1−p)·P⁻. Gains: position-trace reduction (the default, because custody is a position criterion), log-det mutual information ½ ln(det P⁻/det P⁺), or max-eigenvalue reduction.

- **Greedy** (`tasking/greedy.py`). For each slot and sensor, pick the visible and slew-feasible object that maximises priority·gain·(1 + w·TSLO/24 h), given updates already applied in this slot. This is classic submodular sensor management: for log-det gain over a fixed set, greedy achieves at least (1 − 1/e) ≈ 63% of the optimal value.
- **MILP** (`tasking/optimize.py`). A receding horizon of 4 slots solved with `scipy.optimize.milp` (HiGHS). The surrogate freezes covariances at the window start and models diminishing returns with rank weights f_j[m] (the ratio of the m-th sequential marginal gain to the m-th best stand-alone gain) plus a revisit bonus, with one-object-per-sensor-per-slot and slew constraints, explicit rank binaries, a local-search fallback and a per-request time budget (at most 30 s, synchronous). The response labels the surrogate as such.
- **Honest status of the MILP.** It beats greedy on its own surrogate objective, but on a discriminating scenario (500 km / 2 m/s prior, 48 h, `mixed_9`) it is worse on every realised metric: custody 88.3% against 91.8%, summed covariance trace 300k against 228k km², mean TSLO 1.77 h against 1.27 h. PLAN M7's "MILP ≥ greedy" holds only on the surrogate, and greedy is the policy the demo and the architecture studio use.
- **Metrics** (`tasking/metrics.py`). Custody % is the fraction of slot nodes with √tr(P_pos) < 100 km (configurable). Also mean time since last observation (TSLO), utilisation, and `random`, `round_robin` and `null` (no observations) baselines run through the identical engine. The default scenario is 8 xGEO objects × the 9-sensor `mixed_9` network × 48 h from the demo epoch. `preset='ground_only'` honestly reproduces the near-full-Moon case in which no ground site can see any of them.

## Architecture Monte Carlo

`architecture/candidates.py` defines six candidate platforms (GEO, L1 halo, L2 southern halo, DRO, 9:2 NRHO, 3:1 resonant; all library orbits mapped through the instantaneous frame) with an aperture-to-limiting-magnitude rule, m_lim = 18.5 + 5 log10(D/0.5 m) clipped to [14, 23], and four preset architectures.

`architecture/montecarlo.py` scores architectures with common random numbers: identical start-epoch offsets, injected burns and measurement-noise draws per Monte Carlo draw, so small `n_mc` still ranks fairly. Metrics:

- coverage %: object × slot nodes visible to at least one sensor;
- custody %: from the greedy tasker's linear-covariance run against a common stale prior, plus the zero-observation reference `custody_pct_null`;
- revisit time: mean and censored p95;
- maneuver-detection latency, from a documented linear-STM displacement plus sampled-NIS surrogate. The burn's displacement is created at the burn node and tested with the same χ² gate as the detector; its false-alarm rate equals α by construction. It is a bookkeeping surrogate, not a UKF calibration, and `meta.method_notes` says so. The full pipeline per draw would cost minutes per request.

The default maneuver model is 0.5 burns per object per day, 1 to 20 m/s, isotropic directions. Draws run on a persistent 4-process spawn pool with bit-identical results to the sequential path. Measured with `n_mc=4`, horizon 3 d, seed 0 (re-run while writing this document, 1.7 to 4 s per request):

| Architecture | Coverage | Custody | Mean revisit | Burns detected | Mean latency (detected) | p05 to p95 custody |
|---|---|---|---|---|---|---|
| Ground only | 14.0% | 53.7% | 44.7 h | 34.3% | 12.2 h | 39.1 to 70.4% |
| Ground + 2 GEO | 37.6% | 74.4% | 20.5 h | 71.4% | 8.2 h | 71.3 to 77.7% |
| Ground + L2 halo | 74.0% | 90.2% | 9.1 h | 94.3% | 9.0 h | 85.6 to 94.4% |
| Ground + DRO + L1 halo | 75.9% | 91.6% | 11.1 h | 88.6% | 1.5 h | 88.7 to 94.4% |

The zero-observation custody reference is 34.6% for all four. `tests/test_architecture_route.py` prints the values for its own configuration.

The studio page sends its own defaults: `n_mc=8`, 3 days, `seed=20260301`, 20-minute slots, catalog object sizes (`studio/api.ts` does not forward the page's target radius and albedo fields). That run is the one in the README screenshot and table, and the API reproduces it exactly (70 injected burns per architecture). The studio's latency bars use the censored statistics (`detection_latency_censored_*`, a missed burn counts as the rest of the horizon), so they are larger than the detected-only means above:

| Architecture | Coverage | Custody | Mean revisit | Burns detected | Latency, censored mean / p95 | Latency, detected only |
|---|---|---|---|---|---|---|
| Ground only | 7.4% | 45.0% | 59.9 h | 12.9% (9 of 70) | 40.1 / 68.9 h | 13.0 h |
| Ground + 2 GEO | 47.5% | 76.6% | 22.1 h | 65.7% (46) | 16.8 / 59.0 h | 4.0 h |
| Ground + L2 halo | 66.4% | 87.3% | 12.6 h | 81.4% (57) | 12.5 / 46.5 h | 7.2 h |
| Ground + DRO + L1 halo | 79.4% | 93.4% | 8.9 h | 91.4% (64) | 6.3 / 29.8 h | 3.6 h |

The two runs differ because each draw re-samples the start epoch (and with it the Moon's phase) and the burns; with 4 or 8 draws the spread between draws is wide, which is why the studio shows p05 to p95 whiskers.

## Demo scenario and analyst brief

`scenario/demo.py` computes the story end to end with the engines above; nothing is scripted numerically. The hand-chosen inputs (the epoch window, the 30 m/s burn magnitude, the burn time 30 minutes after the last pre-burn ground tracklet, the burn direction chosen from 256 sampled directions as the one passing closest to L1, the 100 m/s planning budget, the custody thresholds) are documented in the bundle under `meta.epoch_choice_rationale`, `meta.burn_rationale` and `meta.assumptions`. A full rebuild takes about 6 s warm.

**Window and premise.** 2026-02-23 to 2026-03-01 UTC, 145 hourly frames, played at 4,320× (about 120 s plus narration holds). SIM-DRO-01 stays within about 12° of the Moon on the sky, so under the 3° to 15° glare model the nine ground sites can track it only on the nights of 23 to 25 February (Moon 33 to 60% illuminated) and are blind from about 11:00 UTC on 25 February through full Moon (3 March). The burn is placed 30 minutes after the last pre-burn ground tracklet, so exactly one post-burn ground observation catches the anomaly before the glare closes the window. The loss is therefore computed, not scripted. Known fragility: that tracklet clears the glare cone by only about 0.1°, so a different glare parameterisation or a 30-minute epoch shift leaves no post-burn ground observation, and the builder then raises an error instead of inventing a detection.

**Burn direction.** Of 256 Fibonacci-sampled 30 m/s directions propagated with the reachability engine, the one whose coasting arc passes closest to L1 (260 km at +41.7 h, refined on the dense solution, against 4,098 km for the unperturbed DRO). In Moon-centred RTN the burn is −29.6 / +5.0 / −1.3 m/s.

**Sequence (all values from `data/demo/scenario_meta.json`).**

1. Routine custody: one ground tracklet every 2 h when a site can see the object; 15 pre-burn ground tracklets; UKF prior 5 km / 0.2 m/s (1σ); cloud σ 7 to 20 km.
2. Burn at 2026-02-25T08:30Z (SIMULATED truth, not visible to the operator).
3. Detection at 10:00Z: the Siding Spring tracklet is 96.5″ off the prediction, NIS 6,178 against the family-wise threshold 18.3 (α = 0.01). Latency 1.5 h. The filter prior is re-opened for an unknown impulsive Δv (σ 25 m/s over the 2 h gap). Cloud σ at detection 101.6 km (UKF √tr P_pos 104.4 km), custody DEGRADED.
4. Ground network blind: of the 86 hourly frames after detection, 79 have an available site blocked by lunar glare (`moon_exclusion`) and 7 have no site with the object above its elevation limit at night (`no_site_available`, not attributed to glare). Custody LOST at 19:00Z (10.5 h after the burn) when the cloud σ (1,500 particles) passes 1,000 km; peak 1,053 km. Moon 64.5% illuminated and 8.9° from the object at loss.
5. Reachability from the last good state (t_ref 08:00Z, 100 m/s budget, 168 h, 1,728 samples = 96 directions × ladder × burn epochs 0, 1, 2 h). Only regions the quiet orbit does not already visit are reported as new. L1 gateway: 24 samples transit from the lunar realm into the Earth realm (1.4% of samples, 8% of rays), earliest +33 h, from 89 m/s. L2 gateway: 3 samples (0.2%), earliest +135 h, from 99 m/s. NRHO corridor of the relay (SIM-NRHO-RELAY-01): not reached by any of the 1,728 samples (at or below the sampling resolution, so a marginal hit at the budget limit cannot be excluded). The alert also reports what does not count: 1,468 samples dip past the x = x_L1 plane inside the neck ball (up to 18,605 km) and return, and 41 reach the Earth realm more than 19,220 km from L1. Closest approach of the reachable set to L1: 433 km at +44 h.
6. Tasking at 20:00Z: the greedy tasker (log-det gain for the escalation, FOV acquisition model, 25-field mosaic budget per 60-minute slot, priority 3 on the protagonist, one slot of re-planning latency) points geo_west, geo_east, l1_halo_obs, dro_obs and nrho_obs at the reachable-set centroid; l2_halo_obs is not tasked (Sun exclusion). Every tasked look is verified against the SIMULATED truth with the visibility model and the mosaic, never assumed. Five looks verified; 38 space tracklets follow.
7. Custody REGAINED at 20:00Z, 1.0 h after the loss and 11.5 h after the burn; σ 0.42 km.
8. Maneuver characterised at 22:00Z from 10 observations: 30.02 ± 0.03 m/s (truth 30.0, +0.07%), direction ± 0.06° (0.11° from truth), epoch 08:30:45Z ± 53 s (+45 s), residual RMS 1.005″, reduced χ² 1.23. Heuristic label: radial (inward).
9. Analyst brief generated (`scenario/brief.py`): template text, glossary first, no LLM, SIMULATED footer.
10. SIMULATED truth marker at 2026-02-27T02:13:35Z: the object passes 260 km from L1 (hourly grid minimum 508 km) but does not transit the L1 neck; it stays in the lunar realm. The bundle says so (`closest_approach` event, `truth_geometry.l1_transit = false`) instead of claiming a gateway entry. Closest approach to the relay: 56,192 km.

Protagonist custody over the window: 93.1% of frames (135 CUSTODY, 9 DEGRADED, 1 LOST). The other ten SIMULATED objects stay in custody 100% of the time except the debris fragment (82.8%, max σ 157.5 km).

Covariance realism in the demo is single-run: `metrics.detection.nees_*` compares one sequential run's mean NEES (4.52) with a band (5.1 to 6.98) that assumes independent epochs. Successive errors are correlated, so a below-band mean is only indicative of a conservative covariance (`nees_note` in the bundle). An 8-seed Monte Carlo through `/api/od/run` with the demo configuration gives pooled NEES of 5.1 to 5.6, which is consistent. The calibrated check is `od/realism.py` with `tests/test_od_realism.py`.

**Bundle format.** `data/demo/scenario.json` (3.7 MB, 145 frames, 33 events) is committed and served by `GET /api/demo/scenario` in about 0.2 s. Time fields: `t` = `t_rel_s` seconds since `meta.t0_utc`, `t_utc` ISO with Z, `t_s` absolute TDB seconds past J2000 (`meta.time_fields`). Build timings: truth 0.43 s, ground observations 0.96 s, reachability 0.98 s, tasking 1.85 s, total 5.99 s.

## Validation and test results

Run from the repo root with `make test`, or directly:

```bash
cd backend && ../.venv/bin/python -m pytest -n 8 -o addopts=""
```

Result on 2026-10-02 at commit `2d8e1ac` (macOS, Python 3.12.4, 45 test files):

```
518 passed, 1 skipped, 27 warnings in 27.81s
```

The skip is a network-only Horizons discovery test. `tests/test_api_contract.py` covers the routing, readiness, 400/503, budget, timestamp, error-detail, rebuild-gating and CORS conventions described in [API.md](API.md). The frontend builds cleanly with `npm run build` (`tsc --noEmit` and `vite build`).

Key measured numbers, each traceable to a test or docstring. Asserted bounds are deliberately looser than the measurements; "≈" values are what the tests print with `-s`.

| Quantity | Measured or asserted | Where |
|---|---|---|
| Jacobi drift, L1 Lyapunov, 10 periods, DOP853 rtol 1e-12 | < 1e-10 asserted; 6.5e-11 measured | `tests/test_cr3bp.py`, PROGRESS.md |
| CR3BP vs DE440s model, L1 Lyapunov, 2 d, rotating frame | < 1% L* (3,844 km) asserted; 475 km measured | `tests/test_propagate.py`, PROGRESS.md |
| Frame round trip (GCRF to rot to GCRF) | relative error < 1e-9; velocity transform vs numerical derivative < 1e-8 nd (6e-11 measured) | `tests/test_frames.py`, DECISIONS.md |
| Lagrange points | L1 0.836915, L2 1.155682, L3 −1.005063 to 1e-5 | `tests/test_cr3bp.py` |
| STM vs finite differences (CR3BP and N-body) | relative error < 1e-6; det Φ = 1 to 1e-8 | `tests/test_cr3bp.py`, `tests/test_ephemeris.py` |
| BodyCache spline vs direct DE440s | < 1 m (Moon and Sun) | `tests/test_ephemeris.py` |
| Orbit library closure \|X(T) − X₀\|∞ at rtol 2.3e-14 | < 1e-10 for all single-shooting records; < 1e-9 for NRHO and multiple shooting; Jacobi drift < 1e-9 | `tests/test_orbits_library.py`, `data/orbits/README.md` |
| L1 Lyapunov vs Koon, Lo, Marsden & Ross | x₀ = 0.8234 (1e-6), ẏ₀ = 0.1263 (1e-3), T = 2.7430 (1e-3), ν > 1000 (JPL 1180.75) | `tests/test_orbits_library.py` |
| 9:2 synodic NRHO | T = 6.5624 d (2/9 synodic month to 1e-6 d), perilune 3,249 km, apolune 71,222 km, ν = 1.32; literature 6.56 d and about 3,200 to 3,366 km | `tests/test_orbits_library.py`, library record |
| JPL catalog re-convergence | \|ΔT\| < 1e-6, \|ΔIC\| < 1e-7, closure < 1e-9; (C, T) curve distance < 2e-4 (6.1e-7 vs the full catalog); ν within 1% | `tests/test_orbits_jpl.py` |
| Visual magnitude hand check | 1 m radius, albedo 0.2, 400,000 km, φ = 0 gives 18.46 mag | `sensors/photometry.py`, `tests/test_sensors_photometry.py` |
| Batch WLS from a 500 km / 5 m/s error, 20 obs over 2 d | converges in ≤ 8 iterations, every component within 3σ, post-fit RMS 0.5 to 1.6″ at 1″ noise | `tests/test_od_batch.py` |
| UKF, 3-day arc | final position error < 50 km; 10-run Monte Carlo NEES/NIS inside the 95% χ² band for ≥ 75 to 80% of epochs | `tests/test_od_ukf.py` |
| Particle cloud | 2,000 particles × 7 d in 0.07 s; per-particle mismatch vs individual 1e-12 propagation ≤ 0.19 km | PROGRESS.md, `od/stacked.py` |
| False-alarm calibration at α = 0.01 | P_fa ∈ [0.003, 0.03] asserted; 0.0100 over 4,800 updates measured; mean NEES ∈ [4.5, 7.5]; run-level false declaration ≤ 6% | `tests/test_maneuver_detection.py`, PROGRESS.md |
| 5 m/s SIMULATED burn | declared on the first post-burn observation (NIS > 100× threshold); within 2 obs asserted | `tests/test_maneuver_detection.py` |
| 10 m/s Δv estimation from 10 post-burn obs | < 20% magnitude and < 20° direction asserted; ≈ 0.2% and 0.14° measured; t_b error < cadence | `tests/test_maneuver_estimation.py`, PROGRESS.md |
| CUSUM threshold for ARL₀ = 100 | empirical ARL₀ ∈ (60, 160) | `tests/test_maneuver_detection.py` |
| Reachability propagation | 2,000 samples × 72 h in 0.2 s | PROGRESS.md |
| Tasking, default scenario | greedy summed covariance trace < 0.6× round-robin and < 0.01× the no-observation reference; TSLO ≤ baselines | `tests/test_tasking_schedulers.py` |
| Tasking, calibration runs (2 space sensors, 8 objects) | trace-greedy custody 88% vs log-det 81%, round-robin 83%, random 80%; with one sensor and strong process noise no myopic rule beat round-robin (48% vs 40%) | `tasking/greedy.py` docstring |
| Timings | 7-day ephemeris propagation < 2 s; library load < 0.5 s; tasking scenario build < 20 s; route timings in [API.md](API.md) | tests, PROGRESS.md |

## Assumptions and limitations

Drawn from module docstrings and DECISIONS.md.

- **Force model.** Point-mass Earth, Moon and Sun with cannonball SRP. No Earth J2 or lunar gravity field (this matters for low lunar orbits and the ELFO object), no other planets, no relativistic terms. Shadows are cylindrical (no penumbra, finite Sun disc neglected).
- **CR3BP orbits are only quasi-periodic in the real force model.** Notional halo and Lyapunov objects get a one-time epoch-velocity correction (5 to 30 m/s, reported as `ephem_fit_dv_mps`) so they stay within about 10,000 km of their reference orbit over the 14-day window. No station-keeping is simulated. Space-based observers fly the CR3BP orbit mapped through the instantaneous frame (they never drift, and are never the truth model).
- **No light-time or stellar aberration** in the measurement model (about 0.7″ and up to 20.5″ respectively). Synthetic observations are self-consistent, but real astrometry would need reducing to geometric directions before ingest. Each measurement carries `light_time_s`, so the correction is a one-liner downstream.
- **Sensor specifications are assumptions.** Public site coordinates with representative aperture-class limiting magnitudes; no atmospheric extinction, sky brightness, seeing, weather or scheduling downtime. The 3° to 15° lunar-glare zone stands in for a sky-brightness model. Objects are diffuse spheres (no glints, light curves or attitude).
- **Observations are synthetic only.** There is no real-observation ingest; all OD, maneuver and tasking results are closed-loop simulations against the catalog truth with 1″ isotropic noise. Real Horizons objects are tracked by the OD routes but never flagged for maneuvers.
- **Process noise** q = 1e-18 km²/s³ and the initial covariances are design assumptions chosen for filter consistency, not measured values.
- **Linearisation limits.** The EKF and UKF become inconsistent for very long gaps with large uncertainty, which is exactly where the particle cloud is the honest representation. Tasking uses linear covariance analysis (no measurement realisations).
- **Earth orientation** on the frontend uses GMST only (no precession, nutation or polar motion). The backend uses astropy's full IAU 2006/2000A chain with bundled (not live) IERS tables.
- **Horizons objects** are served only within their cached spans (demo window 2026-02-15 to 2026-03-31 for 7 contemporaneous spacecraft; historical windows for 18 more). Interpolation error is reported per object (`interp_leave_one_out_max_km`).
- **Custody metric saturation.** With the tasking API default (10 km prior, 48 h) every object stays in custody without observations, so `custody_pct` only discriminates between policies when the prior is stale or sensors are scarce. `null_custody_pct` and `custody_pct_observed` are reported so this is visible.
- **The MILP objective is a surrogate** (frozen covariances, rank-weighted diminishing returns) and is labelled as such.
- **Architecture detection latency** uses the STM plus sampled-NIS surrogate described above, not the full UKF pipeline per draw.
- **Demo covariance realism is single-run** (see the demo section above).
- **The demo depends on a narrow glare margin.** The single post-burn ground tracklet clears the assumed glare cone by about 0.1°.
- **Synchronous API.** Every route runs within the request. The request budgets in [API.md](API.md) keep the worst in-bounds request to about 10 to 15 s on a laptop; larger requests (for example 8 architectures × 7 d at 5-minute slots, or 300 s MILP budgets) are rejected rather than queued. An asynchronous job model is a roadmap item.
- **Deployment hardening.** CORS defaults to local origins only (`SELENE_CORS_ORIGINS` widens it). There is no authentication anywhere, which is fine for the offline demo but not for the accreditation path in PITCH.md. `POST /api/demo/rebuild` is disabled unless explicitly enabled.
- **Docker is unverified** because Docker was not available on the build machine. The Dockerfile installs from `requirements.lock`, copies the committed data and fetches the kernels at build time, and `.dockerignore` excludes local environments and caches, but none of it has been run.
- **Frontend performance.** 60 fps with about 20k particles is the design target on a discrete GPU; it has not been profiled (headless captures use software GL).
- **Scope.** No real-time telescope control, no conjunction assessment, no attribution, no targeting. Real JPL Horizons objects carry the role "reference / custody object"; the word *target* appears only in sensor-geometry code, never for a named spacecraft.

## Data sources and licences

- **JPL DE440s** planetary and lunar ephemeris (`de440s.bsp`, 1849 to 2150) and **`gm_de440.tpc`** gravitational parameters. NASA/JPL NAIF generic kernels, public domain (U.S. Government work). Park, Folkner, Williams & Boggs (2021), *The JPL Planetary and Lunar Ephemerides DE440 and DE441*, AJ 161:105.
- **JPL Horizons** spacecraft state vectors (`https://ssd.jpl.nasa.gov/api/horizons.api`), geocentric ICRF, TDB, cached under `data/horizons/` with per-object provenance (fetch time, span, step, interpolation quality). Mission navigation products republished by JPL SSD; public. 25 objects are cached: CAPSTONE, LRO, Danuri, the Chandrayaan-2 orbiter, THEMIS-B and C, and TESS (contemporaneous with the demo epoch; these 7 appear in the catalog), plus Artemis I and II, SLIM, EQUULEUS, Chandrayaan-1 and 3, GRAIL-A, LADEE, LCROSS, Lunar Prospector, Clementine, SMILE, Luna-25 and Chang'e boosters (historical windows). The Horizons SLIM merged file chained Moon-centred segments as geocentric; the cache loader splits at implausible-speed discontinuities and keeps the longest consistent segment.
- **JPL Three-Body Periodic Orbit Catalog** (`https://ssd-api.jpl.nasa.gov/periodic_orbits.api`, API v1.0). A compact subset with query URLs and access date is in `data/orbits/jpl_reference.json`; public.
- **IERS tables** bundled with astropy (no live download).
- **Software.** FastAPI, numpy, scipy, numba, jplephem, skyfield, astropy, pyarrow, pandas, pydantic and httpx (Python); React, three.js, react-three-fiber, drei, zustand, Recharts and Vite (JavaScript). All under permissive open-source licences; see `backend/pyproject.toml` and `frontend/package.json`.

## References

Module docstrings give the exact equations used from each source.

- Szebehely (1967), *Theory of Orbits*.
- Koon, Lo, Marsden & Ross (2011), *Dynamical Systems, the Three-Body Problem and Space Mission Design* (realms, necks, Hill's regions, transit and non-transit orbits).
- Richardson (1980), Celestial Mechanics 22 (third-order halo expansion).
- Howell (1984), Celestial Mechanics 32 (halo families, stability index).
- Pavlak (2013), PhD thesis, Purdue University (single and multiple shooting).
- Marchand, Howell & Wilson (2007), Journal of Spacecraft and Rockets 44(4).
- Keller (1977); Doedel et al. (2007) (pseudo-arclength continuation).
- Zimovan-Spreen, Howell & Davis (2020), CMDA 132:28 (9:2 NRHO).
- Lee (2019), NASA/TM-2019-220360 (Gateway NRHO reference).
- Hénon (1969) (DROs); Broucke (1968) (stability indices).
- Ely (2005) and Ely & Lieb (2006), Journal of the Astronautical Sciences (elliptical lunar frozen orbits).
- Whitley & Martinez (2016), IEEE Aerospace Conference (staging-orbit comparison).
- Holzinger, Chow & Garretson (2021), *A Primer on Cislunar Space*, AFRL (regions of interest for cislunar SDA).
- Montenbruck & Gill (2000), *Satellite Orbits*; IERS Conventions 2010.
- Hejduk (AMOS 2011), Krag (1974), McCue et al. (1971), Russell (1916), Hapke (2012) (photometry).
- Gooding (1997), CMDA 66 (angles-only IOD); Tommei, Milani & Rossi (2007); DeMars & Jah (2013).
- Tapley, Schutz & Born (2004), *Statistical Orbit Determination*; Marquardt (1963).
- Julier & Uhlmann (2004); Wan & van der Merwe (2000) (unscented filtering); Bell & Cathey (1993), IEEE TAC 38(2) (iterated update).
- Bar-Shalom, Li & Kirubarajan (2001), *Estimation with Applications to Tracking and Navigation* (NEES/NIS).
- Page (1954); Basseville & Nikiforov (1993) (CUSUM, change detection).
- Swinbank & Purser (2006); González (2010) (Fibonacci sphere).
- Williams, Fisher & Willsky (2007); Hero & Cochran (2011) (sensor management).
