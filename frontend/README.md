# SELENE UI (`selene-ui`)

Vite + React 18 + TypeScript + react-three-fiber frontend for SELENE, cislunar space domain awareness.
Dark ops-center aesthetic; every simulated object/event is labelled SIMULATED and attributed to a notional actor.

## Run

All tooling is project-local (no global installs). Node lives in `../.tools/node/bin`.

```bash
export PATH=/Users/ethanli/selene/.tools/node/bin:$PATH   # or use `make dev` from the repo root
cd frontend
npm install --no-audit --no-fund
npm run dev          # http://127.0.0.1:5173, proxies /api -> http://127.0.0.1:8000
npm run typecheck    # tsc --noEmit
npm run build        # tsc --noEmit && vite build -> dist/  (make build copies it to backend/selene/api/static)
```

Deep links (used by the pitch and by the headless smoke test): `/ops?demo=1` auto-plays the story,
`&t=<seconds>` jumps there and pauses, `&frame=inertial` switches the view, `&view=overview|earth|moon|l1|l2` flies
the camera to a preset, `&layers=exclusion,reach,...` switches scene layers on, `&tab=object|events|brief` picks the
dock tab and `&select=<object id>` selects an object.

## Mock mode vs live

`src/api/client.ts` catches network failures / gateway errors, flips a backend-status store to `offline`, and returns
data from `src/api/mock.ts`. HTTP 4xx/5xx with a real response are thrown as `ApiError` (not mocked) so backend bugs
stay visible. The switch is per call: every call records whether it was served `live` or `mock` under a short endpoint
name, and the banner prints the result verbatim, e.g.
`LIVE: catalog, trajectory, orbits, ephemeris, sensors, coverage, demo, OD, maneuver, reachability, tasking, architecture`
once every screen has fetched its data (a fresh session shows the untouched routes as `AVAILABLE` until a panel calls them).
`LIVE` means the screen actually fetched that data in this session; `AVAILABLE (not wired)` means the route exists on
the backend (OpenAPI probe) but nothing on the current screen calls it yet (so a newly registered `/api/od/run` does not
masquerade as live OD until the UI consumes it); `MOCK` is the browser fallback. Routes other tracks have not written
yet are detected once per session from the backend's OpenAPI document (`/openapi.json`, proxied in dev; one 200
request, no 404 noise), with an invalid-POST probe (422 vs 404) as fallback.

**Custody in the idle (non-scenario) view is `unknown`, shown as `CUSTODY NOT EVALUATED`** — no OD or tasking has
evaluated the catalog objects, so no green "held" status is invented; measured custody states appear inside the scenario
(cloud σ thresholds) and in the Analysis tab's OD / tasking results.

**Live-only analysis.** The OD, maneuver, reachability, tasking and architecture calls have **no browser placeholder**:
`api.odRun` & co. (`requestLive` in `src/api/client.ts`) throw on any failure and the panel shows the backend's
`detail` message inline (a 400 "window longer than 14 days", a 404 unknown object, a 500 with its text). The only
remaining browser fallbacks are the idle-view catalog/orbit shapes (CR3BP in the browser), the coverage layout
placeholder and the **demo scenario**, which is an *explicit* fallback: `startDemo()` tries `GET /api/demo/scenario`
first and only when the backend is **unreachable** (`ApiError.unreachable`: a network failure, a 502–504 gateway
answer, or the empty-body 500 the Vite dev proxy returns while uvicorn is down — never a real 4xx/5xx such as a 404
"bundle not built") plays the browser story, titled `[OFFLINE MOCK] …`, tagged OFFLINE MOCK in the HUD, the metrics
card and the brief footer, and announced by a warning toast. A real backend error is shown as a DEMO ERROR tag with
the backend's message.

**Ephemeris range guard.** `src/lib/ephem.ts` serves exact DE440s geometry (rotating basis, Sun direction, Earth
orientation) only for epochs inside the loaded span (± one sample interval); outside it, every accessor falls back to
the mean-element model instead of clamping or extrapolating, and the HUD says "cursor outside the loaded ephemeris
span". `startDemo()` installs the ephemeris for the scenario span (2026-10-01 + 48 h) before the browser mock scenario
is generated, so its ground-visibility model uses the exact basis; the mock scenario cache is keyed by that state.

| Data | Live (backend up) | Mock (backend down) |
|---|---|---|
| Catalog | `GET /api/catalog/objects` (envelope: epoch, counts, disclaimer; 11 SIMULATED + 7 REAL · JPL HORIZONS objects). The catalog epoch becomes the idle timeline start (7-day span). | 5 notional objects riding the browser CR3BP families (`demo/mockScenario.ts`) |
| Object motion (no scenario) | `GET /api/catalog/objects/{id}/trajectory?frame=rot_nd` over the visible span (`demo/useLiveTracks.ts`: 600 s grid, 150 s for lunar orbiters, n ≤ 5000, 6-wide request queue, keyed cache); cubic Hermite interpolation between samples (`lib/tracks.ts`) at every render tick; trails are the last 24 h (2 h for lunar orbiters) of the track | browser CR3BP display propagation from the catalog state |
| Object panel state vector | `…/trajectory?frame=gcrf_km` for the selected object, Hermite-evaluated at the cursor (position + velocity); ranges from the rot_nd track | derived from rotating-frame display samples |
| Orbit families | `GET /api/orbits/families` (9 families, 12–15 members each of 325, 200 samples per member); thinned to ≤ 5 members per family (3 for the resonant families), NRHO-tagged members always drawn, the 9:2 NRHO highlighted and labelled; legend with per-family show/hide and highlight | `src/lib/cr3bp.ts`: browser CR3BP corrector — L1/L2 Lyapunov, L2 southern halo→NRHO, DRO (25 members, closure < 1e-7) |
| Ephemeris / frames | `GET /api/ephemeris/bodies` (DE440s) incl. the per-epoch rotating-frame basis `R`, `d_km`, `r_bary_km` → exact rotating↔inertial transform, Sun direction, Earth orientation (see below) | mean-element formulas (Meeus) in `src/lib/ephem.ts` |
| Sensors | `GET /api/sensors`: 9 ground sites placed on the Earth (lat/lon, carried by Rᵀ·R_z(GMST)), 6 space observers on their host orbits (`GET /api/orbits/records/{id}` for `orbit.source = library:<id>`, phase convention τ = ((t − epoch)/T* + phase·P) mod P as in the backend; GEO hosts at the exact Earth-fixed geostationary point) | `demo/mockNetwork.ts`: 4 notional ground sites + DRO / L1 / GEO observers |
| Coverage page | `POST /api/coverage` with the preset list from `GET /api/coverage/presets` (8 backend presets) | browser model, 5 studio presets |
| Demo scenario | `GET /api/demo/scenario`: the bundle precomputed by the backend engines (DE440s truth, UKF, particle clouds, reachability, greedy tasker, analyst brief); 145 hourly frames over 6 days, `meta.playback_speed` = 4,320× (≈ 2 min wall-clock + narration holds) | `demo/mockScenario.ts`: 48 h story generated from CR3BP dynamics + computed ground observability, labelled OFFLINE MOCK everywhere it appears |
| Analysis · OD | `POST /api/od/run` (presets `GET /api/od/presets`) | none — inline error |
| Analysis · Maneuver | `POST /api/maneuver/detect` with an injectable SIMULATED burn | none — inline error |
| Analysis · Reach | `POST /api/reachability` (+ `GET /api/reachability/regions` for the region definitions) | none — inline error |
| Analysis · Tasking | `POST /api/tasking/schedule` (presets `GET /api/tasking/presets`) | none — inline error |
| Architecture studio | `POST /api/architecture/evaluate` (studio dialect accepted by the route; `GET /api/architecture/presets` behind "Load backend presets") | browser schematic model, tagged SCHEMATIC MODEL (relative comparison only) |

### Frames (exact when the backend is up)

The scene is authored in the Earth–Moon rotating frame, nondimensional (unit = instantaneous Earth–Moon distance,
Moon pinned at (1−μ, 0, 0)); the Earth and Moon spheres are scaled by L*/d(t) so their radii stay true km.
`/api/ephemeris/bodies` serves, per epoch, the basis `R` (columns x̂ ŷ ẑ in GCRF), `d_km` and `r_bary_km`, so that
`r_gcrf = r_bary + d · R · r_rot`. Between hourly epochs `R` is slerped as a quaternion. The **inertial** view applies

`p_display = R₀ᵀ (r_bary(t) + d(t) R(t) p_rot) / L* + (−μ, 0, 0)`

to the whole synodic group (uniform scale d/L*, rotation R₀ᵀR(t), Earth fixed at (−μ,0,0); R₀ = basis at the timeline
start, so both views coincide at t0 and the Earth–Moon line then sweeps about the display pole along the real
ephemeris). Verified against the served ephemeris: the displayed Moon matches the DE440s Moon to < 1 m, the Earth stays
fixed to 1 mm, and basis-converted GCRF trajectory samples match the served rot_nd samples to 3 m. Earth orientation is
`Rᵀ·R_z(GMST)` (precession/nutation neglected); the Sun direction is `Rᵀ(sun − r_bary)`. Without a basis (backend
down) the view falls back to the planar rotation about z by the mean-element Earth–Moon line angle, and the HUD says so.

### Labels

Every scene label (objects, EARTH/MOON, L-points, the 9:2 NRHO tag, ground sites) registers with
`scene/labels.ts`; `scene/LabelDeclutter.tsx` projects them each tick and hides any label whose DOM box would overlap
an already-placed one, by priority (selected object > Earth/Moon > L1/L2 > NRHO > simulated > real > L3–L5 > sites).
The lunar orbiters therefore show one label at the overview zoom and all of them in the Moon close-up.

### What the browser CR3BP is (and is not)

`src/lib/cr3bp.ts` integrates the Earth–Moon CR3BP (μ = 0.012150585) with fixed-step RK4 and adaptive Dormand–Prince
5(4), finds symmetric periodic orbits by single shooting with a finite-difference Jacobian and backtracking, and
continues them in a natural parameter. Seeds: Koon–Lo–Marsden–Ross L1 Lyapunov (x₀ = 0.8234, ẏ₀ = 0.1263; recovered
T = 2.7429), an approximate L2 Lyapunov seed, an approximate 9:2-NRHO-class L2 southern halo (x₀ = 1.0221,
z₀ = −0.1821; T = 1.5121 ≈ 6.57 d) and a large DRO (x₀ = 0.80). It exists so the UI has real shapes offline; the backend
library (DE440s, STM-based corrector, stability indices) is the system of record. In the mock scenario the uncertainty
cloud is 320 Monte-Carlo seeds propagated through the same dynamics, so the "ballooning" after a detected maneuver is
the honest consequence of injecting a 25 m/s (1σ) Δv uncertainty, and the reachability fractions are computed from
those seeds. Nominal measurement residuals and the Δv-estimate error are seeded pseudo-random draws, labelled SIMULATED.

### What the mock scenario computes (nothing about visibility is scripted)

`src/lib/groundVis.ts` is a port of the backend's observing rules (`backend/selene/sensors/constraints.py`,
`photometry.py`): Sun below −12° at the site, target elevation ≥ 20°, Moon–target separation above the
**phase-dependent lunar-glare zone** (3° at new Moon → 15° at full Moon, a modelling assumption shared with the
backend), Earth/Moon shadow, and a Lambertian-sphere visual magnitude against the site's limiting magnitude. The
frame model (`src/lib/ephem.ts`) places the Earth with the J2000 obliquity relative to the Earth–Moon plane and spins
it with GMST (mean elements; the 5° lunar inclination is neglected), so the Moon's declination at each site is
meaningful. The generator evaluates every site at every 10-min frame, schedules one network observation per 2-h slot
from the best visible site, puts the burn 1 h before the last observation preceding the first ≥ 12 h blind gap,
derives custody from the cloud σ (degraded ≥ 200 km, lost ≥ 1 000 km), and verifies the DRO observer's own
Sun/Moon/Earth exclusions before each tasked observation. With `t0 = 2026-10-01` (Moon 73 % illuminated, glare zone
11.7°) the result is: 6 Teide/Socorro observations in computed night windows, burn T+9 h, detection by Socorro at
T+10 h (90.9″ residual), glare entry at T+10.7 h **from the DRO geometry itself** (the pre-burn orbit enters at
T+10.8 h — the brief says so), custody lost at T+15.5 h, SPC-DRO-A regain at T+23.5 h. The events quote the computed
elevation, Moon separation, glare threshold and magnitude of each observation; the custody-lost event lists each
site's blocking reason. The ground glare cone drawn in the scene uses the same phase-dependent half-angle.

## Architecture studio & Coverage

`/architecture` posts the studio dialect straight to `POST /api/architecture/evaluate` (the route accepts
`ground_network` / `sensors[].orbit` / `slew_rate_dps`; `studio/api.ts` clamps `n_mc` ≤ 16 and the horizon ≤ 7 d and
picks the coarsest slot that satisfies the backend's work-unit budget). Results carry the p05–p95 band over the draws
as whiskers on the coverage / custody / revisit bars, the backend's `method` string and `meta.method_notes` (in the
explainer), caps applied, and the LIVE · BACKEND MONTE CARLO tag; "Load backend presets" replaces the list with the
reference architectures from `GET /api/architecture/presets` (GEO longitude is carried through as `lon_deg`). The
browser schematic model answers only when the backend is unreachable (gateway 502–504 or the dev proxy's empty-body
500) and is tagged SCHEMATIC MODEL; a 404 (route missing on a stale backend) or any real error is shown inline. `/coverage` takes
its preset list from `GET /api/coverage/presets` and runs `POST /api/coverage` (see the page header for the live/mock
source line).

## Scenario driver: frames → scene

`GET /api/demo/scenario` returns `{meta, frames[], events[], metrics, brief}` (the browser mock has the same shape).
`demo/normalize.ts` packs cloud points (`clouds[].points` / `points_rot`) to `Float32Array`, maps `custody_ui` /
`sigma_pos_km`, fills the `boresight_rot`/`pointing_rot` alias, keeps `metrics`, derives `playback_speed` when the
bundle does not carry it and tags the bundle's `source`. `startDemo()` loads it into the store (timeline = the bundle
span, 2026-02-23 → 03-01 for the committed bundle), sets the scripted speed (`meta.playback_speed`, 4,320× → 120 s of
wall-clock for the 144 h story, plus the narration holds), selects the protagonist, opens the Events tab and plays.
The timeline's `‹ ev` / `ev ›` buttons jump to the previous / next story event (observations are skipped) and narrate
just that one. The playback loop (`hooks/usePlayback.ts`) crawls at 1/40 speed while a story event is being narrated
(so "lost" → "tasked" → "regained", one simulated hour apart, each get their dwell) and **coasts at 3×** once the cursor
is past the last story event (the silent tail after the brief), so the whole story takes ≈ 2¼ minutes of wall-clock.
The dock switches to the Brief at the `brief_ready` event and again at the end of the story; every dock tab opens
scrolled to its top (the Brief opens on its title and BLUF, not wherever the Events feed was). The KPI strip reads the
frame at the exact cursor, so custody / σ_pos change in the same instant as the badge, caption and ribbon. On every
render tick:

1. `useScenarioFrame()` binary-searches the frame at the cursor (re-computes only when the frame index changes):
   objects get the bracketing positions, 72-frame trails, custody and σ; clouds expose their `Float32Array`; sensors and
   the reachable set pass through. With no scenario it falls back to `useIdleFrames()` (catalog propagated in-browser).
2. `scene/Objects.tsx` interpolates each object between the bracketing frames imperatively in `useFrame` (no React
   re-render per tick); the HTML label (SIMULATED / HORIZONS tag, custody tag) follows the group.
3. `scene/ParticleCloud.tsx` copies the frame's points into a pre-allocated 32 768-point buffer, colours by a
   Mahalanobis-like radius (diagonal sample covariance) and sets the draw range; additive blending; no per-tick allocation.
   Mock clouds carry 10 240 particles nominally and up to 19 200 while the cloud balloons (≈39 MB precomputed for 289 frames).
4. `scene/Cones.tsx` draws FOV cones from `frame.sensors` (space observers always; ground sites while observing, as thin
   needles from the site to the target) and Sun/Moon/Earth exclusion cones (Sun direction from `lib/ephem.ts`; the
   ground lunar-glare cone follows the Moon's phase); `scene/GroundSites.tsx` carries the sites with the Earth's
   orientation (obliquity + GMST) and lights the active ones. Object and L-point markers are constant pixel size
   (`scene/screenScale.ts`), so they never out-size the Moon in close-ups.
5. `useDemoDriver()` subscribes to the store: events crossed by the cursor become toasts (+ feed highlight) exactly at
   their `t`, alert events auto-select their object, events carrying `data.show_layers` switch those scene layers on
   (custody-lost turns on the exclusion cones), the `brief_ready` event brings the Brief tab forward (and so does the
   end of the span). The Events feed lists only events the story has reached ("Upcoming" toggle for presenters), the
   Brief tab shows an "in progress" card until the brief event, and timeline tooltips do not reveal future event text.
6. `scene/ObservationFlashes.tsx` flashes a line from the observing sensor to its target for every realised tracklet
   in the frame's `observations[]` (rows with a residual; linear-covariance tasking "looks" are not flashed), fading
   over ≈1.2 s of wall-clock. Tasked sensors keep their FOV wedge / line of sight (`frame.sensors[].target_id`).
7. The Object tab's **Scenario metrics** card reads `scenario.metrics` (custody and σ timelines, detection latency,
   time to regain, Δv truth vs estimate) and reveals each figure only once the cursor has passed the moment it refers
   to, so the card never spoils the outcome. Both metric spellings the bundles have used are accepted (`t_rel_s` /
   `t_*_rel_s` in current bundles, `t_s` / `t_*_s` in older ones, `*_utc` as the fallback).

Frame toggle: see "Frames" above — exact DE440s basis when the backend is up, planar mean-element rotation otherwise.

## Analysis tab (Ops dock)

All four panels run **the selected object at the time cursor** (floored to the hour; `select` an object in the scene or
the Object list; the scenario protagonist is the default). The epoch is clamped to the catalogue epoch −30 d … +100 d
(the backend extends its cached truth on demand up to ±120 d and fails beyond) and the panel says when it clamps; the
Run buttons stay disabled until the catalogue epoch has arrived. Every panel shows a spinner while running, then a LIVE
badge with the backend's own `timing.total_s` and the round-trip time; errors are the backend's `detail` text, inline.
Results persist across tab switches (`store/useAnalysis.ts`) and are drawn in the scene. Every result is **stamped with
the object and epoch it was computed for** (`for SIM-DRO-01 @ 02-25 10:00Z` next to the LIVE badge); selecting another
object or moving the cursor to another hour shows a STALE notice with a "Re-run for the current target" link, and a
result for another object is never drawn in the scene (the OD cloud and reachable endpoints are gated on the object
match). A re-run aborts the superseded request (`AbortController`), so the backend does not keep computing an answer
nobody will see.

| Panel | Call | Shows |
|---|---|---|
| OD | `POST /api/od/run` with a preset from `GET /api/od/presets` (`ui_quick`, `custody_week`, `ground_only`), sensor set, window | observation yield and why epochs were dropped; IOD / batch LS / UKF position error vs the SIMULATED truth, σ_pos, RMS; σ_pos + truth-error series (log); NIS (χ²₂ 99 % gate) and NEES (χ²₆ 95 % band) per update with the fraction inside the band; particle-cloud custody decay with a horizon slider — the chosen frame is drawn in the scene in blue (`positions_rot`), the amber cloud stays the scenario's |
| Maneuver | `POST /api/maneuver/detect`; controls: inject a SIMULATED burn (0–100 m/s, one of the backend's ten directions, epoch offset), window, cadence | verdict / latency / updates tiles; NIS series with the χ²₂(1−α) gate, the family-wise gate (value in the legend), the burn and declaration epochs, first exceedance per test; detections table; Δv magnitude ± σ, direction error and burn-epoch error against the injected truth; heuristic class (descriptive only) |
| Reach | `POST /api/reachability` (Δv budget 0–200 m/s, horizon 24–168 h, ≈ sample count) | reachable endpoints in the scene coloured by the first region each ray enters (grey = none) plus the no-burn nominal path; region table (fraction of rays, earliest arrival, min Δv, NEW / nominal tags); sensor-pointing hints (peak single-field capture, first window, BEST) |
| Tasking | `POST /api/tasking/schedule` (method greedy · milp · random · round_robin · compare; sensor preset from `GET /api/tasking/presets`; horizon, slot) over the default xGEO set + the selected SIMULATED object | custody %, zero-observation custody reference, mean / max time-since-last-observation, tracklets; Gantt strip per sensor (one block per assigned slot, coloured by object in fixed order; blocked sensors say so); per-object table; for `compare`, custody and mean-TSLO bars + a table for the four policies |

Real (JPL Horizons) objects: OD and maneuver tests are refused by the backend (no synthetic truth / never a "maneuver"
of a real spacecraft) — the panels say so and keep the button disabled; reachability runs as a what-if envelope and the
backend's disclaimer is shown.

## Layout

```
src/
  api/        client.ts (fetch wrapper, mock fallback, per-endpoint LIVE/MOCK registry, OpenAPI probe), types.ts
              (PLAN.md §5 contract + live shapes), mock.ts
  lib/        cr3bp.ts (integrators, corrector, mock families), ephem.ts (DE440s rotating basis, Sun/GMST, inertial
              display transform, mean-element fallback), tracks.ts (Hermite trajectory tracks), groundVis.ts
  store/      useSelene.ts (zustand: frame, time cursor, playback, selection, layers, events, scenario, dock, toasts)
  hooks/      usePlayback.ts (rAF loop advancing tSec by speed*dt)
  api/analysisTypes.ts   live response shapes of /od/run, /maneuver/detect, /reachability, /tasking/schedule,
                         /architecture/presets (verified against real responses)
  store/useAnalysis.ts   per-engine run state (status, data, error, round-trip ms) + scene-overlay switches
  panels/analysis/       AnalysisPanel (sub-tabs) · OdPanel · ManeuverPanel · ReachabilityPanel · TaskingPanel ·
                         common (target = selected object at the time cursor, RunBar with spinner/LIVE/timing/error) ·
                         palette (validated dark categorical palette; regions and objects keep fixed colour slots)
  panels/ScenarioMetrics.tsx  the Object tab's scenario-metrics card
  scene/AnalysisOverlays.tsx  OD particle cloud at the slider epoch (blue) · reachable endpoints coloured by region
  scene/ObservationFlashes.tsx  sensor→target flashes for realised tracklets
  scene/      SceneRoot (Canvas, exact/planar frame group, camera rig, scale probe, label declutter), Bodies
              (day/night Earth & Moon at true scale, L-points), OrbitFamilies (thinning, NRHO highlight), Objects
              (track sampling, trails, labels), labels.ts + LabelDeclutter, frameBus (display matrix, scale bar),
              ParticleCloud, Cones, GroundSites, Reachable, ReferenceGrid, constants
  panels/     TopBar (frame/layers/legend/demo/reset), Timeline (ticks + tooltips), EventsFeed, ObjectPanel
              (+SigmaSparkline), AnalystBrief, Toasts, CameraPresets, Dock
  pages/      OpsPage (/ops), ArchitecturePage (/architecture), CoveragePage (/coverage)
  demo/       normalize.ts, useScenarioFrame.ts, useIdleObjects.ts (live tracks → frames; CR3BP fallback), useLiveTracks.ts
              (trajectory fetch/cache), useDemoDriver.ts, mockScenario.ts, mockNetwork.ts
  styles/     theme.css (CSS variables; IBM Plex with system fallbacks, works offline)
```

Scene units: 1 scene unit = L* = 384 400 km (CR3BP nondimensional length), z up. Camera near 1e-4, far 50.

## Smoke test (headless Chrome, no extra installs)

```bash
"/Applications/Google Chrome.app/Contents/MacOS/Google Chrome" --headless=new --disable-gpu --use-angle=swiftshader \
  --enable-unsafe-swiftshader --window-size=1600,1000 --virtual-time-budget=15000 --enable-logging=stderr --v=0 \
  --user-data-dir=/tmp/selene-chrome --screenshot=/tmp/selene.png "http://127.0.0.1:5173/ops?demo=1&t=93600"
```

## Known limitations

- In mock mode (backend down) the inertial view is a rotation about the ecliptic pole (lunar inclination neglected) and the
  GCRF state vector is derived from display samples with the mean-element frame model (labelled). Live mode is exact
  (DE440s basis); the state vector is the backend truth/Horizons trajectory, not a filter estimate — the OD endpoints
  (not live yet) will hold the filtered states.
- Earth orientation uses GMST only (no precession/nutation/polar motion); ground sites are on a spherical Earth.
- Objects whose backend trajectory cannot be served in the current window (Horizons span, 120-day extension limit) are
  listed with a NO DATA IN WINDOW tag and not drawn.
- `GET /api/sensors/{id}/visibility` currently returns 500 from the backend (duplicate `sensor_id` kwarg in the
  route); the client exposes it but nothing in the UI depends on it yet.
- Mock-scenario filter behaviour is emulated (isotropic 25 m/s Δv inflation after a detection; fixed re-anchor σ after each
  space observation), not run; the live scenario uses the backend UKF.
- The 3°–15° lunar-glare zone is a modelling assumption (shared with the backend) standing in for a sky-brightness model.
- 60 fps with ~20k particles is the design target on a discrete GPU; it has not been profiled (headless runs use software GL).
- Idle-mode motion is a CR3BP display propagation (no Sun, no SRP, no lunar gravity field); objects whose catalog epoch
  is > 60 days from the timeline are skipped.
- Covariance ellipsoids are shown as particle clouds / σ sparkline, not as a 3-D ellipsoid mesh.
- Coverage and Architecture pages are owned by other tracks.
