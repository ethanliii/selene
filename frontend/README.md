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
stay visible. The switch is per call, so the UI moves to live data transparently as endpoints come up.

| Data | Live (backend up) | Mock (backend down) |
|---|---|---|
| Orbit families | `GET /api/orbits/families` | `src/lib/cr3bp.ts`: browser CR3BP corrector — L1/L2 Lyapunov, L2 southern halo→NRHO, DRO (25 members, closure < 1e-7) |
| Catalog | `GET /api/catalog/objects` | 5 notional objects riding those families (`demo/mockScenario.ts`), with `ic_rot` + derived GCRF state |
| Object motion (no scenario) | catalog state → browser CR3BP display propagation | same |
| Ephemeris angles | `GET /api/ephemeris/bodies` (DE440s) → Earth–Moon line, Sun direction | mean-element formulas (Meeus) in `src/lib/ephem.ts` |
| Sensors | `GET /api/sensors` | `demo/mockNetwork.ts`: 4 notional ground sites (public observatory coordinates, assumed specs) + DRO / L1 / GEO observers |
| Demo scenario | `GET /api/demo/scenario` | `demo/mockScenario.ts`: 48 h story generated from CR3BP dynamics + computed ground observability (≈0.4 s) |
| Coverage / OD / tasking / architecture | live | layout placeholders (zeros), never physics |

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

## Scenario driver: frames → scene

`GET /api/demo/scenario` (or the mock) returns `{meta, frames[], events[], brief}`. `demo/normalize.ts` packs cloud
points to `Float32Array`, fills the `boresight_rot`/`pointing_rot` alias, derives `playback_speed` (duration / 120 s)
and `protagonist_id`. `startDemo()` loads it into the store (timeline reset to the scenario span), sets the scripted
speed, selects the protagonist and plays. On every render tick:

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
5. `useDemoDriver()` subscribes to the store: events crossed by the cursor become toasts (+ feed highlight), alert events
   auto-select their object, events carrying `data.show_layers` switch those scene layers on (custody-lost turns on the
   exclusion cones), and reaching the end switches the dock to the Brief (footer shows `generated <UTC>`). The Events
   feed lists only events the story has reached ("Upcoming" toggle for presenters), the Brief tab is withheld until the
   brief event, and timeline tooltips do not reveal future event text.

Frame toggle: `rotating` keeps the Earth–Moon line on +x; `inertial` rotates the whole synodic group about +z (the
ecliptic pole) by θ(t) − θ(t₀), θ = ecliptic longitude of the Moon from the ephemeris vector when loaded (else the mean
lunar longitude). The lunar orbit inclination is not represented (HUD says "ecliptic-plane rotation"). GCRF values shown
in the Object panel apply the obliquity rotation to that ecliptic-aligned frame.

## Layout

```
src/
  api/        client.ts (fetch wrapper, mock fallback, useBackendStatus), types.ts (PLAN.md §5 contract), mock.ts
  lib/        cr3bp.ts (integrators, corrector, mock families), ephem.ts (Earth–Moon/Sun/GMST angles, obliquity, frame
              conversion), groundVis.ts (ground/space observability rules ported from the backend)
  store/      useSelene.ts (zustand: frame, time cursor, playback, selection, layers, events, scenario, dock, toasts)
  hooks/      usePlayback.ts (rAF loop advancing tSec by speed*dt)
  scene/      SceneRoot (Canvas, frame group, camera rig), Bodies (day/night Earth & Moon, Sun light, L-points),
              OrbitFamilies, Objects (+trails), ParticleCloud, Cones, GroundSites, Reachable, ReferenceGrid, constants
  panels/     TopBar (frame/layers/legend/demo/reset), Timeline (ticks + tooltips), EventsFeed, ObjectPanel
              (+SigmaSparkline), AnalystBrief, Toasts, CameraPresets, Dock
  pages/      OpsPage (/ops), ArchitecturePage (/architecture), CoveragePage (/coverage)
  demo/       normalize.ts, useScenarioFrame.ts, useIdleObjects.ts, useDemoDriver.ts, mockScenario.ts, mockNetwork.ts
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

- Inertial view is a rotation about the ecliptic pole (lunar inclination neglected); the GCRF state vector in the Object
  panel is derived from rotating-frame display samples with the same mean-element frame model (labelled). The backend OD
  endpoints hold the filtered states.
- Mock-scenario filter behaviour is emulated (isotropic 25 m/s Δv inflation after a detection; fixed re-anchor σ after each
  space observation), not run; the live scenario uses the backend UKF.
- The 3°–15° lunar-glare zone is a modelling assumption (shared with the backend) standing in for a sky-brightness model.
- Object labels can overlap at the overview zoom; there is no label collision avoidance (toggle Labels or use a preset).
- 60 fps with ~20k particles is the design target on a discrete GPU; it has not been profiled (headless runs use software GL).
- Idle-mode motion is a CR3BP display propagation (no Sun, no SRP, no lunar gravity field); objects whose catalog epoch
  is > 60 days from the timeline are skipped.
- Covariance ellipsoids are shown as particle clouds / σ sparkline, not as a 3-D ellipsoid mesh.
- Coverage and Architecture pages are owned by other tracks.
