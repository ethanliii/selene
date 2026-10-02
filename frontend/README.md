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

Start the API with `make api` (or `make dev` for both). If the backend is down the UI still renders in
**mock mode**: `src/api/client.ts` catches network failures / 502–504 from the proxy, flips a backend-status
store to `offline`, and returns the minimal placeholder structures from `src/api/mock.ts`. The banner and
viewport then show **BACKEND OFFLINE — MOCK DATA**. HTTP 4xx/5xx with a real response are thrown as
`ApiError` (not mocked) so backend bugs stay visible.

## Layout

```
src/
  api/        client.ts (fetch wrapper, mock fallback, useBackendStatus), types.ts (PLAN.md §5 contract), mock.ts
  store/      useSelene.ts (zustand: frame, time cursor, playback, selection, layers, events, scenario)
  hooks/      usePlayback.ts (rAF loop advancing tSec by speed*dt)
  scene/      SceneRoot (Canvas, controls, stars, frame group), Bodies (Earth/Moon/Sun/Lagrange), ReferenceGrid,
              OrbitFamilies, Objects (+trails), ParticleCloud, Cones (SensorFOVCone, ExclusionCone), constants.ts
  panels/     TopBar, Timeline, EventsFeed, ObjectPanel, AnalystBrief, Dock
  pages/      OpsPage (/ops), ArchitecturePage (/architecture), CoveragePage (/coverage)
  demo/       useScenarioFrame.ts (frame lookup at the time cursor -> scene objects/clouds)
  styles/     theme.css (CSS variables; IBM Plex with system fallbacks, works offline)
```

Scene units: 1 scene unit = L* = 384 400 km (CR3BP nondimensional length). Camera near 1e-4, far 50.
Everything physical is authored in the Earth–Moon rotating frame; the Inertial toggle currently rotates that
group about +z at 2π/(27.28 d)·t as a **placeholder** until the ephemeris endpoint drives the true Earth–Moon line.

## Pending work for later agents (TODO markers in code)

- `scene/OrbitFamilies.tsx` — family coloring, hover/tooltips, decimation.
- `scene/Objects.tsx` — interpolation between frames, covariance ellipsoid, Horizons vs. SIMULATED styling.
- `scene/ParticleCloud.tsx` — crossfade between frames, color by age/weight.
- `scene/Cones.tsx` — FOV cones from frame sensors; exclusion cones from ephemeris Sun/Moon/Earth directions.
- `scene/SceneRoot.tsx` — true inertial (GCRF) view from `/api/ephemeris/bodies`; Sun direction from ephemeris.
- `demo/useScenarioFrame.ts` — interpolation, trails, sensor/exclusion cone derivation.
- `panels/ObjectPanel.tsx` — covariance trace from UKF, observation history, maneuver history, reachability quick-look.
- `pages/ArchitecturePage.tsx` — sensor palette, Monte Carlo run, side-by-side Recharts comparison.
- `pages/CoveragePage.tsx` — heatmap renderer for `POST /api/coverage` with time slider.
