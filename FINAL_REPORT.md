# SELENE — Final Report (build session of 2026-10-02)

Status: **MVP + investor demo built and working**; the last UI wiring pass (Analysis panels) was still in progress when the session's usage limit was reached — see "What is left".

## What was built
- **Dynamics engine** — Earth–Moon CR3BP (numba), DE440s ephemeris N-body (Earth+Moon+Sun, SRP, shadows), exact instantaneous rotating↔GCRF frames, STMs. Jacobi drift 6.5e-11 over 10 periods; CR3BP vs ephemeris 475 km over 2 days.
- **Periodic-orbit library** — 325 orbits in 9 families (L1/L2 Lyapunov, L1/L2 halo N/S incl. NRHOs, DRO, 3:1 & 2:1 resonant), shooting + pseudo-arclength continuation, stability indices; validated against JPL's periodic-orbit catalog (|ΔT| ≤ 3e-12 on re-convergence). 9:2 NRHO: T = 6.5624 d, perilune 3,249 km.
- **Catalog** — 11 SIMULATED notional objects + 7 real spacecraft from cached JPL Horizons ephemerides (CAPSTONE, LRO, KPLO, Chandrayaan-2, ARTEMIS P1/P2, …), ephemeris-model truth.
- **Sensors** — 9 notional ground sites (assumed specs) + 6 space observers; Lambertian photometry (hand-verified), Sun/Moon/Earth exclusion, lunar-glare model, eclipse; cislunar coverage heatmap with blind-spot reasons.
- **Orbit determination** — angles-only IOD by two-range shooting under ephemeris dynamics, batch WLS, UKF (stacked sigma points, iterated update), EKF, particle clouds, NEES/NIS realism checks.
- **Maneuver detection** — NIS / windowed NIS / gap re-fit Mahalanobis / CUSUM; Pfa 0.0100 at α = 0.01; 5 m/s burn detected on the first post-burn observation; Δv estimate error ~0.2 % / 0.1°.
- **Reachability** — stacked Δv-set propagation (2000 × 72 h in 0.2 s), 9 named regions with a conservative neck-transit definition of the L1/L2 gateways, sensor pointing hints.
- **Sensor tasking** — information-gain greedy scheduler, receding-horizon MILP (HiGHS), random/round-robin baselines, custody % and time-since-last-obs metrics.
- **Architecture trade studio** — Monte Carlo over 6 candidate platforms / 4 presets: coverage, custody, revisit, detection latency (e.g. ground only 14 % coverage / 54 % custody → ground + DRO + L1 halo 76 % / 92 %).
- **Demo scenario** — 6-day story computed by the real engines: 30 m/s unannounced burn on 2026-02-25, detected +1.5 h, custody lost in lunar glare (σ ≈ 1,053 km), DRO observer tasked, custody regained +1 h, Δv characterised (+0.07 %, 0.11°), auto-generated analyst brief.
- **Frontend** — React/Three.js ops console: 3D Earth–Moon scene (rotating/inertial), families, objects + trails, particle clouds, FOV/exclusion cones, timeline with custody ribbon and story pips, events, KPI strip, presenter captions, Coverage page, Architecture Studio (live Monte Carlo), "▶ Demo Scenario" playing the real bundle.
- **Docs** — README.md (setup, architecture, math, limitations), PITCH.md (168 cited sources, fact-checked), PLAN.md, DECISIONS.md, PROGRESS.md.

## How to run
```
make setup      # venv + node_modules + JPL kernels (fetched if missing)
make dev        # API :8000 + Vite :5173     (or: make build && make demo → single uvicorn on :8000)
make test       # backend suite
```
Works offline from `data/` (DE440s cache, Horizons samples, orbit library, demo bundle). Docker files are provided but unverified (Docker not installed here).

## Test results
`cd backend && ../.venv/bin/python -m pytest -q -n 8` → **518 passed, 1 skipped** (last run this session). Frontend: `npm run build` and `tsc --noEmit` clean as of commit `05e4048`.

## What is left (cut off by the usage limit)
1. **Frontend Analysis panels (OD / Maneuver / Reachability / Tasking) wiring** — a subagent was mid-implementation; its uncommitted edits under `frontend/src/` currently fail `tsc`. Either finish them or `git checkout -- frontend` to return to the last green UI (commit `05e4048`, which already plays the real demo bundle and runs the live architecture Monte Carlo).
2. Minor honesty edits flagged by the last verifier: README gateway-semantics paragraph (lines ~245/261), "rays vs samples" wording in the brief, and the "post-burn energy allows transit" label (true but misleading since the DRO's Jacobi constant is already below C_L2).
3. Re-check the demo headline reachability number against the verifier's independent re-derivation (it argues the 24/1728 L1 transits are still near the classifier threshold); the alert/brief text already reports closest approach and counts verbatim.
4. Docker build verification on a machine with Docker.

## Known limitations
Point-mass force model (no lunar gravity field / J2), no light-time or aberration, assumed sensor specs, cylindrical shadows, CR3BP motion for space-observer platforms, synthetic observations only, detection latency in the architecture studio uses a documented STM+NIS surrogate.

## Suggested next steps
Real observation ingest (commercial telescope networks), lunar gravity field + J2 in the truth model, asynchronous job API for long Monte Carlo runs, hosted-payload sensor study, accreditation path (see PITCH.md §7–8).
