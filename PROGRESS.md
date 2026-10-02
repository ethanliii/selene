# PROGRESS

| Milestone | Status | Notes |
|-----------|--------|-------|
| 0 Skeleton & tooling | done | pyproject, Makefile, FastAPI app w/ lazy routers, health test, Docker |
| 1 Dynamics | done | CR3BP (numba), DE440s ephemeris N-body + SRP, instantaneous rotating frame, Trajectory; 40 tests. Jacobi drift 6.5e-11/10 periods; CR3BP vs ephemeris 475 km over 2 d |
| 2 Orbit library | done | 325 periodic orbits in 9 families (L1/L2 Lyapunov, L1/L2 halo N/S incl. NRHOs, DRO, 3:1 & 2:1 resonant); 9:2 NRHO located (T=6.5624 d, perilune 3249 km); validated vs JPL catalog; notional catalog (11 SIMULATED + 7 Horizons) + catalog/orbits/ephemeris routes |
| 3 Sensors & coverage | done | 9 notional ground sites + 6 space observers, Lambertian photometry (hand-verified incl. π factor), Sun/Moon/Earth exclusion, lunar-glare model, eclipse, coverage heatmap route (<1.5 s) |
| 4 Orbit determination | done | angles-only IOD (two-range shooting under DE440s dynamics, analytic Jacobian), batch WLS, stacked-sigma-point UKF + iterated update, EKF, particle clouds (2000×7 d in 0.07 s), NEES/NIS realism MC; POST /api/od/run ~1 s |
| 5 Maneuver detection | done | per-update/windowed NIS, gap re-fit Mahalanobis, NEES monitor, CUSUM; Pfa 0.0100 at α=0.01 (4800 updates); 5 m/s burn detected on first post-burn obs; Δv est. error ~0.2 % / 0.14° with 10 obs; POST /api/maneuver/detect |
| 6 Reachability | done | 9 named regions (L1/L2 gateway, NRHO corridor, S-pole approach, LLO, GEO return, escape), stacked propagation 2000×72 h in 0.2 s, impact termination by event, sensor pointing hints; POST /api/reachability ~0.3 s |
| 7 Sensor tasking | done | linear-covariance custody engine, greedy log-det scheduler, receding-horizon MILP (HiGHS) with rank-weighted diminishing returns, random/round-robin baselines, custody %/TSLO metrics; POST /api/tasking/schedule ~2 s |
| 8 Architecture studio | done | 6 candidate platforms, 4 presets, common-random-number Monte Carlo: coverage %, greedy-tasker custody %, revisit, detection latency (documented STM+NIS surrogate); POST /api/architecture/evaluate ~2-10 s. e.g. n_mc=4, horizon 3 d, seed 0: ground only 14.0 % coverage / 53.7 % custody → ground + DRO + L1 halo 75.9 % / 91.6 % |
| 9 API + demo scenario | done | 6-day scripted story computed by the real engines (burn 30 m/s on 2026-02-25, detected +1.5 h, lost in lunar glare σ 1053 km, DRO observer tasked, regained +1 h, Δv est. +0.07 %/0.11°); 3.7 MB bundle, 145 frames, 33 events (incl. a closest-approach event; L1/L2 'gateway' = neck transit, not proximity), template analyst brief; GET /api/demo/scenario 0.16 s |
| 10 Frontend | done | Ops scene layers (families, objects+trails, particle clouds, FOV/exclusion cones), panels, demo driver (mock-first); Coverage page; Architecture Studio page — live wiring to backend pending |
| 11 Docs & final review | done (see FINAL_REPORT.md for residuals) | README refreshed; adversarial backend review applied (JSON 404s under /api, 400s for bad input, per-request budgets, Z-suffixed timestamps, API contract test); 511 tests |


Data: JPL Horizons cache for 25 spacecraft (CAPSTONE NRHO, LRO, KPLO, Chandrayaan-2, ARTEMIS P1/P2, ...) in data/horizons (offline).
