# PROGRESS

| Milestone | Status | Notes |
|-----------|--------|-------|
| 0 Skeleton & tooling | done | pyproject, Makefile, FastAPI app w/ lazy routers, health test, Docker |
| 1 Dynamics | done | CR3BP (numba), DE440s ephemeris N-body + SRP, instantaneous rotating frame, Trajectory; 40 tests. Jacobi drift 6.5e-11/10 periods; CR3BP vs ephemeris 475 km over 2 d |
| 2 Orbit library | done | 342 periodic orbits in 9 families (L1/L2 Lyapunov, L1/L2 halo N/S incl. NRHOs, DRO, 3:1 & 2:1 resonant); 9:2 NRHO located (T=6.5624 d, perilune 3249 km); validated vs JPL catalog; notional catalog (11 SIMULATED + 7 Horizons) + catalog/orbits/ephemeris routes |
| 3 Sensors & coverage | done | 9 notional ground sites + 6 space observers, Lambertian photometry (hand-verified incl. π factor), Sun/Moon/Earth exclusion, lunar-glare model, eclipse, coverage heatmap route (<1.5 s) |
| 4 Orbit determination | pending | |
| 5 Maneuver detection | pending | |
| 6 Reachability | pending | |
| 7 Sensor tasking | pending | |
| 8 Architecture studio | pending | |
| 9 API + demo scenario | pending | |
| 10 Frontend | in progress | Ops scene layers (families, objects+trails, particle clouds, FOV/exclusion cones), panels, demo driver (mock-first); Coverage page; Architecture Studio page — live wiring to backend pending |
| 11 Docs & final review | pending | |


Data: JPL Horizons cache for 25 spacecraft (CAPSTONE NRHO, LRO, KPLO, Chandrayaan-2, ARTEMIS P1/P2, ...) in data/horizons (offline).
