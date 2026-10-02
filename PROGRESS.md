# PROGRESS

| Milestone | Status | Notes |
|-----------|--------|-------|
| 0 Skeleton & tooling | done | pyproject, Makefile, FastAPI app w/ lazy routers, health test, Docker |
| 1 Dynamics | done | CR3BP (numba), DE440s ephemeris N-body + SRP, instantaneous rotating frame, Trajectory; 40 tests. Jacobi drift 6.5e-11/10 periods; CR3BP vs ephemeris 475 km over 2 d |
| 2 Orbit library | pending | |
| 3 Sensors & coverage | pending | |
| 4 Orbit determination | pending | |
| 5 Maneuver detection | pending | |
| 6 Reachability | pending | |
| 7 Sensor tasking | pending | |
| 8 Architecture studio | pending | |
| 9 API + demo scenario | pending | |
| 10 Frontend | scaffold | Vite+React+TS+R3F app, theme, routes, store, typed API client, mock mode, basic 3D scene |
| 11 Docs & final review | pending | |


Data: JPL Horizons cache for 25 spacecraft (CAPSTONE NRHO, LRO, KPLO, Chandrayaan-2, ARTEMIS P1/P2, ...) in data/horizons (offline).
