## Operating mode: FULLY AUTONOMOUS
- Never ask me questions or wait for approval. I am not watching. Make the most reasonable decision yourself, record it in DECISIONS.md with a one-line rationale, and keep going.
- First write PLAN.md (architecture, milestones, acceptance criteria per milestone). Then execute every milestone in order without stopping.
- After each milestone: run the tests, start the app, fix anything broken, update PROGRESS.md, and make a git commit with a clear message. Do not move on while tests fail.
- If something is blocked (an API is down, a dependency won't install, a data source needs credentials), use the cached or simulated fallback, note it in DECISIONS.md, and continue. Never stop the whole build over one blocker.
- Skip anything requiring accounts or secrets (e.g. Space-Track); use simulated data instead.
- Work in this folder only. No global installs (use a venv and local node_modules), no sudo, nothing outside the project.
- Use subagents in parallel where work is independent (e.g. frontend vs. dynamics engine), then integrate.
- Before finishing, do a final review pass: run the full test suite, launch the app, run the demo scenario end to end, and fix any defects.
- When completely done, write FINAL_REPORT.md covering what was built, how to run it, test results, known limitations, and suggested next steps. Only then stop.

You are building the MVP + investor demo for "SELENE", a defense startup providing cislunar space domain awareness (SDA): maintaining custody of objects between GEO and the Moon (xGEO), detecting unannounced maneuvers, predicting where objects could go, and designing optimal sensor architectures. Customers: the U.S. Space Force (Cislunar Coordination Office, Space Delta 2), AFRL, defense primes bidding on cislunar SDA programs, and NASA/commercial lunar operators who need traffic safety. This is a defensive awareness and safety product. It does NOT do weapons targeting or engagement planning.

Why this is hard, and therefore valuable: existing SDA tools assume two-body Keplerian motion (TLE/SGP4). In cislunar space, three-body dynamics are chaotic, the search volume is enormous, observations are sparse, and Sun/Moon exclusion zones blind sensors. SELENE is built natively for this regime.

## Before coding
Enter plan mode. Propose the architecture, milestones, and the governing equations (CR3BP, ephemeris model, filters). Then build milestone by milestone, and verify each one (tests pass, app runs) before continuing.

## Stack
- Backend: Python 3.11+, FastAPI, numpy, scipy (solve_ivp with DOP853/Radau), numba where hot, skyfield + jplephem with JPL DE440s ephemerides, astropy for frames and time. Parquet/SQLite cache.
- Frontend: Vite + React + TypeScript + react-three-fiber (Three.js) for a 3D Earth-Moon scene; Recharts for charts. Dark ops-center aesthetic, legible for a classified-briefing-style demo.
- One command to run (`make dev` or `docker compose up`), working offline from cached data.

## Data
- Real ephemerides: JPL DE440s for Earth/Moon/Sun. Query the JPL Horizons API for cislunar and lunar-orbiting spacecraft it publishes (discover what's available and cache it).
- Notional objects: synthetic spacecraft placed in published orbit types (L1/L2 halos, a 9:2 NRHO, DROs, elliptical lunar frozen orbits, resonant orbits touring the Lagrange points). Label them clearly as SIMULATED, and attribute simulated events to a "notional actor", never to a real country or spacecraft.

## Core engines (each a module with unit tests)
1. Dynamics: CR3BP in the Earth-Moon rotating frame, plus a higher-fidelity ephemeris model (Earth + Moon + Sun point masses from DE440s, optional solar radiation pressure). Frame transforms between rotating, Earth-centered inertial (J2000/GCRF), and Moon-centered frames. Tests: Jacobi constant conservation in CR3BP; agreement between CR3BP and ephemeris model over short spans.
2. Periodic-orbit library: compute L1/L2 Lyapunov and halo families, NRHOs, and DROs via single/multiple shooting differential correction and natural-parameter/pseudo-arclength continuation. Store initial conditions, periods, and stability indices. Tests: periodicity closure error below tolerance; compare against values from published literature (cite sources in comments).
3. Sensor models: ground optical telescopes (configurable global sites) and space-based observers (GEO, L1/L2 halo, DRO). Model photometric detectability from object size, albedo, phase angle, and range to visual magnitude vs. limiting magnitude; Sun/Moon/Earth exclusion angles; Earth shadow and eclipse; field of view and slew constraints. Output a coverage heatmap of the cislunar volume over time that shows the blind spots.
4. Orbit determination: angles-only (RA/Dec) initial orbit determination adapted for cislunar dynamics, then batch least squares and an unscented Kalman filter using the ephemeris dynamics. Include covariance realism checks; particle-cloud uncertainty propagation to visualize how custody decays without observations.
5. Maneuver detection: measurement residual and Mahalanobis-distance tests, filter-consistency monitoring (NEES/NIS), and delta-v magnitude/direction estimation, with configurable false-alarm rates.
6. Reachability: given an estimated state and an assumed delta-v budget, sample reachable sets over 24–168 hours and highlight which high-value regions (L1/L2 gateways, NRHO corridor, lunar south pole approaches, GEO belt return paths) become reachable. Purpose: deciding where to point sensors to maintain custody.
7. Sensor tasking: schedule a heterogeneous sensor network to maximize custody (minimize the summed covariance trace / maximize expected information gain) under visibility constraints. Start with greedy information-theoretic scheduling, then compare with a simple optimization approach. Report custody percentage and mean time-since-last-observation per object.
8. Architecture trade studio: let the user place hypothetical space-based sensors in candidate orbits (GEO, L1 halo, L2 halo, DRO, resonant orbits) and score each architecture on coverage %, custody %, revisit time, and maneuver-detection latency via Monte Carlo. This is the sales tool for government architecture studies.

## UI
- 3D scene with toggles for rotating frame vs. inertial frame: Earth, Moon, Lagrange points, the periodic-orbit families, objects with trails, uncertainty particle clouds, sensor fields of view, and Sun/Moon exclusion cones.
- Timeline scrubber with playback speed; events feed (maneuver detected, custody lost/regained, entered sensitive region).
- Object panel: state, orbit type, covariance ellipsoid, observation history, maneuver history, custody status.
- Architecture studio page: drag sensors onto orbits, run Monte Carlo, compare architectures side by side.
- "Demo scenario" button running a scripted 2-minute story: a notional object sits quietly in a DRO; it performs an unannounced burn; ground telescopes lose it in lunar glare and its uncertainty cloud balloons; reachability shows it could reach the L1 gateway corridor near a notional allied relay in NRHO; the SELENE tasker redirects a DRO-based observer; custody is regained and the maneuver characterized; an auto-generated plain-English analyst brief appears.

## Quality bar
- Real physics, SI units, documented assumptions. No invented numbers presented as real.
- Honest metrics, even when custody is poor (poor custody is the market problem, not something to hide).

## Deliverables
- README covering setup, architecture, the dynamics/filters used, and assumptions and limitations.
- PITCH.md covering:
  - The problem, citing public statements and programs (Space Force cislunar strategy, Cislunar Coordination Office, AFRL Oracle).
  - Why now, and the customer list.
  - Competitive landscape for the founders to research and verify.
  - The moat: native cislunar dynamics, sensor tasking, and an architecture design tool.
  - Go-to-market: SBIR/STTR and AFWERX/SpaceWERX pathways to program of record.
  - Roadmap: real observation ingest from commercial telescope networks; a hosted payload for a space-based sensor; accreditation path.

Start with the plan, then build Milestone 1.
