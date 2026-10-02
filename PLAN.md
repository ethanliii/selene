# SELENE — Cislunar Space Domain Awareness: Build Plan

> MVP + investor demo. Defensive awareness & traffic-safety product. No targeting / engagement planning.

## 0. Goals and non-goals

**Goal.** A working, offline-capable, one-command app that demonstrates native cislunar SDA:
custody maintenance of xGEO objects, unannounced-maneuver detection, reachability, sensor tasking,
and an architecture trade studio — with real physics (CR3BP + DE440s ephemeris), honest metrics,
and a scripted 2-minute demo story.

**Non-goals.** Weapons targeting, engagement planning, attribution to real actors, real-time
telescope control. All synthetic objects/events are labelled `SIMULATED` and attributed to a
"notional actor".

## 1. Architecture

```
selene/
├── backend/                      Python 3.12, FastAPI
│   └── selene/
│       ├── constants.py          GM (DE440 gm_de440.tpc), CR3BP μ, L*, T*, SI conversions
│       ├── time.py               UTC <-> TDB (astropy), epoch helpers
│       ├── dynamics/
│       │   ├── cr3bp.py          EOM, Jacobi, STM, Lagrange points, event helpers (numba)
│       │   ├── ephemeris.py      DE440s loader (jplephem), Earth+Moon+Sun point-mass EOM, SRP, STM
│       │   ├── frames.py         rotating <-> GCRF (EME2000/J2000) <-> Moon-centered inertial
│       │   └── propagate.py      solve_ivp wrappers (DOP853 default, Radau option), dense output
│       ├── orbits/
│       │   ├── shooting.py       single/multiple shooting differential correction
│       │   ├── continuation.py   natural-parameter + pseudo-arclength continuation
│       │   ├── families.py       L1/L2 Lyapunov, L1/L2 halo (N/S), NRHO (9:2), DRO, resonant, frozen
│       │   ├── stability.py      monodromy eigenvalues, stability index ν = ½(λ+1/λ)
│       │   └── library.py        cached catalog (data/orbits/*.parquet + json)
│       ├── objects/
│       │   ├── notional.py       SIMULATED spacecraft catalog on published orbit types
│       │   └── horizons.py       JPL Horizons client + cache (CAPSTONE, LRO, KPLO, ... as available)
│       ├── sensors/
│       │   ├── sites.py          ground optical sites (configurable), geodetic -> GCRF
│       │   ├── observers.py      space-based observers (GEO, L1/L2 halo, DRO) riding library orbits
│       │   ├── photometry.py     visual magnitude from size/albedo/phase/range; limiting mag
│       │   ├── constraints.py    Sun/Moon/Earth exclusion, Earth shadow/eclipse, elevation, FOV, slew
│       │   ├── visibility.py     per (sensor, object, t) visibility + measurement generation (RA/Dec)
│       │   └── coverage.py       cislunar volume grid coverage heatmap over time (blind spots)
│       ├── od/
│       │   ├── measurements.py   angles-only model h(x), Jacobian H (numerical), noise models
│       │   ├── iod.py            angles-only IOD via two-range shooting under ephemeris dynamics
│       │   ├── batch.py          batch weighted least squares w/ STM, a-priori, iteration control
│       │   ├── ukf.py            unscented Kalman filter (ephemeris dynamics, process noise)
│       │   ├── particles.py      particle-cloud uncertainty propagation (custody decay)
│       │   └── realism.py        NEES/NIS covariance realism checks, chi-square bounds
│       ├── maneuver/
│       │   ├── detection.py      residual/Mahalanobis + NIS/NEES monitors, configurable α (Pfa)
│       │   └── estimation.py     impulsive Δv magnitude/direction/epoch estimation
│       ├── reachability/
│       │   ├── sampling.py       Δv sphere/magnitude/epoch sampling, propagation 24–168 h
│       │   └── regions.py        high-value regions (L1/L2 gateways, NRHO corridor, S-pole, GEO return)
│       ├── tasking/
│       │   ├── information.py    expected information gain (log-det), covariance update ops
│       │   ├── greedy.py         greedy info-theoretic scheduler
│       │   ├── optimize.py       horizon assignment via MILP (scipy.optimize.milp) / local search
│       │   └── metrics.py        custody %, mean time-since-last-obs, covariance trace series
│       ├── architecture/
│       │   ├── candidates.py     candidate sensor orbits (GEO, L1/L2 halo, DRO, resonant)
│       │   └── montecarlo.py     score: coverage %, custody %, revisit, maneuver-detection latency
│       ├── scenario/
│       │   ├── demo.py           scripted 2-minute story -> timeline frames + events
│       │   └── brief.py          plain-English analyst brief generator (template, no LLM needed)
│       └── api/
│           ├── app.py            FastAPI app, CORS, static mount of built frontend
│           ├── schemas.py        pydantic models (API contract, §5)
│           └── routes/           catalog, orbits, ephemeris, sensors, coverage, od, maneuver,
│                                 reachability, tasking, architecture, demo
├── backend/tests/                pytest, one file per module
├── frontend/                     Vite + React + TS + react-three-fiber + drei + zustand + Recharts
│   └── src/
│       ├── api/                  typed client for §5 contract
│       ├── store/                zustand: time cursor, playback, frame mode, selection, events
│       ├── scene/                Earth, Moon, Sun light, Lagrange pts, orbit families, objects+trails,
│       │                         particle clouds, sensor FOV cones, exclusion cones, frame toggle
│       ├── panels/               Timeline, EventsFeed, ObjectPanel, AnalystBrief, CoveragePanel
│       ├── pages/                Ops (default), ArchitectureStudio, Coverage
│       └── demo/                 demo-scenario driver (plays /api/demo/scenario)
├── data/
│   ├── cache/                    de440s.bsp, gm_de440.tpc, horizons raw (gitignored)
│   ├── orbits/                   committed compact orbit catalog
│   ├── horizons/                 committed compact Horizons samples (small JSON)
│   └── demo/                     committed precomputed demo scenario bundle (offline fallback)
├── Makefile                      make setup / dev / test / build / demo
├── docker-compose.yml
├── README.md, PITCH.md, DECISIONS.md, PROGRESS.md, FINAL_REPORT.md
```

**Runtime.** `make dev` starts uvicorn (`:8000`) and Vite (`:5173`, proxying `/api`). `make build`
builds the frontend into `backend/selene/api/static/` so `uvicorn` alone serves everything.
Everything works offline from `data/` (DE440s + cached Horizons + precomputed catalog/demo).

**Heavy computation strategy.** Precompute the orbit library, coverage grids, and the demo scenario
once (`make precompute`) and commit the compact results; API calls for OD/tasking/reachability run
live but are bounded (≤ a few seconds) by design (coarse grids, numba kernels, cached STMs).

## 2. Governing equations

### 2.1 CR3BP (Earth–Moon, rotating/synodic frame, nondimensional)
μ = GM_M / (GM_E + GM_M) (DE440: GM_E = 398600.435507, GM_M = 4902.800118 km³/s² → μ ≈ 0.0121505).
L* = 384 400 km (mean distance), T* = sqrt(L*³/(GM_E+GM_M)) ≈ 3.7519e5 s (so one synodic rev = 2π T*).

Pseudo-potential U = ½(x²+y²) + (1−μ)/r₁ + μ/r₂, r₁ = |(x+μ, y, z)|, r₂ = |(x−1+μ, y, z)|.

ẍ − 2ẏ = ∂U/∂x, ÿ + 2ẋ = ∂U/∂y, z̈ = ∂U/∂z. Jacobi constant C = 2U − (ẋ²+ẏ²+ż²) (conserved).

STM: Φ̇ = A(t)Φ, A = [[0, I],[U_xx, 2Ω]], Ω = [[0,1,0],[−1,0,0],[0,0,0]]. Lagrange points from the
quintic ∂U/∂x = 0 on the x-axis (L1 ≈ 0.83692, L2 ≈ 1.15568, L3 ≈ −1.00506) and the equilateral L4/L5.

### 2.2 Ephemeris model (Earth-centered GCRF, km, s)
r̈ = −GM_E r/|r|³ − Σ_{j∈{Moon,Sun}} GM_j [ (r − r_j)/|r − r_j|³ + r_j/|r_j|³ ] + a_SRP,
positions r_j(t) from DE440s (jplephem, TDB). Optional SRP: a_SRP = ν · P☉ · C_R · (A/m) · (1 AU/|r−r☉|)² · (r−r☉)/|r−r☉|,
P☉ = 4.56e−6 N/m², ν = shadow factor (cylindrical Earth/Moon shadow). Variational equations via
numerical/analytic gradient of the point-mass terms for the STM.

### 2.3 Frames
Instantaneous Earth–Moon rotating frame from the DE440s Earth–Moon state (r, v):
x̂ = r/|r|, ẑ = (r×v)/|r×v|, ŷ = ẑ×x̂, ω = |r×v|/|r|²; origin at the Earth–Moon barycenter;
lengths scaled by the instantaneous |r| so the Moon sits at (1−μ, 0, 0). Velocity transform
includes the ω×r and scale-rate terms. GCRF ≈ J2000 for display; Moon-centered inertial = GCRF
axes translated to the Moon.

### 2.4 Periodic orbits
Symmetric single shooting: X₀ = (x₀, 0, z₀, 0, ẏ₀, 0), integrate to the next y = 0 crossing (event),
drive ẋ(T/2) = ż(T/2) = 0 with Newton steps using the STM and f(X(T/2)). Lyapunov: free ẏ₀ (x₀ fixed).
Halo: free (x₀, ẏ₀) with z₀ fixed, or (z₀, ẏ₀) with x₀ fixed. DRO: planar, retrograde, x₀ fixed, free ẏ₀.
Multiple shooting for sensitive NRHOs. Continuation: natural parameter then pseudo-arclength.
Stability index ν = ½(λ_max + 1/λ_max) from the monodromy matrix. Literature checks (cited in code):
Richardson (1980) third-order halo seed; Koon–Lo–Marsden–Ross (2011) L1 Lyapunov IC
(x₀ = 0.8234, ẏ₀ = 0.1263, T ≈ 2.7430); Zimovan-Spreen/Howell/Davis (2020) and Lee (2019,
NASA Gateway white paper) 9:2 NRHO (period ≈ 6.56 d, perilune radius ≈ 3200–3400 km).

### 2.5 Sensors
Visual magnitude (diffuse sphere, phase angle φ, range R, radius ρ, albedo a):
m = m☉ − 2.5 log₁₀( a · ρ² · F(φ) / R² ), F(φ) = (2/3π)[(π−φ)cos φ + sin φ] · (2/π)... (implemented
as the standard Lambertian-sphere phase function p(φ) = (2/(3π²))[(π−φ)cos φ + sin φ] with
m = −26.74 − 2.5 log₁₀( a ρ² p(φ) / R² ), R and ρ in the same units). Detect if m ≤ m_lim.
Constraints: Sun elevation < −12° at ground sites, target elevation > el_min, angular separations
from Sun/Moon/Earth-limb > exclusion, target not in Earth/Moon shadow (cylindrical), FOV & slew.

### 2.6 Estimation
Measurement: angles-only RA/Dec in GCRF from observer position (ground: site + Earth rotation
via astropy; space: library orbit). H by central differences.
Batch WLS: δx = (HᵀWH + P₀⁻¹)⁻¹(HᵀWδy + P₀⁻¹(x̄ − x)). UKF: scaled sigma points (α=1e-3, β=2, κ=0),
ephemeris propagation, additive process noise Q (acceleration PSD). Particles: N samples of the
posterior propagated with the ephemeris model. Realism: NEES ε = eᵀP⁻¹e over Monte Carlo runs
against χ² bounds; NIS similarly on innovations.

### 2.7 Maneuver detection
Per-measurement NIS d² = νᵀS⁻¹ν with threshold χ²_{m}(1−α); windowed/cumulative NIS; Mahalanobis
distance between predicted and post-fit states; Δv estimation by least squares over (Δv ∈ ℝ³, t_b
on a grid) minimizing post-burn residuals under ephemeris dynamics.

### 2.8 Reachability, tasking, architecture
Reachability: Fibonacci-sphere directions × magnitudes ≤ budget × burn epochs, propagate 24–168 h,
classify against named regions; report fraction and earliest-arrival time per region.
Tasking: expected information gain I = ½ ln(det P⁻ / det P⁺) with the linearized angles update;
greedy per time slot with slew/visibility constraints; compare with a horizon MILP assignment
(scipy `milp`). Metrics: custody % (trace P position < threshold), mean time since last obs.
Architecture: Monte Carlo over object sets/maneuver epochs for each candidate architecture →
coverage %, custody %, revisit time, maneuver-detection latency (mean/95th).

## 3. Milestones and acceptance criteria

| # | Milestone | Acceptance criteria |
|---|-----------|---------------------|
| 0 | Skeleton & tooling | `make test` runs; `make dev` serves `/api/health`; repo on GitHub |
| 1 | Dynamics | Jacobi drift < 1e-10 over 10 periods (DOP853 rtol 1e-12); CR3BP vs ephemeris position error < ~1% of L* over 2 days for an L1 Lyapunov; frame round-trip error < 1e-9 (relative); Lagrange points match literature to 1e-5 |
| 2 | Orbit library | Closure error < 1e-10 (nondim) for every stored orbit; L1 Lyapunov period matches KLMR to 1e-3; NRHO period 6.4–6.7 d, perilune radius 3000–3600 km; DRO family spans ≥ 20 members; families cached to `data/orbits` |
| 3 | Sensors | magnitude model matches hand calc; Sun/Moon exclusion & shadow unit tests; coverage grid endpoint returns a heatmap showing lunar-glare and daylight blind spots |
| 4 | OD | IOD recovers truth within a few % range on synthetic 3-obs arcs; batch LS converges to truth within 3σ; UKF NEES within χ² 95% bounds on ≥ 20 Monte Carlo runs; particle cloud grows without obs |
| 5 | Maneuver detection | Pfa ≈ configured α on no-maneuver runs (±50% at α=0.01, N=200); detects 5 m/s burn within 2 post-burn obs; Δv estimate magnitude within 20%, direction within 20° |
| 6 | Reachability | Sampling returns region hit fractions; 0 m/s budget never leaves the nominal; monotone growth with budget and horizon |
| 7 | Tasking | Greedy beats random on summed trace; MILP ≥ greedy on the horizon objective; metrics reported honestly |
| 8 | Architecture studio | Monte Carlo scores for ≥ 4 candidate architectures; results reproducible with seeds |
| 9 | API + demo scenario | `/api/demo/scenario` returns the full story (frames, events, brief) in < 2 s from cache; all route tests pass |
| 10 | Frontend | 3D scene w/ frame toggle, families, objects+trails, particle clouds, FOV/exclusion cones; timeline + events + object panel; Architecture Studio; Demo button plays the story |
| 11 | Docs & final review | README, PITCH, FINAL_REPORT; full test suite green; app launched; demo run end-to-end |

## 4. Execution strategy
- M0–M1 sequentially (foundation). Then parallel subagents: {M2 orbits, M3 sensors, frontend scaffold}.
  Then {M4 OD, M5 maneuver, M6 reachability}. Then {M7 tasking, M8 architecture, M9 API/demo}.
  Then {M10 frontend}, then M11. Each wave: implement → independent adversarial verify → fix.
- Every milestone ends with `make test` green, app start check, PROGRESS.md update, git commit + push.

## 5. API contract (summary; pydantic in `api/schemas.py`)
- `GET /api/health` → `{status, version, offline:bool}`
- `GET /api/catalog/objects` → `[{id, name, kind: "simulated"|"horizons", orbit_type, actor:"notional", state_gcrf_km, epoch_utc}]`
- `GET /api/orbits/families` → `{families:[{name, members:[{id, ic, period, jacobi, stability, samples_rot:[[x,y,z]...]}]}]}`
- `GET /api/ephemeris/bodies?t0&t1&n` → `{epochs, earth, moon, sun (GCRF km), rot_frame: {moon, l1..l5}}`
- `GET /api/sensors` → `{ground:[...], space:[...]}`
- `POST /api/coverage` `{t0, t1, n_t, grid}` → `{grid, values[t][i]}`
- `POST /api/od/run` `{object_id, t0, t1, sensors, method}` → `{iod, batch, ukf: {epochs, states, covs, nis}, particles}`
- `POST /api/maneuver/detect` `{object_id, ...}` → `{detections:[{t, nis, dv_est, dir, sigma}]}`
- `POST /api/reachability` `{object_id, dv_budget_mps, horizon_h}` → `{points, regions:[{name, fraction, earliest_h}]}`
- `POST /api/tasking/schedule` `{objects, sensors, t0, t1, method}` → `{schedule, custody_pct, mean_tslo_h, trace_series}`
- `POST /api/architecture/evaluate` `{architectures:[{name, sensors:[...]}], n_mc}` → `{scores:[...]}`
- `GET /api/demo/scenario` → `{meta, frames:[{t, objects, clouds, sensors, events}], events, brief}`
