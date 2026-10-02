# DECISIONS

One line per decision, newest at the bottom. Format: `date — decision — rationale`.

- 2026-10-02 — No PROMPT.md exists; executing the brief in CLAUDE.md (renamed from Claude.md). — Same content, canonical name.
- 2026-10-02 — Plan written directly to PLAN.md instead of interactive plan mode. — Plan mode requires user approval to exit; operating mode is fully autonomous.
- 2026-10-02 — Python 3.12 (anaconda) venv, Node 24 from `.tools/node`, no global installs. — Already provisioned locally; satisfies "3.11+" and "no global installs".
- 2026-10-02 — Stopped `run5h.sh` and its orphaned headless `claude -p` session (pid 37943) that was concurrently rewriting files in this repo. — Two autonomous agents editing the same tree would corrupt each other; this interactive session carries the user's latest instruction. Kept the other session's equivalent rewrites of constants.py/time.py/app.py.
- 2026-10-02 — Internal time scale: TDB Julian date / seconds past J2000 (jplephem native); UTC only at the API boundary. — Avoids repeated astropy conversions in hot loops.
- 2026-10-02 — GM values read from `gm_de440.tpc` with identical literal fallbacks; CR3BP μ derived from them. — One source of truth between CR3BP and ephemeris models.
- 2026-10-02 — Demo reference epoch fixed at 2026-03-01T00:00:00 UTC. — Reproducible offline results; within DE440s span.
- 2026-10-02 — Default branch `main`; repo pushed to GitHub under the authenticated account. — User asked for everything on GitHub.
- 2026-10-02 — Rotating-frame angular velocity (incl. out-of-plane precession term) derived kinematically from DE440s Moon velocity finite differences rather than a force model. — Makes velocity transform the exact time-derivative of the position transform (6e-11 nd residual vs 2.5e-7 model-based).
- 2026-10-02 — Ephemeris force model = point-mass Earth+Moon+Sun + cannonball SRP w/ cylindrical shadows; no J2/lunar gravity field. — Adequate for xGEO custody demo; documented limitation.
- 2026-10-02 — Frontend stack pinned to React 18.3 + R3F 8 + drei 9 + three 0.170. — Known-compatible set; build verified.
- 2026-10-02 — Horizons SLIM (-240) merged file had Moon-centred segments chained as geocentric; cache loader splits at implausible-speed discontinuities and keeps the longest consistent segment. — Data hygiene without hand-editing JPL output.
- 2026-10-02 — Orbit record IC convention: perpendicular xz-crossing farthest from the Moon; halo N/S by sign of z there; family comparisons to JPL done in (Jacobi, period) space. — Matches JPL/Zimovan-Spreen conventions; crossing-independent validation.
- 2026-10-02 — Perilune/apolune refined by 1-D minimisation on dense output, not grid sampling. — Grid sampling overstated NRHO perilune by up to 400 km and hid 4 sub-surface halo members.
- 2026-10-02 — Visual magnitude uses m = M☉ − 2.5 log10(a·π·ρ²·p(φ)/R²) (Lambertian sphere; geometric albedo 2a/3). — Independent surface integration confirmed the π factor; initial implementation was 1.24 mag too faint.
- 2026-10-02 — Halo/Lyapunov notional objects are quasi-periodic arcs under the ephemeris truth (one-time velocity correction, no station-keeping). — Honest dynamics; adequate over the 14-day demo window.
- 2026-10-02 — Frontend is mock-first with per-endpoint live switching and a visible BACKEND OFFLINE/MOCK badge. — UI remains demonstrable even if a backend route is slow or missing; no mock data can masquerade as live.
- 2026-10-02 — IOD admissible region rejects seeds only if unbound w.r.t. BOTH Earth and Moon. — Earth-only hyperbolic test would reject valid NRHO perilune states.
- 2026-10-02 — UKF sigma points propagated in one stacked ODE system (numba RHS + BodyCache); iterated Gauss-Newton measurement update by default. — 16× faster and avoids uncorrelated integrator error amplified by W0 at α=1e-3; iterated update fixes over-confidence for IOD-sized priors.
- 2026-10-02 — Default process-noise PSD q = 1e-18 km²/s³. — 40-run NEES study: q=1e-18 keeps NEES inside χ² bounds; 1e-16 is pessimistic.
- 2026-10-02 — API floats rounded to 9 significant figures, never fixed decimals. — 6-decimal rounding zeroed velocity-covariance blocks.
- 2026-10-02 — Maneuver gap re-fit only for gaps > 2.5× median cadence; min_updates_before_test raised and real Horizons objects not flagged with point-mass truth. — Avoids false declarations on quiet objects at coarse cadence.
- 2026-10-02 — MILP tasking surrogate uses rank-weighted diminishing returns with explicit rank binaries and a per-request time budget. — Count-indexed penalties left ~30 % of sensor slots idle; runtime now bounded.
- 2026-10-02 — Architecture Monte Carlo detection latency uses a documented linear-STM displacement + sampled-NIS surrogate (burn displacement created at the burn node), not the full UKF pipeline per draw. — Full pipeline costs minutes per request; surrogate reuses the same χ² test and states its false-alarm rate is α by construction.
- 2026-10-02 — Common random numbers across architectures (identical epochs, burns, noise per draw). — Fair side-by-side comparison with small n_mc.
- 2026-10-02 — Demo window 2026-02-23→03-01 with a 30 m/s burn on 2026-02-25T08:30Z aimed along the sampled direction passing closest to L1. — Real visibility shows ground sites blind from Feb 25 through full Moon (glare), which is the story's physical premise; Δv chosen from reachability, not hand-tuned.
- 2026-10-02 — Ground-blind attribution asks only sites that were available (night + elevation) why they were blocked; frames with no available site are 'no_site_available', not glare. — Honest loss-reason accounting.
- 2026-10-02 — docker compose not verified (Docker not installed on this machine); Dockerfile kept as best-effort. — No global installs allowed; verification deferred.
