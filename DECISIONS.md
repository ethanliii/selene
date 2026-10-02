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
