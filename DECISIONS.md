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
