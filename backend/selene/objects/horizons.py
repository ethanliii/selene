"""JPL Horizons client + offline cache for real cislunar / lunar-orbiting spacecraft.

What this module does
---------------------
* **Discovery** (network): asks the JPL Horizons API which spacecraft it knows about
  (``COMMAND='SPACECRAFT*'`` wildcard + a list of mission-name probes) and probes a
  curated candidate list of cislunar objects for state-vector coverage around the demo
  epoch (2026-03-01, window 2026-02-15 .. 2026-03-31 UTC).
* **Caching** (network): for each candidate with data, downloads geocentric ICRF state
  vectors (``CENTER='500@399'``, ``REF_PLANE='FRAME'``, ``REF_SYSTEM='ICRF'``,
  ``VEC_TABLE='2'``, ``OUT_UNITS='KM-S'``, ``TIME_TYPE='TDB'``), stores the raw response
  under ``data/cache/horizons/<id>.txt`` (git-ignored) and a compact rounded JSON under
  ``data/horizons/<id>.json`` (committed, < 1 MB each), plus ``data/horizons/index.json``.
* **Offline use** (no network): :func:`load_horizons_objects` reads only the committed JSON
  and returns :class:`HorizonsObject` instances with interpolated ``state_at(t)``.

Units and conventions
---------------------
* Positions in **km**, velocities in **km/s**, time as **TDB seconds past J2000.0**
  (``t = (JD_TDB - 2451545.0) * 86400``), frame **ICRF/J2000 axes, Earth-centred**
  (Horizons centre ``500@399`` = geocentre). JSON values are rounded to 1e-3 km and
  1e-6 km/s (position/velocity) and 1e-3 s (time) to keep files small.
* Horizons spacecraft trajectories are the mission navigation teams' reconstructed /
  predicted ephemerides as republished by JPL SSD. They are *real data*, not simulated;
  nothing in this module invents states.

Interpolation
-------------
Because Horizons gives both position and velocity, ``state_at`` uses piecewise cubic
**Hermite** interpolation on each axis (``scipy.interpolate.CubicHermiteSpline``), which is
far more accurate than a position-only spline at a given sample spacing; a position-only
``CubicSpline`` is available via ``method="spline"``. Sample steps are chosen per object
(10 min for low lunar orbiters whose period is ~2 h and for the CAPSTONE NRHO / ARTEMIS
P1-P2 perilune passes, 5 min for Artemis II, 1 h for slow xGEO and historical objects; see
``CANDIDATES``). The leave-one-out interpolation error (effectively a 2x-step test) is
computed at refresh time and stored in the metadata so users can judge fidelity honestly.

Data hygiene
------------
Horizons "merged" spacecraft files occasionally contain non-physical rows: samples past a
lunar impact / re-entry (inside a body), or a mis-chained segment that jumps hundreds of
thousands of km between consecutive samples (e.g. ``SLIM_merged`` 2023-12-26..2024-01-08,
whose rows satisfy |r - 2 r_Moon| ~ 2-6 km, i.e. a Moon-centred segment reported as
geocentric). :func:`trim_inside_bodies` drops the former; :func:`split_at_discontinuities`
cuts the table wherever the implied chord speed |dr|/dt exceeds ``MAX_PLAUSIBLE_SPEED_KMS``
(12 km/s: nothing bound in the Earth-Moon system moves that fast) and keeps the longest
physically consistent segment. Everything dropped is counted in the metadata. No value is
ever "corrected" - rows are only kept or dropped.

Politeness: <= 1 request/s, 20 s timeout, 2 retries. All network failures are logged and
the existing cache is kept untouched. ``--rebuild-from-cache`` re-derives the compact JSON
from the raw responses without any network access.

References
----------
* Giorgini, J.D. et al., "JPL's On-Line Solar System Data Service", BAAS 28(3), 1158 (1996).
* Horizons API docs: https://ssd-api.jpl.nasa.gov/doc/horizons.html
* Horizons manual (spacecraft IDs, ``SPACECRAFT*`` wildcard): https://ssd.jpl.nasa.gov/horizons/manual.html
"""
from __future__ import annotations

import argparse
import json
import logging
import re
import sys
import time as _time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

from selene import time as stime
from selene.constants import CACHE_DIR, DATA_DIR, DE440S_PATH, GEO_RADIUS_KM, R_EARTH, R_MOON

log = logging.getLogger("selene.horizons")

HORIZONS_API = "https://ssd.jpl.nasa.gov/api/horizons.api"
HORIZONS_DATA_DIR = DATA_DIR / "horizons"
HORIZONS_CACHE_DIR = CACHE_DIR / "horizons"

#: Demo epoch and discovery window (UTC ISO).
DEMO_EPOCH_UTC = "2026-03-01T00:00:00"
WINDOW_UTC = ("2026-02-15T00:00:00", "2026-03-31T00:00:00")
FALLBACK_SPAN_DAYS = 45.0

REQUEST_INTERVAL_S = 1.0
REQUEST_TIMEOUT_S = 20.0
REQUEST_RETRIES = 2

#: Geocentric-range bounds (km) used to decide whether an object is in the
#: Earth-Moon / xGEO regime SELENE cares about. Objects whose maximum geocentric range
#: exceeds the upper bound (Sun-Earth L1/L2 halos at ~1.5-2.1e6 km, heliocentric cruise)
#: are recorded in the index but not cached.
CISLUNAR_MAX_RANGE_KM = 2.0e6
CISLUNAR_MIN_APOGEE_KM = 1.0e5   # must reach well beyond GEO (42 164 km) to be "xGEO"

# Selenocentric range thresholds for regime labelling (km).
LUNAR_ORBIT_MAX_SELENO_KM = 30_000.0    # LRO, Chandrayaan-2, Danuri, ARTEMIS P1/P2 (apolune < ~20 000 km)
LUNAR_HALO_MAX_SELENO_KM = 100_000.0    # CAPSTONE 9:2 NRHO (apolune ~70 000 km)

#: Largest chord speed |r(k+1)-r(k)| / dt (km/s) a real Earth-Moon-system trajectory can show
#: between consecutive samples. LEO circular speed is 7.8 km/s and escape speed at LEO
#: 11.2 km/s; the chord speed of a smooth arc never exceeds the true peak speed, so anything
#: above 12 km/s is a table discontinuity (merged-ephemeris artifact), not motion.
MAX_PLAUSIBLE_SPEED_KMS = 12.0


# ---------------------------------------------------------------------------
# Candidate list
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Candidate:
    """A Horizons spacecraft to probe.

    scope: "cislunar" -> cache if it has data in / near the window and passes the range rule;
           "context"  -> probe only (1-day step) to record its regime/range in the index.
    step:  Horizons STEP_SIZE string used for the cached table.
    """

    id: int
    name: str
    regime_hint: str
    step: str = "1 h"
    scope: str = "cislunar"
    note: str = ""


CANDIDATES: tuple[Candidate, ...] = (
    # --- expected active in the window ---------------------------------------------------
    Candidate(-1176, "CAPSTONE", "nrho", "10 m", note="12U CubeSat in the Gateway 9:2 L2 southern NRHO; Horizons data through 2026-08-14 (mission completed 2026-07)."),
    Candidate(-85, "LRO", "lunar_orbit", "10 m", note="Lunar Reconnaissance Orbiter, ~50 km x 200 km polar lunar orbit, period ~113 min."),
    Candidate(-152, "Chandrayaan-2 Orbiter", "lunar_orbit", "10 m", note="ISRO, ~100 km polar lunar orbit."),
    Candidate(-155, "Danuri (KPLO)", "lunar_orbit", "10 m", note="Korea Pathfinder Lunar Orbiter, ~100 km polar lunar orbit."),
    Candidate(-192, "ARTEMIS-P1 (THEMIS-B)", "lunar_orbit", "10 m", note="Highly elliptical, high-inclination lunar orbit (~26 h period) since 2011."),
    Candidate(-193, "ARTEMIS-P2 (THEMIS-C)", "lunar_orbit", "10 m", note="Highly elliptical lunar orbit (~26 h period) since 2011."),
    Candidate(-95, "TESS", "xgeo_heo", "1 h", note="2:1 lunar-resonant HEO (perigee ~1e5 km, apogee ~3.7e5 km): a real xGEO custody case."),
    # --- 2026 missions that may begin after the window (cache first 45 d if so) -----------
    Candidate(-1024, "Artemis II (Orion)", "cislunar_transit", "5 m", note="Crewed hybrid free-return lunar flyby; designation 2026-069A."),
    Candidate(-168540, "SWC-1", "cislunar_transit", "1 h", note="Designation 2026-069C (same launch as Artemis II); regime classified from data."),
    Candidate(-463, "SMILE", "xgeo_heo", "1 h", note="ESA/CAS Solar wind Magnetosphere Ionosphere Link Explorer, HEO (apogee ~1.2e5 km); designation 2026-109A."),
    Candidate(-169792, "LINK", "unknown", "1 h", note="Designation 2026-152A; regime classified from data."),
    # --- completed lunar missions (expected: no data in window) --------------------------
    Candidate(-169, "Chandrayaan-3 Propulsion Module", "lunar_orbit", "1 h", note="Returned to high Earth orbit in late 2023."),
    Candidate(-158, "Chandrayaan-3 Lander (Vikram)", "lunar_lander", "1 h"),
    Candidate(-153, "Chandrayaan-2 Lander", "lunar_lander", "1 h"),
    Candidate(-86, "Chandrayaan-1", "lunar_orbit", "1 h"),
    Candidate(-12, "LADEE", "lunar_orbit", "1 h"),
    Candidate(-75, "OMOTENASHI", "cislunar_transit", "1 h"),
    Candidate(-101, "EQUULEUS", "cislunar_transit", "1 h", note="Flew to Sun-Earth L2 after Artemis I."),
    Candidate(-125, "ICPS (Artemis I upper stage)", "cislunar_transit", "1 h"),
    Candidate(-164, "Lunar Flashlight", "cislunar_transit", "1 h", scope="context"),
    Candidate(-240, "SLIM", "lunar_lander", "1 h"),
    Candidate(-242, "Lunar Trailblazer", "cislunar_transit", "1 h", scope="context", note="Lost after launch 2025-02-27; predicted heliocentric drift only."),
    Candidate(-1023, "Artemis I (Orion)", "cislunar_transit", "1 h"),
    Candidate(-182, "NEA Scout", "cislunar_transit", "1 h"),
    Candidate(-70007, "BioSentinel", "cislunar_transit", "1 h", scope="context", note="Artemis I secondary; heliocentric."),
    Candidate(-10001001, "Odin (AstroForge)", "cislunar_transit", "1 h", scope="context", note="Launched 2025-02-27 with IM-2; contact lost."),
    Candidate(-78000, "Chang'e 5-T1 booster (WE0913A)", "cislunar_debris", "1 h", note="Uncontrolled upper stage (initially mis-identified as a Falcon 9 stage) that impacted the lunar far side 2022-03-04 ~12:25 UTC; the Horizons file extends a few minutes past impact, so the final sample lies inside the Moon."),
    Candidate(-139459, "Chang'e 3 booster", "cislunar_debris", "1 h"),
    Candidate(-143846, "Chang'e 4 booster", "cislunar_debris", "1 h"),
    Candidate(-9901492, "Luna-25 stage", "cislunar_debris", "1 h"),
    Candidate(-25, "Lunar Prospector", "lunar_orbit", "1 h"),
    Candidate(-177, "GRAIL-A (Ebb)", "lunar_orbit", "1 h"),
    Candidate(-18, "LCROSS Shepherd", "cislunar_transit", "1 h"),
    Candidate(-40, "Clementine", "lunar_orbit", "1 h"),
    # --- active but outside the Earth-Moon regime (Sun-Earth L1/L2, ~1.5-2.1e6 km) --------
    Candidate(-9, "ESCAPADE-Blue", "sun_earth_l2", "1 d", scope="context", note="Loitering near Sun-Earth L2 until the late-2026 Earth flyby."),
    Candidate(-10, "ESCAPADE-Gold", "sun_earth_l2", "1 d", scope="context"),
    Candidate(-170, "James Webb Space Telescope", "sun_earth_l2", "1 d", scope="context"),
    Candidate(-43, "IMAP", "sun_earth_l1", "1 d", scope="context"),
    Candidate(-231, "SWFO-L1", "sun_earth_l1", "1 d", scope="context"),
    Candidate(-171, "Carruthers Geocorona Observatory", "sun_earth_l1", "1 d", scope="context"),
)

#: Mission names searched by string (Horizons returns candidate lists for ambiguous names
#: or a small-body "No matches found" page when it knows no such major body).
NAME_PROBES: tuple[str, ...] = (
    "GATEWAY", "Queqiao", "Danuri", "LADEE", "ARTEMIS", "Lunar Pathfinder", "Chang", "Hakuto",
    "Peregrine", "Odysseus", "Nova-C", "Athena", "Blue Ghost", "Trailblazer", "LunaH", "EQUULEUS",
    "OMOTENASHI", "ICPS", "Orion", "Artemis II", "Chandrayaan", "SLIM", "Lunar Flashlight", "TESS",
    "IM-2", "Griffin", "Resilience", "THEMIS", "Luna", "SMILE",
)

# Hand-curated reasons for names Horizons does not carry (verified 2026-10-02 probe).
_UNKNOWN_NAME_REASONS = {
    "GATEWAY": "Not yet launched (Gateway PPE/HALO); no Horizons record.",
    "Queqiao": "CNSA relay satellites (Queqiao-1 at EM-L2 halo, Queqiao-2 in lunar frozen orbit) are not published by Horizons.",
    "Lunar Pathfinder": "ESA/SSTL relay not in Horizons.",
    "Hakuto": "ispace HAKUTO-R M1 (2023) / M2 Resilience (2025) landers crashed; never in Horizons.",
    "Peregrine": "Astrobotic Peregrine (2024) re-entered; not in Horizons.",
    "Nova-C": "Intuitive Machines IM-1 Odysseus (2024) / IM-2 Athena (2025) ended on the surface; not in Horizons.",
    "Odysseus": "Horizons string match resolves to asteroid (1143) Odysseus, not the IM-1 lander.",
    "Athena": "Not in Horizons (IM-2 lander ended 2025-03).",
    "Blue Ghost": "Firefly Blue Ghost M1 (landed 2025-03-02, ended 2025-03-16) not in Horizons.",
    "LunaH": "LunaH-Map (Artemis I CubeSat) not in Horizons.",
    "IM-2": "Not in Horizons.",
    "Griffin": "Horizons string match resolves to asteroid (4995) Griffin; Astrobotic Griffin lander not in Horizons.",
    "Resilience": "ispace M2 Resilience not in Horizons (crashed 2025-06-05).",
}


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------
@dataclass
class HorizonsObject:
    """Cached Horizons state-vector table for one spacecraft.

    Attributes
    ----------
    id : Horizons body ID (negative for spacecraft).
    name : Horizons target name.
    t_tdb_s : (N,) TDB seconds past J2000.0, strictly increasing.
    states : (N, 6) geocentric ICRF [x, y, z (km), vx, vy, vz (km/s)].
    span : (start_utc_iso, end_utc_iso) of the cached table.
    notes : free-text provenance notes.
    metadata : dict with regime, step, ranges, fetched_utc, span_note, ... (see JSON).
    """

    id: int
    name: str
    t_tdb_s: np.ndarray
    states: np.ndarray
    span: tuple[str, str]
    notes: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    _hermite: Any = field(default=None, repr=False, compare=False)
    _spline: Any = field(default=None, repr=False, compare=False)

    # -- basic properties --------------------------------------------------
    def __len__(self) -> int:
        return int(self.t_tdb_s.shape[0])

    @property
    def t0(self) -> float:
        return float(self.t_tdb_s[0])

    @property
    def t1(self) -> float:
        return float(self.t_tdb_s[-1])

    @property
    def regime(self) -> str:
        return str(self.metadata.get("regime", "unknown"))

    @property
    def kind(self) -> str:
        return "horizons"

    @property
    def contemporaneous(self) -> bool:
        """True if the cached span overlaps the discovery window (object active at the demo epoch era)."""
        return bool(self.metadata.get("contemporaneous", True))

    def in_span(self, t_s: float) -> bool:
        return self.t0 <= float(t_s) <= self.t1

    @property
    def geocentric_range_km(self) -> np.ndarray:
        return np.linalg.norm(self.states[:, :3], axis=1)

    # -- interpolation -----------------------------------------------------
    def _build_hermite(self):
        from scipy.interpolate import CubicHermiteSpline

        return CubicHermiteSpline(self.t_tdb_s, self.states[:, :3], self.states[:, 3:], axis=0)

    def _build_spline(self):
        from scipy.interpolate import CubicSpline

        return CubicSpline(self.t_tdb_s, self.states, axis=0)

    def states_at(self, t_s: np.ndarray, method: str = "hermite") -> np.ndarray:
        """Vectorised interpolation. Rows outside the span are NaN."""
        t = np.atleast_1d(np.asarray(t_s, dtype=float))
        out = np.full((t.shape[0], 6), np.nan)
        ok = (t >= self.t0) & (t <= self.t1)
        if not ok.any():
            return out
        if method == "hermite":
            if self._hermite is None:
                self._hermite = self._build_hermite()
            out[ok, :3] = self._hermite(t[ok])
            out[ok, 3:] = self._hermite.derivative()(t[ok])
        elif method == "spline":
            if self._spline is None:
                self._spline = self._build_spline()
            out[ok] = self._spline(t[ok])
        else:
            raise ValueError(f"unknown interpolation method {method!r}")
        return out

    def state_at(self, t_s: float, method: str = "hermite", strict: bool = False) -> np.ndarray | None:
        """Geocentric ICRF state [km, km/s] at TDB seconds past J2000.

        Returns ``None`` outside the cached span (or raises ``ValueError`` if ``strict``).
        """
        t = float(t_s)
        if not self.in_span(t):
            if strict:
                raise ValueError(
                    f"{self.name} ({self.id}): t={t:.1f} s outside cached span [{self.t0:.1f}, {self.t1:.1f}]"
                )
            return None
        return self.states_at(np.array([t]), method=method)[0]

    def state_at_utc(self, utc, **kw) -> np.ndarray | None:
        return self.state_at(stime.seconds_since_j2000_tdb(utc), **kw)

    def leave_one_out_error_km(self, method: str = "hermite") -> float:
        """Max position error when each interior sample is predicted from its neighbours
        (an interpolation test at twice the nominal step; the nominal-step error is
        roughly 1/16 of this for a cubic)."""
        n = len(self)
        if n < 6:
            return float("nan")
        idx = np.arange(1, n - 1, 2)
        keep = np.ones(n, dtype=bool)
        keep[idx] = False
        sub = HorizonsObject(self.id, self.name, self.t_tdb_s[keep], self.states[keep], self.span)
        pred = sub.states_at(self.t_tdb_s[idx], method=method)
        err = np.linalg.norm(pred[:, :3] - self.states[idx, :3], axis=1)
        return float(np.nanmax(err))

    # -- (de)serialisation -------------------------------------------------
    def to_json_dict(self) -> dict[str, Any]:
        return {
            "id": int(self.id),
            "name": self.name,
            "source": "JPL Horizons",
            "api": HORIZONS_API,
            "fetched_utc": self.metadata.get("fetched_utc"),
            "center": "Earth (500@399)",
            "frame": "ICRF",
            "units": "km, km/s",
            "time": "TDB seconds past J2000.0 (JD_TDB 2451545.0)",
            "step": self.metadata.get("step"),
            "regime": self.metadata.get("regime"),
            "span_utc": list(self.span),
            "span_note": self.metadata.get("span_note"),
            "n_samples": len(self),
            "metadata": {k: v for k, v in self.metadata.items() if k not in ("fetched_utc", "step", "regime", "span_note")},
            "notes": list(self.notes),
            "t_tdb_s": [round(float(t), 3) for t in self.t_tdb_s],
            "states": [
                [round(float(s[0]), 3), round(float(s[1]), 3), round(float(s[2]), 3),
                 round(float(s[3]), 6), round(float(s[4]), 6), round(float(s[5]), 6)]
                for s in self.states
            ],
        }

    @classmethod
    def from_json_dict(cls, d: dict[str, Any]) -> "HorizonsObject":
        t = np.asarray(d["t_tdb_s"], dtype=float)
        s = np.asarray(d["states"], dtype=float).reshape(-1, 6)
        if t.shape[0] != s.shape[0]:
            raise ValueError(f"Horizons JSON for {d.get('id')}: {t.shape[0]} epochs vs {s.shape[0]} states")
        meta = dict(d.get("metadata", {}))
        for k in ("fetched_utc", "step", "regime", "span_note"):
            if d.get(k) is not None:
                meta[k] = d[k]
        meta.setdefault("source", d.get("source", "JPL Horizons"))
        return cls(
            id=int(d["id"]),
            name=str(d["name"]),
            t_tdb_s=t,
            states=s,
            span=(str(d["span_utc"][0]), str(d["span_utc"][1])),
            notes=list(d.get("notes", [])),
            metadata=meta,
        )

    def save_json(self, path: Path) -> int:
        path.parent.mkdir(parents=True, exist_ok=True)
        txt = json.dumps(self.to_json_dict(), separators=(",", ":"))
        path.write_text(txt)
        return len(txt.encode())


# ---------------------------------------------------------------------------
# Offline loading
# ---------------------------------------------------------------------------
def load_index(data_dir: Path | str | None = None) -> dict[str, Any]:
    p = Path(data_dir or HORIZONS_DATA_DIR) / "index.json"
    if not p.exists():
        return {"objects": [], "not_cached": [], "name_searches": []}
    return json.loads(p.read_text())


def load_horizons_objects(data_dir: Path | str | None = None, contemporaneous_only: bool = False,
                          at_utc: str | None = None) -> list[HorizonsObject]:
    """Read every committed ``<id>.json`` (no network).

    Order (see :func:`catalog_sort_key`): objects active in the demo window first, then
    historical fallback tables; within each group by Horizons ID ascending numerically
    (most negative first), so with the shipped cache ``objs[0]`` is CAPSTONE (-1176) and the
    2014 LADEE table (-12) is never the headline object.

    contemporaneous_only: keep only objects whose Horizons span overlaps the discovery window
        (drops historical fallback spans such as LADEE 2014 or the Chang'e 5-T1 booster 2022).
    at_utc: keep only objects whose cached span contains this UTC epoch.
    """
    d = Path(data_dir or HORIZONS_DATA_DIR)
    objs: list[HorizonsObject] = []
    for p in sorted(d.glob("*.json")):
        if p.name == "index.json":
            continue
        try:
            objs.append(HorizonsObject.from_json_dict(json.loads(p.read_text())))
        except Exception as exc:  # noqa: BLE001 - a corrupt file must not break the catalog
            log.warning("skipping unreadable Horizons file %s: %s", p, exc)
    if contemporaneous_only:
        objs = [o for o in objs if o.metadata.get("contemporaneous", True)]
    if at_utc is not None:
        t = stime.seconds_since_j2000_tdb(at_utc)
        objs = [o for o in objs if o.in_span(t)]
    objs.sort(key=lambda o: catalog_sort_key(o.id, o.contemporaneous))
    return objs


def catalog_sort_key(obj_id: int, contemporaneous: bool) -> tuple[int, int]:
    """Catalog order: demo-window objects before historical ones, then ID ascending
    (negative spacecraft IDs: -1176 CAPSTONE < -193 < -192 < -155 < -152 < -95 < -85)."""
    return (0 if contemporaneous else 1, int(obj_id))


def get_horizons_object(obj_id: int, data_dir: Path | str | None = None) -> HorizonsObject | None:
    p = Path(data_dir or HORIZONS_DATA_DIR) / f"{int(obj_id)}.json"
    if not p.exists():
        return None
    return HorizonsObject.from_json_dict(json.loads(p.read_text()))


# ---------------------------------------------------------------------------
# DE440s Moon helper (self-contained; used for regime classification and tests)
# ---------------------------------------------------------------------------
_SPK = None


def moon_position_gcrf_km(t_tdb_s: np.ndarray, bsp_path: Path | str | None = None) -> np.ndarray:
    """Moon (301) position relative to Earth (399) in ICRF km from DE440s via jplephem.

    Both bodies are stored relative to the Earth-Moon barycentre (3) in DE440s, so
    r_moon/earth = r(3->301) - r(3->399)."""
    global _SPK
    from jplephem.spk import SPK

    path = Path(bsp_path or DE440S_PATH)
    if _SPK is None or getattr(_SPK, "_selene_path", None) != str(path):
        _SPK = SPK.open(str(path))
        _SPK._selene_path = str(path)  # type: ignore[attr-defined]
    jd = stime.J2000_JD + np.asarray(t_tdb_s, dtype=float) / stime.DAY_S
    r_m = _SPK[3, 301].compute(jd)
    r_e = _SPK[3, 399].compute(jd)
    return (r_m - r_e).T  # (N, 3)


def selenocentric_range_km(obj: HorizonsObject) -> np.ndarray:
    try:
        rm = moon_position_gcrf_km(obj.t_tdb_s)
    except Exception as exc:  # noqa: BLE001
        log.warning("DE440s unavailable for selenocentric range (%s)", exc)
        return np.full(len(obj), np.nan)
    return np.linalg.norm(obj.states[:, :3] - rm, axis=1)


def classify_regime(geo_r: np.ndarray, sel_r: np.ndarray, hint: str = "unknown") -> str:
    """Label the regime from the data (hint only breaks ties for transit objects)."""
    gmax, gmin = float(np.nanmax(geo_r)), float(np.nanmin(geo_r))
    smax = float(np.nanmax(sel_r)) if np.isfinite(sel_r).any() else float("nan")
    if hint.startswith("sun_earth") and gmin > 1.0e6:
        return hint                      # Sun-Earth L1/L2 halo/Lissajous (~1.5e6 km), never cislunar
    if gmax > CISLUNAR_MAX_RANGE_KM:
        return "beyond_cislunar"
    if np.isfinite(smax):
        if smax < LUNAR_ORBIT_MAX_SELENO_KM:
            return "lunar_orbit"
        if smax < LUNAR_HALO_MAX_SELENO_KM:
            return "nrho" if hint == "nrho" else "lunar_halo"
    if gmin < 3.0 * R_EARTH and gmax >= 3.0e5:
        return "cislunar_transit"
    if gmin > GEO_RADIUS_KM and gmax >= CISLUNAR_MIN_APOGEE_KM:
        return "xgeo_heo" if hint != "cislunar_transit" else "cislunar_transit"
    if gmax >= CISLUNAR_MIN_APOGEE_KM:
        return "xgeo_heo"
    return "below_xgeo"


def is_cislunar_regime(regime: str) -> bool:
    return regime in {"lunar_orbit", "nrho", "lunar_halo", "cislunar_transit", "xgeo_heo"}


def interp_quality(loo_hermite_km: float) -> str:
    """Honest label for ``state_at`` fidelity from the 2x-step leave-one-out error:
    'good' (< 20 km at 2x step, i.e. ~1 km at the nominal step), 'coarse' (< 500 km),
    otherwise 'samples_only' (use the stored samples; do not trust interpolated states)."""
    if not np.isfinite(loo_hermite_km):
        return "unknown"
    if loo_hermite_km < 20.0:
        return "good"
    if loo_hermite_km < 500.0:
        return "coarse"
    return "samples_only"


# ---------------------------------------------------------------------------
# Horizons text parsing (pure functions; unit-tested offline)
# ---------------------------------------------------------------------------
_SPAN_ERR_RE = re.compile(
    r'No ephemeris for target "(?P<name>.+?)" (?P<which>prior to|after) A\.D\. '
    r"(?P<when>\d{4}-[A-Za-z]{3}-\d{2} \d{2}:\d{2}:\d{2}(?:\.\d+)?)\s*(?P<scale>TDB|UT|TT)?"
)
_TARGET_RE = re.compile(r"Target body name:\s*(?P<name>.+?)\s*\((?P<id>-?\d+)\)\s*(?:\{source:\s*(?P<src>[^}]*)\})?")
_MULTI_RE = re.compile(r"^\s*(-?\d+)\s+(.+?)\s{2,}(\S+)?", re.M)


def parse_horizons_calendar_tdb(s: str) -> datetime:
    """'2026-APR-02 01:58:32.3050' -> naive datetime (TDB calendar)."""
    s = s.strip()
    main, _, frac = s.partition(".")
    dt = datetime.strptime(main.title(), "%Y-%b-%d %H:%M:%S")
    if frac:
        dt += timedelta(microseconds=int(round(float("0." + frac) * 1e6)))
    return dt


def format_horizons_time(dt: datetime) -> str:
    return dt.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]


def parse_span_error(text: str) -> tuple[str, datetime] | None:
    """Return ('prior to'|'after', bound datetime TDB) if Horizons reports a span miss."""
    m = _SPAN_ERR_RE.search(text or "")
    if not m:
        return None
    return m.group("which"), parse_horizons_calendar_tdb(m.group("when"))


def parse_vectors_csv(text: str) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Parse a Horizons VECTORS table (``VEC_TABLE=2``, ``CSV_FORMAT=YES``).

    Returns (t_tdb_s (N,), states (N,6) km & km/s, header-info dict). The first CSV
    column must be JDTDB (we request ``TIME_TYPE='TDB'``); this is asserted."""
    if "$$SOE" not in text or "$$EOE" not in text:
        raise ValueError("no $$SOE/$$EOE block in Horizons response")
    head, _, rest = text.partition("$$SOE")
    body, _, _tail = rest.partition("$$EOE")
    hdr_lines = [ln for ln in head.splitlines() if ln.strip()]
    col_line = next((ln for ln in reversed(hdr_lines) if "JD" in ln and "," in ln), "")
    if "JDTDB" not in col_line:
        raise ValueError(f"expected a JDTDB column header, got: {col_line[:80]!r}")
    cols = [c.strip() for c in col_line.split(",") if c.strip()]
    ix = {c: i for i, c in enumerate(cols)}
    need = ["X", "Y", "Z", "VX", "VY", "VZ"]
    if not all(k in ix for k in need):
        raise ValueError(f"vector columns missing in header {cols}")
    rows_t, rows_s = [], []
    for ln in body.splitlines():
        if not ln.strip():
            continue
        parts = [p.strip() for p in ln.split(",")]
        jd = float(parts[0])
        rows_t.append(stime.et_from_jd_tdb(jd))
        rows_s.append([float(parts[ix[k]]) for k in need])
    t = np.asarray(rows_t, dtype=float)
    s = np.asarray(rows_s, dtype=float).reshape(-1, 6)
    info: dict[str, Any] = {}
    m = _TARGET_RE.search(head)
    if m:
        info["target_name"] = m.group("name").strip()
        info["target_id"] = int(m.group("id"))
        if m.group("src"):
            info["trajectory_source"] = m.group("src").strip()
    m2 = re.search(r"Center body name:\s*(.+?)\s*\((\d+)\)", head)
    if m2:
        info["center"] = f"{m2.group(1).strip()} ({m2.group(2)})"
    m3 = re.search(r"Reference frame\s*:\s*(\S+)", head)
    if m3:
        info["frame"] = m3.group(1)
    return t, s, info


def parse_name_search(text: str) -> dict[str, Any]:
    """Classify a COMMAND='<name>' object-data response."""
    txt = text or ""
    if "Multiple major-bodies match" in txt:
        matches = []
        for ln in txt.splitlines():
            m = re.match(r"^\s*(-?\d+)\s+(.+?)\s{2,}(.*)$", ln)
            if m and "ID#" not in ln:
                matches.append({"id": int(m.group(1)), "name": m.group(2).strip(), "other": m.group(3).strip()})
        return {"status": "multiple", "matches": matches}
    if "No matches found" in txt:
        return {"status": "unknown", "matches": []}
    if "Small-body Index Search Results" in txt or re.search(r"^JPL/HORIZONS\s+\d+ ", txt, re.M):
        # resolved to an asteroid/comet record, not a spacecraft
        first = next((ln for ln in txt.splitlines() if "JPL/HORIZONS" in ln), "")
        return {"status": "small_body", "matches": [], "detail": first.strip()}
    m = re.search(r"Revised:.*?\s(-\d+)\s*$", txt, re.M)
    if m:
        hdr = next((ln for ln in txt.splitlines() if "Revised:" in ln), "")
        return {"status": "single", "matches": [{"id": int(m.group(1)), "name": hdr.strip()}]}
    return {"status": "unparsed", "matches": [], "detail": txt[:200]}


# ---------------------------------------------------------------------------
# Network client
# ---------------------------------------------------------------------------
class HorizonsClient:
    """Thin, polite httpx client (<= 1 req/s, retries, 20 s timeout)."""

    def __init__(self, interval_s: float = REQUEST_INTERVAL_S, timeout_s: float = REQUEST_TIMEOUT_S,
                 retries: int = REQUEST_RETRIES):
        import httpx

        self._httpx = httpx
        self.interval_s = interval_s
        self.timeout_s = timeout_s
        self.retries = retries
        self._last = 0.0
        self.n_requests = 0

    def _throttle(self) -> None:
        wait = self.interval_s - (_time.monotonic() - self._last)
        if wait > 0:
            _time.sleep(wait)
        self._last = _time.monotonic()

    def get(self, params: dict[str, str]) -> dict[str, Any]:
        """Return the parsed JSON ({'result': text, 'error': ...}); raises on transport failure."""
        last_exc: Exception | None = None
        for attempt in range(self.retries + 1):
            self._throttle()
            try:
                r = self._httpx.get(HORIZONS_API, params={"format": "json", **params}, timeout=self.timeout_s)
                self.n_requests += 1
                if r.status_code >= 500:
                    raise self._httpx.HTTPStatusError(f"HTTP {r.status_code}", request=r.request, response=r)
                try:
                    return r.json()
                except ValueError:
                    return {"result": r.text, "error": None if r.status_code == 200 else f"HTTP {r.status_code}"}
            except Exception as exc:  # noqa: BLE001 - retry any transport error
                last_exc = exc
                log.warning("Horizons request failed (attempt %d/%d): %s", attempt + 1, self.retries + 1, exc)
        assert last_exc is not None
        raise last_exc

    def object_data(self, command: str) -> str:
        j = self.get({"COMMAND": f"'{command}'", "OBJ_DATA": "YES", "MAKE_EPHEM": "NO"})
        return j.get("result") or j.get("error") or ""

    def vectors(self, obj_id: int, start_tdb: datetime, stop_tdb: datetime, step: str) -> dict[str, Any]:
        params = {
            "COMMAND": f"'{int(obj_id)}'",
            "OBJ_DATA": "YES",
            "MAKE_EPHEM": "YES",
            "EPHEM_TYPE": "VECTORS",
            "CENTER": "'500@399'",
            "REF_PLANE": "FRAME",
            "REF_SYSTEM": "ICRF",
            "VEC_TABLE": "2",
            "VEC_CORR": "NONE",
            "OUT_UNITS": "KM-S",
            "TIME_TYPE": "TDB",
            "CSV_FORMAT": "YES",
            "VEC_LABELS": "NO",
            "START_TIME": f"'{format_horizons_time(start_tdb)}'",
            "STOP_TIME": f"'{format_horizons_time(stop_tdb)}'",
            "STEP_SIZE": f"'{step}'",
        }
        return self.get(params)


def _utc_iso_to_tdb_datetime(utc_iso: str) -> datetime:
    """UTC ISO -> naive datetime on the TDB calendar (what Horizons wants with TIME_TYPE=TDB)."""
    from astropy.time import Time

    t = Time(utc_iso, scale="utc").tdb
    return datetime.strptime(t.iso[:23], "%Y-%m-%d %H:%M:%S.%f")


def resolve_and_fetch(client: HorizonsClient, cand: Candidate, window_utc: tuple[str, str] = WINDOW_UTC,
                      fallback_days: float = FALLBACK_SPAN_DAYS, max_tries: int = 5) -> dict[str, Any]:
    """Fetch a VECTORS table for the window; adapt the span from Horizons' own
    'prior to / after' errors when the object does not cover it.

    Returns {'status': 'ok'|'no_data'|'error', 'text', 't', 'states', 'info', 'span_note',
    'avail_start'|'avail_end' (TDB datetimes when learned)}.
    """
    w0, w1 = (_utc_iso_to_tdb_datetime(window_utc[0]), _utc_iso_to_tdb_datetime(window_utc[1]))
    start, stop = w0, w1
    avail_start: datetime | None = None
    avail_end: datetime | None = None
    span_note = ""
    margin = timedelta(seconds=1)
    fb = timedelta(days=fallback_days)
    for _ in range(max_tries):
        j = client.vectors(cand.id, start, stop, cand.step)
        text = (j.get("result") or "") + ("\n" + j["error"] if j.get("error") else "")
        if "$$SOE" in text:
            try:
                t, s, info = parse_vectors_csv(text)
            except ValueError as exc:
                return {"status": "error", "text": text, "detail": str(exc), "span_note": span_note}
            if t.shape[0] < 2:
                return {"status": "no_data", "text": text, "detail": "fewer than 2 samples", "span_note": span_note}
            got_days = (float(t[-1]) - float(t[0])) / stime.DAY_S
            if avail_start is not None and avail_end is not None and got_days < fallback_days - 0.5:
                span_note = (f"Horizons data available only {format_horizons_time(avail_start)} .. "
                             f"{format_horizons_time(avail_end)} TDB ({got_days:.1f} d, outside the window); "
                             "entire available span cached.")
            contemporaneous = not ((avail_end is not None and avail_end <= w0) or
                                   (avail_start is not None and avail_start >= w1))
            return {"status": "ok", "text": text, "t": t, "states": s, "info": info, "span_note": span_note,
                    "avail_start": avail_start, "avail_end": avail_end, "contemporaneous": contemporaneous}
        se = parse_span_error(text)
        if se is None:
            detail = (j.get("error") or text.strip().splitlines()[-1:] or ["unknown response"])
            return {"status": "error", "text": text, "detail": str(detail)[:200], "span_note": span_note}
        which, bound = se
        if which == "after":            # data ends before our stop
            avail_end = bound
            if bound <= w0:
                # Ends before the window: take the last `fallback_days` of availability.
                stop = bound - margin
                start = stop - fb
                span_note = (f"Horizons data ends {format_horizons_time(bound)} TDB, before the window; "
                             f"cached the last {fallback_days:g} days available.")
            else:
                stop = bound - margin
                span_note = span_note or f"Horizons data ends {format_horizons_time(bound)} TDB; window clipped."
        else:                           # 'prior to': data starts after our start
            avail_start = bound
            if bound >= w1:
                start = bound + margin
                stop = start + fb
                span_note = (f"Horizons data starts {format_horizons_time(bound)} TDB, after the window; "
                             f"cached the first {fallback_days:g} days available.")
            else:
                start = bound + margin
                span_note = span_note or f"Horizons data starts {format_horizons_time(bound)} TDB; window clipped."
        if avail_start and avail_end and avail_end <= avail_start + timedelta(minutes=5):
            return {"status": "no_data", "text": text, "detail": "available span is empty", "span_note": span_note}
        if stop <= start:
            return {"status": "no_data", "text": text, "detail": "resolved span is empty", "span_note": span_note}
    return {"status": "error", "text": "", "detail": "could not resolve span", "span_note": span_note}


# ---------------------------------------------------------------------------
# Refresh / discovery
# ---------------------------------------------------------------------------
def _now_utc_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def trim_inside_bodies(t: np.ndarray, s: np.ndarray) -> tuple[np.ndarray, np.ndarray, dict[str, int]]:
    """Drop samples whose position lies inside the Earth or the Moon.

    Horizons trajectory files for impactors / re-entering stages run a few minutes past the
    event (e.g. Chang'e 5-T1 booster lunar impact 2022-03-04, Luna-25 stage re-entry), so
    the final rows are not spacecraft states. Such rows are removed and counted."""
    geo = np.linalg.norm(s[:, :3], axis=1)
    try:
        sel = np.linalg.norm(s[:, :3] - moon_position_gcrf_km(t), axis=1)
    except Exception:  # noqa: BLE001 - no kernel: only the Earth test applies
        sel = np.full(t.shape[0], np.inf)
    inside_earth = geo < R_EARTH
    inside_moon = sel < R_MOON
    keep = ~(inside_earth | inside_moon)
    return t[keep], s[keep], {"inside_earth": int(inside_earth.sum()), "inside_moon": int(inside_moon.sum())}


def split_at_discontinuities(t: np.ndarray, s: np.ndarray,
                             max_speed_kms: float = MAX_PLAUSIBLE_SPEED_KMS
                             ) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]:
    """Cut the table where consecutive samples imply a chord speed above ``max_speed_kms``
    and keep the longest physically consistent segment (ties -> the latest one).

    Returns (t_kept, s_kept, dropped) where ``dropped`` lists each discarded segment as
    {start_utc, end_utc, n_samples, jump_km, implied_speed_kms, moon_offset_signature}.
    ``moon_offset_signature`` is True when the segment's |r - 2 r_Moon| stays below
    ``LUNAR_ORBIT_MAX_SELENO_KM``, the fingerprint of a Moon-centred SPK segment that was
    chained as if geocentric (observed in Horizons' ``SLIM_merged`` file). The rows are
    dropped, never re-centred: that would be inventing a correction Horizons did not publish.
    """
    n = t.shape[0]
    if n < 2:
        return t, s, []
    dt = np.diff(t)
    chord = np.linalg.norm(np.diff(s[:, :3], axis=0), axis=1) / dt
    # A splice shows a chord speed far above the speeds Horizons reports at both ends of the
    # interval; a smooth arc never does (chord <= mean speed <= peak speed). The second test
    # keeps the rule valid for heliocentric context objects whose Earth-relative speed can
    # legitimately exceed ``max_speed_kms`` (e.g. Lunar Trailblazer, BioSentinel at 1 d step).
    v = np.linalg.norm(s[:, 3:], axis=1)
    v_end = np.maximum(v[:-1], v[1:])
    cuts = np.where((chord > max_speed_kms) & (chord > 2.0 * v_end))[0]   # break between k and k+1
    if cuts.size == 0:
        return t, s, []
    bounds = [0, *(int(k) + 1 for k in cuts), n]
    segments = [(bounds[i], bounds[i + 1]) for i in range(len(bounds) - 1)]
    best = max(range(len(segments)), key=lambda i: (segments[i][1] - segments[i][0], i))
    try:
        rm = moon_position_gcrf_km(t)
        off2 = np.linalg.norm(s[:, :3] - 2.0 * rm, axis=1)
    except Exception:  # noqa: BLE001 - no kernel: signature unknown
        off2 = np.full(n, np.nan)
    dropped = []
    for i, (a, b) in enumerate(segments):
        if i == best:
            continue
        k = a - 1 if a > 0 else b - 1      # the cut that separates this segment
        seg_off = off2[a:b]
        dropped.append({
            "start_utc": stime.tdb_jd_to_utc_iso(stime.jd_tdb_from_et(float(t[a])), 0),
            "end_utc": stime.tdb_jd_to_utc_iso(stime.jd_tdb_from_et(float(t[b - 1])), 0),
            "n_samples": int(b - a),
            "jump_km": round(float(chord[k] * dt[k]), 1),
            "implied_speed_kms": round(float(chord[k]), 3),
            "moon_offset_signature": bool(np.isfinite(seg_off).all() and seg_off.max() < LUNAR_ORBIT_MAX_SELENO_KM),
        })
    a, b = segments[best]
    return t[a:b], s[a:b], dropped


def build_object(cand: Candidate, fetched: dict[str, Any]) -> HorizonsObject:
    t, s, info = fetched["t"], fetched["states"], fetched["info"]
    t, s, trimmed = trim_inside_bodies(t, s)
    if t.shape[0] < 2:
        raise ValueError(f"{cand.name}: no samples above the Earth/Moon surface")
    t, s, dropped_segments = split_at_discontinuities(t, s)
    if t.shape[0] < 2:
        raise ValueError(f"{cand.name}: no physically consistent segment")
    span = (stime.tdb_jd_to_utc_iso(stime.jd_tdb_from_et(float(t[0])), 0),
            stime.tdb_jd_to_utc_iso(stime.jd_tdb_from_et(float(t[-1])), 0))
    obj = HorizonsObject(id=cand.id, name=info.get("target_name", cand.name), t_tdb_s=t, states=s, span=span)
    geo = obj.geocentric_range_km
    sel = selenocentric_range_km(obj)
    regime = classify_regime(geo, sel, cand.regime_hint)
    obj.metadata = {
        "fetched_utc": fetched.get("fetched_utc") or _now_utc_iso(),
        "step": cand.step,
        "regime": regime,
        "regime_hint": cand.regime_hint,
        "span_note": fetched.get("span_note") or "covers the discovery window",
        "contemporaneous": bool(fetched.get("contemporaneous", True)),
        "window_utc": list(WINDOW_UTC),
        "trajectory_source": info.get("trajectory_source"),
        "center": info.get("center", "Earth (399)"),
        "frame": info.get("frame", "ICRF"),
        "geocentric_range_km": [round(float(np.min(geo)), 1), round(float(np.max(geo)), 1)],
        "selenocentric_range_km": ([round(float(np.nanmin(sel)), 1), round(float(np.nanmax(sel)), 1)]
                                   if np.isfinite(sel).any() else None),
        "interp_leave_one_out_max_km": {
            "hermite": round(obj.leave_one_out_error_km("hermite"), 3),
            "spline": round(obj.leave_one_out_error_km("spline"), 3),
        },
        "designation_note": cand.note,
        "trimmed_samples": trimmed,
        "dropped_segments": dropped_segments,
        "max_chord_speed_kms": round(float(np.max(np.linalg.norm(np.diff(s[:, :3], axis=0), axis=1) / np.diff(t))), 3),
    }
    if fetched.get("rebuilt_utc"):
        obj.metadata["rebuilt_utc"] = fetched["rebuilt_utc"]
    obj.metadata["interp_quality"] = interp_quality(obj.metadata["interp_leave_one_out_max_km"]["hermite"])
    obj.notes = [
        "REAL DATA: JPL Horizons state vectors (mission navigation ephemeris republished by JPL SSD).",
        "Geocentric ICRF/J2000 axes; km, km/s; time = TDB seconds past J2000.0.",
        "Values rounded to 1e-3 km / 1e-6 km/s / 1e-3 s for compactness.",
        "Interpolate with cubic Hermite (position+velocity); leave-one-out error (2x step) in metadata.",
    ]
    if cand.note:
        obj.notes.append(cand.note)
    n_trim = trimmed["inside_earth"] + trimmed["inside_moon"]
    if n_trim:
        obj.notes.append(f"{n_trim} Horizons sample(s) located inside the Earth/Moon (past re-entry or impact) were dropped.")
    for seg in dropped_segments:
        why = (" (|r - 2 r_Moon| < 30 000 km throughout: a Moon-centred segment chained as geocentric)"
               if seg["moon_offset_signature"] else "")
        obj.notes.append(
            f"DROPPED {seg['n_samples']} Horizons samples {seg['start_utc']}..{seg['end_utc']}: table discontinuity of "
            f"{seg['jump_km']:.0f} km between consecutive samples (implied {seg['implied_speed_kms']:.0f} km/s > "
            f"{MAX_PLAUSIBLE_SPEED_KMS:g} km/s){why}; only the longest physically consistent segment is kept.")
    return obj


def discover_names(client: HorizonsClient, names: Iterable[str] = NAME_PROBES) -> list[dict[str, Any]]:
    out = []
    for q in names:
        try:
            res = parse_name_search(client.object_data(q))
        except Exception as exc:  # noqa: BLE001
            res = {"status": "error", "matches": [], "detail": str(exc)}
        entry = {"query": q, **res}
        if res["status"] in ("unknown", "small_body") and q in _UNKNOWN_NAME_REASONS:
            entry["reason"] = _UNKNOWN_NAME_REASONS[q]
        out.append(entry)
        log.info("name probe %-18s -> %s %s", q, res["status"], [m["id"] for m in res.get("matches", [])][:6])
    return out


def discover_spacecraft_list(client: HorizonsClient) -> list[dict[str, Any]]:
    """All spacecraft Horizons knows (COMMAND='SPACECRAFT*' wildcard listing)."""
    res = parse_name_search(client.object_data("SPACECRAFT*"))
    return res.get("matches", [])


def _index_entries(cand: Candidate, fetched: dict[str, Any], data_dir: Path
                   ) -> tuple[str, dict[str, Any]]:
    """Turn one fetched/parsed table into ('cached'|'not_cached', index entry), writing
    ``<id>.json`` when the object is cached. Shared by :func:`refresh` and
    :func:`rebuild_from_cache` so both paths produce identical index rows."""
    try:
        obj = build_object(cand, fetched)
    except ValueError as exc:
        return "not_cached", {"id": cand.id, "name": cand.name, "status": "no_data", "reason": str(exc)}
    geo = obj.metadata["geocentric_range_km"]
    regime = obj.metadata["regime"]
    base = {"id": cand.id, "name": obj.name, "regime": regime, "span_utc": list(obj.span),
            "n_samples": len(obj), "step": cand.step, "geocentric_range_km": geo,
            "selenocentric_range_km": obj.metadata["selenocentric_range_km"],
            "span_note": obj.metadata["span_note"], "contemporaneous": obj.metadata["contemporaneous"],
            "trajectory_source": obj.metadata["trajectory_source"]}
    if cand.scope != "cislunar" or not is_cislunar_regime(regime):
        if regime.startswith("sun_earth"):
            why = (f"Sun-Earth {regime[-2:].upper()} regime (geocentric range {geo[0]:.0f}-{geo[1]:.0f} km): "
                   "beyond Earth-Moon scope; probed for context only")
        elif geo[1] > CISLUNAR_MAX_RANGE_KM:
            why = ("outside Earth-Moon/xGEO regime (max geocentric range "
                   f"{geo[1]:.0f} km > {CISLUNAR_MAX_RANGE_KM:.0f} km): heliocentric / escape trajectory")
        elif regime == "below_xgeo":
            why = f"never beyond {CISLUNAR_MIN_APOGEE_KM:.0f} km geocentric (max {geo[1]:.0f} km): not an xGEO object"
        else:
            why = f"regime '{regime}' not cached (scope={cand.scope})"
        log.info("%-34s %-9d -> found, not cached: %s", obj.name, cand.id, why)
        return "not_cached", {**base, "status": "found_not_cached", "reason": why, "note": cand.note or None}
    size = obj.save_json(data_dir / f"{cand.id}.json")
    entry = {**base, "status": "cached", "file": f"{cand.id}.json", "size_bytes": size,
             "interp_leave_one_out_max_km": obj.metadata["interp_leave_one_out_max_km"],
             "interp_quality": obj.metadata["interp_quality"],
             "trimmed_samples": obj.metadata["trimmed_samples"],
             "dropped_segments": obj.metadata["dropped_segments"]}
    log.info("%-34s %-9d -> cached %d samples %s..%s r=[%.0f, %.0f] km (%s)", obj.name, cand.id, len(obj),
             obj.span[0], obj.span[1], geo[0], geo[1], regime)
    return "cached", entry


def _write_index(data_dir: Path, cached: list[dict[str, Any]], not_cached: list[dict[str, Any]],
                 name_searches: list[dict[str, Any]], sc_list: list[dict[str, Any]],
                 candidates: Sequence[Candidate], window_utc: tuple[str, str], *,
                 run: dict[str, Any], partial: bool) -> dict[str, Any]:
    """Merge this run's rows into the existing index and write it.

    ``partial`` (``--only`` / rebuild runs) keeps previous rows for objects not touched now.
    Provenance is cumulative: ``requests_made_total`` sums every run, ``refresh_history``
    lists each run, and ``first_generated_utc`` is kept from the first full refresh, so the
    index never pretends a 15-request partial run produced the whole catalogue."""
    prev = load_index(data_dir)
    prev_cat = prev.get("spacecraft_catalogue")
    if partial:
        probed = {e["id"] for e in cached} | {e["id"] for e in not_cached}
        cached += [e for e in prev.get("objects", []) if e["id"] not in probed and (data_dir / e.get("file", "")).exists()]
        not_cached += [e for e in prev.get("not_cached", []) if e["id"] not in probed]
    # an object can only be in one list: the entry from *this* run wins
    nc_ids = {n["id"] for n in not_cached}
    cached[:] = [e for e in cached if e["id"] not in nc_ids]
    if not name_searches:
        name_searches = prev.get("name_searches", [])
    catalogue = ({"count": len(sc_list), "note": "COMMAND='SPACECRAFT*' listing; full list in data/cache/horizons/spacecraft_list.json",
                  "sample": [m for m in sc_list if m["id"] in {c.id for c in candidates}]}
                 if sc_list else (prev_cat or {"count": 0, "note": "not fetched", "sample": []}))
    cached.sort(key=lambda e: catalog_sort_key(e["id"], e.get("contemporaneous", True)))
    history = list(prev.get("refresh_history", []))
    if not history and prev.get("generated_utc"):      # index written before history existed
        history.append({"utc": prev["generated_utc"], "kind": "refresh", "requests": prev.get("requests_made", 0),
                        "note": "pre-history index (requests figure is that of the last partial run)"})
    history.append(run)
    now = _now_utc_iso()
    index = {
        "generated_utc": now,
        "first_generated_utc": prev.get("first_generated_utc") or prev.get("generated_utc") or now,
        "source": "JPL Horizons",
        "api": HORIZONS_API,
        "demo_epoch_utc": DEMO_EPOCH_UTC,
        "window_utc": list(window_utc),
        "fallback_span_days": FALLBACK_SPAN_DAYS,
        "center": "Earth (500@399)",
        "frame": "ICRF",
        "units": "km, km/s, TDB seconds past J2000",
        "requests_made_this_run": int(run.get("requests", 0)),
        "requests_made_total": int(prev.get("requests_made_total", prev.get("requests_made", 0))) + int(run.get("requests", 0)),
        "refresh_history": history,
        "provenance_note": ("The index is a merge of every refresh/rebuild run listed in refresh_history; "
                            "per-object fetched_utc lives in each <id>.json."),
        "cislunar_rule": {"max_geocentric_range_km": CISLUNAR_MAX_RANGE_KM,
                          "min_apogee_km": CISLUNAR_MIN_APOGEE_KM,
                          "lunar_orbit_max_selenocentric_km": LUNAR_ORBIT_MAX_SELENO_KM,
                          "lunar_halo_max_selenocentric_km": LUNAR_HALO_MAX_SELENO_KM,
                          "max_plausible_chord_speed_kms": MAX_PLAUSIBLE_SPEED_KMS},
        "n_cached": len(cached),
        "objects": cached,
        "not_cached": not_cached,
        "name_searches": name_searches,
        "spacecraft_catalogue": catalogue,
    }
    (data_dir / "index.json").write_text(json.dumps(index, indent=1))
    log.info("index written: %d cached, %d not cached, %d requests this run (%d total)",
             len(cached), len(not_cached), index["requests_made_this_run"], index["requests_made_total"])
    return index


def refresh(data_dir: Path | str | None = None, cache_dir: Path | str | None = None,
            candidates: Sequence[Candidate] = CANDIDATES, window_utc: tuple[str, str] = WINDOW_UTC,
            do_discovery: bool = True, only_ids: set[int] | None = None) -> dict[str, Any]:
    """Probe candidates, cache what is available, write index.json. Offline-safe:
    on a transport failure the existing cache is left untouched and the error logged."""
    data_dir = Path(data_dir or HORIZONS_DATA_DIR)
    cache_dir = Path(cache_dir or HORIZONS_CACHE_DIR)
    data_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)

    try:
        client = HorizonsClient()
        # cheap connectivity check (also documents the catalogue in the raw cache)
        sc_list = discover_spacecraft_list(client) if do_discovery else []
    except Exception as exc:  # noqa: BLE001
        log.error("Horizons unreachable (%s); keeping existing cache in %s", exc, data_dir)
        idx = load_index(data_dir)
        idx["last_refresh_attempt_utc"] = _now_utc_iso()
        idx["last_refresh_error"] = str(exc)
        return idx

    if sc_list:
        (cache_dir / "spacecraft_list.json").write_text(json.dumps(sc_list, indent=1))
        log.info("Horizons lists %d spacecraft", len(sc_list))

    cached: list[dict[str, Any]] = []
    not_cached: list[dict[str, Any]] = []
    probed_ids: list[int] = []
    for cand in candidates:
        if only_ids and cand.id not in only_ids:
            continue
        probed_ids.append(cand.id)
        try:
            fetched = resolve_and_fetch(client, cand, window_utc)
        except Exception as exc:  # noqa: BLE001 - network died mid-run; keep going offline-safe
            log.error("fetch failed for %s (%d): %s", cand.name, cand.id, exc)
            not_cached.append({"id": cand.id, "name": cand.name, "status": "error", "reason": f"network error: {exc}"})
            continue
        raw_path = cache_dir / f"{cand.id}.txt"
        if fetched.get("text"):
            raw_path.write_text(fetched["text"])
            _write_raw_sidecar(raw_path, cand, fetched)
        if fetched["status"] != "ok":
            reason = fetched.get("detail", "")
            if fetched["status"] == "no_data":
                reason = "no data in window; " + reason
            elif "prior to" in (fetched.get("text") or "") or "after" in (fetched.get("text") or ""):
                reason = reason or "span error"
            entry = {"id": cand.id, "name": cand.name, "status": fetched["status"], "reason": reason,
                     "span_note": fetched.get("span_note") or None, "note": cand.note or None}
            not_cached.append(entry)
            log.info("%-34s %-9d -> %s (%s)", cand.name, cand.id, fetched["status"], reason)
            continue
        kind, entry = _index_entries(cand, fetched, data_dir)
        (cached if kind == "cached" else not_cached).append(entry)

    name_searches = discover_names(client) if do_discovery else []
    run = {"utc": _now_utc_iso(), "kind": "refresh", "requests": client.n_requests,
           "discovery": bool(do_discovery), "only_ids": sorted(only_ids) if only_ids else None,
           "n_probed": len(probed_ids), "n_cached_this_run": len(cached)}
    return _write_index(data_dir, cached, not_cached, name_searches, sc_list, candidates, window_utc,
                        run=run, partial=bool(only_ids))


def _write_raw_sidecar(raw_path: Path, cand: Candidate, fetched: dict[str, Any]) -> None:
    """Store the span resolution next to the raw text so an offline rebuild can reproduce
    span_note / contemporaneous without re-asking Horizons."""
    side = {"id": cand.id, "fetched_utc": _now_utc_iso(), "status": fetched.get("status"),
            "span_note": fetched.get("span_note") or "", "contemporaneous": fetched.get("contemporaneous"),
            "avail_start_tdb": format_horizons_time(fetched["avail_start"]) if fetched.get("avail_start") else None,
            "avail_end_tdb": format_horizons_time(fetched["avail_end"]) if fetched.get("avail_end") else None}
    raw_path.with_suffix(".meta.json").write_text(json.dumps(side, indent=1))


def rebuild_from_cache(data_dir: Path | str | None = None, cache_dir: Path | str | None = None,
                       candidates: Sequence[Candidate] = CANDIDATES, window_utc: tuple[str, str] = WINDOW_UTC,
                       only_ids: set[int] | None = None) -> dict[str, Any]:
    """Re-derive ``data/horizons/<id>.json`` and the index rows from the raw responses in
    ``data/cache/horizons/<id>.txt`` - no network. Used after a change to the hygiene rules
    (trimming, discontinuity splitting, regime labels) so the committed JSON can be
    regenerated deterministically from the archived Horizons text.

    span_note / contemporaneous come from the ``<id>.meta.json`` sidecar when present, else
    from the previous index row, else are recomputed from the span vs. the window."""
    data_dir = Path(data_dir or HORIZONS_DATA_DIR)
    cache_dir = Path(cache_dir or HORIZONS_CACHE_DIR)
    data_dir.mkdir(parents=True, exist_ok=True)
    prev = load_index(data_dir)
    prev_rows = {e["id"]: e for e in prev.get("objects", []) + prev.get("not_cached", [])}
    w0 = stime.seconds_since_j2000_tdb(window_utc[0])
    w1 = stime.seconds_since_j2000_tdb(window_utc[1])
    cached: list[dict[str, Any]] = []
    not_cached: list[dict[str, Any]] = []
    n_rebuilt = 0
    for cand in candidates:
        if only_ids and cand.id not in only_ids:
            continue
        raw = cache_dir / f"{cand.id}.txt"
        if not raw.exists():
            log.info("%-34s %-9d -> no raw cache; skipped", cand.name, cand.id)
            continue
        text = raw.read_text()
        if "$$SOE" not in text:
            continue                                   # a span-error / no-data response: index row unchanged
        try:
            t, s, info = parse_vectors_csv(text)
        except ValueError as exc:
            log.warning("raw cache for %s unparsable: %s", cand.id, exc)
            continue
        side_p = raw.with_suffix(".meta.json")
        side = json.loads(side_p.read_text()) if side_p.exists() else {}
        row = prev_rows.get(cand.id, {})
        prev_json = data_dir / f"{cand.id}.json"
        prev_meta = json.loads(prev_json.read_text()) if prev_json.exists() else {}
        span_note = side.get("span_note") or row.get("span_note") or prev_meta.get("span_note") or ""
        if span_note == "covers the discovery window":
            span_note = ""
        contemporaneous = side.get("contemporaneous")
        if contemporaneous is None:
            contemporaneous = row.get("contemporaneous", prev_meta.get("metadata", {}).get("contemporaneous"))
        if contemporaneous is None:
            contemporaneous = bool(float(t[-1]) >= w0 and float(t[0]) <= w1)
        fetched = {"status": "ok", "text": text, "t": t, "states": s, "info": info, "span_note": span_note,
                   "contemporaneous": bool(contemporaneous),
                   "fetched_utc": side.get("fetched_utc") or prev_meta.get("fetched_utc"),
                   "rebuilt_utc": _now_utc_iso()}
        kind, entry = _index_entries(cand, fetched, data_dir)
        (cached if kind == "cached" else not_cached).append(entry)
        n_rebuilt += 1
    run = {"utc": _now_utc_iso(), "kind": "rebuild_from_cache", "requests": 0,
           "only_ids": sorted(only_ids) if only_ids else None, "n_rebuilt": n_rebuilt}
    return _write_index(data_dir, cached, not_cached, [], [], candidates, window_utc, run=run, partial=True)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m selene.objects.horizons",
                                 description="Discover + cache JPL Horizons cislunar spacecraft ephemerides.")
    ap.add_argument("--refresh", action="store_true", help="query Horizons and rewrite data/horizons/*.json")
    ap.add_argument("--rebuild-from-cache", action="store_true",
                    help="offline: regenerate data/horizons/*.json + index rows from the raw data/cache/horizons/*.txt")
    ap.add_argument("--window", nargs=2, metavar=("START_UTC", "STOP_UTC"), default=list(WINDOW_UTC))
    ap.add_argument("--only", nargs="*", type=int, default=None, help="restrict to these Horizons IDs")
    ap.add_argument("--no-discovery", action="store_true", help="skip name probes / SPACECRAFT* listing")
    ap.add_argument("--data-dir", default=None)
    ap.add_argument("--cache-dir", default=None)
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(levelname)s %(name)s: %(message)s")
    only = set(args.only) if args.only else None
    if args.refresh:
        idx = refresh(args.data_dir, args.cache_dir, window_utc=(args.window[0], args.window[1]),
                      do_discovery=not args.no_discovery, only_ids=only)
        print(f"cached {idx.get('n_cached', 0)} objects -> {Path(args.data_dir or HORIZONS_DATA_DIR)}")
        return 0
    if args.rebuild_from_cache:
        idx = rebuild_from_cache(args.data_dir, args.cache_dir, window_utc=(args.window[0], args.window[1]), only_ids=only)
        print(f"rebuilt from raw cache: {idx.get('n_cached', 0)} objects -> {Path(args.data_dir or HORIZONS_DATA_DIR)}")
        return 0
    objs = load_horizons_objects(args.data_dir)
    if not objs:
        print("no cached Horizons objects; run with --refresh (network required)")
        return 1
    for o in objs:
        g = o.metadata.get("geocentric_range_km", ["?", "?"])
        print(f"{o.id:>9d}  {o.name:<34s} {o.regime:<17s} {o.span[0]}..{o.span[1]}  N={len(o):5d}  r=[{g[0]}, {g[1]}] km")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
