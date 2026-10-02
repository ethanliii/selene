"""Time utilities: UTC <-> TDB (Barycentric Dynamical Time) and Julian dates.

jplephem expects TDB Julian dates. We use astropy for the UTC->TDB conversion
(includes leap seconds and the ~1.7 ms periodic TDB-TT terms).
"""
from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
from astropy.time import Time

J2000_JD = 2451545.0  # TDB JD of J2000.0 epoch
DAY_S = 86400.0


def utc_to_time(utc) -> Time:
    """Accepts ISO string, datetime, or astropy Time -> astropy Time (scale utc)."""
    if isinstance(utc, Time):
        return utc
    if isinstance(utc, datetime):
        if utc.tzinfo is not None:
            utc = utc.astimezone(timezone.utc).replace(tzinfo=None)
        return Time(utc, scale="utc")
    return Time(str(utc), scale="utc")


def utc_to_tdb_jd(utc) -> float:
    """UTC (ISO string / datetime / Time) -> TDB Julian date (float)."""
    return float(utc_to_time(utc).tdb.jd)


def tdb_jd_to_utc_iso(jd_tdb: float, precision: int = 3) -> str:
    return Time(jd_tdb, format="jd", scale="tdb").utc.isot[: 19 + (precision + 1 if precision else 0)]


def tdb_jd_to_utc_datetime(jd_tdb: float) -> datetime:
    return Time(jd_tdb, format="jd", scale="tdb").utc.to_datetime(timezone=timezone.utc)


def seconds_since_j2000_tdb(utc) -> float:
    """Seconds of TDB elapsed since J2000.0 for a UTC epoch."""
    return (utc_to_tdb_jd(utc) - J2000_JD) * DAY_S


def jd_tdb_from_et(et_s: float) -> float:
    """Ephemeris time (TDB seconds past J2000) -> TDB JD."""
    return J2000_JD + et_s / DAY_S


def et_from_jd_tdb(jd: float) -> float:
    return (jd - J2000_JD) * DAY_S


def utc_grid(utc0, utc1, n: int) -> np.ndarray:
    """n evenly spaced TDB JDs between two UTC epochs (inclusive)."""
    return np.linspace(utc_to_tdb_jd(utc0), utc_to_tdb_jd(utc1), int(n))
