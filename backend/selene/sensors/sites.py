"""Ground optical sites: configuration dataclass, a notional global network, and GCRF site states.

**Specification disclaimer.**  The default network below places notional telescopes at
publicly known optical-observatory *locations* (coordinates are public).  The apertures,
limiting magnitudes, fields of view and slew rates are **ASSUMED representative values for the
aperture class** — they are *not* the specifications of any real instrument at those sites.
Rule of thumb used (dark site, ~1 s exposure, SNR ≈ 6 on a point source):
0.5 m ≈ 18.5 mag, 1.0 m ≈ 19.5–20 mag, 1.3–1.5 m ≈ 20.5 mag, 3.5 m ≈ 22 mag.

Site state in GCRF
------------------
``site_gcrf`` uses astropy's ``EarthLocation.get_gcrs_posvel`` (IAU 2006/2000A precession-
nutation, Earth rotation angle, polar motion from the bundled IERS tables; network access is
disabled so the app stays offline).  Output is Earth-centered GCRS position [km] and velocity
[km/s]; GCRS ≈ GCRF/J2000 at the 10 mas level, far below the 1 arcsec measurement noise used
here.  The geodetic zenith direction (needed for elevation and Sun-elevation tests) is obtained
from the same transform by differencing a point 1 km above the site.  Results are cached per
(site, time-grid) because the astropy transform costs ~1 ms per epoch batch setup.

Time convention: ``t_s`` = TDB seconds past J2000.0 (as everywhere in ``selene.dynamics``).
"""
from __future__ import annotations

import hashlib
import threading
from collections import OrderedDict
from dataclasses import asdict, dataclass, field

import numpy as np

from selene.time import J2000_JD

__all__ = [
    "GroundSite",
    "DEFAULT_SITES",
    "SPEC_DISCLAIMER",
    "get_site",
    "site_gcrf",
    "site_frame",
    "SiteFrame",
]

SPEC_DISCLAIMER = (
    "Site coordinates are public observatory locations; apertures, limiting magnitudes, FOV and "
    "slew rates are ASSUMED representative values for the aperture class (0.5 m ~ 18.5 mag, "
    "1 m ~ 19.5-20 mag), NOT the specifications of real instruments at those sites."
)


@dataclass(frozen=True)
class GroundSite:
    """A ground-based optical telescope (notional specs, see module docstring)."""

    id: str
    name: str
    lat_deg: float
    lon_deg: float            # east-positive
    alt_m: float
    aperture_m: float
    limiting_mag: float
    fov_deg: float
    slew_rate_deg_s: float
    min_elevation_deg: float = 20.0
    sun_elev_max_deg: float = -12.0   # astronomical twilight; Sun must be below this elevation
    notes: str = ""
    kind: str = field(default="ground", init=False)

    def as_dict(self) -> dict:
        d = asdict(self)
        d["kind"] = "ground"
        d["spec_note"] = "ASSUMED representative specs; location public"
        return d


# Coordinates: public observatory locations (WGS-84, approximate to ~0.01 deg; altitude of the
# summit/site).  Specs: ASSUMED (see SPEC_DISCLAIMER).
DEFAULT_SITES: tuple[GroundSite, ...] = (
    GroundSite("haleakala", "Haleakala, HI (notional 1 m)", 20.7082, -156.2568, 3055.0,
               1.0, 19.8, 2.0, 3.0, notes="Maui summit site; assumed 1 m-class wide-field"),
    GroundSite("socorro", "Socorro, NM (notional 1 m)", 33.8172, -106.6599, 1510.0,
               1.0, 19.5, 2.0, 3.0, notes="New Mexico desert site; assumed 1 m-class"),
    GroundSite("mt_lemmon", "Mt. Lemmon, AZ (notional 1.5 m)", 32.4420, -110.7893, 2791.0,
               1.5, 20.5, 1.0, 2.0, notes="Catalina Mountains; assumed 1.5 m-class"),
    GroundSite("cerro_tololo", "Cerro Tololo, Chile (notional 1 m)", -30.1690, -70.8063, 2200.0,
               1.0, 19.8, 2.0, 3.0, notes="Southern hemisphere; assumed 1 m-class"),
    GroundSite("teide", "Teide, Canary Islands (notional 1 m)", 28.3000, -16.5097, 2390.0,
               1.0, 19.8, 2.0, 3.0, notes="Atlantic longitude gap filler; assumed 1 m-class"),
    GroundSite("sutherland", "Sutherland, South Africa (notional 1 m)", -32.3783, 20.8105, 1798.0,
               1.0, 19.8, 2.0, 3.0, notes="African longitude; assumed 1 m-class"),
    GroundSite("siding_spring", "Siding Spring, Australia (notional 0.5 m)", -31.2733, 149.0644, 1165.0,
               0.5, 18.5, 3.0, 5.0, notes="Australian longitude; assumed 0.5 m wide-field"),
    GroundSite("diego_garcia", "Diego Garcia (notional 1 m)", -7.4117, 72.4522, 10.0,
               1.0, 19.3, 2.0, 3.0, notes="Indian Ocean low-altitude site; slightly worse sky assumed"),
    GroundSite("ascension", "Ascension Island (notional 0.5 m)", -7.9697, -14.3936, 90.0,
               0.5, 18.3, 3.0, 5.0, notes="Mid-Atlantic island; assumed 0.5 m wide-field"),
)

_SITES_BY_ID = {s.id: s for s in DEFAULT_SITES}


def get_site(site_id: str) -> GroundSite:
    try:
        return _SITES_BY_ID[site_id]
    except KeyError:
        raise KeyError(f"unknown ground site {site_id!r}; known: {sorted(_SITES_BY_ID)}") from None


@dataclass
class SiteFrame:
    """Cached site geometry on a time grid (all GCRF, Earth-centered)."""

    t_s: np.ndarray        # (N,)
    pos_km: np.ndarray     # (N,3)
    vel_km_s: np.ndarray   # (N,3)
    zenith: np.ndarray     # (N,3) unit geodetic zenith direction


# ---------------------------------------------------------------------------
# cache
# ---------------------------------------------------------------------------
_CACHE: "OrderedDict[tuple, SiteFrame]" = OrderedDict()
_CACHE_MAX = 128
_LOCK = threading.Lock()


def _grid_key(t: np.ndarray) -> tuple:
    if t.size == 0:
        return (0, 0.0, 0.0, "")
    return (int(t.size), float(t[0]), float(t[-1]), hashlib.sha1(t.tobytes()).hexdigest())


def _astropy_frame(lat_deg: float, lon_deg: float, alt_m: float, t: np.ndarray) -> SiteFrame:
    import warnings

    from astropy.coordinates import EarthLocation
    from astropy.time import Time
    from astropy.utils import iers

    # Offline by design: never try to download IERS-A; degrade to the bundled tables with a
    # warning instead of raising if an epoch falls outside them.
    with iers.conf.set_temp("auto_download", False), iers.conf.set_temp("iers_degraded_accuracy", "warn"):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            obstime = Time(J2000_JD, t / 86400.0, format="jd", scale="tdb")
            loc = EarthLocation.from_geodetic(lon=lon_deg, lat=lat_deg, height=alt_m)
            up = EarthLocation.from_geodetic(lon=lon_deg, lat=lat_deg, height=alt_m + 1000.0)
            p, v = loc.get_gcrs_posvel(obstime)
            p_up, _ = up.get_gcrs_posvel(obstime)
    pos = p.xyz.to_value("km").T.reshape(-1, 3)
    vel = v.xyz.to_value("km/s").T.reshape(-1, 3)
    zen = p_up.xyz.to_value("km").T.reshape(-1, 3) - pos
    zen /= np.linalg.norm(zen, axis=1, keepdims=True)
    return SiteFrame(t.copy(), pos, vel, zen)


def site_frame(site: GroundSite, t_s) -> SiteFrame:
    """Position, velocity and zenith of ``site`` at TDB seconds ``t_s`` (scalar or (N,)), cached."""
    t = np.atleast_1d(np.asarray(t_s, dtype=np.float64)).ravel()
    if t.size == 0:
        e = np.zeros((0, 3))
        return SiteFrame(t, e, e, e)
    key = (site.id, site.lat_deg, site.lon_deg, site.alt_m, _grid_key(t))
    with _LOCK:
        hit = _CACHE.get(key)
        if hit is not None:
            _CACHE.move_to_end(key)
            return hit
    fr = _astropy_frame(site.lat_deg, site.lon_deg, site.alt_m, t)
    with _LOCK:
        _CACHE[key] = fr
        while len(_CACHE) > _CACHE_MAX:
            _CACHE.popitem(last=False)
    return fr


def site_gcrf(site: GroundSite, t_s) -> tuple[np.ndarray, np.ndarray]:
    """GCRF (Earth-centered) position [km] and velocity [km/s] of a ground site.

    Scalar ``t_s`` -> ((3,), (3,)); array (N,) -> ((N,3), (N,3)).
    """
    scalar = np.ndim(t_s) == 0
    fr = site_frame(site, t_s)
    if scalar:
        return fr.pos_km[0], fr.vel_km_s[0]
    return fr.pos_km, fr.vel_km_s


def clear_cache() -> None:
    with _LOCK:
        _CACHE.clear()
