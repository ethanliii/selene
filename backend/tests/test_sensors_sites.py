import numpy as np
import pytest

from selene.constants import R_EARTH, R_EARTH_POLAR
from selene.dynamics.ephemeris import get_ephemeris
from selene.sensors import constraints as C
from selene.sensors.sites import DEFAULT_SITES, SPEC_DISCLAIMER, get_site, site_frame, site_gcrf
from selene.time import seconds_since_j2000_tdb


def test_default_network_is_sane():
    assert 7 <= len(DEFAULT_SITES) <= 9
    ids = [s.id for s in DEFAULT_SITES]
    assert len(set(ids)) == len(ids)
    for s in DEFAULT_SITES:
        assert -90 < s.lat_deg < 90 and -180 <= s.lon_deg <= 180
        assert 0.3 <= s.aperture_m <= 4.0
        assert 17.0 <= s.limiting_mag <= 22.0
        assert s.min_elevation_deg == 20.0 and s.sun_elev_max_deg == -12.0
    assert "ASSUMED" in SPEC_DISCLAIMER
    assert get_site("haleakala").name.startswith("Haleakala")
    with pytest.raises(KeyError):
        get_site("nope")


def test_site_gcrf_magnitude_and_velocity():
    t = seconds_since_j2000_tdb("2026-03-01T00:00:00")
    site = get_site("haleakala")
    p, v = site_gcrf(site, t)
    r = np.linalg.norm(p)
    assert R_EARTH_POLAR < r < R_EARTH + 5.0          # geocentric distance incl. 3 km altitude
    speed = np.linalg.norm(v)
    # Earth rotation: omega * r * cos(geocentric lat) ~ 0.435 km/s at 20.7 N
    assert speed == pytest.approx(7.292115e-5 * r * np.cos(np.deg2rad(20.6)), rel=0.02)
    assert abs(np.dot(p, v)) / (r * speed) < 1e-3     # velocity perpendicular to position


def test_site_gcrf_vectorised_and_cached():
    t0 = seconds_since_j2000_tdb("2026-03-01T00:00:00")
    ts = t0 + np.linspace(0.0, 86164.0905, 25)        # one sidereal day
    site = get_site("socorro")
    fr1 = site_frame(site, ts)
    fr2 = site_frame(site, ts)
    assert fr1 is fr2                                  # cache hit
    p, v = site_gcrf(site, ts)
    assert p.shape == (25, 3) and v.shape == (25, 3)
    # after one sidereal day the site returns to (almost exactly) its inertial direction
    ang = np.degrees(np.arccos(np.dot(p[0], p[-1]) / (np.linalg.norm(p[0]) * np.linalg.norm(p[-1]))))
    assert ang < 0.05
    # and after 12 h it is on the far side (~180 deg about the pole)
    ang12 = np.degrees(np.arccos(np.dot(p[0], p[12]) / (np.linalg.norm(p[0]) * np.linalg.norm(p[12]))))
    assert ang12 > 100.0
    assert np.allclose(np.linalg.norm(fr1.zenith, axis=1), 1.0)
    # zenith is close to the geocentric radial direction (geodetic-geocentric offset < 0.2 deg)
    cosang = np.sum(fr1.zenith * p, axis=1) / np.linalg.norm(p, axis=1)
    assert np.all(np.degrees(np.arccos(cosang)) < 0.2)


def _astropy_sun_alt(site, utc_iso):
    import warnings

    from astropy.coordinates import AltAz, EarthLocation, get_sun
    from astropy.time import Time
    from astropy.utils import iers

    with iers.conf.set_temp("auto_download", False), iers.conf.set_temp("iers_degraded_accuracy", "warn"):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            loc = EarthLocation.from_geodetic(lon=site.lon_deg, lat=site.lat_deg, height=site.alt_m)
            t = Time(utc_iso, scale="utc")
            alt = get_sun(t).transform_to(AltAz(obstime=t, location=loc)).alt.deg
    return float(alt)


def test_haleakala_sun_elevation_midnight_and_noon_against_astropy():
    site = get_site("haleakala")
    # Haleakala lon -156.26 deg -> local solar midnight ~ 10:25 UTC, noon ~ 22:25 UTC
    for utc, expect_positive in (("2026-03-01T10:25:00", False), ("2026-03-01T22:25:00", True)):
        t = seconds_since_j2000_tdb(utc)
        p, _ = site_gcrf(site, t)
        fr = site_frame(site, t)
        sun = get_ephemeris().position("sun", t)
        el = np.degrees(C.sun_elevation(p, fr.zenith[0], sun))
        ref = _astropy_sun_alt(site, utc)
        assert (el > 0) == expect_positive
        assert (ref > 0) == expect_positive
        # astropy's AltAz includes refraction=0 by default and the same geometric model; agree to 0.3 deg
        assert el == pytest.approx(ref, abs=0.3)
        if expect_positive:
            assert el > 40.0      # early March noon at 20.7 N: Sun ~ 62 deg high
        else:
            assert el < -60.0


def test_dro_like_target_near_moon_is_within_10deg_of_moon_from_earth():
    """A DRO sits ~70 000 km from the Moon; from Earth that is < ~10 deg, so lunar glare matters."""
    t = seconds_since_j2000_tdb("2026-03-01T00:00:00")
    eph = get_ephemeris()
    moon = eph.position("moon", t)
    # point 70 000 km from the Moon, perpendicular to the Earth-Moon line (worst case for separation)
    perp = np.cross(moon, [0, 0, 1.0])
    perp /= np.linalg.norm(perp)
    tgt = moon + 70_000.0 * perp
    site = get_site("haleakala")
    p, _ = site_gcrf(site, t)
    sep = np.degrees(C.angular_separation(tgt - p, moon - p))
    assert 8.0 < sep < 12.0
    # a target 3 x closer (NRHO-apolune class distances) is even tighter
    tgt2 = moon + 20_000.0 * perp
    assert np.degrees(C.angular_separation(tgt2 - p, moon - p)) < 4.0
