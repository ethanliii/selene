import numpy as np
import pytest

from selene.constants import R_EARTH, R_MOON
from selene.sensors import constraints as C
from selene.sensors.reasons import (
    REASON_DAYLIGHT,
    REASON_EARTH_EXCLUSION,
    REASON_IN_SHADOW,
    REASON_LOW_ELEVATION,
    REASON_MOON_EXCLUSION,
    REASON_SUN_EXCLUSION,
    popcount,
    primary_reason,
    reason_names,
)

SUN = np.array([1.5e8, 0.0, 0.0])   # Sun on +x


def _dir(angle_deg, dist=1.0):
    a = np.deg2rad(angle_deg)
    return dist * np.array([np.cos(a), np.sin(a), 0.0])


def test_angular_separation():
    assert C.angular_separation([1, 0, 0], [0, 1, 0]) == pytest.approx(np.pi / 2)
    assert C.angular_separation([1, 0, 0], [-1, 0, 0]) == pytest.approx(np.pi)
    assert C.angular_separation([1, 0, 0], [1, 1e-9, 0]) == pytest.approx(1e-9, rel=1e-6)
    assert C.angular_separation(np.ones((4, 1, 3)), np.ones((1, 6, 3))).shape == (4, 6)


def test_sun_exclusion():
    obs = np.array([0.0, 4.2e4, 0.0])
    # target 20 deg from the Sun direction as seen from obs -> violated for 30 deg exclusion, ok for 15
    tgt = obs + 1e5 * _dir(20.0)
    assert C.sun_exclusion(obs, tgt, SUN, np.deg2rad(30))
    assert not C.sun_exclusion(obs, tgt, SUN, np.deg2rad(15))


def test_earth_exclusion_from_limb_and_occultation():
    obs = np.array([-4e5, 0.0, 0.0])          # 400 000 km from Earth on -x
    rho = np.arcsin(R_EARTH / 4e5)            # Earth angular radius ~0.91 deg
    excl = np.deg2rad(15.0)
    # line of sight 10 deg from the Earth's centre: 10 - 0.91 < 15 -> violated
    tgt = obs + 8e5 * _dir(10.0)
    assert C.earth_exclusion(obs, tgt, excl)
    # 16.5 deg from centre -> 15.6 from limb -> OK
    tgt = obs + 8e5 * _dir(16.5)
    assert not C.earth_exclusion(obs, tgt, excl)
    # directly behind the Earth -> occulted even with zero exclusion
    tgt = np.array([4e5, 0.0, 0.0])
    assert C.earth_exclusion(obs, tgt, 0.0)
    # in front of the Earth disc (between observer and Earth): bright-Earth background -> excluded too
    tgt = np.array([-2e5, 0.0, 0.0])
    assert C.earth_exclusion(obs, tgt, 0.0)
    # same range but 5 deg off the disc with zero exclusion -> fine
    assert not C.earth_exclusion(obs, obs + 2e5 * _dir(5.0), 0.0)
    assert np.rad2deg(rho) == pytest.approx(0.914, abs=0.01)


def test_moon_exclusion():
    moon = np.array([3.844e5, 0.0, 0.0])
    obs = np.zeros(3)
    excl = np.deg2rad(5.0)
    tgt = moon + np.array([0.0, 2.0e4, 0.0])    # 3 deg off the Moon's centre -> violated
    assert C.moon_exclusion(obs, tgt, moon, excl)
    tgt = moon + np.array([0.0, 6.0e4, 0.0])    # 8.9 deg -> ok
    assert not C.moon_exclusion(obs, tgt, moon, excl)
    assert C.moon_exclusion(obs, moon + np.array([1e4, 0, 0]), moon, 0.0)   # behind the Moon


def test_cylindrical_shadows():
    # Earth shadow: anti-Sun direction is -x
    assert C.in_earth_shadow(np.array([-4.2e4, 0.0, 0.0]), SUN)
    assert not C.in_earth_shadow(np.array([+4.2e4, 0.0, 0.0]), SUN)
    assert not C.in_earth_shadow(np.array([-4.2e4, R_EARTH + 10.0, 0.0]), SUN)
    assert C.in_earth_shadow(np.array([-4.2e4, R_EARTH - 10.0, 0.0]), SUN)
    moon = np.array([0.0, 3.844e5, 0.0])
    assert C.in_moon_shadow(moon + np.array([-5000.0, 0.0, 0.0]), moon, SUN)
    assert not C.in_moon_shadow(moon + np.array([+5000.0, 0.0, 0.0]), moon, SUN)
    # the shadow axis is tilted ~0.15 deg toward -y here (Sun not exactly on the Moon's x axis),
    # shifting the axis ~13 km at 5000 km downstream, so step 50 km outside the radius
    assert not C.in_moon_shadow(moon + np.array([-5000.0, R_MOON + 50.0, 0.0]), moon, SUN)


def test_elevation_and_sun_elevation():
    site = np.array([R_EARTH, 0.0, 0.0])
    zen = np.array([1.0, 0.0, 0.0])
    assert C.elevation_angle(site, zen, np.array([4e5, 0, 0])) == pytest.approx(np.pi / 2)
    assert C.elevation_angle(site, zen, np.array([R_EARTH, 4e5, 0])) == pytest.approx(0.0, abs=1e-12)
    assert C.sun_elevation(site, zen, SUN) > 0
    assert C.sun_elevation(site, zen, -SUN) < 0


def test_illuminated_fraction():
    moon = np.array([0.0, 3.844e5, 0.0])
    # quadrature (Sun-Moon-Earth angle = 90 deg + atan(3.844e5/1.5e8) = 90.15 deg -> k = 0.5013)
    assert C.illuminated_fraction(SUN, moon) == pytest.approx(0.5013, abs=1e-3)
    full = np.array([-3.844e5, 0.0, 0.0])
    assert C.illuminated_fraction(SUN, full) == pytest.approx(1.0, abs=1e-4)        # opposition
    new = np.array([3.844e5, 0.0, 0.0])
    assert C.illuminated_fraction(SUN, new) == pytest.approx(0.0, abs=1e-4)


def test_lunar_glare_full_moon_2deg_excluded_30deg_ok():
    site = np.array([R_EARTH, 0.0, 0.0])
    moon = np.array([-3.844e5, 0.0, 0.0])          # full Moon (opposite the Sun)
    d = 3.844e5 + R_EARTH
    tgt2 = site + d * _dir(180.0 - 2.0)             # 2 deg from the Moon
    tgt30 = site + d * _dir(180.0 - 30.0)
    assert C.lunar_glare_exclusion(site, tgt2, moon, SUN)
    assert not C.lunar_glare_exclusion(site, tgt30, moon, SUN)
    # at new Moon the threshold shrinks to 3 deg: 2 deg still excluded, 4 deg ok
    moon_new = np.array([3.844e5, 0.0, 0.0])
    tgt4 = site + d * _dir(4.0)
    assert not C.lunar_glare_exclusion(site, tgt4, moon_new, SUN)
    assert C.lunar_glare_exclusion(site, site + d * _dir(2.0), moon_new, SUN)
    thr = C.lunar_glare_threshold(np.array([0.0, 0.5, 1.0]))
    assert np.rad2deg(thr) == pytest.approx([3.0, 9.0, 15.0])


def test_reason_masks_and_helpers():
    moon = np.array([0.0, 3.844e5, 0.0])
    obs = np.array([0.0, -4.2e4, 0.0])
    # target hidden in Earth shadow and close to Earth limb as seen from obs
    tgt = np.array([-1.0e4, 0.0, 0.0])
    m = C.space_reasons(obs, tgt, SUN, moon, sun_excl_deg=30, earth_excl_deg=15, moon_excl_deg=5)
    assert m & REASON_IN_SHADOW
    assert m & REASON_EARTH_EXCLUSION
    assert not (m & REASON_SUN_EXCLUSION)
    # ground: daylight site looking low
    site = np.array([R_EARTH, 0.0, 0.0])
    zen = np.array([1.0, 0.0, 0.0])
    # put the Moon below this site's horizon so only daylight + elevation can trip
    moon_below = np.array([-3.844e5, 0.0, 0.0])
    g = C.ground_reasons(site, zen, np.array([R_EARTH + 100, 4e5, 0]), SUN, moon_below,
                         min_elevation_deg=20, sun_elev_max_deg=-12)
    assert g & REASON_DAYLIGHT and g & REASON_LOW_ELEVATION
    assert set(reason_names(int(g))) == {"daylight", "low_elevation"}
    assert popcount(np.array([0, 1, 3, 255])).tolist() == [0, 1, 2, 8]
    assert primary_reason(REASON_DAYLIGHT | REASON_MOON_EXCLUSION) == REASON_MOON_EXCLUSION
