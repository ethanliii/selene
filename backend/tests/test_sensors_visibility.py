import numpy as np
import pytest

from selene.dynamics.ephemeris import get_ephemeris
from selene.dynamics.frames import DEMO_EPOCH_TDB_S
from selene.sensors.observers import get_observer
from selene.sensors.reasons import REASON_MOON_EXCLUSION, REASON_TOO_FAINT
from selene.sensors.sites import get_site
from selene.sensors.visibility import (
    ARCSEC,
    Measurement,
    measurement_jacobian,
    measurement_model,
    observe,
    visibility,
)


def test_measurement_model_and_analytic_jacobian():
    obs = np.array([1000.0, -2000.0, 500.0])
    tgt = np.array([3.0e5, 2.0e5, -1.0e5])
    ra, dec = measurement_model(obs, tgt)
    rho = tgt - obs
    assert ra == pytest.approx(np.arctan2(rho[1], rho[0]) % (2 * np.pi))
    assert dec == pytest.approx(np.arcsin(rho[2] / np.linalg.norm(rho)))
    H = measurement_jacobian(obs, tgt)
    assert H.shape == (2, 3)
    # central differences
    Hn = np.zeros((2, 3))
    h = 1e-2
    for k in range(3):
        d = np.zeros(3)
        d[k] = h
        rp, dp = measurement_model(obs, tgt + d)
        rm, dm = measurement_model(obs, tgt - d)
        Hn[0, k] = (rp - rm) / (2 * h)
        Hn[1, k] = (dp - dm) / (2 * h)
    assert np.allclose(H, Hn, rtol=1e-6, atol=1e-12)
    # broadcasting over N
    HH = measurement_jacobian(np.tile(obs, (5, 1)), np.tile(tgt, (5, 1)))
    assert HH.shape == (5, 2, 3)


def test_visibility_result_shape_and_ground_blindness_near_moon():
    t = DEMO_EPOCH_TDB_S + np.arange(0, 7 * 86400.0, 3600.0)
    moon = get_ephemeris().position("moon", t)
    tgt = moon + np.array([0.0, 0.0, 15_000.0])   # 15 000 km above the Moon (~2.2 deg from Earth)
    res = visibility(get_site("haleakala"), tgt, t, 1.0, 0.2)
    n = t.size
    assert res.visible.shape == (n,) and res.magnitude.shape == (n,) and res.reasons.shape == (n,)
    assert res.range_km.shape == (n,) and res.phase_deg.shape == (n,)
    assert np.all(res.range_km > 3.3e5) and np.all((res.phase_deg >= 0) & (res.phase_deg <= 180))
    # always inside the lunar-glare zone from the ground -> never visible, and the mask says why
    assert not res.visible.any()
    assert np.all(res.reasons & REASON_MOON_EXCLUSION)
    d = res.as_dict()
    assert d["fraction_visible"] == 0.0 and len(d["reason_names"]) == n
    # a DRO-based observer close to the Moon sees the same target most of the time
    res2 = visibility(get_observer("dro_obs"), tgt, t, 1.0, 0.2)
    assert res2.visible.mean() > 0.5


def test_too_faint_flag_depends_on_size():
    t = DEMO_EPOCH_TDB_S + np.arange(0, 2 * 86400.0, 1800.0)
    moon = get_ephemeris().position("moon", t)
    # target 150 000 km from the Moon toward +z: well outside glare; magnitude decides
    tgt = moon + np.array([0.0, 0.0, 150_000.0])
    site = get_site("mt_lemmon")           # 20.5 mag assumed
    big = visibility(site, tgt, t, 2.0, 0.3)
    small = visibility(site, tgt, t, 0.1, 0.1)
    assert not big.magnitude.min() > site.limiting_mag  # a 2 m object is bright enough at times
    assert np.all(small.reasons & REASON_TOO_FAINT)     # a 10 cm object never is
    assert np.all(small.magnitude > big.magnitude)


def test_observe_noise_statistics_and_fields():
    t = DEMO_EPOCH_TDB_S + 5 * 3600.0
    obs = get_observer("dro_obs")
    moon = get_ephemeris().position("moon", t)
    state = np.concatenate([moon + np.array([0.0, 0.0, 15_000.0]), np.zeros(3)])
    rng = np.random.default_rng(1)
    ms = [observe(obs, state, t, sigma_arcsec=1.0, rng=rng) for _ in range(400)]
    assert all(isinstance(m, Measurement) for m in ms)
    m0 = ms[0]
    assert m0.sensor_id == "dro_obs" and m0.sigma_rad == pytest.approx(ARCSEC)
    assert m0.observer_pos_gcrf.shape == (3,) and m0.light_time_s > 0
    dra = np.array([(m.ra_rad - m.truth["ra_rad"]) * np.cos(m.dec_rad) for m in ms])
    ddec = np.array([m.dec_rad - m.truth["dec_rad"] for m in ms])
    assert np.std(dra) == pytest.approx(ARCSEC, rel=0.2)
    assert np.std(ddec) == pytest.approx(ARCSEC, rel=0.2)
    assert abs(np.mean(ddec)) < 0.2 * ARCSEC
    assert "ra_rad" in m0.as_dict()


def test_observe_returns_none_when_not_visible():
    t = DEMO_EPOCH_TDB_S + 5 * 3600.0
    moon = get_ephemeris().position("moon", t)
    state = np.concatenate([moon + np.array([0.0, 0.0, 15_000.0]), np.zeros(3)])
    site = get_site("haleakala")
    assert observe(site, state, t, rng=np.random.default_rng(0)) is None
    m = observe(site, state, t, rng=np.random.default_rng(0), require_visible=False)
    assert m is not None and not m.visible and m.reasons & REASON_MOON_EXCLUSION


def test_jacobian_polar_singularity_is_finite():
    # line of sight along +z: RA is undefined; the Jacobian must be finite (zeros), never NaN
    H = measurement_jacobian(np.zeros(3), np.array([0.0, 0.0, 1e5]))
    assert H.shape == (2, 3) and np.all(np.isfinite(H))
    assert np.all(H[0] == 0.0) and H[1, 2] == 0.0
    # away from the pole the guard is a no-op
    H2 = measurement_jacobian(np.zeros(3), np.array([1e5, 2e5, 3e4]))
    assert np.all(np.isfinite(H2)) and np.any(H2 != 0)


def test_empty_time_grid_returns_empty_results():
    from selene.sensors.sites import site_gcrf

    p, v = site_gcrf(get_site("teide"), np.array([]))
    assert p.shape == (0, 3) and v.shape == (0, 3)
    res = visibility(get_site("teide"), np.zeros((0, 3)), np.array([]))
    assert res.visible.shape == (0,) and res.as_dict()["fraction_visible"] == 0.0
    res = visibility(get_observer("dro_obs"), np.zeros((0, 3)), np.array([]))
    assert res.magnitude.shape == (0,)
