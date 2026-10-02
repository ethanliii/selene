import numpy as np
import pytest

from selene.dynamics.frames import DEMO_EPOCH_TDB_S as T0
from selene.od.measurements import (
    ARCSEC,
    ObservationSet,
    jacobian_state,
    los_unit,
    noise_cov,
    predict,
    resolve_sensors,
    residual,
    simulate_observations,
    wrap_angle,
)
from selene.od.types import FilterRun
from selene.sensors.visibility import Measurement


def test_ra_wrap_around():
    # measured just above 0, predicted just below 2π: the residual must be tiny and positive
    ra_m, ra_p = 1e-6, 2 * np.pi - 1e-6
    nu = residual(ra_m, 0.3, ra_p, 0.3)
    assert nu[0] == pytest.approx(2e-6 * np.cos(0.3), rel=1e-9)
    assert nu[1] == 0.0
    # and the other way round
    nu2 = residual(ra_p, 0.3, ra_m, 0.3)
    assert nu2[0] == pytest.approx(-2e-6 * np.cos(0.3), rel=1e-9)
    assert np.allclose(wrap_angle([np.pi + 0.1, -np.pi - 0.1, 7 * np.pi]), [-np.pi + 0.1, np.pi - 0.1, np.pi])
    # vectorised
    nu3 = residual(np.array([0.1, 6.2]), np.array([0.0, 0.0]), np.array([6.2, 0.1]), np.array([0.0, 0.0]))
    assert nu3.shape == (2, 2) and nu3[0, 0] > 0 > nu3[1, 0]


def test_jacobian_vs_finite_differences():
    obs = np.array([30_000.0, -10_000.0, 5_000.0])
    x = np.array([-2.6e5, 2.0e5, 9.8e4, -1.0, -0.5, -0.3])
    H = jacobian_state(x, obs)
    assert H.shape == (2, 6) and np.all(H[:, 3:] == 0.0)
    ra0, dec0 = predict(x, obs)
    Hn = np.zeros((2, 6))
    h = 1e-2
    for k in range(6):
        d = np.zeros(6)
        d[k] = h
        rp, dp = predict(x + d, obs)
        rm, dm = predict(x - d, obs)
        Hn[:, k] = residual(rp, dp, rm, dm) / (2 * h)       # scaled residual derivative (with the RA wrap)
    assert np.allclose(H, Hn, rtol=1e-6, atol=1e-13)
    # consistency: the scaled residual of a small displacement equals H·δx to first order
    dx = np.array([3.0, -2.0, 1.0, 0, 0, 0])
    r1, d1 = predict(x + dx, obs)
    assert np.allclose(residual(r1, d1, ra0, dec0), H @ dx, rtol=1e-4)
    # LOS unit vector round trip
    L = los_unit(ra0, dec0)
    assert np.allclose(L, (x[:3] - obs) / np.linalg.norm(x[:3] - obs))


def test_noise_cov_and_sensor_resolution():
    m = Measurement(0.0, "x", 0.0, 0.0, 2 * ARCSEC, np.zeros(3), np.zeros(3))
    assert np.allclose(noise_cov(m), (2 * ARCSEC) ** 2 * np.eye(2))
    assert len(resolve_sensors("all")) == 15 and len(resolve_sensors("ground")) == 9 and len(resolve_sensors("space")) == 6
    assert [s.id for s in resolve_sensors(["dro_obs", "haleakala"])] == ["dro_obs", "haleakala"]
    with pytest.raises(KeyError):
        resolve_sensors(["no_such_sensor"])


def test_simulate_observations_visibility_and_determinism():
    grid = T0 + np.arange(0, 24 * 3600.0, 3600.0)
    a = simulate_observations("SIM-DRO-01", ["dro_obs", "haleakala"], grid, 1.0, np.random.default_rng(5), True)
    b = simulate_observations("SIM-DRO-01", ["dro_obs", "haleakala"], grid, 1.0, np.random.default_rng(5), True)
    assert isinstance(a, ObservationSet) and len(a) > 0
    assert [m.ra_rad for m in a] == [m.ra_rad for m in b]           # seeded -> reproducible
    assert all(a[i].t_s <= a[i + 1].t_s for i in range(len(a) - 1))   # time-ordered
    s = a.summary()
    assert s["n_candidates"] == 48 and s["n_measurements"] + len(a.dropped) == 48
    # the ground site is blind (lunar glare / daylight) at the demo epoch; the DRO observer sees it
    assert "haleakala" in s["dropped_by_sensor"] and s["by_sensor"].get("dro_obs", 0) > 10
    assert set(s["dropped_by_reason"]) & {"moon_exclusion", "daylight", "low_elevation", "too_faint"}
    # noise statistics: on-sky residual vs the stored noiseless truth ~ N(0, σ²)
    nu = np.array([residual(m.ra_rad, m.dec_rad, m.truth["ra_rad"], m.truth["dec_rad"]) for m in a]) / ARCSEC
    assert abs(nu.mean()) < 0.6 and 0.5 < nu.std() < 1.6
    # geometry-only mode keeps every epoch and flags them
    c = simulate_observations("SIM-DRO-01", ["haleakala"], grid[:5], 1.0, None, False)
    assert len(c) == 5 and not all(m.visible for m in c) and c.dropped == []


def test_filter_run_container_roundtrip():
    n = 3
    fr = FilterRun(np.arange(n) * 10.0 + T0, np.zeros((n, 6)), np.tile(np.eye(6), (n, 1, 1)), np.zeros((n, 6)),
                   np.tile(np.eye(6), (n, 1, 1)), np.zeros((n, 2)), np.tile(np.eye(2), (n, 1, 1)), np.ones(n), [], "X")
    assert np.allclose(fr.sigma_pos_km, np.sqrt(3.0))
    d = fr.to_dict()
    assert d["n"] == 3 and len(d["epochs"]) == 3 and d["epochs"][0].startswith("2026-03-01")
    assert len(d["covs"][0]) == 6 and d["sigma_pos_km"][0] == pytest.approx(np.sqrt(3.0))
