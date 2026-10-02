import numpy as np
import pytest
from scipy.stats import chi2

from selene.dynamics.ephemeris import EphemParams
from selene.dynamics.frames import DEMO_EPOCH_TDB_S as T0
from selene.objects.catalog import get_catalog
from selene.od.batch import batch_least_squares, propagate_with_stm_to
from selene.od.measurements import simulate_observations

OID = "SIM-DRO-01"


def _setup(seed=2, n=20, days=2.0, sensors=("dro_obs", "l1_halo_obs", "geo_west")):
    cat = get_catalog()
    tr = cat.truth(OID)
    params = EphemParams(srp=True, cr_area_mass=tr.params.cr_area_mass)
    rng = np.random.default_rng(seed)
    grid = T0 + np.linspace(0, days * 86400.0, n)
    obs = simulate_observations(OID, list(sensors), grid, 1.0, rng, True)
    # n measurements spread over the FULL span (not the first n, which would shorten the arc)
    keep = sorted(set(np.linspace(0, len(obs) - 1, n).astype(int)))
    obs = [obs[i] for i in keep]
    xt = tr.at(T0)
    d = rng.standard_normal(6)
    d[:3] *= 500.0 / np.linalg.norm(d[:3])       # 500 km
    d[3:] *= 5e-3 / np.linalg.norm(d[3:])        # 5 m/s
    return tr, params, obs, xt, xt + d


def test_batch_converges_from_500km_5mps_within_3sigma():
    tr, params, obs, xt, x0 = _setup()
    assert len(obs) == 20
    assert (obs[-1].t_s - obs[0].t_s) / 86400.0 > 1.9          # the arc really spans ~2 days
    res = batch_least_squares(obs, x0, T0, params=params, object_id=OID)
    assert res.converged and res.iterations <= 8
    err = res.x - xt
    sig = np.sqrt(np.diag(res.P))
    assert np.all(np.abs(err) < 3.0 * sig), f"err/sigma = {err / sig}"
    m2 = float(err @ np.linalg.solve(res.P, err))
    assert m2 < chi2.ppf(0.9973, 6)
    assert 0.5 < res.rms_arcsec < 1.6                      # post-fit RMS at the 1" noise level
    assert res.history[0]["rms_arcsec"] > 50 > res.history[-1]["rms_arcsec"]
    d = res.as_dict()
    assert len(d["residuals_arcsec"]) == 20 and d["converged"] is True


def test_batch_with_apriori_and_few_obs():
    tr, params, obs, xt, x0 = _setup(seed=4, n=4, days=0.5)
    P0 = np.diag([500.0 ** 2] * 3 + [5e-3 ** 2] * 3)
    res = batch_least_squares(obs[:3], x0, T0, P0=P0, x_apriori=x0, params=params)
    assert res.converged
    err = res.x - xt
    assert float(err @ np.linalg.solve(res.P, err)) < chi2.ppf(0.9973, 6)
    # the a-priori bounds the solution covariance from above
    assert np.all(np.diag(res.P) <= np.diag(P0) * (1 + 1e-9))


def test_propagate_with_stm_handles_both_sides_and_duplicates():
    tr, params, *_ = _setup(n=3, days=0.1)
    x0 = tr.at(T0)
    times = np.array([T0 + 3600.0, T0 - 1800.0, T0 + 3600.0, T0])
    X, Phi = propagate_with_stm_to(x0, T0, times, params)
    assert np.allclose(X[0], X[2]) and np.allclose(X[3], x0) and np.allclose(Phi[3], np.eye(6))
    assert np.allclose(X[1], tr.at(T0 - 1800.0), atol=1e-6)
    # STM vs finite differences (velocity column)
    dv = np.zeros(6)
    dv[4] = 1e-6
    Xp, _ = propagate_with_stm_to(x0 + dv, T0, times[:1], params)
    assert np.allclose((Xp[0] - X[0]) / 1e-6, Phi[0][:, 4], rtol=1e-4, atol=1e-3)


def test_batch_noise_floor_is_reported_as_converged():
    """Seed 107 (20 obs / 48 h) reaches the numerical noise floor: two successive costs are identical and
    no strictly descending step exists.  That is convergence, not failure (was misreported before)."""
    cat = get_catalog()
    tr = cat.truth(OID)
    params = EphemParams(srp=True, cr_area_mass=tr.params.cr_area_mass)
    rng = np.random.default_rng(107)
    grid = T0 + np.linspace(0, 2 * 86400.0, 20)
    obs = simulate_observations(OID, ["dro_obs", "l1_halo_obs", "geo_west"], grid, 1.0, rng, True)
    keep = np.linspace(0, len(obs) - 1, 20).astype(int)
    obs = [obs[i] for i in keep]
    xt = tr.at(T0)
    d = rng.standard_normal(6)
    d[:3] *= 500.0 / np.linalg.norm(d[:3])
    d[3:] *= 5e-3 / np.linalg.norm(d[3:])
    res = batch_least_squares(obs, xt + d, T0, params=params)
    assert res.converged, res.history
    err = res.x - xt
    assert float(err @ np.linalg.solve(res.P, err)) < chi2.ppf(0.9973, 6)
