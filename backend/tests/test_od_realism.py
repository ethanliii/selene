import numpy as np
from scipy.stats import chi2

from selene.od.realism import RealismResult, chi2_mean_bounds, nees


def test_chi2_mean_bounds_cover_synthetic_chi2_means():
    rng = np.random.default_rng(0)
    n_runs, dof, conf = 10, 6, 0.95
    lo, hi = chi2_mean_bounds(dof, n_runs, conf)
    assert lo < dof < hi
    means = chi2.rvs(dof, size=(20_000, n_runs), random_state=rng).mean(axis=1)
    inside = np.mean((means >= lo) & (means <= hi))
    assert abs(inside - conf) < 0.01
    # single-run bounds reduce to the plain χ² quantiles
    lo1, hi1 = chi2_mean_bounds(2, 1, 0.95)
    assert np.isclose(lo1, chi2.ppf(0.025, 2)) and np.isclose(hi1, chi2.ppf(0.975, 2))


def test_nees_definition_and_result_summary():
    e = np.array([[1.0, 0, 0, 0, 0, 0], [0, 2.0, 0, 0, 0, 0]])
    P = np.tile(np.eye(6), (2, 1, 1))
    P[1, 1, 1] = 4.0
    assert np.allclose(nees(e, P), [1.0, 1.0])
    rng = np.random.default_rng(1)
    K, M = 30, 10
    r = RealismResult(M, 0.95, np.arange(K) * 100.0, chi2.rvs(6, size=(M, K), random_state=rng),
                      chi2.rvs(2, size=(M, K), random_state=rng), np.ones((M, K)), chi2_mean_bounds(6, M), chi2_mean_bounds(2, M))
    s = r.summary()
    assert s["nees_inside_fraction"] > 0.8 and s["verdict"] == "consistent"
    r_bad = RealismResult(M, 0.95, np.arange(K) * 100.0, 4 * chi2.rvs(6, size=(M, K), random_state=rng),
                          chi2.rvs(2, size=(M, K), random_state=rng), np.ones((M, K)), chi2_mean_bounds(6, M), chi2_mean_bounds(2, M))
    assert "over-confident" in r_bad.summary()["verdict"]
