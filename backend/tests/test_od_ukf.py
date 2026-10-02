import time

import numpy as np
import pytest

from selene.dynamics.ephemeris import EphemParams
from selene.dynamics.frames import DEMO_EPOCH_TDB_S as T0
from selene.objects.catalog import get_catalog
from selene.od.measurements import simulate_observations
from selene.od.realism import chi2_mean_bounds, default_P0, nees, run_realism_mc
from selene.od.ukf import UKFConfig, ekf_run, process_noise, sigma_points, ukf_run

OID = "SIM-DRO-01"


def _arc(n_obs=30, days=3.0, seed=3, sensors=("dro_obs", "geo_west")):
    cat = get_catalog()
    tr = cat.truth(OID)
    params = EphemParams(srp=True, cr_area_mass=tr.params.cr_area_mass)
    rng = np.random.default_rng(seed)
    grid = T0 + np.linspace(0, days * 86400.0, n_obs)
    obs = simulate_observations(OID, list(sensors), grid, 1.0, rng, True)
    keep = sorted(set(np.linspace(0, len(obs) - 1, n_obs).astype(int)))   # spread over the full span
    obs = [obs[i] for i in keep]
    P0 = default_P0(100.0, 1.0)
    x0 = tr.at(T0) + np.linalg.cholesky(P0) @ rng.standard_normal(6)
    return tr, params, obs, x0, P0


def test_ukf_30_obs_3_days_accuracy_and_consistency():
    tr, params, obs, x0, P0 = _arc()
    assert len(obs) >= 25
    assert (obs[-1].t_s - obs[0].t_s) / 86400.0 > 2.9          # the arc really spans ~3 days
    run = ukf_run(obs, x0, P0, T0, params=params, object_id=OID)
    assert len(run) == len(obs)
    e = run.truth_error(tr.at)
    assert np.linalg.norm(e[-1, :3]) < 50.0
    assert np.all(np.isfinite(run.nis)) and 0.5 < run.nis.mean() < 5.0
    ne = nees(e, run.P)
    # a single run's NEES series is strongly time-correlated (one prior draw), so only a sanity
    # bound is asserted here; the Monte-Carlo tests below carry the statistical consistency check
    lo, hi = chi2_mean_bounds(6, 1, 0.95)
    assert np.mean((ne > lo) & (ne < hi)) > 0.4 and ne.mean() < 3 * 6.0, (ne.round(1).tolist())
    assert np.all(run.sigma_pos_km[1:] <= run.sigma_pos_km[:-1] * 50)   # no blow-up
    d = run.to_dict()
    assert len(d["states"]) == len(obs) and len(d["covs"][0]) == 6 and d["meta"]["filter"] == "ukf"


def test_ukf_stacked_vs_loop_and_ekf_agree_and_timing():
    tr, params, obs, x0, P0 = _arc(n_obs=100, days=7.0, sensors=("dro_obs",))
    cfg_s = UKFConfig(propagation="stacked")
    t = time.perf_counter()
    rs = ukf_run(obs, x0, P0, T0, params=params, config=cfg_s)
    dt_s = time.perf_counter() - t
    assert dt_s < 20.0
    re = ekf_run(obs, x0, P0, T0, params=params)
    # UKF (α=1e-3) and EKF are the same linearisation to first order
    assert np.linalg.norm(rs.x[-1, :3] - re.x[-1, :3]) < 1.0
    assert np.allclose(rs.sigma_pos_km, re.sigma_pos_km, rtol=0.2)
    rl = ukf_run(obs[:20], x0, P0, T0, params=params, config=UKFConfig(propagation="loop"))
    assert np.linalg.norm(rl.x[-1, :3] - rs.x[19, :3]) < 5.0


def test_ukf_process_noise_and_sigma_points():
    Q = process_noise(100.0, 1e-12)
    assert Q[0, 0] == pytest.approx(1e-12 * 100 ** 3 / 3) and Q[0, 3] == pytest.approx(1e-12 * 100 ** 2 / 2)
    assert Q[3, 3] == pytest.approx(1e-10) and np.allclose(Q, Q.T)
    P = np.diag([4.0, 1.0, 1.0, 1e-6, 1e-6, 1e-6])
    chi = sigma_points(np.zeros(6), P, 2.0)
    assert chi.shape == (13, 6) and chi[1, 0] == pytest.approx(4.0) and chi[7, 0] == pytest.approx(-4.0)


def test_ukf_inflation_hook_is_applied():
    tr, params, obs, x0, P0 = _arc(n_obs=10, days=1.0)
    calls = []

    def inflate(ctx):
        calls.append(ctx["k"])
        if ctx["k"] == 3:
            return ctx["P_pred"] * 100.0
        return None

    run = ukf_run(obs, x0, P0, T0, params=params, inflate=inflate)
    base = ukf_run(obs, x0, P0, T0, params=params)
    assert calls == list(range(len(obs)))
    assert run.meta["inflated"][3] is True and not any(run.meta["inflated"][:3])
    assert run.sigma_pos_pred_km[3] > 5 * base.sigma_pos_pred_km[3]
    assert run.nis[3] == pytest.approx(base.nis[3])              # stored NIS is the pre-inflation statistic
    assert run.meta["nis_post_inflation"][3] < run.nis[3]


def test_nees_monte_carlo_10_runs_inside_bounds():
    r = run_realism_mc(10, OID, T0, T0 + 3 * 86400.0, ["dro_obs", "geo_west"], cadence_s=3 * 86400.0 / 15,
                       sigma_arcsec=1.0, seed=0)
    s = r.summary()
    lo, hi = r.nees_bounds
    assert lo < s["nees_overall_mean"] < hi, s
    assert s["nees_inside_fraction"] >= 0.75, s
    assert s["nis_inside_fraction"] >= 0.75, s
    assert r.nees_runs.shape[0] == 10 and r.nis_runs.shape == r.nees_runs.shape


@pytest.mark.slow
def test_nees_monte_carlo_20_runs_inside_bounds():
    r = run_realism_mc(20, OID, T0, T0 + 3 * 86400.0, ["dro_obs", "geo_west"], cadence_s=3 * 86400.0 / 15,
                       sigma_arcsec=1.0, seed=11)
    s = r.summary()
    lo, hi = r.nees_bounds
    assert lo < s["nees_overall_mean"] < hi, s
    assert s["nees_inside_fraction"] >= 0.8, s


def test_ukf_regenerate_for_update_false_runs():
    tr, params, obs, x0, P0 = _arc(n_obs=8, days=1.0)
    run = ukf_run(obs, x0, P0, T0, params=params, config=UKFConfig(regenerate_for_update=False))
    base = ukf_run(obs, x0, P0, T0, params=params)
    assert len(run) == len(obs) and np.all(np.isfinite(run.nis))
    assert np.linalg.norm(run.x[-1, :3] - base.x[-1, :3]) < 1.0      # q ≈ 0: both variants agree


def test_iterated_update_fixes_large_prior_overconfidence():
    """IOD-sized prior (500 km / 5 m/s) seen from a close observer (dro_obs, ρ ≈ 75 000 km): the angle
    nonlinearity ≈ ½(δ/ρ)² ≈ 4.6″ ≫ 1″ noise.  Single-pass UKF/EKF is grossly over-confident; the
    iterated (Gauss-Newton) update restores consistency.  10 runs, 3 days."""
    P0 = default_P0(500.0, 5.0)
    kw = dict(cadence_s=3 * 86400.0 / 15, sigma_arcsec=1.0, P0=P0, seed=0)
    bad = run_realism_mc(10, OID, T0, T0 + 3 * 86400.0, ["dro_obs"], config=UKFConfig(iterated_update=0), **kw)
    good = run_realism_mc(10, OID, T0, T0 + 3 * 86400.0, ["dro_obs"], config=UKFConfig(), **kw)
    lo, hi = good.nees_bounds
    sb, sg = bad.summary(), good.summary()
    assert sb["nees_overall_mean"] > hi, sb                       # the defect is real (measured ≈ 120 for 20 runs)
    assert lo < sg["nees_overall_mean"] < hi, sg
    assert sg["nees_inside_fraction"] >= 0.75, sg
    assert sg["rms_final_pos_err_km"] < sb["rms_final_pos_err_km"]
    assert good.meta["alpha"] == 1e-3
