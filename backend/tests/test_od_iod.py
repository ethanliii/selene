"""Two-range shooting IOD.  The 3-observation ground arcs use ``respect_visibility=False``
(geometry only) because at the demo epoch (two days before full Moon) every ground site is
blinded by lunar glare for these objects — see test_od_measurements; a separate test uses the
physically visible space-based observations."""
import time

import numpy as np
import pytest
from scipy.stats import chi2

from selene.dynamics.ephemeris import EphemParams
from selene.dynamics.frames import DEMO_EPOCH_TDB_S as T0
from selene.objects.catalog import get_catalog
from selene.od.iod import admissible, iod_two_range
from selene.od.measurements import simulate_observations

OID = "SIM-DRO-01"


def _params(oid=OID):
    return EphemParams(srp=True, cr_area_mass=get_catalog().truth(oid).params.cr_area_mass)


def _ground_arc(oid, seed):
    rng = np.random.default_rng(seed)
    m1 = simulate_observations(oid, ["haleakala"], T0 + np.array([0.0, 6 * 3600.0]), 1.0, rng, respect_visibility=False)
    m2 = simulate_observations(oid, ["cerro_tololo"], T0 + np.array([12 * 3600.0]), 1.0, rng, respect_visibility=False)
    return list(m1) + list(m2)


def test_iod_three_obs_ground_arc_12h_over_noise_seeds():
    """Spec targets (range at t2 within 5 %, velocity within 10 %) must hold for every noise seed, not one
    flattering draw; the covariance is checked as a mean Mahalanobis² over the seeds (16-seed study:
    7.7 vs the χ²₆ mean 6, inside the 95 % band [4.4, 7.8])."""
    cat = get_catalog()
    m2s = []
    for seed in (1, 2, 6, 7):          # seed 6 is the worst of the 16-seed study (1780 km error)
        meas = _ground_arc(OID, seed)
        t = time.perf_counter()
        res = iod_two_range(meas, _params())
        elapsed = time.perf_counter() - t
        assert elapsed < 10.0, f"IOD took {elapsed:.1f} s"
        assert res.best is not None and res.best.converged and res.best.admissible
        assert res.meta["square_problem"] and res.meta["quality"] == "ok" and res.meta["n_roots"] == 1
        truth = cat.state_at(OID, res.t_ref_s)
        obs2 = meas[1].observer_pos_gcrf
        rho_true = np.linalg.norm(truth[:3] - obs2)
        rho_est = np.linalg.norm(res.best.x_ref[:3] - obs2)
        assert abs(rho_est - rho_true) / rho_true < 0.05, f"seed {seed}: range error {100 * abs(rho_est - rho_true) / rho_true:.2f} %"
        v_err = np.linalg.norm(res.best.x_ref[3:] - truth[3:]) / np.linalg.norm(truth[3:])
        assert v_err < 0.10, f"seed {seed}: velocity error {100 * v_err:.1f} %"
        e = res.best.x_ref - truth
        m2s.append(float(e @ np.linalg.solve(res.best.P_ref, e)))
        d = res.as_dict()
        assert d["n_obs"] == 3 and d["best"]["rms_arcsec"] < 1e-3   # square problem: interior residual ~ 0
        assert res.n_rejected > 0                                   # admissible-region pruning did something
    # first-order covariance: the mean over 4 seeds of a χ²₆ variable lies in [1.9, 12.3] at 99 %
    lo, hi = chi2.ppf(0.005, 24) / 4, chi2.ppf(0.995, 24) / 4
    assert lo < np.mean(m2s) < hi, m2s


def test_iod_square_problem_flags_ambiguity_and_contains_truth():
    """L2 halo, same 12 h two-site arc: two exact admissible roots exist (47 500 km apart) and nothing in the
    data orders them — the result must say so rather than present a single 'best' with a tight covariance."""
    cat = get_catalog()
    for oid in ("SIM-L2-HALO-01", "SIM-NRHO-RELAY-01"):
        meas = _ground_arc(oid, 1)
        res = iod_two_range(meas, _params(oid))
        assert res.meta["square_problem"] and res.meta["n_grid"] == 10
        roots = [c for c in res.candidates if c.converged and c.admissible]
        assert len(roots) >= 2 and res.meta["ambiguous"] is True and res.meta["quality"] == "ambiguous", res.meta
        truth = cat.state_at(oid, res.t_ref_s)
        errs = [np.linalg.norm(c.x_ref[:3] - truth[:3]) for c in roots]
        sigs = [np.sqrt(np.trace(c.P_ref[:3, :3])) for c in roots]
        k = int(np.argmin(errs))
        assert errs[k] < 3.0 * sigs[k] + 100.0, (oid, errs, sigs)      # the true root is among the candidates
        assert max(errs) > 10_000.0                                     # and a wrong exact root really exists
        assert res.elapsed_s < 10.0


def test_iod_single_observer_ill_conditioned_is_not_converged():
    """dro_obs alone, 3 obs over 24 h: the interior residual is nearly flat along a valley in (ρ1, ρ3);
    the honest output is 'not converged' with a huge covariance, never a confident state."""
    obs = simulate_observations(OID, ["dro_obs"], T0 + np.array([0.0, 12 * 3600.0, 24 * 3600.0]), 1.0,
                                np.random.default_rng(0), True)
    assert len(obs) == 3
    res = iod_two_range(list(obs), _params())
    assert res.meta["quality"] == "not_converged" and not res.best.converged
    assert np.sqrt(np.trace(res.best.P_ref[:3, :3])) > 10_000.0


def test_iod_five_obs_space_observers_visible():
    cat = get_catalog()
    grid = T0 + np.linspace(0, 18 * 3600.0, 5)
    obs = simulate_observations(OID, ["dro_obs", "geo_west"], grid, 1.0, np.random.default_rng(3), True)
    sel = obs[:: max(1, len(obs) // 5)][:5]
    assert len(sel) >= 4
    res = iod_two_range(sel, _params())
    assert res.best is not None and res.best.converged
    truth = cat.state_at(OID, res.t_ref_s)
    assert np.linalg.norm(res.best.x_ref[:3] - truth[:3]) < 0.01 * np.linalg.norm(truth[:3])
    assert res.best.rms_arcsec < 5.0                              # overdetermined: residuals at the noise level
    assert not res.meta["square_problem"] and res.meta["quality"] == "ok" and res.meta["n_grid"] == 6
    # keep_inadmissible refines at least as many basins
    res2 = iod_two_range(sel, _params(), keep_inadmissible=True, n_refine=8)
    assert res2.meta["n_refined"] >= res.meta["n_refined"] and res2.best.converged


def test_admissibility_bound_to_moon_not_earth():
    """An NRHO perilune-like state is hyperbolic w.r.t. the Earth but bound to the Moon -> admissible."""
    cat = get_catalog()
    x = cat.state_at("SIM-NRHO-RELAY-01", T0)
    ok, e_e, e_m = admissible(x, T0)
    assert ok
    # an absurd 10 km/s state is unbound to both
    bad = x.copy()
    bad[3:] *= 8.0
    assert not admissible(bad, T0)[0]


def test_iod_input_validation():
    with pytest.raises(ValueError):
        iod_two_range([], _params())
