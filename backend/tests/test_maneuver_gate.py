"""M5 regression tests for the verifier findings: filter-health gate, gap definition relative to
the cadence, post-gap arc consistency (burn inside the arc, not inside the gap), per-test alpha,
CUSUM p-value semantics, regime-aware prior and the real-object refusal.

Scenarios use the compact EKF on SIM-DRO-01 (and the OD UKF on SIM-ELFO-01 where noted).
"""
from __future__ import annotations

import numpy as np
import pytest
from scipy.stats import chi2

from selene.dynamics.ephemeris import EphemParams
from selene.dynamics.frames import DEMO_EPOCH_TDB_S as T0
from selene.maneuver._simple_ekf import run_ekf
from selene.maneuver.config import DetectorConfig, FilterConfig
from selene.maneuver.detection import (
    STATUS_VALUES,
    cusum_test,
    detect,
    establish_baseline,
    find_gaps,
    gap_refit_test,
)
from selene.maneuver.pipeline import resolve_prior, run_scenario
from selene.maneuver.synthetic import (
    Burn,
    direction_unit,
    generate_measurements,
    initial_covariance,
    sample_initial_state,
    truth_with_burn,
)
from selene.maneuver.types import FilterRun
from selene.objects.catalog import get_catalog
from selene.sensors.observers import get_observer

D = 86400.0
CAD = 6 * 3600.0


def _fake_run(nis, dt_s=CAD):
    n = len(nis)
    t = np.arange(n, dtype=float) * dt_s
    z6 = np.zeros((n, 6)); I6 = np.tile(np.eye(6), (n, 1, 1))
    return FilterRun(t_s=t, x=z6, P=I6, x_pred=z6, P_pred=I6, innov=np.zeros((n, 2)), S=np.tile(np.eye(2), (n, 1, 1)),
                     nis=np.asarray(nis, float), meas=[], object_id="fake", meta={"filter": "fake"})


@pytest.fixture(scope="module")
def dro():
    cat = get_catalog()
    e = cat.get("SIM-DRO-01")
    ph = e.physical
    return {"params": EphemParams(srp=True, cr_area_mass=ph["cr_area_mass_m2_kg"]), "x0": cat.state_at("SIM-DRO-01", T0),
            "sensors": [get_observer(i) for i in ("dro_obs", "l1_halo_obs", "geo_west")], "phys": ph,
            "P0": initial_covariance(20.0, 2.0)}


def _ekf_case(dro, burn, epochs, seed=0):
    rng = np.random.default_rng(seed)
    truth = truth_with_burn(dro["x0"], T0, T0 + 7 * D, burn, dro["params"])
    meas = generate_measurements(truth.at, epochs, dro["sensors"], 1.0, rng, dro["phys"]["radius_m"], dro["phys"]["albedo"])
    xs = sample_initial_state(truth.at(T0), dro["P0"], rng)
    return truth, meas, run_ekf(xs, dro["P0"], T0, meas, params=dro["params"], q_psd=1e-18)


# ---------------------------------------------------------------------------
# filter-health gate on synthetic NIS sequences
# ---------------------------------------------------------------------------
def test_baseline_search_and_statuses():
    cfg = DetectorConfig(alpha=0.01, cusum_enabled=False)
    # (i) divergent from the start: never a consistent baseline -> no verdict, nothing declared
    rep = detect(_fake_run([0.0, 70.0, 2000.0, 9000.0, 4000.0, 800.0, 700.0, 1200.0, 690.0, 820.0]), cfg)
    assert rep.status == "filter_not_converged" and not rep.declared and rep.first_declared_t_s is None
    assert rep.summary["filter_health"]["baseline_established"] is False and rep.summary["n_tested"] == 0
    # (ii) quiet baseline then a jump: declared, tests start right after the baseline
    nis = np.full(20, 1.5); nis[12] = 400.0
    rep = detect(_fake_run(nis), cfg)
    assert rep.status == "maneuver_declared" and rep.declared
    b = rep.summary["filter_health"]["baseline_end_index"]
    assert b == 4 and rep.summary["test_start_index"] == 5
    assert rep.first_declared_t_s == 12 * CAD
    # (iii) too few updates for a baseline, (iv) empty run, (v) gate disabled
    assert detect(_fake_run([1.0, 1.0]), cfg).status == "insufficient_updates"
    assert detect(_fake_run([]), cfg).status == "no_observations"
    rep0 = detect(_fake_run([0.0, 70.0, 2000.0, 9000.0, 4000.0]), DetectorConfig(baseline_updates=0, cusum_enabled=False))
    assert rep0.summary["filter_health"]["baseline_updates"] == 0 and rep0.summary["n_tested"] == 3
    assert set(STATUS_VALUES) >= {rep.status, rep0.status}
    # short runs: the baseline shrinks to max(3, n//2) so 3 quiet daily points still establish custody
    rep6 = detect(_fake_run([1.0, 1.8, 2.1, 371.0, 7659.0, 2363.0], dt_s=D), cfg)
    assert rep6.status == "maneuver_declared" and rep6.first_declared_t_s == 3 * D


def test_establish_baseline_function():
    thr = chi2.isf(0.05, 10)
    nis = np.array([30.0, 30.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
    assert establish_baseline(nis, 0, 5, 0.05, 2) == 6  # first 5-window with sum <= thr is 2..6
    assert establish_baseline(np.full(8, thr / 5 * 1.5), 0, 5, 0.05, 2) is None
    assert establish_baseline(nis, 0, 0, 0.05, 2) == -1  # disabled


def test_late_baseline_is_flagged_not_declared():
    """A filter that is inconsistent for a long time and settles late gets a warning; a final
    isolated exceedance is still tested (operationally nothing else is available)."""
    nis = np.concatenate([np.full(12, 40.0), np.full(6, 1.0), [40.0]])
    rep = detect(_fake_run(nis), DetectorConfig(cusum_enabled=False))
    h = rep.summary["filter_health"]
    assert h["baseline_established"] and h["baseline_delay_updates"] >= 5
    assert any("late" in w for w in h["warnings"])


# ---------------------------------------------------------------------------
# per-test alpha, CUSUM semantics, gap definition
# ---------------------------------------------------------------------------
def test_per_test_alpha_and_cusum_pvalue():
    cfg = DetectorConfig(alpha=0.01, alpha_nis=0.05, alpha_window=0.001, alpha_gap=0.02)
    assert (cfg.a_nis, cfg.a_window, cfg.a_gap) == (0.05, 0.001, 0.02)
    assert DetectorConfig().a_window == 0.01
    with pytest.raises(ValueError):
        DetectorConfig(alpha_nis=1.5)
    nis = np.full(30, 2.0); nis[20:] = 8.0
    det = cusum_test(_fake_run(nis), DetectorConfig(cusum_h=8.0, min_updates_before_test=0))
    assert det and det[0].p_value is None and det[0].confidence is None
    assert det[0].extra["false_alarm_rate_per_update"] == pytest.approx(1 / 200.0)
    assert det[0].as_dict()["p_value"] is None


def test_find_gaps_relative_to_cadence():
    t = np.arange(6) * D  # uniform 24-h cadence
    assert find_gaps(t, 12 * 3600) == [1, 2, 3, 4, 5]            # absolute threshold alone: every interval
    assert find_gaps(t, 12 * 3600, cadence_factor=2.5) == []     # relative to the cadence: none
    t2 = np.concatenate([np.arange(12) * CAD, 12 * CAD + 36 * 3600 + np.arange(5) * CAD])
    assert find_gaps(t2, 12 * 3600, cadence_factor=2.5) == [12]


# ---------------------------------------------------------------------------
# post-gap arc consistency: a burn INSIDE the post-gap arc must not be attributed to the gap
# ---------------------------------------------------------------------------
def test_gap_refit_skips_arc_that_contains_the_burn(dro):
    gap_end = T0 + 3 * D + 36 * 3600
    epochs = np.concatenate([T0 + CAD * np.arange(1, 13), gap_end + CAD * np.arange(0, 12)])
    tb = gap_end + 1.5 * CAD  # between the 2nd and 3rd post-gap observations
    pre = truth_with_burn(dro["x0"], T0, tb + 1.0, None, dro["params"])
    burn = Burn(tb, 5e-3 * direction_unit("prograde", pre.at(tb), tb))
    truth, meas, run = _ekf_case(dro, burn, epochs, seed=7)
    gaps = find_gaps(run.t_s, 12 * 3600, 2.5)
    assert len(gaps) == 1 and run.t_s[gaps[0]] == pytest.approx(gap_end)  # first post-gap observation
    cfg = DetectorConfig(alpha=0.01)
    skipped = []
    dets = gap_refit_test(run, cfg, dro["params"], skipped=skipped)
    assert dets == [] and len(skipped) == 1 and skipped[0]["arc_consistent"] is False
    assert skipped[0]["arc_chi2"] > skipped[0]["arc_chi2_threshold"]
    rep = detect(run, cfg, truth=truth.at, params=dro["params"])
    post = [m.t_s for m in meas if m.t_s > tb]
    assert rep.declared and rep.status == "maneuver_declared"
    assert rep.first_declared_t_s >= post[0] and rep.first_declared_t_s <= post[1]
    assert rep.summary["counts"]["gap_refit"] == 0 and len(rep.summary["gaps_skipped_arc_inconsistent"]) == 1


def test_gap_refit_still_detects_burn_inside_gap_with_prior_free_consistency(dro):
    epochs = np.concatenate([T0 + CAD * np.arange(1, 13), T0 + 3 * D + 36 * 3600 + CAD * np.arange(0, 12)])
    tb = T0 + 3.5 * D
    pre = truth_with_burn(dro["x0"], T0, tb + 1.0, None, dro["params"])
    burn = Burn(tb, 3e-3 * direction_unit("radial_out", pre.at(tb), tb))
    truth, meas, run = _ekf_case(dro, burn, epochs, seed=7)
    d = gap_refit_test(run, DetectorConfig(alpha=0.01), dro["params"])
    assert len(d) == 1 and d[0].extra["arc_consistent"] is True and d[0].extra["arc_dof"] == 10
    assert d[0].extra["arc_chi2"] < d[0].extra["arc_chi2_threshold"]


# ---------------------------------------------------------------------------
# pipeline: coarse cadence (issue 1), quiet lunar orbiter (issue 2), real objects (issue 3)
# ---------------------------------------------------------------------------
def test_pipeline_24h_cadence_burn_detected_after_not_before(dro):
    res = run_scenario("SIM-DRO-01", T0, T0 + 7 * D, None, 1440 * 60.0, 1.0,
                       {"t_s": T0 + 74 * 3600.0, "magnitude_mps": 5.0, "direction": "prograde"}, seed=0, prefer_filter="ekf")
    tb = res.truth_block()
    assert res.report.status == "maneuver_declared" and tb["detected"] is True and tb["premature_declaration"] is False
    assert tb["detection_latency_s"] > 0 and tb["detected_after_n_post_burn_obs"] == 1
    assert res.report.summary["counts"]["gap_refit"] == 0 and res.report.summary["n_gaps"] == 0
    assert abs(tb["estimate_error"]["magnitude_error_pct"]) < 20 and tb["estimate_error"]["direction_error_deg"] < 20
    assert res.estimate.window[0] < res.burn.t_s < res.estimate.window[1]


def test_regime_prior_resolution():
    cat = get_catalog()
    sp, sv, lab = resolve_prior(cat.state_at("SIM-ELFO-01", T0), T0, FilterConfig())
    assert (sp, sv, lab) == (2.0, 0.1, "moon_bound")
    sp, sv, lab = resolve_prior(cat.state_at("SIM-DRO-01", T0), T0, FilterConfig())
    assert (sp, sv, lab) == (20.0, 2.0, "cislunar")
    sp, sv, lab = resolve_prior(cat.state_at("SIM-ELFO-01", T0), T0, FilterConfig(sigma_pos0_km=5.0, sigma_vel0_mps=0.2))
    assert (sp, sv) == (5.0, 0.2) and "explicit" in lab


def test_quiet_lunar_orbiter_is_not_declared_with_reference_ekf_at_wide_prior():
    """Issue 2: the STM EKF diverges on SIM-ELFO-01 from a 20 km / 2 m/s prior; the gate must
    report non-convergence instead of a maneuver."""
    res = run_scenario("SIM-ELFO-01", T0, T0 + 7 * D, None, CAD, 1.0, None, seed=0, prefer_filter="ekf",
                       filt_cfg=FilterConfig(sigma_pos0_km=20.0, sigma_vel0_mps=2.0))
    assert res.filter_used == "simple_ekf" and res.config["prior"]["regime"].startswith("moon_bound")
    assert not res.report.declared and res.estimate is None
    assert res.report.status in ("filter_not_converged", "filter_inconsistent")
    assert res.report.summary["filter_health"]["warnings"]


def test_quiet_lunar_orbiter_default_path_not_declared():
    """Issue 2 with the route defaults (auto filter, auto prior)."""
    res = run_scenario("SIM-ELFO-01", T0, T0 + 7 * D, None, CAD, 1.0, None, seed=0)
    assert res.config["prior"] == {**res.config["prior"], "sigma_pos0_km": 2.0, "sigma_vel0_mps": 0.1, "regime": "moon_bound"}
    assert not res.report.declared and res.report.status in ("quiet", "filter_not_converged", "filter_inconsistent")
    if res.filter_used != "simple_ekf":  # the OD UKF is consistent there
        assert res.report.status == "quiet" and res.report.summary["nees"]["mean_nees"] < 20


def test_real_objects_are_refused():
    cat = get_catalog()
    real = [i for i in cat.ids() if cat.get(i).kind != "simulated"] if hasattr(cat, "ids") else []
    if not real:
        pytest.skip("no Horizons objects cached")
    with pytest.raises(ValueError, match="SIMULATED"):
        run_scenario(real[0], T0, T0 + 2 * D, None, CAD, 1.0, None)
    with pytest.raises(ValueError, match="SIMULATED"):
        run_scenario(real[0], T0, T0 + 2 * D, None, CAD, 1.0, {"t_s": T0 + D, "magnitude_mps": 5.0, "direction": "prograde"})


def test_blind_network_reports_no_observations():
    res = run_scenario("SIM-DRO-01", T0, T0 + 7 * D, ["haleakala", "cerro_tololo", "teide", "siding_spring"], CAD, 1.0, None, seed=0)
    assert len(res.meas) == 0 and res.report.status == "no_observations" and not res.report.declared
    assert res.meta["custody_established"] is False
