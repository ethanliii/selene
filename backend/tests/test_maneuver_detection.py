"""M5 maneuver detection: Pfa calibration, Pd on a 5 m/s burn, RA wrap, gap re-fit, NEES, CUSUM.

Synthetic pipeline: SIM-DRO-01 truth (ephemeris model + SRP) with an optional injected burn,
1″ angles at 6-h cadence from the notional space observers (dro_obs, l1_halo_obs, geo_west are
offered, but with ``max_per_epoch=1`` and dro_obs first in the list — it sees every epoch — the
measurements come from dro_obs alone, i.e. this is a single-observer calibration), and the
compact STM EKF in ``selene.maneuver._simple_ekf`` (independent of the OD track's UKF).
"""
from __future__ import annotations

import json
import time

import numpy as np
import pytest
from scipy.stats import chi2

from selene.dynamics.ephemeris import EphemParams
from selene.dynamics.frames import DEMO_EPOCH_TDB_S as T0
from selene.maneuver._simple_ekf import innovation, run_ekf, wrap_angle
from selene.maneuver.config import DetectorConfig
from selene.maneuver.detection import (
    Detection,
    cusum_test,
    cusum_threshold_for_arl,
    detect,
    find_gaps,
    gap_refit_test,
    nees_monitor,
    nis_test,
    windowed_nis_test,
)
from selene.maneuver.synthetic import (
    Burn,
    direction_unit,
    generate_measurements,
    initial_covariance,
    renoise,
    sample_initial_state,
    truth_with_burn,
)
from selene.maneuver.types import FilterRun
from selene.objects.catalog import get_catalog
from selene.sensors.observers import get_observer
from selene.sensors.visibility import Measurement

D = 86400.0
CADENCE = 6 * 3600.0
SENSOR_IDS = ("dro_obs", "l1_halo_obs", "geo_west")
Q_PSD = 1e-18


@pytest.fixture(scope="module")
def case():
    cat = get_catalog()
    e = cat.get("SIM-DRO-01")
    ph = e.physical
    params = EphemParams(srp=True, cr_area_mass=ph["cr_area_mass_m2_kg"])
    x0 = cat.state_at("SIM-DRO-01", T0)
    sensors = [get_observer(i) for i in SENSOR_IDS]
    epochs = T0 + CADENCE * np.arange(1, 29)  # 7 days
    return {"params": params, "x0": x0, "sensors": sensors, "epochs": epochs, "phys": ph, "P0": initial_covariance(20.0, 2.0)}


def _run_case(case, burn, seed=0, epochs=None, q=Q_PSD):
    rng = np.random.default_rng(seed)
    truth = truth_with_burn(case["x0"], T0, T0 + 7 * D, burn, case["params"])
    meas = generate_measurements(truth.at, case["epochs"] if epochs is None else epochs, case["sensors"], 1.0, rng,
                                 case["phys"]["radius_m"], case["phys"]["albedo"])
    xs = sample_initial_state(truth.at(T0), case["P0"], rng)
    run = run_ekf(xs, case["P0"], T0, meas, params=case["params"], q_psd=q, object_id="SIM-DRO-01")
    return truth, meas, run


def _fake_run(nis, t=None):
    n = len(nis)
    t = np.arange(n, dtype=float) * CADENCE if t is None else np.asarray(t, float)
    z6 = np.zeros((n, 6)); I6 = np.tile(np.eye(6), (n, 1, 1))
    return FilterRun(t_s=t, x=z6, P=I6, x_pred=z6, P_pred=I6, innov=np.zeros((n, 2)), S=np.tile(np.eye(2), (n, 1, 1)),
                     nis=np.asarray(nis, float), meas=[], object_id="fake", meta={"filter": "fake"})


# ---------------------------------------------------------------------------
# RA wrap
# ---------------------------------------------------------------------------
def test_wrap_angle_and_ra_wrap_in_innovation():
    assert wrap_angle(np.pi + 0.1) == pytest.approx(-np.pi + 0.1)
    assert wrap_angle(-0.1) == pytest.approx(-0.1)
    # predicted LOS just below RA = 2π, measured just above 0: innovation must be +2δ·cos(dec), not ≈ -2π
    delta = 2e-5
    obs = np.zeros(3)
    r = 3.8e5 * np.array([np.cos(-delta), np.sin(-delta), 0.3])
    dec = np.arcsin(r[2] / np.linalg.norm(r))
    m = Measurement(t_s=0.0, sensor_id="x", ra_rad=delta, dec_rad=dec, sigma_rad=1e-6,
                    observer_pos_gcrf=obs, observer_vel_gcrf=np.zeros(3))
    nu, H = innovation(m, np.concatenate([r, np.zeros(3)]))
    assert nu[0] == pytest.approx(2 * delta * np.cos(dec), rel=1e-6)
    assert abs(nu[1]) < 1e-12
    assert H.shape == (2, 6) and np.all(H[:, 3:] == 0)
    # and the mirrored case
    r2 = 3.8e5 * np.array([np.cos(delta), np.sin(delta), 0.3])
    m2 = Measurement(t_s=0.0, sensor_id="x", ra_rad=2 * np.pi - delta, dec_rad=dec, sigma_rad=1e-6,
                     observer_pos_gcrf=obs, observer_vel_gcrf=np.zeros(3))
    nu2, _ = innovation(m2, np.concatenate([r2, np.zeros(3)]))
    assert nu2[0] == pytest.approx(-2 * delta * np.cos(dec), rel=1e-6)


def test_ra_wrap_end_to_end_no_false_alarm_when_los_crosses_ra_zero(case):
    """Rotate the observer geometry so the line of sight crosses RA = 0 during the arc; a
    wrap bug would produce a ~2π innovation and a huge NIS at the crossing."""
    truth, meas, run = _run_case(case, None, seed=3)
    # shift all RA by the same offset so the arc straddles 0/2π (observer positions rotated accordingly)
    ra0 = np.array([m.ra_rad for m in meas])
    off = (2 * np.pi - np.median(ra0))
    Rz = np.array([[np.cos(off), -np.sin(off), 0], [np.sin(off), np.cos(off), 0], [0, 0, 1]])
    meas2 = []
    for m in meas:
        meas2.append(Measurement(t_s=m.t_s, sensor_id=m.sensor_id, ra_rad=(m.ra_rad + off) % (2 * np.pi), dec_rad=m.dec_rad,
                                 sigma_rad=m.sigma_rad, observer_pos_gcrf=Rz @ m.observer_pos_gcrf,
                                 observer_vel_gcrf=Rz @ m.observer_vel_gcrf, truth={}))
    ra2 = np.array([m.ra_rad for m in meas2])
    assert (ra2 < 0.5).any() and (ra2 > 2 * np.pi - 0.5).any()  # straddles the wrap
    x0r = np.concatenate([Rz @ run.meta["x0"][:3], Rz @ run.meta["x0"][3:]])
    # rotated problem is not a solution of the real force model, so use a generous q and only check the wrap
    run2 = run_ekf(x0r, case["P0"], T0, meas2, params=case["params"], q_psd=1e-12)
    assert np.all(np.abs(run2.innov[:, 0]) < 0.01), "RA innovation jumped by ~2π at the wrap"


# ---------------------------------------------------------------------------
# Pfa calibration
# ---------------------------------------------------------------------------
def _pfa(case, n_runs, alpha=0.01, seed=0):
    rng = np.random.default_rng(seed)
    truth = truth_with_burn(case["x0"], T0, T0 + 7 * D, None, case["params"])
    base = generate_measurements(truth.at, case["epochs"], case["sensors"], 1.0, rng, case["phys"]["radius_m"], case["phys"]["albedo"])
    assert len(base) >= 20
    thr = chi2.isf(alpha, 2)
    n_exceed = n_total = 0
    nees_means = []
    declared = 0
    cfg = DetectorConfig(alpha=alpha, cusum_enabled=False)
    for _ in range(n_runs):
        m = renoise(base, rng)
        xs = sample_initial_state(truth.at(T0), case["P0"], rng)
        run = run_ekf(xs, case["P0"], T0, m, params=case["params"], q_psd=Q_PSD)
        nis = run.nis[cfg.min_updates_before_test:]
        n_exceed += int(np.sum(nis > thr)); n_total += nis.size
        _, s = nees_monitor(run, truth.at, 0.05, cfg.min_updates_before_test)
        nees_means.append(s["mean_nees"])
        declared += int(detect(run, cfg, params=case["params"]).declared)
    return n_exceed / n_total, n_total, float(np.mean(nees_means)), declared / n_runs


def test_pfa_calibration_100_runs(case):
    tic = time.perf_counter()
    pfa, n, nees, pdecl = _pfa(case, 100)
    elapsed = time.perf_counter() - tic
    print(f"\nPfa(alpha=0.01) = {pfa:.4f} over {n} updates; mean NEES = {nees:.2f}; run-level false declaration = {pdecl:.2f}; {elapsed:.1f}s")
    assert 0.003 <= pfa <= 0.03
    # consistent filter: NEES mean near 6 (χ²₆); bounds generous because q adds a little conservatism
    assert 4.5 <= nees <= 7.5
    assert pdecl <= 0.06  # family-wise (Šidák) declaration rate ≈ alpha; binomial(100, 0.01) P(X>=7) < 1e-4
    assert elapsed < 60


@pytest.mark.slow
def test_pfa_calibration_200_runs(case):
    pfa, n, nees, _ = _pfa(case, 200, seed=1)
    print(f"\nPfa(alpha=0.01) = {pfa:.4f} over {n} updates; mean NEES = {nees:.2f}")
    assert 0.003 <= pfa <= 0.03


# ---------------------------------------------------------------------------
# Pd on a 5 m/s burn; no detection at 0 m/s
# ---------------------------------------------------------------------------
def test_detects_5mps_burn_within_two_observations(case):
    tb = T0 + 3.3 * D
    pre = truth_with_burn(case["x0"], T0, tb + 1.0, None, case["params"])
    u = direction_unit("prograde", pre.at(tb), tb)
    burn = Burn(tb, 5e-3 * u)
    truth, meas, run = _run_case(case, burn, seed=5)
    rep = detect(run, DetectorConfig(alpha=0.01), truth=truth.at, params=case["params"])
    assert rep.declared
    post = [m.t_s for m in meas if m.t_s > tb]
    assert rep.first_declared_t_s <= post[1], "must declare within 2 post-burn observations"
    nis_d = [d for d in rep.by_test("nis") if d.t_s >= tb]
    assert nis_d and nis_d[0].t_s == post[0] and nis_d[0].statistic > 100 * nis_d[0].threshold
    # nothing is declared before the burn (family-wise threshold), and the first declaration is a NIS exceedance
    assert all(d.t_s > tb for d in rep.detections if d.test != "nees" and d.statistic > rep.summary["thresholds_familywise"].get(d.test, np.inf))
    assert rep.summary["counts"]["nees"] > 0  # the EKF absorbed the burn: NEES jumps after it
    tb_block = rep.as_dict()
    json.dumps(tb_block)  # serialisable


def test_zero_burn_is_not_declared(case):
    truth, meas, run = _run_case(case, Burn(T0 + 3.3 * D, np.zeros(3)), seed=11)
    assert truth.burn is None or np.linalg.norm(truth.burn.dv_kms) == 0
    rep = detect(run, DetectorConfig(alpha=0.01), truth=truth.at, params=case["params"])
    assert not rep.declared
    assert rep.summary["counts"]["nis"] <= 2
    assert rep.summary["counts"]["nis_window"] == 0 and rep.summary["counts"]["gap_refit"] == 0


# ---------------------------------------------------------------------------
# individual tests on synthetic NIS sequences
# ---------------------------------------------------------------------------
def test_nis_and_windowed_tests_on_synthetic_sequence():
    nis = np.full(20, 1.5)
    nis[10:15] = 7.0  # each below χ²₂(0.99)=9.21 but the 5-window sum 35 ≫ χ²₁₀(0.99)=23.2
    run = _fake_run(nis)
    assert nis_test(run, 0.01) == []
    w = windowed_nis_test(run, 0.01, window=5)
    # first full window exceeding: indices 8..12 sum to 1.5+1.5+7+7+7 = 24 > 23.2; peak at index 14 (35)
    assert w and w[0].index == 12 and w[0].test == "nis_window" and w[0].dof == 10
    assert w[0].statistic == pytest.approx(24.0) and max(d.statistic for d in w) == pytest.approx(35.0)
    assert w[0].threshold == pytest.approx(chi2.isf(0.01, 10))
    assert 0 <= w[0].p_value < 0.01 and w[0].confidence > 0.99
    nis[3] = 20.0
    d = nis_test(_fake_run(nis), 0.01)
    assert len(d) == 1 and d[0].index == 3 and d[0].p_value == pytest.approx(chi2.sf(20.0, 2))
    assert isinstance(d[0], Detection)


def test_cusum_arl_calibration_and_alarm():
    h = cusum_threshold_for_arl(100.0, 1.0, 2, n_mc=1000, seed=0)
    assert h > 0
    rng = np.random.default_rng(42)
    rls = []
    for _ in range(300):
        s = 0.0
        for i in range(5000):
            s = max(0.0, s + rng.chisquare(2) - 3.0)
            if s > h:
                rls.append(i + 1)
                break
        else:
            rls.append(5000)
    arl = float(np.mean(rls))
    print(f"\nCUSUM h={h:.3f} empirical ARL0={arl:.1f} (target 100)")
    assert 60 < arl < 160
    nis = np.full(30, 2.0); nis[20:] = 8.0
    det = cusum_test(_fake_run(nis), DetectorConfig(cusum_h=h, min_updates_before_test=0))
    assert det and det[0].index >= 20 and det[0].test == "cusum"


# ---------------------------------------------------------------------------
# gap re-fit (lunar-glare style outage)
# ---------------------------------------------------------------------------
def test_gap_refit_detects_burn_inside_long_gap(case):
    epochs = np.concatenate([T0 + CADENCE * np.arange(1, 13), T0 + 3 * D + 36 * 3600 + CADENCE * np.arange(0, 12)])
    tb = T0 + 3.5 * D
    pre = truth_with_burn(case["x0"], T0, tb + 1.0, None, case["params"])
    burn = Burn(tb, 3e-3 * direction_unit("radial_out", pre.at(tb), tb))
    truth, meas, run = _run_case(case, burn, seed=7, epochs=epochs)
    gaps = find_gaps(run.t_s, 12 * 3600)
    assert len(gaps) == 1
    cfg = DetectorConfig(alpha=0.01, gap_hours_for_refit=12.0)
    d = gap_refit_test(run, cfg, case["params"])
    assert len(d) == 1 and d[0].test == "gap_refit" and d[0].dof == 6
    assert d[0].extra["gap_hours"] > 24 and d[0].statistic > d[0].threshold
    assert d[0].extra["fit_converged"] and d[0].extra["arc_consistent"] is True
    # same geometry, no burn -> no gap detection
    truth0, meas0, run0 = _run_case(case, None, seed=7, epochs=epochs)
    assert gap_refit_test(run0, cfg, case["params"]) == []


def test_nees_monitor_consistent_without_burn(case):
    truth, meas, run = _run_case(case, None, seed=21)
    dets, s = nees_monitor(run, truth.at, 0.05, 2)
    assert s["n_epochs"] >= 20 and s["dof"] == 6
    assert len(dets) <= 3  # 5 % level on ~24 epochs
    assert s["pos_err_km"].shape == (len(run),)
