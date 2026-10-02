"""Monte Carlo architecture scoring: presets, direction of improvement, reproducibility, surrogate validation.

The detection surrogate is validated on the three things it actually asserts (not on its own
sampled false-alarm rate, which equals alpha by construction):
  * the linear burn displacement against a nonlinear propagation of the burned truth,
  * the replayed pre-update covariance against the tasker engine's own sigma series,
  * the NIS threshold against ``selene.maneuver.detection.nis_test``.
"""
from __future__ import annotations

import time

import numpy as np
import pytest
from scipy.stats import chi2

from selene.architecture.candidates import Architecture, SensorSpec, preset_architectures
from selene.architecture.montecarlo import EvalConfig, ManeuverModel, burn_displacement, evaluate

N_MC = 2
HORIZON_DAYS = 2.0


@pytest.fixture(scope="module")
def preset_result():
    t = time.perf_counter()
    res = evaluate(preset_architectures(), n_mc=N_MC, horizon_days=HORIZON_DAYS, seed=0)
    res.timing["wall_s"] = time.perf_counter() - t
    return res


def _row(res, name):
    return next(s for s in res.scores if s.name == name).stats


def test_presets_evaluate_fast_and_report_all_metrics(preset_result):
    res = preset_result
    print(f"\nPRESETS n_mc={N_MC}, {HORIZON_DAYS} d: {res.timing['wall_s']:.1f} s wall ({res.timing['executor']}); "
          f"per draw {[round(x, 2) for x in res.timing['per_draw_s']]}")
    assert res.timing["wall_s"] < 60.0
    assert [s.name for s in res.scores] == ["Ground only", "Ground + 2 GEO", "Ground + L2 halo", "Ground + DRO + L1 halo"]
    assert len(res.object_ids) == 11 and all(o.startswith("SIM-") for o in res.object_ids)
    for s in res.scores:
        st = s.stats
        print(f"  {s.name:24s} coverage {st['coverage_pct']:5.1f} %  custody {st['custody_pct']:5.1f} % (null {st['custody_pct_null']:.1f})  "
              f"revisit {st['revisit_mean_h']:5.1f} h (p95 obs {st['revisit_p95_h']}, censored {st['revisit_p95_censored_h']:.1f})  "
              f"latency {st['detection_latency_mean_h']} h (censored {st['detection_latency_censored_mean_h']:.2f})  "
              f"detected {st['n_detected']}/{st['n_burns']} (nis {st['detections_by_nis']}, miss {st['detections_by_miss']})  "
              f"FA {st['n_false_alarms']}/{st['n_quiet_obs_sampled']}")
        for key in ("coverage_pct", "custody_pct", "custody_pct_null", "revisit_mean_h", "revisit_max_h", "revisit_p95_h",
                    "revisit_p95_censored_h", "revisit_censored_objects_pct", "detection_latency_censored_mean_h", "n_burns",
                    "n_detected", "detections_pct", "n_space_sensors", "total_space_aperture_m", "false_alarm_rate_per_obs",
                    "replay_sigma_max_rel_err", "n_draws"):
            assert key in st, key
        assert 0.0 <= st["coverage_pct"] <= 100.0 and 0.0 <= st["custody_pct"] <= 100.0
        assert st["custody_pct"] >= st["custody_pct_null"] - 1e-9   # observations never hurt a covariance analysis
        assert st["n_draws"] == N_MC and st["n_burns"] > 0
        assert 0.0 < st["revisit_mean_h"] <= HORIZON_DAYS * 24.0 + 1e-9
        assert st["revisit_p95_h"] is not None and 0.0 < st["revisit_p95_h"] <= st["revisit_p95_censored_h"] + 1e-9
        assert len(s.per_object) == 11
        assert s.stats["coverage_pct_p05"] <= s.stats["coverage_pct"] <= s.stats["coverage_pct_p95"]
    # common random numbers: every architecture saw the same burns
    burns = [[(b.object_id, b.t_burn_s, tuple(b.dv_kms)) for d in s.draws for b in d.burns] for s in res.scores]
    assert all(b == burns[0] for b in burns)


def test_adding_space_sensors_improves_every_axis(preset_result):
    """Direction check: DRO + L1 halo (and L2 halo) vs ground only.  Numbers are printed, not hidden."""
    g = _row(preset_result, "Ground only")
    for name in ("Ground + L2 halo", "Ground + DRO + L1 halo"):
        a = _row(preset_result, name)
        print(f"\n{name} vs ground: coverage {g['coverage_pct']:.1f} -> {a['coverage_pct']:.1f} %, custody {g['custody_pct']:.1f} -> "
              f"{a['custody_pct']:.1f} %, censored latency {g['detection_latency_censored_mean_h']:.2f} -> "
              f"{a['detection_latency_censored_mean_h']:.2f} h, detections {g['detections_pct']:.0f} -> {a['detections_pct']:.0f} %, "
              f"revisit p95 (observed) {g['revisit_p95_h']:.2f} -> {a['revisit_p95_h']:.2f} h")
        assert a["coverage_pct"] > g["coverage_pct"]
        assert a["custody_pct"] > g["custody_pct"]
        assert a["detection_latency_censored_mean_h"] < g["detection_latency_censored_mean_h"]
        assert a["detections_pct"] >= g["detections_pct"]
        assert a["revisit_mean_h"] < g["revisit_mean_h"]


def test_detections_are_post_burn_and_latency_is_consistent(preset_result):
    """Every detection happens at a scheduled observation strictly after the burn node; the latency equals
    the node time minus the burn time; undetected burns carry the remaining horizon as censoring time."""
    slot_h = preset_result.config.slot_min / 60.0
    n_checked = 0
    for s in preset_result.scores:
        for d in s.draws:
            for b in d.burns:
                assert b.censored_h == pytest.approx(HORIZON_DAYS * 24.0 - (b.t_burn_s - d.t0_s) / 3600.0, abs=1e-6)
                if b.detected:
                    n_checked += 1
                    assert b.detect_how in ("nis", "miss") and b.detect_sensor
                    assert b.latency_h >= slot_h - 1e-9 and b.latency_h <= b.censored_h + 1e-9
                    # the latency is an integer number of slots (detections happen at slot nodes)
                    assert abs(b.latency_h / slot_h - round(b.latency_h / slot_h)) < 1e-6
                else:
                    assert np.isnan(b.latency_h)
    assert n_checked > 0


def test_reproducible_for_a_seed_and_different_for_another():
    archs = [Architecture("G", [], True), Architecture("G+DRO", [SensorSpec("dro", aperture_m=0.4)], True)]
    r1 = evaluate(archs, n_mc=2, horizon_days=1.0, seed=7)
    r2 = evaluate(archs, n_mc=2, horizon_days=1.0, seed=7)
    r3 = evaluate(archs, n_mc=2, horizon_days=1.0, seed=8)
    keys = ("coverage_pct", "custody_pct", "revisit_mean_h", "revisit_p95_h", "detection_latency_censored_mean_h", "n_burns",
            "n_detected", "n_false_alarms")
    for s1, s2 in zip(r1.scores, r2.scores):
        assert {k: s1.stats[k] for k in keys} == {k: s2.stats[k] for k in keys}
        assert s1.per_object == s2.per_object
    assert [m["t0_tdb_s"] for m in r1.draws_meta] == [m["t0_tdb_s"] for m in r2.draws_meta]
    assert [m["t0_tdb_s"] for m in r1.draws_meta] != [m["t0_tdb_s"] for m in r3.draws_meta]
    # a thread pool does not change the answer (per-draw generators)
    r4 = evaluate(archs, n_mc=2, horizon_days=1.0, seed=7, workers=2, executor="thread")
    assert {k: r4.scores[1].stats[k] for k in keys} == {k: r1.scores[1].stats[k] for k in keys}


def test_process_pool_is_bit_identical_or_falls_back():
    """The spawn process pool must reproduce the sequential answer exactly; if the environment cannot start
    worker processes the engine must fall back to the sequential path and say so."""
    archs = [Architecture("G+L2", [SensorSpec("l2_halo_S")], True)]
    t = time.perf_counter()
    seq = evaluate(archs, n_mc=3, horizon_days=1.0, seed=5)
    t_seq = time.perf_counter() - t
    t = time.perf_counter()
    par = evaluate(archs, n_mc=3, horizon_days=1.0, seed=5, workers=3, executor="process")
    t_par = time.perf_counter() - t
    print(f"\nprocess pool: sequential {t_seq:.1f} s, pool {t_par:.1f} s ({par.timing['executor']}; note={par.timing.get('note')})")
    keys = ("coverage_pct", "custody_pct", "revisit_mean_h", "detection_latency_censored_mean_h", "n_burns", "n_detected", "n_false_alarms")
    assert {k: par.scores[0].stats[k] for k in keys} == {k: seq.scores[0].stats[k] for k in keys}
    assert [b.as_dict() for d in par.scores[0].draws for b in d.burns] == [b.as_dict() for d in seq.scores[0].draws for b in d.burns]
    assert par.timing["executor"].startswith("process") or "note" in par.timing


def test_burn_displacement_matches_nonlinear_propagation():
    """The surrogate's linear displacement dx(t_k) = Phi(t_k, t_b)[0, dv] (started at the burn node) must agree
    with a nonlinear propagation of the burned truth.  A displacement propagated from t0 instead (the
    bug the verifiers found) is 10-100x too large at the first post-burn node."""
    from selene.dynamics.ephemeris import propagate_ephemeris
    from selene.dynamics.frames import DEMO_EPOCH_TDB_S
    from selene.objects.catalog import get_catalog
    from selene.tasking.information import build_tracks
    from selene.tasking.scenario import slot_grid

    cat = get_catalog()
    t0 = DEMO_EPOCH_TDB_S + 3.0 * 86400.0
    t_nodes = slot_grid(t0, t0 + 1.0 * 86400.0, 20.0)
    tracks = build_tracks(["SIM-DRO-01", "SIM-NRHO-RELAY-01", "SIM-L1-HALO-01"], t_nodes)
    kb = 10
    dv = np.array([0.0, 0.01, 0.0])     # 10 m/s along +y GCRF
    for tr in tracks:
        lin = burn_displacement(tr, kb, dv)
        assert np.all(lin[: kb + 1] == 0.0)                      # nothing before / at the burn node
        x_b = tr.x_nodes[kb].copy()
        x_b[3:] += dv
        sol = propagate_ephemeris(x_b, float(t_nodes[kb]), t_eval_s=t_nodes[kb:], params=cat.truth(tr.object_id).params,
                                  rtol=1e-11, atol=1e-11)
        nonlin = sol.y.T - tr.x_nodes[kb:]
        for n in (1, 3, 18, len(t_nodes) - 1 - kb):             # 20 min, 1 h, 6 h, ~20.7 h after the burn
            a, b = lin[kb + n, :3], nonlin[n, :3]
            rel = np.linalg.norm(a - b) / np.linalg.norm(b)
            print(f"\n{tr.object_id:18s} +{n * 20:4d} min: linear {np.linalg.norm(a):8.2f} km, nonlinear {np.linalg.norm(b):8.2f} km, rel err {rel:.1e}")
            assert rel < 1e-2
        # one slot after a 10 m/s burn the displacement is |dv| * dt = 12 km, not hundreds of km
        assert np.linalg.norm(lin[kb + 1, :3]) == pytest.approx(0.01 * 1200.0, rel=1e-3)
        # a displacement wrongly propagated from t0 would be far larger at the first post-burn node
        wrong = np.concatenate([np.zeros(3), dv])
        for k in range(kb + 1):
            wrong = tr.phi[k] @ wrong
        assert np.linalg.norm(wrong[:3]) > 5.0 * np.linalg.norm(lin[kb + 1, :3])


def test_surrogate_replay_matches_engine_and_threshold_matches_detector(preset_result):
    """(1) The pre-update covariance the surrogate replays reproduces the tasker engine's own post-update sigma at
    every observed node (so S = H P- H^T + R is built from the covariance the schedule was actually planned with).
    (2) The NIS threshold is the one ``maneuver.detection.nis_test`` applies (chi2.isf(alpha, 2))."""
    from selene.maneuver.detection import nis_test
    from selene.od.types import FilterRun

    for s in preset_result.scores:
        assert s.stats["replay_sigma_max_rel_err"] < 1e-9, (s.name, s.stats["replay_sigma_max_rel_err"])
        assert s.stats["n_quiet_obs_sampled"] > 0
    model = preset_result.maneuver_model
    assert model.nis_threshold == pytest.approx(chi2.isf(model.alpha, 2), rel=1e-12)
    # feed a two-sample NIS series straddling the surrogate's threshold through the real detector
    thr = model.nis_threshold
    n = 2
    run = FilterRun(t_s=np.array([0.0, 60.0]), x=np.zeros((n, 6)), P=np.zeros((n, 6, 6)), x_pred=np.zeros((n, 6)),
                    P_pred=np.zeros((n, 6, 6)), innov=np.zeros((n, 2)), S=np.tile(np.eye(2), (n, 1, 1)),
                    nis=np.array([0.999 * thr, 1.001 * thr]), meas=[None, None], object_id="SIM-TEST")
    det = nis_test(run, alpha=model.alpha, m=2)
    assert [d.index for d in det] == [1] and det[0].threshold == pytest.approx(thr, rel=1e-12)


def test_zero_maneuver_rate_reports_no_burns():
    arch = [Architecture("G+DRO+L1", [SensorSpec("dro"), SensorSpec("l1_halo")], True)]
    quiet = evaluate(arch, n_mc=1, horizon_days=1.0, seed=1, maneuver_model=ManeuverModel(rate_per_object_per_day=0.0, alpha=0.05))
    st = quiet.scores[0].stats
    assert st["n_burns"] == 0 and st["detections_pct"] is None and st["detection_latency_mean_h"] is None
    assert st["detection_latency_censored_mean_h"] is None
    assert st["n_quiet_obs_sampled"] > 0
    # the sampled false-alarm rate is alpha by construction: it is reported, not used as validation
    assert 0.0 <= st["false_alarm_rate_per_obs"] <= 1.0


def test_large_burns_are_detected_faster_than_small_ones():
    arch = [Architecture("G+DRO+L1", [SensorSpec("dro"), SensorSpec("l1_halo")], True)]
    small = evaluate(arch, n_mc=2, horizon_days=2.0, seed=3,
                     maneuver_model=ManeuverModel(rate_per_object_per_day=5.0, dv_mps_range=(0.05, 0.1)))
    large = evaluate(arch, n_mc=2, horizon_days=2.0, seed=3,
                     maneuver_model=ManeuverModel(rate_per_object_per_day=5.0, dv_mps_range=(20.0, 50.0)))
    s, l = small.scores[0].stats, large.scores[0].stats
    print(f"\nsmall burns (0.05-0.1 m/s): {s['n_detected']}/{s['n_burns']} detected, censored latency {s['detection_latency_censored_mean_h']:.2f} h; "
          f"large (20-50 m/s): {l['n_detected']}/{l['n_burns']}, {l['detection_latency_censored_mean_h']:.2f} h")
    assert s["n_burns"] == l["n_burns"] == 22   # rate 5/day x 2 d -> every object burns (Poisson(10) >= 1 w.p. ~1)
    assert l["detections_pct"] >= s["detections_pct"]
    assert l["detection_latency_censored_mean_h"] < s["detection_latency_censored_mean_h"]


def test_target_population_override_changes_photometric_coverage():
    """target_radius_m / target_albedo (the studio page's reference population) override the catalog physical
    parameters: a 5 m bright population is seen far more often than a 0.3 m dark one."""
    arch = [Architecture("G", [], True)]
    big = evaluate(arch, n_mc=1, seed=2, config=EvalConfig(horizon_days=1.0, target_radius_m=5.0, target_albedo=0.6))
    small = evaluate(arch, n_mc=1, seed=2, config=EvalConfig(horizon_days=1.0, target_radius_m=0.3, target_albedo=0.05))
    cb, cs = big.scores[0].stats["coverage_pct"], small.scores[0].stats["coverage_pct"]
    print(f"\ntarget override: 5 m / 0.6 -> coverage {cb:.1f} %, 0.3 m / 0.05 -> {cs:.1f} %")
    assert cb > cs
    assert big.config.as_dict()["target_note"].startswith("reference object population")


def test_validation_errors():
    with pytest.raises(ValueError):
        evaluate([], n_mc=1)
    with pytest.raises(ValueError):
        evaluate([Architecture("a", [], True), Architecture("a", [], True)], n_mc=1, horizon_days=1.0)
    with pytest.raises(ValueError):
        evaluate([Architecture("a", [], True)], n_mc=0, horizon_days=1.0)
    with pytest.raises(ValueError):
        evaluate([Architecture("none", [], ground=False)], n_mc=1, horizon_days=1.0)
    with pytest.raises(ValueError):   # real Horizons objects are refused (no simulated burns on real spacecraft)
        evaluate([Architecture("a", [], True)], n_mc=1, horizon_days=1.0, object_set="all")
    with pytest.raises(ValueError):
        EvalConfig(horizon_days=20.0)
    with pytest.raises(ValueError):   # phasing span + horizon must stay inside the 14-day cached truth window
        EvalConfig(horizon_days=7.0, phasing_span_days=12.0)
    assert EvalConfig(horizon_days=7.0, phasing_span_days=7.0).phasing_span() == pytest.approx(7.0 * 86400.0)
    assert EvalConfig(horizon_days=7.0).phasing_span() == pytest.approx(7.0 * 86400.0)
    with pytest.raises(ValueError):
        EvalConfig(target_radius_m=-1.0)
    with pytest.raises(ValueError):
        ManeuverModel(dv_mps_range=(5.0, 1.0))
    with pytest.raises(KeyError):
        evaluate([Architecture("a", [], True)], n_mc=1, horizon_days=1.0, object_set=["SIM-NOPE"])
    with pytest.raises(ValueError):
        evaluate([Architecture("a", [], True)], n_mc=2, horizon_days=1.0, workers=2, executor="gpu")
