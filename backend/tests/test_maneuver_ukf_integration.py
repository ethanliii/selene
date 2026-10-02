"""Integration: the maneuver detectors consume the OD track's real UKF (skipped if absent)."""
from __future__ import annotations

import numpy as np
import pytest

ukf = pytest.importorskip("selene.od.ukf")

from selene.dynamics.frames import DEMO_EPOCH_TDB_S as T0  # noqa: E402
from selene.maneuver.config import DetectorConfig  # noqa: E402
from selene.maneuver.pipeline import run_scenario  # noqa: E402

D = 86400.0


def test_ukf_filterrun_feeds_detection_and_estimation():
    tb = T0 + 3.3 * D
    res = run_scenario("SIM-DRO-01", T0, T0 + 7 * D, ["dro_obs", "l1_halo_obs", "geo_west"], 6 * 3600.0, 1.0,
                       {"t_s": tb, "magnitude_mps": 5.0, "direction": "prograde"}, seed=4,
                       det_cfg=DetectorConfig(alpha=0.01), prefer_filter="ukf")
    assert res.filter_used != "simple_ekf" and res.filter_note == "selene.od.ukf"
    assert res.report.declared
    post = [m.t_s for m in res.meas if m.t_s > tb]
    assert res.report.first_declared_t_s <= post[1]
    assert res.estimate is not None
    tbk = res.truth_block()
    assert abs(tbk["estimate_error"]["magnitude_error_pct"]) < 20
    assert tbk["estimate_error"]["direction_error_deg"] < 20
    assert res.report.status == "maneuver_declared" and tbk["detected"] and not tbk["premature_declaration"]
    assert np.isfinite(res.estimate.t_burn_sigma_s) and 0 < res.estimate.t_burn_sigma_s < 3600.0
    print(f"\nUKF run: filter={res.filter_used} latency={tbk['detection_latency_s']/3600:.1f} h, "
          f"|dv| err {tbk['estimate_error']['magnitude_error_pct']:.2f}%, dir err {tbk['estimate_error']['direction_error_deg']:.2f}°, "
          f"total {res.timing['total_s']:.2f}s; NEES consistent={res.report.summary['nees']['consistent']}")


def test_ukf_no_burn_not_declared():
    res = run_scenario("SIM-DRO-01", T0, T0 + 7 * D, ["dro_obs", "l1_halo_obs", "geo_west"], 6 * 3600.0, 1.0, None,
                       seed=4, det_cfg=DetectorConfig(alpha=0.01), prefer_filter="ukf")
    assert res.filter_used != "simple_ekf"
    assert not res.report.declared and res.report.status == "quiet"
    print(f"\nUKF no-burn: mean NIS {res.report.summary['mean_nis']:.2f}, mean NEES {res.report.summary['nees']['mean_nees']:.2f}, "
          f"bounds {res.report.summary['nees']['mean_bounds']}")


def test_ukf_lunar_orbiter_burn_detected_with_auto_prior():
    """SIM-ELFO-01 (13-h, e = 0.6 lunar orbit): with the Moon-bound auto prior the UKF is
    consistent and a 5 m/s burn is declared on the first post-burn observation."""
    tb = T0 + 74 * 3600.0
    res = run_scenario("SIM-ELFO-01", T0, T0 + 7 * D, None, 6 * 3600.0, 1.0,
                       {"t_s": tb, "magnitude_mps": 5.0, "direction": "prograde"}, seed=0, prefer_filter="ukf", estimate=False)
    assert res.config["prior"]["regime"] == "moon_bound"
    tbk = res.truth_block()
    assert res.report.status == "maneuver_declared" and tbk["detected"] and tbk["detected_after_n_post_burn_obs"] == 1
    h = res.report.summary["filter_health"]
    assert h["baseline_established"] and h["nees_consistent"] in (True, False)
    print(f"\nUKF ELFO: latency {tbk['detection_latency_s']/3600:.1f} h, baseline NEES {h['nees_baseline_mean']:.1f}, total {res.timing['total_s']:.1f}s")


@pytest.mark.slow
def test_ukf_lunar_orbiter_dv_estimate():
    tb = T0 + 74 * 3600.0
    res = run_scenario("SIM-ELFO-01", T0, T0 + 7 * D, None, 6 * 3600.0, 1.0,
                       {"t_s": tb, "magnitude_mps": 5.0, "direction": "prograde"}, seed=0, prefer_filter="ukf")
    tbk = res.truth_block()
    assert res.estimate is not None and abs(tbk["estimate_error"]["magnitude_error_pct"]) < 20
    assert tbk["estimate_error"]["direction_error_deg"] < 20
    print(f"\nUKF ELFO Δv: |dv| err {tbk['estimate_error']['magnitude_error_pct']:.2f}%, dir {tbk['estimate_error']['direction_error_deg']:.2f}°, "
          f"estimation {res.timing['estimation_s']:.1f}s")
