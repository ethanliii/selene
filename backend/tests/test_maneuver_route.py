"""POST /api/maneuver/detect contract tests (synthetic, SIMULATED burns only)."""
from __future__ import annotations

import pytest

BODY = {
    "object_id": "SIM-DRO-01",
    "t0": "2026-03-01T00:00:00",
    "t1": "2026-03-07T00:00:00",
    "sensors": ["dro_obs", "l1_halo_obs", "geo_west"],
    "cadence_min": 360,
    "sigma_arcsec": 1.0,
    "alpha": 0.01,
    "seed": 3,
}


def test_detect_with_injected_burn(client):
    body = dict(BODY, injected={"t_burn_utc": "2026-03-04T02:00:00", "magnitude_mps": 5.0, "direction": "prograde"})
    r = client.post("/api/maneuver/detect", json=body)
    assert r.status_code == 200, r.text
    d = r.json()
    assert d["object_id"] == "SIM-DRO-01" and d["kind"] == "simulated" and d["label"] == "SIMULATED"
    assert d["filter"]["name"] in ("ukf", "simple_ekf", "ekf") and d["filter"]["n_updates"] >= 15
    assert d["filter"]["source_note"]
    assert d["declared"] is True and d["detections"] and d["status"] == "maneuver_declared"
    assert d["truth"]["magnitude_mps"] == pytest.approx(5.0)
    # detected AFTER the burn (a declaration before it would be a false alarm, not a detection)
    assert d["truth"]["detected"] is True and d["truth"]["premature_declaration"] is False
    assert d["truth"]["detection_latency_s"] > 0 and 1 <= d["truth"]["detected_after_n_post_burn_obs"] <= 2
    assert d["truth"]["note"].startswith("SIMULATED")
    assert d["summary"]["filter_health"]["baseline_established"] is True
    sd = d["filter"]["sigma_diag_km_mps"]
    assert len(sd) == d["filter"]["n_updates"] and all(v > 0 for v in sd[5]) and sd[5][3] < 10.0  # velocity σ in m/s survives rounding
    assert d["config"]["prior"]["regime"] == "cislunar" and d["config"]["prior"]["sigma_pos0_km"] == 20.0
    est = d["dv_estimate"]
    assert est is not None and est["n_obs"] >= 2
    assert abs(d["truth"]["estimate_error"]["magnitude_error_pct"]) < 20
    assert d["truth"]["estimate_error"]["direction_error_deg"] < 20
    assert est["classification"]["heuristic"] is True and est["t_burn_utc"]
    assert set(d["timing"]) >= {"truth_s", "measurements_s", "filter_s", "detection_s", "estimation_s", "total_s"}
    assert d["summary"]["alpha"] == 0.01 and d["summary"]["threshold_nis"] == pytest.approx(9.2103, abs=1e-3)
    for det in d["detections"][:3]:
        assert {"t_utc", "test", "statistic", "threshold", "p_value", "confidence"} <= set(det)


def test_detect_no_burn_is_quiet_with_ekf(client):
    body = dict(BODY, filter="ekf")
    r = client.post("/api/maneuver/detect", json=body)
    assert r.status_code == 200, r.text
    d = r.json()
    assert d["filter"]["name"] == "simple_ekf"
    assert d["declared"] is False and d["status"] == "quiet" and d["status_note"]
    assert d["dv_estimate"] is None and d["truth"] is None
    assert d["summary"]["counts"]["nis_window"] == 0 and d["summary"]["counts"]["gap_refit"] == 0
    assert d["summary"]["nees"]["consistent"] in (True, False)  # reported honestly either way


def test_detect_component_dv_and_zero_burn(client):
    body = dict(BODY, injected={"t_burn_utc": "2026-03-04T02:00:00", "dv_mps": [0.0, 0.0, 0.0]}, filter="ekf", seed=8)
    r = client.post("/api/maneuver/detect", json=body)
    assert r.status_code == 200, r.text
    d = r.json()
    assert d["declared"] is False and d["truth"]["magnitude_mps"] == 0.0 and d["truth"]["detected"] is False


def test_detect_validation_errors(client):
    assert client.post("/api/maneuver/detect", json=dict(BODY, object_id="NOPE")).status_code == 404
    assert client.post("/api/maneuver/detect", json=dict(BODY, t1="2026-02-01T00:00:00")).status_code == 400
    assert client.post("/api/maneuver/detect", json=dict(BODY, t1="2026-06-01T00:00:00")).status_code == 400
    bad_dir = dict(BODY, injected={"t_burn_utc": "2026-03-04T02:00:00", "magnitude_mps": 5.0, "direction": "sideways"})
    assert client.post("/api/maneuver/detect", json=bad_dir).status_code == 400
    outside = dict(BODY, injected={"t_burn_utc": "2026-03-09T02:00:00", "magnitude_mps": 5.0, "direction": "prograde"})
    assert client.post("/api/maneuver/detect", json=outside).status_code == 400
    incomplete = dict(BODY, injected={"t_burn_utc": "2026-03-04T02:00:00", "magnitude_mps": 5.0})
    assert client.post("/api/maneuver/detect", json=incomplete).status_code == 400
    assert client.post("/api/maneuver/detect", json=dict(BODY, bogus=1)).status_code == 422
    assert client.post("/api/maneuver/detect", json=dict(BODY, sensors=["nope"])).status_code == 404


def test_real_objects_refused_with_and_without_burn(client):
    """Issue 3: no maneuver verdict is ever produced for a real (Horizons) spacecraft."""
    cat = client.get("/api/catalog/objects", params={"kind": "horizons"}).json()
    if not cat["objects"]:
        pytest.skip("no Horizons objects cached")
    oid = cat["objects"][0]["id"]
    body = dict(BODY, object_id=oid, injected={"t_burn_utc": "2026-03-04T02:00:00", "magnitude_mps": 5.0, "direction": "prograde"})
    r = client.post("/api/maneuver/detect", json=body)
    assert r.status_code == 400 and "SIMULATED" in r.json()["detail"]
    r = client.post("/api/maneuver/detect", json=dict(BODY, object_id=oid))
    assert r.status_code == 400 and "SIMULATED" in r.json()["detail"] and "Horizons" in r.json()["detail"]


def test_quiet_lunar_orbiter_default_request_not_declared(client):
    """Issue 2: SIM-ELFO-01 with route defaults must not yield a maneuver (auto prior + gate)."""
    r = client.post("/api/maneuver/detect", json={"object_id": "SIM-ELFO-01", "t0": "2026-03-01T00:00:00", "seed": 0})
    assert r.status_code == 200, r.text
    d = r.json()
    assert d["declared"] is False and d["dv_estimate"] is None
    assert d["status"] in ("quiet", "filter_not_converged", "filter_inconsistent")
    assert d["config"]["prior"]["regime"] == "moon_bound" and d["config"]["prior"]["sigma_vel0_mps"] == 0.1
    # explicit wide prior with the reference EKF: the gate reports non-convergence, never a maneuver
    r = client.post("/api/maneuver/detect", json={"object_id": "SIM-ELFO-01", "t0": "2026-03-01T00:00:00", "seed": 0,
                                                  "filter": "ekf", "sigma_pos0_km": 20.0, "sigma_vel0_mps": 2.0})
    d = r.json()
    assert d["declared"] is False and d["status"] in ("filter_not_converged", "filter_inconsistent")
    assert d["summary"]["filter_health"]["warnings"]


def test_blind_network_status(client):
    r = client.post("/api/maneuver/detect", json=dict(BODY, sensors=["haleakala", "cerro_tololo", "teide", "siding_spring"]))
    assert r.status_code == 200, r.text
    d = r.json()
    assert d["status"] == "no_observations" and d["declared"] is False and d["filter"]["n_updates"] == 0
    assert "blind" in d["status_note"]


def test_24h_cadence_burn_not_attributed_before_it_happened(client):
    """Issue 1: at a coarse cadence the gap re-fit must not declare before the burn."""
    # 7-day span: 3 post-burn daily observations (96, 144, 168 h; 120 h is not visible) — the
    # 6-day default leaves only 2, the estimator's bare minimum, and a ≈ 50 % magnitude error
    body = dict(BODY, t1="2026-03-08T00:00:00", cadence_min=1440, seed=0, sensors=None, filter="ekf",
                injected={"t_burn_utc": "2026-03-04T02:00:00", "magnitude_mps": 5.0, "direction": "prograde"})
    r = client.post("/api/maneuver/detect", json=body)
    assert r.status_code == 200, r.text
    d = r.json()
    assert d["declared"] is True and d["truth"]["premature_declaration"] is False
    assert d["truth"]["detection_latency_s"] > 0 and d["truth"]["detected_after_n_post_burn_obs"] == 1
    assert abs(d["truth"]["estimate_error"]["magnitude_error_pct"]) < 20
    assert d["truth"]["estimate_error"]["direction_error_deg"] < 20
