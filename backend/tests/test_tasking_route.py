"""FastAPI tests for POST /api/tasking/schedule and GET /api/tasking/presets."""
from __future__ import annotations

import time

import pytest


def test_presets(client):
    r = client.get("/api/tasking/presets")
    assert r.status_code == 200
    b = r.json()
    assert "default" in b["sensor_presets"] and "ground_only" in b["sensor_presets"]
    assert len(b["default_object_ids"]) == 8 and "compare" in b["methods"]
    assert isinstance(b["milp_available"], bool)


def test_default_schedule_under_20s(client):
    t = time.perf_counter()
    r = client.post("/api/tasking/schedule", json={})
    dt = time.perf_counter() - t
    assert r.status_code == 200, r.text
    b = r.json()
    print(f"\nROUTE default greedy: {dt:.1f} s wall, {len(r.content)/1e3:.0f} kB, custody {b['overall']['custody_pct']:.1f} %, "
          f"{b['overall']['n_observations']} obs, timing {b['timing']}")
    assert dt < 20.0
    assert b["method"] == "greedy"
    assert b["config"]["n_slots"] == 144 and len(b["config"]["object_ids"]) == 8 and len(b["config"]["sensor_ids"]) == 9
    assert b["config"]["q_psd_km2_s3"] == pytest.approx(1e-18)
    assert len(b["epochs_utc"]) == 145 and b["epochs_utc"][0].startswith("2026-03-01T00:00")
    assert len(b["per_object"]) == 8 and len(b["sensors"]) == 9
    assert len(b["schedule"]) == b["overall"]["n_observations"] > 0
    e = b["schedule"][0]
    for key in ("slot", "t_s", "sensor_id", "object_id", "gain", "p_acq", "sigma_before_km", "sigma_after_km", "visible_reasons"):
        assert key in e
    o = b["per_object"][0]
    for key in ("id", "custody_pct", "mean_tslo_h", "final_sigma_km", "sigma_series_km", "n_obs"):
        assert key in o
    assert len(o["sigma_series_km"]) == 145
    for key in ("custody_pct", "custody_pct_null", "custody_pct_observed", "mean_tslo_h", "summed_trace_final_km2",
                "summed_trace_series_km2", "custody_note"):
        assert key in b["overall"]
    # the default saturates: the zero-observation reference is also 100 % and the response says so
    assert b["overall"]["custody_pct_null"] == 100.0 and "custody_pct_null" in b["assumptions"][6]
    assert b["comparison"] == []
    assert any("SIMULATED" in a or "simulated" in a for a in [b["disclaimer"]])
    assert b["assumptions"]


def test_compare_mode(client):
    t = time.perf_counter()
    r = client.post("/api/tasking/schedule", json={"method": "compare", "include_series": False, "horizon_slots": 3})
    dt = time.perf_counter() - t
    assert r.status_code == 200, r.text
    b = r.json()
    methods = [row["method"] for row in b["comparison"]]
    print(f"\nROUTE compare: {dt:.1f} s wall; " + "; ".join(
        f"{row['method']}: custody {row['custody_pct']:.1f} %, mean trace {row['summed_trace_mean_km2']:.1f} km2, final {row['summed_trace_final_km2']:.2f}"
        for row in b["comparison"]))
    assert dt < 20.0
    assert methods == ["greedy", "milp", "random", "round_robin"]
    assert b["method"].startswith("compare")
    assert "sigma_series_km" not in b["per_object"][0] and "summed_trace_series_km2" not in b["overall"]
    rows = {row["method"]: row for row in b["comparison"]}
    assert rows["milp"]["objective_milp_total"] >= rows["milp"]["objective_greedy_surrogate_total"]
    assert rows["milp"]["n_filled_idle"] == 0 and rows["milp"]["n_greedy_kept"] == 0 and not rows["milp"]["budget_exhausted"]
    assert rows["greedy"]["summed_trace_mean_km2"] <= rows["random"]["summed_trace_mean_km2"]
    assert rows["greedy"]["summed_trace_mean_km2"] <= rows["round_robin"]["summed_trace_mean_km2"]
    assert all(row["custody_pct_null"] == 100.0 for row in b["comparison"])


def test_milp_route_is_bounded_by_time_budget(client):
    """horizon 12 over 48 h ran ~80 s unbounded (review); with time_budget_s the route returns in
    budget and flags the greedy fallback."""
    t = time.perf_counter()
    r = client.post("/api/tasking/schedule", json={"method": "milp", "horizon_slots": 12, "time_budget_s": 6.0, "include_series": False})
    dt = time.perf_counter() - t
    assert r.status_code == 200, r.text
    b = r.json()
    print(f"\nROUTE milp horizon 12 / budget 6 s: {dt:.1f} s wall, budget_exhausted={b['overall']['budget_exhausted']}, "
          f"{b['overall']['n_budget_exhausted']} greedy slots, timing {b['timing']}")
    assert dt < 20.0
    assert b["overall"]["budget_exhausted"] is True and b["overall"]["n_budget_exhausted"] > 0
    assert b["overall"]["n_observations"] > 0
    # the horizon cap itself
    r = client.post("/api/tasking/schedule", json={"method": "milp", "horizon_slots": 24})
    assert r.status_code == 422


def test_ground_only_reports_zero_observations_honestly(client):
    r = client.post("/api/tasking/schedule", json={"preset": "ground_only", "include_series": False, "t1": "2026-03-02T00:00:00"})
    assert r.status_code == 200, r.text
    b = r.json()
    assert b["overall"]["n_observations"] == 0 and b["overall"]["n_objects_never_observed"] == 8
    assert b["overall"]["custody_pct_observed"] == 0.0
    assert all(s["n_obs"] == 0 and s["blocked_reason_counts"]["moon_exclusion"] > 0 for s in b["sensors"])


def test_explicit_objects_and_sensors_and_methods(client):
    body = {"object_ids": ["SIM-DRO-01", "SIM-NRHO-RELAY-01"], "sensor_ids": ["dro_obs"], "t1": "2026-03-01T06:00:00",
            "slot_min": 30, "method": "round_robin", "include_series": False, "custody_threshold_km": 5.0}
    r = client.post("/api/tasking/schedule", json=body)
    assert r.status_code == 200, r.text
    b = r.json()
    assert b["method"] == "round_robin" and b["config"]["n_slots"] == 12 and b["config"]["custody_threshold_km"] == 5.0
    assert {o["id"] for o in b["per_object"]} == {"SIM-DRO-01", "SIM-NRHO-RELAY-01"}
    r = client.post("/api/tasking/schedule", json={**body, "method": "milp", "horizon_slots": 2})
    assert r.status_code == 200 and r.json()["method"] in ("milp", "local_search")
    r = client.post("/api/tasking/schedule", json={**body, "method": "random", "seed": 3})
    assert r.status_code == 200
    # duplicate ids are collapsed (they would double-count every metric)
    r = client.post("/api/tasking/schedule", json={**body, "object_ids": ["SIM-DRO-01", "SIM-DRO-01"]})
    assert r.status_code == 200 and r.json()["overall"]["n_objects"] == 1
    # acquisition models and search tiles are accepted and echoed
    r = client.post("/api/tasking/schedule", json={**body, "acquisition": "irf", "search_tiles": 9})
    assert r.status_code == 200 and r.json()["config"]["acquisition"] == "irf" and r.json()["config"]["search_tiles"] == 9


@pytest.mark.parametrize("body,status", [
    ({"method": "bogus"}, 422),
    ({"preset": "nope"}, 400),
    ({"object_ids": ["NOPE"]}, 400),
    ({"t1": "2026-02-01T00:00:00"}, 400),
    ({"slot_min": 1, "t1": "2026-03-10T00:00:00"}, 400),
    ({"t0": "not a date"}, 400),
])
def test_validation_errors(client, body, status):
    r = client.post("/api/tasking/schedule", json=body)
    assert r.status_code == status, r.text
