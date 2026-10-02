"""POST /api/reachability, GET /api/reachability/regions, /ladder."""
from __future__ import annotations

import time

import numpy as np
import pytest


def test_regions_endpoint(client):
    r = client.get("/api/reachability/regions")
    assert r.status_code == 200
    d = r.json()
    keys = [x["key"] for x in d["regions"]]
    assert {"l1_gateway", "l2_gateway", "nrho_corridor", "south_pole_approach", "llo_shell", "geo_belt_return",
            "earth_return_escape", "lunar_impact"} <= set(keys)
    for x in d["regions"]:
        assert x["geometry"]["type"] and x["why_it_matters"]
    assert "disclaimer" in d


def test_ladder_endpoint(client):
    r = client.get("/api/reachability/ladder", params={"dv_budget_mps": 37})
    assert r.status_code == 200
    assert r.json()["magnitudes_mps"] == [2.0, 5.0, 10.0, 20.0, 37.0]


def test_default_request_shape_and_timing(client):
    t = time.perf_counter()
    r = client.post("/api/reachability", json={"object_id": "SIM-DRO-01"})
    dt = time.perf_counter() - t
    assert r.status_code == 200, r.text
    d = r.json()
    print(f"default /api/reachability: {dt:.2f} s wall, n_samples={d['samples']['n']}, timing={d['timing']}")
    assert dt < 15.0
    assert d["object"]["id"] == "SIM-DRO-01" and d["object"]["label"] == "SIMULATED"
    assert d["config"]["dv_budget_mps"] == 50.0 and d["config"]["horizon_h"] == 168.0     # week-scale default
    n = d["samples"]["n"]
    assert n == d["config"]["n_dirs"] * len(d["config"]["magnitudes_mps"]) * 4 == d["config"]["n_samples_actual"]
    for k in ("end_rot", "end_gcrf", "dv_dir_gcrf", "hit_regions", "hit_region", "primary_region", "terminated",
              "termination_t_h"):
        assert len(d["samples"][k]) == n
    assert d["samples"]["hit_region"] == d["samples"]["primary_region"]
    assert len(d["samples"]["end_rot"][0]) == 3
    assert "paths" not in d["samples"]
    assert d["nominal"]["t_h"][0] == 0.0 and d["nominal"]["t_h"][-1] == 168.0
    assert len(d["nominal"]["rot"]) == len(d["nominal"]["t_h"])
    reg = {x["key"]: x for x in d["regions"]}
    assert set(reg) >= {"l1_gateway", "l2_gateway", "nrho_corridor", "geo_belt_return"}
    # with the week-scale default the demo table is not empty: L2 is on the nominal path (min_dv 0),
    # L1 and the NRHO corridor open up with 50 m/s
    assert reg["l2_gateway"]["nominal_hits"] and reg["l2_gateway"]["min_dv_mps"] == 0.0
    assert reg["l1_gateway"]["newly_reachable"] and reg["l1_gateway"]["min_dv_mps"] > 0
    assert reg["l1_gateway"]["refined_min_dv"]["dv_mps"] <= reg["l1_gateway"]["min_dv_mps"]
    for x in reg.values():
        assert 0.0 <= x["fraction"] <= 1.0 and 0.0 <= x["sample_fraction"] <= 1.0
        if x["n_hit"] == 0 and not x["nominal_hits"]:
            assert x["earliest_h"] is None and x["min_dv_mps"] is None
        if x["refined_min_dv"]:
            assert x["refined_min_dv"]["dv_mps"] <= x["min_dv_mps"] + 1e-9
    assert d["envelope"][0]["t_h"] == 0.0 and d["envelope"][-1]["t_h"] == 168.0
    assert "semi_axes_km" in d["meta"]["envelope_definitions"]
    assert d["envelope"][-1]["max_radius_km"] > d["envelope"][1]["max_radius_km"]
    hints = d["sensor_hints"]
    assert len(hints["steps"]) == len(d["envelope"])
    assert hints["best_overall"] in hints["summary"]
    assert "ASSUMED" in d["disclaimer"]
    assert d["timing"]["route_total_s"] > 0


def test_paths_zero_budget_and_overrides(client):
    r = client.post("/api/reachability", json={"object_id": "SIM-DRO-01", "dv_budget_mps": 0, "horizon_h": 24,
                                               "n_dirs": 3, "include_paths": True, "include_hints": False,
                                               "refine": False, "path_dt_h": 12})
    assert r.status_code == 200, r.text
    d = r.json()
    n = d["samples"]["n"]
    assert n == 3 * 1 * 4
    p = d["samples"]["paths"]
    assert p["t_h"] == [0.0, 12.0, 24.0]
    assert np.asarray(p["rot"]).shape == (n, 3, 3)
    nom = np.asarray(d["nominal"]["rot"])
    assert np.allclose(np.asarray(p["rot"]), nom[None], atol=1e-4)
    assert d["nominal"]["hit_regions"] == [k for k, x in {x["key"]: x for x in d["regions"]}.items() if x["n_hit"]]
    assert all(x["newly_reachable"] is False for x in d["regions"])
    assert d["sensor_hints"]["steps"] == []


def test_real_object_is_allowed_with_what_if_disclaimer(client):
    cat = client.get("/api/catalog/objects", params={"kind": "horizons"}).json()
    if not cat["objects"]:
        pytest.skip("no Horizons objects cached at the demo epoch")
    oid = cat["objects"][0]["id"]
    r = client.post("/api/reachability", json={"object_id": oid, "n_dirs": 4, "include_hints": False, "refine": False,
                                               "horizon_h": 24, "burn_epochs_h": [0]})
    assert r.status_code == 200, r.text
    assert "REAL" in r.json()["disclaimer"]


def test_validation_errors(client):
    assert client.post("/api/reachability", json={"object_id": "NOPE"}).status_code == 404
    assert client.post("/api/reachability", json={"object_id": "SIM-DRO-01", "horizon_h": 12}).status_code == 422
    assert client.post("/api/reachability", json={"object_id": "SIM-DRO-01", "horizon_h": 24,
                                                  "burn_epochs_h": [0, 48]}).status_code == 400
    assert client.post("/api/reachability", json={"object_id": "SIM-DRO-01", "sensors": ["nope"], "n_dirs": 2}).status_code == 400
    assert client.post("/api/reachability", json={"object_id": "SIM-DRO-01", "bogus": 1}).status_code == 422
    # malformed magnitude lists are 400s, never 500s
    for mags, budget in (([-5], 50), ([], 50), ([500], 10), ([0, -1], 50)):
        r = client.post("/api/reachability", json={"object_id": "SIM-DRO-01", "magnitudes_mps": mags, "n_dirs": 3,
                                                   "dv_budget_mps": budget, "horizon_h": 24})
        assert r.status_code == 400, (mags, budget, r.status_code, r.text[:200])
        assert "magnitudes_mps" in r.json()["detail"]
    # explicit n_dirs that would exceed the total-sample cap is refused up front
    r = client.post("/api/reachability", json={"object_id": "SIM-DRO-01", "n_dirs": 4000, "dv_budget_mps": 1000})
    assert r.status_code == 400 and "cap" in r.json()["detail"]


def test_sample_cap_and_seed_compat(client):
    """n_samples at the maximum with the full ladder is clamped to the server cap (never exceeded),
    and a client sending the legacy 'seed' field is accepted."""
    r = client.post("/api/reachability", json={"object_id": "SIM-DRO-01", "n_samples": 20000, "dv_budget_mps": 1000,
                                               "horizon_h": 24, "include_hints": False, "refine": False, "seed": 7})
    assert r.status_code == 200, r.text[:300]
    d = r.json()
    assert d["samples"]["n"] <= 20000 and d["config"]["n_dirs_clamped"] is True
    assert d["samples"]["n"] == d["config"]["n_dirs"] * len(d["config"]["magnitudes_mps"]) * 4
    assert d["timing"]["route_total_s"] < 15.0
