import time

import numpy as np


def test_od_run_default_pipeline(client):
    t = time.perf_counter()
    r = client.post("/api/od/run", json={"sensors": ["dro_obs", "geo_west"], "cadence_min": 180,
                                         "particles": {"n": 300, "t_grid_h": 48, "step_h": 12}})
    dt = time.perf_counter() - t
    assert r.status_code == 200, r.text
    assert dt < 30.0
    d = r.json()
    for key in ("iod", "batch", "ukf", "particles", "truth_error_km", "timing", "observations"):
        assert key in d
    assert d["observations"]["n_used"] >= 5
    assert d["iod"]["best"]["converged"] and d["iod"]["best"]["truth_error_pos_km"] < 5000
    assert d["batch"]["converged"] and d["batch"]["truth_error_pos_km"] < 50
    u = d["ukf"]
    assert u["n"] == len(d["truth_error_km"]) == len(u["epochs"]) == len(u["states"]) == len(u["covs"]) == len(u["nis"])
    assert len(u["sigma_pos_km"]) == u["n"] and d["truth_error_km"][-1] < 50
    p = d["particles"]
    assert len(p["frames"]) == 5 and len(p["frames"][0]["positions_km"]) == 300
    assert p["custody"]["sigma_pos_km_end"] >= p["custody"]["sigma_pos_km_start"]
    assert d["timing"]["total_s"] < 30
    # covariances survive the JSON round trip (significant-figure rounding, not fixed decimals)
    P = np.array(u["covs"][-1])
    assert np.all(np.diag(P)[3:] > 0) and np.linalg.det(P) > 0
    assert abs(np.sqrt(np.trace(P[3:, 3:])) - u["sigma_vel_km_s"][-1]) < 1e-6 * u["sigma_vel_km_s"][-1]
    assert np.linalg.det(np.array(d["batch"]["cov"])) > 0 and np.linalg.det(np.array(d["iod"]["best"]["P_ref"])) > 0
    assert u["meta"]["q_psd_km2_s3"] == 1e-18 and u["meta"]["iterated_update"] == 5
    m0 = d["observations"]["measurements"][0]
    assert abs(m0["sigma_rad"] - np.pi / 648000.0) < 1e-12
    assert d["iod"]["quality"] == "ok" and d["iod"]["usable_as_seed"] is True
    # deterministic with the same seed
    r2 = client.post("/api/od/run", json={"sensors": ["dro_obs", "geo_west"], "cadence_min": 180,
                                          "particles": {"n": 300, "t_grid_h": 48, "step_h": 12}})
    assert r2.json()["ukf"]["states"][-1] == u["states"][-1]


def test_od_run_ukf_only_and_ground_blind_spot(client):
    r = client.post("/api/od/run", json={"method": "ukf", "sensors": "space", "cadence_min": 240, "particles": None})
    assert r.status_code == 200
    d = r.json()
    assert d["iod"] is None and d["batch"] is None and d["particles"] is None and d["ukf"]["n"] > 0
    r = client.post("/api/od/run", json={"sensors": "ground", "cadence_min": 120})
    assert r.status_code == 200
    d = r.json()
    assert d["observations"]["n_used"] == 0 and d["ukf"] is None
    assert any("No usable observations" in n for n in d["notes"])
    assert "moon_exclusion" in d["observations"]["dropped_by_reason"]


def test_od_run_errors_and_presets(client):
    assert client.post("/api/od/run", json={"object_id": "nope"}).status_code == 404
    assert client.post("/api/od/run", json={"t1": "2026-02-01T00:00:00"}).status_code == 400
    assert client.post("/api/od/run", json={"sensors": ["no_such"]}).status_code == 400
    r = client.get("/api/od/presets")
    assert r.status_code == 200 and "ui_quick" in r.json()["presets"]


def test_od_run_14_day_window_and_iod_ambiguity_path(client):
    # exactly 14 days in UTC must be accepted (the cap used to be evaluated in TDB and reject it by 0.2 ms)
    r = client.post("/api/od/run", json={"method": "ukf", "sensors": ["geo_west"], "cadence_min": 24 * 60,
                                         "t0": "2026-03-01T00:00:00", "t1": "2026-03-15T00:00:00", "particles": None})
    assert r.status_code == 200, r.text
    assert client.post("/api/od/run", json={"t0": "2026-03-01T00:00:00", "t1": "2026-03-15T00:10:00"}).status_code == 400
    # three-observation IOD on the L2 halo: square problem; the route reports the quality flag and the batch fit
    # is started from every exact root (the post-fit cost discriminates)
    r = client.post("/api/od/run", json={"object_id": "SIM-L2-HALO-01", "method": "all", "sensors": ["dro_obs", "geo_west"],
                                         "cadence_min": 360, "iod_max_obs": 3, "particles": None})
    assert r.status_code == 200, r.text
    d = r.json()
    assert d["iod"]["square_problem"] is True and d["iod"]["quality"] in ("ok", "ambiguous", "not_converged")
    assert "candidates" in d["iod"] and all("truth_error_pos_km" in c for c in d["iod"]["candidates"])
    if d["iod"]["quality"] == "ambiguous":
        assert len(d["batch"]["starts_tried"]) >= 2
    assert d["batch"]["converged"] and d["batch"]["truth_error_pos_km"] < 100
