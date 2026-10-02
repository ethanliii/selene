"""API contract review fixes: SPA fallback vs /api 404s, health readiness, 400-not-500 on bad input, request budgets,
timestamp normalisation (trailing Z), unquoted error details, rebuild gating and CORS defaults."""
from __future__ import annotations

import time

import pytest

from selene.api import app as app_module
from selene.api.routes import architecture as arch_route
from selene.api.routes import catalog as catalog_route
from selene.api.routes import od as od_route
from selene.api.routes import reachability as reach_route


# ---------------------------------------------------------------------------
# routing / SPA fallback
def test_unknown_api_paths_are_json_404_not_the_spa_shell(client):
    r = client.get("/api/nonexistent")
    assert r.status_code == 404 and r.headers["content-type"].startswith("application/json")
    assert "unknown API route" in r.json()["detail"]
    r = client.post("/api/nonexistent", json={})
    assert r.status_code == 404                                  # not 405 from a GET-only catch-all
    r = client.get("/api/catalog/objects/SIM-DRO-01/nope")
    assert r.status_code == 404 and r.headers["content-type"].startswith("application/json")


def test_trailing_slash_redirects_to_the_canonical_api_path(client):
    r = client.get("/api/sensors/", follow_redirects=False)
    assert r.status_code == 307 and r.headers["location"].endswith("/api/sensors")
    r = client.get("/api/health/?x=1", follow_redirects=False)
    assert r.status_code == 307 and r.headers["location"].endswith("/api/health?x=1")
    assert client.get("/api/health/").status_code == 200           # followed


def test_spa_shell_still_served_for_ui_routes(client):
    if not app_module.STATIC_DIR.exists():
        pytest.skip("frontend not built")
    r = client.get("/ops")
    assert r.status_code == 200 and r.headers["content-type"].startswith("text/html")


# ---------------------------------------------------------------------------
# health / readiness
def test_health_reports_data_readiness_and_routes(client):
    b = client.get("/api/health").json()
    assert b["status"] == "ok" and b["ephemeris"] == "de440s.bsp"
    assert b["data"]["present"]["de440s_kernel"] is True and b["data"]["required_missing"] == []
    assert isinstance(b["offline"], bool) and "offline_meaning" in b
    assert set(b["routers_loaded"]) >= {"catalog", "coverage", "od", "reachability", "tasking", "architecture", "demo"}
    assert "/api/coverage" in b["routes"] and "/api/od/run" in b["routes"] and len(b["routes"]) > 20
    assert "*" not in b["cors_origins"]


def test_missing_kernel_is_503_with_hint_not_500(client, monkeypatch):
    def boom():
        raise FileNotFoundError("DE440s kernel not found at /nowhere/data/cache/de440s.bsp; run `make setup` to download it.")

    monkeypatch.setattr(catalog_route, "kernel_span_tdb_s", boom)
    r = client.get("/api/ephemeris/bodies?n=2")
    assert r.status_code == 503, r.text
    b = r.json()
    assert b["status"] == "degraded" and "make setup" in b["hint"] and "data" in b


def test_health_degraded_when_kernel_missing(client, monkeypatch, tmp_path):
    monkeypatch.setattr(app_module, "DE440S_PATH", tmp_path / "missing.bsp")
    b = client.get("/api/health").json()
    assert b["status"] == "degraded" and "de440s_kernel" in b["data"]["required_missing"] and b["ephemeris"] is None


# ---------------------------------------------------------------------------
# 400 instead of 500 on bad input
def test_visibility_bad_or_reversed_window_is_400(client):
    r = client.get("/api/sensors/haleakala/visibility", params={"object_id": "SIM-DRO-01", "t0": "garbage"})
    assert r.status_code == 400 and "could not parse t0" in r.json()["detail"]
    r = client.get("/api/sensors/haleakala/visibility", params={"object_id": "SIM-DRO-01", "t0": "2026-03-01T00:00:00Z",
                                                                 "t1": "2026-02-01T00:00:00Z"})
    assert r.status_code == 400 and "t1 must be after t0" in r.json()["detail"]


def test_visibility_echoes_normalised_utc_and_both_time_arrays(client):
    r = client.get("/api/sensors/haleakala/visibility", params={"object_id": "SIM-DRO-01", "n": 4})
    assert r.status_code == 200, r.text
    b = r.json()
    assert b["t0"] == b["t0_utc"] == "2026-03-01T00:00:00.000Z" and b["t1"] == b["t1_utc"] and b["t1"].endswith("Z")
    assert b["tdb_s"] == b["t_s"] and len(b["epochs_utc"]) == 4 and all(e.endswith("Z") for e in b["epochs_utc"])


def test_coverage_bad_input_is_4xx_with_field_errors(client):
    assert client.post("/api/coverage", json={"t0": "garbage"}).status_code == 400
    r = client.post("/api/coverage", json={"network": {"ground_ids": [], "space": [{"id": "x"}]}, "n_t": 4})
    assert r.status_code == 422 and "platform_orbit" in r.text and "__init__" not in r.text
    r = client.post("/api/coverage", json={"network": {"ground_ids": [], "space": [{"id": "x", "platform_orbit": "geo", "bogus": 1}]}, "n_t": 4})
    assert r.status_code == 422 and "bogus" in r.text
    # a misspelt top-level key must not silently run the default network
    r = client.post("/api/coverage", json={"preset": "full", "n_t": 4})
    assert r.status_code == 422 and "preset" in r.text
    r = client.post("/api/coverage", json={"network": "bogus"})
    assert r.status_code == 400 and not r.json()["detail"].startswith('"')


def test_coverage_echoes_normalised_window_and_epochs(client):
    r = client.post("/api/coverage", json={"n_t": 3, "t0": "2026-03-01T00:00:00"})
    assert r.status_code == 200, r.text
    b = r.json()
    assert b["meta"]["t0_utc"] == "2026-03-01T00:00:00.000Z" and b["meta"]["t1_utc"].endswith("Z")
    p = b["per_time"][0]
    assert p["t"] == p["tdb_s"] and p["utc"] == b["epochs_utc"][0] == b["meta"]["t0_utc"]
    assert len(b["tdb_s"]) == len(b["epochs_utc"]) == 3


def test_epoch_outside_kernel_span_is_400_everywhere(client):
    for url, kw in (("/api/ephemeris/bodies", dict(params={"t0": "1500-01-01T00:00:00Z"})),
                    ("/api/catalog/objects/SIM-DRO-01/trajectory", dict(params={"t0": "2200-01-01T00:00:00Z"}))):
        r = client.get(url, **kw)
        assert r.status_code == 400, (url, r.text)
        assert "DE440s" in r.json()["detail"] and "1849" in r.json()["detail"]
    r = client.post("/api/coverage", json={"t0": "1500-01-01T00:00:00Z", "n_t": 4})
    assert r.status_code == 400 and "DE440s" in r.json()["detail"]


def test_timestamps_carry_the_utc_designator(client):
    assert client.get("/api/ephemeris/bodies?n=2").json()["epochs"][0].endswith("Z")
    tr = client.get("/api/catalog/objects/SIM-DRO-01/trajectory?n=2").json()
    assert tr["t0_utc"].endswith("Z") and tr["epochs_utc"][0].endswith("Z")
    assert client.get("/api/catalog/objects/SIM-DRO-01").json()["epoch_utc"].endswith("Z")
    assert client.get("/api/catalog/objects").json()["epoch_utc"].endswith("Z")


def test_error_details_are_not_double_quoted(client):
    for url, body in (("/api/od/run", {"sensors": ["nope"]}), ("/api/tasking/schedule", {"preset": "bogus"}),
                      ("/api/tasking/schedule", {"object_ids": ["nope"]}), ("/api/maneuver/detect", {"sensors": ["nope"]})):
        r = client.post(url, json=body)
        assert r.status_code in (400, 404), (url, r.text)
        d = r.json()["detail"]
        assert isinstance(d, str) and not d.startswith('"') and not d.startswith("'"), (url, d)


# ---------------------------------------------------------------------------
# request budgets
def test_od_budget_rejects_runaway_shapes_and_clamps_particle_export(client):
    r = client.post("/api/od/run", json={"t1": "2026-03-15T00:00:00Z", "cadence_min": 5})
    assert r.status_code == 400 and "visibility evaluations" in r.json()["detail"]
    assert client.post("/api/od/run", json={"max_obs": 2000}).status_code == 422
    r = client.post("/api/od/run", json={"method": "ukf", "sensors": ["geo_west"], "cadence_min": 720,
                                         "particles": {"n": 50, "t_grid_h": 336, "step_h": 0.5, "max_export": 2000}})
    assert r.status_code == 200, r.text
    b = r.json()
    assert b["t0_utc"].endswith("Z") and len(b["caps_applied"]) == 2
    assert len(b["particles"]["frames"]) <= od_route.MAX_PARTICLE_FRAMES
    assert len(b["particles"]["frames"]) * len(b["particles"]["frames"][0]["positions_km"]) <= od_route.MAX_PARTICLE_POINTS
    assert len(r.content) < 8e6


def test_reachability_path_points_are_capped(client):
    t = time.perf_counter()
    r = client.post("/api/reachability", json={"object_id": "SIM-DRO-01", "n_samples": 4000, "include_paths": True,
                                               "path_dt_h": 1, "include_hints": False, "refine": False})
    assert r.status_code == 200, r.text
    b = r.json()
    n = b["samples"]["n"]
    n_steps = len(b["samples"]["paths"]["t_h"])
    assert n * n_steps <= reach_route.MAX_PATH_POINTS * 1.05
    assert b["config"]["caps_applied"] and "path_dt_h" in b["config"]["caps_applied"][0]
    assert len(r.content) < 15e6 and time.perf_counter() - t < 60.0


def test_architecture_per_draw_budget_rejects_unparallelisable_shapes(client):
    archs = [{"name": f"a{i}", "ground": True} for i in range(8)]
    r = client.post("/api/architecture/evaluate", json={"architectures": archs, "n_mc": 16, "horizon_days": 7.0, "slot_min": 5})
    assert r.status_code == 400 and "work units per draw" in r.json()["detail"]
    # the budget rule accounts for the worker count: n_mc never exceeds workers x (per-worker budget / per-draw units)
    assert arch_route.max_n_mc_for(8, 7.0, 20.0, workers=4) <= 4 * (arch_route.MAX_UNITS_PER_WORKER // (8 * 504))
    assert arch_route.max_n_mc_for(4, 2.0, 20.0, workers=4) >= 8        # the advertised default stays reachable
    assert arch_route.max_n_mc_for(4, 7.0, 20.0, workers=1) == 2         # sequential fallback is budgeted too
    assert client.get("/api/architecture/presets").json()["limits"]["work_units_per_worker"] == arch_route.MAX_UNITS_PER_WORKER


def test_maneuver_span_boundary_and_observation_budget(client):
    r = client.post("/api/maneuver/detect", json={"t0": "2026-03-01T00:00:00Z", "t1": "2026-03-31T00:00:00Z", "cadence_min": 1440})
    assert r.status_code == 200, r.text                                  # exactly 30 days is accepted
    r = client.post("/api/maneuver/detect", json={"t1": "2026-03-30T23:59:00Z", "cadence_min": 30, "max_obs_per_epoch": 6})
    assert r.status_code == 400 and "filter updates" in r.json()["detail"]


def test_orbit_families_payload_cap(client):
    r = client.get("/api/orbits/families?max_members=0&n_samples=2000")
    assert r.status_code == 400 and "payload cap" in r.json()["detail"]
    r = client.get("/api/orbits/families?max_members=0&n_samples=200")
    assert r.status_code == 200 and len(r.content) < 7e6


def test_tasking_time_budget_is_capped(client):
    assert client.post("/api/tasking/schedule", json={"time_budget_s": 300}).status_code == 422


# ---------------------------------------------------------------------------
# demo rebuild gating and CORS
def test_demo_rebuild_is_gated_by_default(client, monkeypatch):
    monkeypatch.delenv("SELENE_DEMO_SCENARIO", raising=False)
    monkeypatch.delenv("SELENE_ALLOW_REBUILD", raising=False)
    r = client.post("/api/demo/rebuild", params={"fast": "true"})
    assert r.status_code == 403 and "SELENE_ALLOW_REBUILD" in r.json()["detail"]


def test_cors_defaults_to_local_origins_only(client):
    r = client.options("/api/coverage", headers={"Origin": "http://evil.example", "Access-Control-Request-Method": "POST"})
    assert r.headers.get("access-control-allow-origin") is None
    r = client.options("/api/coverage", headers={"Origin": "http://localhost:5173", "Access-Control-Request-Method": "POST"})
    assert r.status_code == 200 and r.headers.get("access-control-allow-origin") == "http://localhost:5173"


def test_cors_env_override(monkeypatch):
    monkeypatch.setenv("SELENE_CORS_ORIGINS", "https://ops.example, https://b.example")
    assert app_module.cors_origins() == ["https://ops.example", "https://b.example"]
    monkeypatch.setenv("SELENE_CORS_ORIGINS", "*")
    assert app_module.cors_origins() == ["*"]
