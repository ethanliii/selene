"""FastAPI tests for POST /api/architecture/evaluate and GET /api/architecture/presets.

Covers both request dialects (canonical and the studio page's ``frontend/src/studio/types.ts`` shape),
the studio-page response fields, n_mc clamping with reporting, the truth-window guard and schema errors.
"""
from __future__ import annotations

import time

import pytest

STUDIO_SCORE_KEYS = ("revisit_h", "detect_latency_h_mean", "detect_latency_h_p95", "n_mc", "n_sensors", "undetected_pct", "never_observed_pct")


def test_presets_endpoint(client):
    r = client.get("/api/architecture/presets")
    assert r.status_code == 200, r.text
    b = r.json()
    assert [p["name"] for p in b["presets"]] == ["Ground only", "Ground + 2 GEO", "Ground + L2 halo", "Ground + DRO + L1 halo"]
    assert {p["id"] for p in b["platforms"]} == {"geo", "l1_halo", "l2_halo_S", "dro", "nrho_9_2", "resonant_3_1"}
    assert "L2_halo" in next(p for p in b["platforms"] if p["id"] == "l2_halo_S")["aliases"]
    assert len(b["ground_network"]) == 9
    assert "coverage_pct" in b["metric_definitions"] and "detection_latency_mean_h" in b["metric_definitions"]
    assert b["defaults"]["n_mc"] == 8 and b["limits"]["n_mc"] >= 8
    # the advertised defaults are reachable: the presets at the default horizon allow >= the default n_mc
    ex = b["limits"]["max_n_mc_examples"]
    assert ex["4 presets, 2 d, 20-min slots"] >= b["defaults"]["n_mc"]
    assert ex["4 presets, 7 d, 20-min slots"] >= 8
    assert "clamped" in b["limits"]["rule"]
    assert "SIMULATED" in b["disclaimer"] and "notional actor" in b["disclaimer"]
    assert any("SATURATION" in n for n in b["method_notes"]) and any("by construction" in n for n in b["method_notes"])


def test_evaluate_presets_default_body(client):
    t = time.perf_counter()
    r = client.post("/api/architecture/evaluate", json={"n_mc": 2, "horizon_days": 2.0, "seed": 0})
    dt = time.perf_counter() - t
    assert r.status_code == 200, r.text
    b = r.json()
    print(f"\nROUTE presets n_mc=2: {dt:.1f} s wall ({b['meta']['timing']['executor']}), {len(r.content) / 1e3:.0f} kB")
    assert dt < 60.0
    names = [s["name"] for s in b["scores"]]
    assert names == ["Ground only", "Ground + 2 GEO", "Ground + L2 halo", "Ground + DRO + L1 halo"]
    for s in b["scores"]:
        for key in ("coverage_pct", "custody_pct", "custody_pct_null", "revisit_mean_h", "revisit_p95_h", "revisit_p95_censored_h",
                    "detection_latency_mean_h", "detection_latency_p95_h", "detection_latency_censored_mean_h", "detections_pct",
                    "n_space_sensors", "total_space_aperture_m", "notes", "n_burns", "n_detected", "false_alarm_rate_per_obs",
                    "replay_sigma_max_rel_err", *STUDIO_SCORE_KEYS):
            assert key in s, key
        for key in STUDIO_SCORE_KEYS:
            assert isinstance(s[key], (int, float)) and s[key] is not None, key
        assert s["revisit_h"] == s["revisit_mean_h"] and s["n_mc"] == 2
        assert s["detect_latency_h_mean"] == s["detection_latency_censored_mean_h"]
        assert s["undetected_pct"] == pytest.approx(100.0 - s["detections_pct"], abs=1e-3)
        print(f"  {s['name']:24s} cov {s['coverage_pct']:5.1f}  cust {s['custody_pct']:5.1f}  rev {s['revisit_mean_h']:5.1f} h "
              f"(p95 {s['revisit_p95_h']})  lat {s['detection_latency_mean_h']}  det {s['detections_pct']} %  n_space {s['n_space_sensors']}")
    assert set(b["per_object"]) == set(names)
    assert len(b["per_object"]["Ground only"]) == 11
    m = b["meta"]
    assert m["n_mc"] == 2 and m["seed"] == 0 and len(m["draws"]) == 2 and m["draws"][0]["t0_utc"].startswith("2026-03")
    assert m["n_mc_requested"] == 2 and m["caps_applied"] == []
    assert m["method_notes"] and any("SURROGATE" in n for n in m["method_notes"])
    assert m["config"]["q_psd_km2_s3"] == pytest.approx(1e-18)
    assert m["timing"]["route_total_s"] > 0
    assert "SIMULATED" in b["disclaimer"] and b["mock"] is False and "SELENE Monte Carlo" in b["method"]
    g, d = b["scores"][0], b["scores"][3]
    assert d["coverage_pct"] > g["coverage_pct"] and d["custody_pct"] > g["custody_pct"]


def test_evaluate_studio_page_dialect_and_n_mc_clamp(client):
    """The architecture studio page (frontend/src/studio/types.ts) sends ground_network / orbit / slew_rate_dps /
    target_* and asks for 100 draws: the route must accept it, clamp n_mc and say so."""
    body = {
        "architectures": [
            {"name": "A · Ground + L2 halo", "ground_network": True,
             "sensors": [{"orbit": "L2_halo", "aperture_m": 0.4, "limiting_mag": 18.5, "fov_deg": 2.0, "slew_rate_dps": 1.0, "phase": 0.0}]},
            {"name": "B · Ground + GEO + DRO", "ground_network": True,
             "sensors": [{"orbit": "GEO", "aperture_m": 0.3, "limiting_mag": 18.0, "fov_deg": 3.0, "slew_rate_dps": 1.0, "phase": 0.0},
                         {"orbit": "DRO", "aperture_m": 0.4, "limiting_mag": 18.5, "fov_deg": 2.0, "slew_rate_dps": 1.0, "phase": 0.25}]},
        ],
        "n_mc": 100, "horizon_days": 1.0, "seed": 20260301, "target_radius_m": 1.5, "target_albedo": 0.2,
    }
    t = time.perf_counter()
    r = client.post("/api/architecture/evaluate", json=body)
    dt = time.perf_counter() - t
    assert r.status_code == 200, r.text
    b = r.json()
    print(f"\nROUTE studio dialect, n_mc 100 requested: {dt:.1f} s, effective n_mc {b['meta']['n_mc']} ({b['meta']['timing']['executor']})")
    assert dt < 60.0
    assert b["meta"]["n_mc_requested"] == 100 and 1 <= b["meta"]["n_mc"] <= 16
    assert b["meta"]["caps_applied"] and "n_mc 100 ->" in b["meta"]["caps_applied"][0] and "Caps applied" in b["method"]
    assert b["meta"]["config"]["target_radius_m"] == 1.5 and b["meta"]["config"]["target_albedo"] == 0.2
    assert [s["name"] for s in b["scores"]] == ["A · Ground + L2 halo", "B · Ground + GEO + DRO"]
    for s in b["scores"]:
        for key in STUDIO_SCORE_KEYS:
            assert isinstance(s[key], (int, float)), key
        assert s["n_mc"] == b["meta"]["n_mc"]
    assert b["scores"][0]["architecture"]["sensors"][0]["platform"] == "l2_halo_S"
    assert [x["platform"] for x in b["scores"][1]["architecture"]["sensors"]] == ["geo", "dro"]
    assert b["scores"][1]["n_sensors"] == 2 + 9


def test_evaluate_custom_architectures_reproducible(client):
    body = {
        "architectures": [
            {"name": "Ground only", "ground": True, "sensors": []},
            {"name": "Ground + DRO", "ground": True,
             "sensors": [{"platform": "dro", "aperture_m": 0.5, "fov_deg": 2.0, "slew_rate_deg_s": 1.0}]},
            {"name": "Space only GEO pair", "ground": False,
             "sensors": [{"platform": "geo", "limiting_mag": 18.0, "lon_deg": -100.0}, {"platform": "geo", "limiting_mag": 18.0, "lon_deg": 60.0}]},
        ],
        "n_mc": 2, "horizon_days": 1.0, "seed": 11, "custody_threshold_km": 50.0,
        "maneuver_model": {"rate_per_object_per_day": 1.0, "dv_mps_range": [2.0, 10.0]},
    }
    r1 = client.post("/api/architecture/evaluate", json=body)
    r2 = client.post("/api/architecture/evaluate", json={**body, "workers": 1})   # sequential path, same answer
    assert r1.status_code == 200, r1.text
    assert r2.status_code == 200, r2.text
    b1, b2 = r1.json(), r2.json()
    strip = lambda b: {s["name"]: {k: v for k, v in s.items() if "runtime" not in k} for s in b["scores"]}
    assert strip(b1) == strip(b2)
    assert b1["meta"]["custody_threshold_km"] == 50.0
    assert b1["scores"][1]["total_space_aperture_m"] == 0.5 and b1["scores"][1]["n_space_sensors"] == 1
    assert b1["scores"][2]["architecture"]["n_ground_sites"] == 0
    assert b1["scores"][1]["coverage_pct"] > b1["scores"][0]["coverage_pct"]


@pytest.mark.parametrize("bad", [
    {"architectures": [{"name": "x", "sensors": [{"platform": "mars_orbit"}]}], "n_mc": 1},
    {"architectures": [{"name": "x", "sensors": [{"orbit": "Mars"}]}], "n_mc": 1},                       # unknown studio orbit
    {"architectures": [{"name": "x", "sensors": [{"orbit": "DRO", "platform": "dro"}]}], "n_mc": 1},     # alias and canonical together
    {"architectures": [{"name": "x", "sensors": [{"platform": "dro", "aperture_cm": 40}]}], "n_mc": 1},
    {"architectures": [{"name": "x", "sensors": [{"platform": "dro", "fov_deg": 0}]}], "n_mc": 1},
    {"architectures": [{"name": "x", "ground": True, "ground_network": True, "sensors": []}], "n_mc": 1},
    {"n_mc": 0},
    {"horizon_days": 30},
    {"horizon_days": 7.0, "phasing_span_days": 12.0, "n_mc": 1},                                          # outside the truth window
    {"target_radius_m": 0.0, "n_mc": 1},
    {"maneuver_model": {"dv_mps_range": [10, 1]}, "n_mc": 1},
    {"object_set": "all", "n_mc": 1},                                   # real Horizons objects refused
    {"architectures": [{"name": "none", "ground": False, "sensors": []}], "n_mc": 1},
    {"architectures": [{"name": "a", "sensors": []}, {"name": "a", "sensors": []}], "n_mc": 1},
    {"executor": "gpu", "n_mc": 1},
])
def test_evaluate_rejects_bad_requests(client, bad):
    r = client.post("/api/architecture/evaluate", json=bad)
    assert r.status_code in (400, 422), r.text


def test_phasing_span_inside_window_is_accepted(client):
    r = client.post("/api/architecture/evaluate", json={"architectures": [{"name": "g", "sensors": []}], "n_mc": 1,
                                                        "horizon_days": 1.0, "phasing_span_days": 13.0, "seed": 1})
    assert r.status_code == 200, r.text
    d = r.json()["meta"]["draws"][0]
    assert 0.0 <= d["t0_offset_days"] <= 13.0
