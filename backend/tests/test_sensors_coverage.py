import time

import numpy as np
import pytest

from selene.constants import MU
from selene.dynamics.frames import DEMO_EPOCH_TDB_S
from selene.sensors.coverage import NETWORK_PRESETS, compute_coverage, make_grid, resolve_network
from selene.sensors.reasons import REASON_MOON_EXCLUSION, REASON_NAMES, REASON_TOO_FAINT

T0 = DEMO_EPOCH_TDB_S
T1 = T0 + 7 * 86400.0
MOON = np.array([1 - MU, 0.0, 0.0])


def test_grid_shapes():
    g = make_grid("2d")
    assert g.shape == (64, 56) and g.points_rot.shape == (64 * 56, 3)
    assert g.xs[0] == -1.6 and g.xs[-1] == 1.6 and g.ys[0] == -1.4 and g.ys[-1] == 1.4
    assert np.all(g.points_rot[:, 2] == 0)
    g3 = make_grid("3d")
    assert g3.shape == (24, 20, 9) and g3.points_rot.shape == (24 * 20 * 9, 3)
    assert g3.zs[0] == -0.4 and g3.zs[-1] == 0.4
    with pytest.raises(ValueError):
        make_grid("4d")


def test_resolve_network_forms():
    assert len(resolve_network("ground_only")) == len(NETWORK_PRESETS["ground_only"][0])
    custom = resolve_network({"ground_ids": ["haleakala"], "space": ["dro_obs", {"id": "x", "name": "x", "platform_orbit": "geo", "geo_longitude_deg": 10.0}]})
    assert [s.id for s in custom] == ["haleakala", "dro_obs", "x"]
    with pytest.raises(KeyError):
        resolve_network("no_such_preset")


@pytest.fixture(scope="module")
def ground_only():
    tic = time.perf_counter()
    res = compute_coverage(T0, T1, 168, "ground_only", "2d")
    res.meta["_wall_s"] = time.perf_counter() - tic
    return res


@pytest.fixture(scope="module")
def with_l2(ground_only):
    return compute_coverage(T0, T1, 168, "ground_plus_l2_halo", "2d")


@pytest.fixture(scope="module")
def with_dro(ground_only):
    return compute_coverage(T0, T1, 168, "ground_plus_dro", "2d")


def test_default_2d_run_is_fast_and_well_formed(ground_only):
    res = ground_only
    assert res.meta["_wall_s"] < 5.0, f"coverage took {res.meta['_wall_s']:.2f}s"
    n_cells = 64 * 56
    assert res.coverage.shape == (n_cells,) and res.per_time_pct.shape == (168,)
    assert res.reason_dominant.shape == (n_cells,) and res.reason_fraction.shape == (n_cells, 8)
    assert np.all((res.coverage >= 0) & (res.coverage <= 1))
    assert np.all((res.per_time_pct >= 0) & (res.per_time_pct <= 100))
    # consistency: mean over cells of coverage == mean over time of per-time pct
    assert np.mean(res.coverage) * 100 == pytest.approx(np.mean(res.per_time_pct), abs=1e-9)
    # rows of reason_fraction plus coverage sum to 1 (each time step is covered or has one reason)
    assert np.allclose(res.reason_fraction.sum(axis=1) + res.coverage, 1.0)
    # honest ground-only picture (Lambert sphere, 1 m radius, albedo 0.2 -> 18.46 mag at 400 000 km,
    # zero phase): roughly a third of the +/-1.6 L* slice is covered on average; the far side of
    # the domain and high-phase geometries are too faint for the assumed 0.5-1.5 m apertures
    assert 0.15 < np.mean(res.coverage) < 0.6
    names = [REASON_NAMES.get(int(b), "covered") for b in res.reason_dominant]
    assert names.count("too_faint") > 0.3 * n_cells


def test_ground_only_blind_near_moon_and_space_observer_improves(ground_only, with_l2, with_dro):
    near_g = ground_only.coverage_near(MOON, 0.1)
    near_l2 = with_l2.coverage_near(MOON, 0.1)
    near_dro = with_dro.coverage_near(MOON, 0.1)
    print(f"\ncoverage within 0.1 L* of the Moon: ground_only={near_g:.3f} +L2 halo={near_l2:.3f} +DRO={near_dro:.3f}")
    print(f"mean coverage: ground_only={ground_only.mean_coverage_pct:.1f}% +L2={with_l2.mean_coverage_pct:.1f}% +DRO={with_dro.mean_coverage_pct:.1f}%")
    assert near_g < 0.05                       # lunar glare + faintness: essentially blind
    assert max(near_l2, near_dro) > near_g + 0.2
    assert near_dro > near_g and near_l2 > near_g
    assert with_dro.mean_coverage_pct > ground_only.mean_coverage_pct
    # the blind spot near the Moon is attributed to lunar glare, not daylight
    sel = np.linalg.norm(ground_only.grid.points_rot - MOON, axis=1) <= 0.1
    dom = ground_only.reason_dominant[sel]
    assert np.mean(dom == REASON_MOON_EXCLUSION) > 0.5
    assert not np.any(dom == 1)                # REASON_DAYLIGHT never the chosen explanation here


def test_coverage_near_earth_is_too_faint_or_covered(ground_only):
    # cells at ~GEO are bright (close) but inside Earth shadow part of the time
    res = ground_only
    geo_cells = np.linalg.norm(res.grid.points_rot - np.array([-MU, 0, 0]), axis=1) < 0.15
    assert np.any(geo_cells)
    assert np.mean(res.coverage[geo_cells]) > 0.3
    far = np.linalg.norm(res.grid.points_rot - np.array([-MU, 0, 0]), axis=1) > 1.5
    assert np.mean(res.reason_dominant[far] == REASON_TOO_FAINT) > 0.8


def test_3d_grid_runs():   # ~0.5 s at 24 steps; no 'slow' marker needed
    res = compute_coverage(T0, T0 + 2 * 86400.0, 24, "ground_plus_dro", "3d")
    assert res.coverage.shape == (24 * 20 * 9,)
    assert res.grid.as_dict()["zs"][0] == -0.4


# ---------------------------------------------------------------------------
# API
# ---------------------------------------------------------------------------
def test_sensors_endpoint(client):
    r = client.get("/api/sensors")
    assert r.status_code == 200
    body = r.json()
    assert 7 <= len(body["ground"]) <= 9 and len(body["space"]) >= 4
    assert "ASSUMED" in body["note"]
    assert body["ground"][0]["kind"] == "ground" and body["space"][0]["kind"] == "space"
    assert all("orbit" in s for s in body["space"])
    assert body["reason_bits"]["moon_exclusion"] == REASON_MOON_EXCLUSION
    r = client.get("/api/sensors/haleakala/visibility", params={"object_id": "nope"})
    assert r.status_code == 404 and "detail" in r.json()
    r = client.get("/api/sensors/no_such/visibility", params={"object_id": "x"})
    assert r.status_code == 404


def test_coverage_endpoint_shape(client):
    r = client.get("/api/coverage/presets")
    assert r.status_code == 200 and "ground_plus_dro" in r.json()
    req = {"t0": "2026-03-01T00:00:00", "t1": "2026-03-03T00:00:00", "n_t": 12,
           "network": "ground_plus_dro", "grid": "2d", "radius_m": 1.0, "albedo": 0.2}
    r = client.post("/api/coverage", json=req)
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["grid"]["shape"] == [64, 56] and len(body["grid"]["xs"]) == 64
    assert len(body["coverage"]) == 64 * 56 == len(body["reason_dominant"]) == len(body["reason_dominant_name"])
    assert len(body["per_time"]) == 12 and {"t", "pct"} <= set(body["per_time"][0])
    assert set(body["reason_fraction"]) == set(REASON_NAMES.values())
    assert body["meta"]["network"] == "ground_plus_dro" and body["meta"]["n_cells"] == 3584
    # custom network with an inline GEO observer
    req["network"] = {"ground_ids": ["haleakala", "teide"],
                      "space": [{"id": "g", "name": "g", "platform_orbit": "geo", "geo_longitude_deg": 0.0}]}
    req["n_t"] = 4
    r = client.post("/api/coverage", json=req)
    assert r.status_code == 200, r.text
    assert r.json()["meta"]["sensor_ids"] == ["haleakala", "teide", "g"]
    r = client.post("/api/coverage", json={**req, "network": "bogus"})
    assert r.status_code == 400
