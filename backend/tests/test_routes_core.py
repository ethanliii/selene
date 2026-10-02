"""FastAPI TestClient tests for /api/catalog, /api/orbits and /api/ephemeris."""
from __future__ import annotations

import time

import numpy as np
import pytest

from selene.constants import MU, R_MOON
from selene.dynamics.cr3bp import lagrange_points
from selene.dynamics.ephemeris import get_ephemeris
from selene.dynamics.frames import DEMO_EPOCH_UTC

MAX_BYTES = int(1.5e6)


def _timed(client, url):
    t = time.perf_counter()
    r = client.get(url)
    return r, time.perf_counter() - t


# ---------------------------------------------------------------------------
# catalog
def test_catalog_objects(client):
    r = client.get("/api/catalog/objects")
    assert r.status_code == 200
    body = r.json()
    assert body["n_simulated"] >= 8 and body["n_real"] >= 1
    assert "SIMULATED" in body["disclaimer"]
    ids = set()
    for o in body["objects"]:
        ids.add(o["id"])
        assert o["kind"] in ("simulated", "horizons")
        assert len(o["state_gcrf_km"]) == 6 and len(o["state_rot_nd"]) == 6
        assert o["epoch_utc"].startswith("2026-03-01T00:00:00")
        if o["kind"] == "simulated":
            assert o["label"] == "SIMULATED" and "notional" in o["actor"]
            assert o["physical"]["radius_m"] > 0
            assert o["truth"]["window_tdb_s"][1] > o["truth"]["window_tdb_s"][0]
        else:
            assert o["is_real"] and o["label"].startswith("REAL")
            assert o["physical"] is None
    assert {"SIM-DRO-01", "SIM-NRHO-RELAY-01", "SIM-ELFO-01"} <= ids
    r2 = client.get("/api/catalog/objects?kind=horizons")
    assert r2.status_code == 200 and all(o["kind"] == "horizons" for o in r2.json()["objects"])
    assert len(r.content) < MAX_BYTES


def test_catalog_object_detail_and_404(client):
    r = client.get("/api/catalog/objects/SIM-DRO-01")
    assert r.status_code == 200
    o = r.json()
    assert o["id"] == "SIM-DRO-01" and o["orbit_type"] == "DRO"
    assert 300_000 <= o["geocentric_range_km"] <= 480_000
    assert o["orbit_record"]["family"] == "DRO"
    assert client.get("/api/catalog/objects/NOPE").status_code == 404
    assert client.get("/api/catalog/objects/NOPE/trajectory").status_code == 404


def test_catalog_trajectory_frames(client):
    r = client.get("/api/catalog/objects/SIM-NRHO-RELAY-01/trajectory")
    assert r.status_code == 200
    b = r.json()
    assert b["frame"] == "gcrf_km" and b["n"] == 200
    assert len(b["states"]) == 200 and len(b["epochs_utc"]) == 200 and len(b["tdb_s"]) == 200
    assert b["epochs_utc"][0].startswith("2026-03-01T00:00:00")
    assert b["epochs_utc"][-1].startswith("2026-03-08T00:00:00")
    st = np.array(b["states"])
    moon = get_ephemeris().moon_state(np.array(b["tdb_s"]))[:, :3]
    d = np.linalg.norm(st[:, :3] - moon, axis=1)
    assert d.min() > R_MOON + 100 and d.max() < 75_000

    r = client.get("/api/catalog/objects/SIM-NRHO-RELAY-01/trajectory?n=50&frame=rot_nd")
    b = r.json()
    assert b["frame"] == "rot_nd"
    st = np.array(b["states"])
    assert np.all(np.abs(st[:, 0] - (1 - MU)) < 0.1) and np.all(np.abs(st[:, :3]) < 2)

    r = client.get(f"/api/catalog/objects/SIM-ELFO-01/trajectory?n=100&frame=moon_km&t0={DEMO_EPOCH_UTC}Z")
    b = r.json()
    assert b["frame"] == "moon_km"
    rr = np.linalg.norm(np.array(b["states"])[:, :3], axis=1)
    assert rr.min() > 1800 and rr.max() < 12_000

    assert client.get("/api/catalog/objects/SIM-DRO-01/trajectory?frame=bogus").status_code == 422
    assert client.get("/api/catalog/objects/SIM-DRO-01/trajectory?t0=notatime").status_code == 400
    assert client.get("/api/catalog/objects/SIM-DRO-01/trajectory?t0=2026-03-02T00:00:00&t1=2026-03-01T00:00:00").status_code == 400


def test_catalog_trajectory_horizons_object(client):
    hz = client.get("/api/catalog/objects?kind=horizons").json()["objects"][0]
    r = client.get(f"/api/catalog/objects/{hz['id']}/trajectory?n=24&t1=2026-03-03T00:00:00")
    assert r.status_code == 200
    b = r.json()
    assert b["kind"] == "horizons" and b["n"] == 24
    assert np.isfinite(np.array(b["states"])).all()
    # entirely outside the cached span -> 400 with a clear message
    r = client.get(f"/api/catalog/objects/{hz['id']}/trajectory?t0=2030-01-01T00:00:00&t1=2030-01-02T00:00:00")
    assert r.status_code == 400


# ---------------------------------------------------------------------------
# orbits
def test_orbit_families(client):
    r, dt_cold = _timed(client, "/api/orbits/families")
    assert r.status_code == 200
    assert len(r.content) < MAX_BYTES
    b = r.json()
    assert b["mu"] == pytest.approx(MU)
    names = {f["name"] for f in b["families"]}
    assert {"L1_lyapunov", "L2_lyapunov", "L1_halo_N", "L1_halo_S", "L2_halo_N", "L2_halo_S", "DRO",
            "resonant_3:1", "resonant_2:1"} <= names
    for f in b["families"]:
        located = [m for m in f["members"] if "_nrho_" in m["id"]]
        assert len(f["members"]) - len(located) <= 12
        assert f["n_returned"] == len(f["members"]) and f["n_total"] >= f["n_returned"]
        for m in f["members"]:
            assert len(m["ic"]) == 6 and len(m["samples_rot"]) == 200
            assert np.allclose(m["samples_rot"][0], m["ic"][:3], atol=1e-5)
            assert np.allclose(m["samples_rot"][-1], m["ic"][:3], atol=1e-4)   # closes after one period
            assert m["period"] > 0 and m["stability"] >= 1.0 - 1e-6
    l2s = next(f for f in b["families"] if f["name"] == "L2_halo_S")
    nrho = [m for m in l2s["members"] if "NRHO_9:2" in m["tags"]]
    assert len(nrho) == 1 and 6.4 <= nrho[0]["period_days"] <= 6.7
    assert 3000 <= nrho[0]["params"]["perilune_km"] <= 3600
    r, dt_warm = _timed(client, "/api/orbits/families")
    assert dt_warm < 2.0, dt_warm


def test_orbit_families_filter_and_params(client):
    r = client.get("/api/orbits/families?family=DRO&n_samples=50&max_members=5")
    assert r.status_code == 200
    b = r.json()
    assert len(b["families"]) == 1 and b["families"][0]["name"] == "DRO"
    assert len(b["families"][0]["members"]) == 5
    assert all(len(m["samples_rot"]) == 50 for m in b["families"][0]["members"])
    assert client.get("/api/orbits/families?family=nope").status_code == 404


def test_orbit_record(client):
    r = client.get("/api/orbits/records/L2_halo_S_nrho_9_2?n_samples=100&gcrf_epoch=" + DEMO_EPOCH_UTC)
    assert r.status_code == 200
    b = r.json()
    assert 6.4 <= b["period_days"] <= 6.7 and "NRHO_9:2" in b["tags"]
    assert len(b["samples_rot"]) == 100 and len(b["samples_gcrf_km"]) == 100
    g = np.linalg.norm(np.array(b["samples_gcrf_km"]), axis=1)
    assert 300_000 < g.min() and g.max() < 480_000
    assert client.get("/api/orbits/records/NOPE").status_code == 404


# ---------------------------------------------------------------------------
# ephemeris
def test_ephemeris_bodies_matches_de440s(client):
    r, _ = _timed(client, "/api/ephemeris/bodies?n=25")
    assert r.status_code == 200
    b = r.json()
    n = 25
    for key in ("epochs", "tdb_s", "earth", "moon", "sun"):
        assert len(b[key]) == n, key
    assert b["source"] == "de440s"
    assert b["epochs"][0].startswith("2026-03-01T00:00:00")
    assert b["epochs"][-1].startswith("2026-03-15T00:00:00")
    ts = np.array(b["tdb_s"])
    eph = get_ephemeris()
    moon_ref = eph.moon_state(ts)[:, :3]
    assert np.allclose(np.array(b["moon"]), moon_ref, atol=2e-3)   # rounded to 1 m
    sun_ref = eph.position("sun", ts, "earth")
    assert np.allclose(np.array(b["sun"]), sun_ref, atol=0.2)
    assert np.all(np.array(b["earth"]) == 0)

    # rotating-frame basis: orthonormal, x̂ along Earth->Moon, Moon reconstructed at (1-μ,0,0)
    R = np.array(b["basis"]["R"]).reshape(n, 3, 3)
    d = np.array(b["basis"]["d_km"])
    rb = np.array(b["basis"]["r_bary_km"])
    for i in range(n):
        assert np.allclose(R[i].T @ R[i], np.eye(3), atol=1e-9)
        assert np.allclose(R[i][:, 0], moon_ref[i] / np.linalg.norm(moon_ref[i]), atol=1e-9)
    assert np.allclose(d, np.linalg.norm(moon_ref, axis=1), atol=1e-3)
    assert np.allclose(rb, MU * moon_ref, atol=1e-3)
    moon_from_basis = rb + d[:, None] * R[:, :, 0] * (1 - MU)
    assert np.allclose(moon_from_basis, moon_ref, atol=1e-2)
    assert np.allclose(np.array(b["rot_frame"]["moon"]), moon_ref, atol=1e-2)
    # L1 lies on the Earth-Moon line between the bodies at the CR3BP fraction of the instantaneous distance
    L = lagrange_points(MU)
    l1 = np.array(b["rot_frame"]["l1"])
    expect = rb + d[:, None] * R[:, :, 0] * L[0, 0]
    assert np.allclose(l1, expect, atol=1e-2)
    assert np.all(np.linalg.norm(l1, axis=1) < np.linalg.norm(moon_ref, axis=1))
    for k in ("l2", "l3", "l4", "l5"):
        assert len(b["rot_frame"][k]) == n
    assert b["lagrange_rot_nd"]["l1"][0] == pytest.approx(L[0, 0], abs=1e-6)
    # inverse map: a GCRF point back to nd using the served basis equals the frames module
    from selene.dynamics import frames

    p = moon_ref[3] * 0.5 + np.array([0.0, 20_000.0, 5000.0])
    nd_served = R[3].T @ (p - rb[3]) / d[3]
    nd_ref = frames.gcrf_pos_to_rot(p, float(ts[3]))
    assert np.allclose(nd_served, nd_ref, atol=1e-7)


def test_ephemeris_bodies_defaults_size_and_speed(client):
    r, dt_cold = _timed(client, "/api/ephemeris/bodies")
    assert r.status_code == 200
    b = r.json()
    assert len(b["epochs"]) == 337
    assert len(r.content) < MAX_BYTES
    r, dt_warm = _timed(client, "/api/ephemeris/bodies")
    assert dt_warm < 2.0, dt_warm
    assert client.get("/api/ephemeris/bodies?n=1").status_code == 200
    assert client.get("/api/ephemeris/bodies?t0=garbage").status_code == 400
    assert client.get("/api/ephemeris/bodies?t0=2026-03-02T00:00:00&t1=2026-03-01T00:00:00&n=5").status_code == 400


def test_core_endpoints_warm_timing(client):
    for url in ("/api/catalog/objects", "/api/catalog/objects/SIM-DRO-01/trajectory",
                "/api/orbits/families", "/api/ephemeris/bodies"):
        client.get(url)
        _, dt = _timed(client, url)
        assert dt < 2.0, (url, dt)
