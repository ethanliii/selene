"""Offline tests for the JPL Horizons cache (``selene.objects.horizons``).

Everything here reads only the committed ``data/horizons/*.json`` and the local DE440s
kernel; nothing touches the network unless ``SELENE_NETWORK_TESTS=1`` is set, in which
case the single ``@pytest.mark.network`` test runs.
"""
from __future__ import annotations

import json
import os
from datetime import datetime

import numpy as np
import pytest

from selene import time as stime
from selene.constants import DE440S_PATH, R_EARTH, R_MOON
from selene.objects import horizons as H

OBJECTS = H.load_horizons_objects()
INDEX = H.load_index()
BY_ID = {o.id: o for o in OBJECTS}

# Plausibility bounds (km) per regime. Lunar orbiters / NRHO / Lagrange-point objects stay
# within ~ one Earth-Moon distance +/- the lunar Hill sphere (300 000 .. 2 000 000 km).
# xGEO HEOs (TESS, boosters on lunar-impact trajectories) and cislunar transits (Artemis II)
# legitimately dip toward Earth, so their lower bound is only "above the surface".
GEO_BOUNDS = {
    "lunar_orbit": (3.0e5, 2.0e6),
    "nrho": (3.0e5, 2.0e6),
    "lunar_halo": (3.0e5, 2.0e6),
    "xgeo_heo": (R_EARTH, 2.0e6),
    "cislunar_transit": (R_EARTH, 2.0e6),
}


# ---------------------------------------------------------------------------
# index / loading
# ---------------------------------------------------------------------------
def test_index_exists_and_lists_data():
    assert (H.HORIZONS_DATA_DIR / "index.json").exists(), "data/horizons/index.json missing: run --refresh"
    assert INDEX["source"] == "JPL Horizons"
    assert INDEX["n_cached"] >= 1 and len(INDEX["objects"]) == INDEX["n_cached"]
    # at least one object genuinely covers the demo window (not just a historical fallback)
    assert any(e.get("contemporaneous") for e in INDEX["objects"])
    # the index documents what was NOT found, with reasons
    assert INDEX["not_cached"] and all(e.get("reason") for e in INDEX["not_cached"])
    assert INDEX["name_searches"] and all("status" in e for e in INDEX["name_searches"])


def test_index_provenance_is_cumulative():
    """The index is a merge of several runs; it must say so instead of reporting only the
    last partial run's request count."""
    assert INDEX["requests_made_total"] >= INDEX["requests_made_this_run"] >= 0
    hist = INDEX["refresh_history"]
    assert hist and all(h.get("kind") in ("refresh", "rebuild_from_cache") for h in hist)
    assert sum(int(h.get("requests", 0)) for h in hist) == INDEX["requests_made_total"]
    assert INDEX["first_generated_utc"] <= INDEX["generated_utc"]
    assert "provenance_note" in INDEX


def test_every_indexed_object_loads_and_matches_file():
    assert OBJECTS, "no Horizons objects cached"
    for e in INDEX["objects"]:
        assert e["id"] in BY_ID, f"index lists {e['id']} but no JSON loaded"
        o = BY_ID[e["id"]]
        assert len(o) == e["n_samples"]
        assert list(o.span) == e["span_utc"]
        assert (H.HORIZONS_DATA_DIR / e["file"]).stat().st_size < 1_000_000, f"{e['file']} exceeds 1 MB"
    # cached / not_cached are disjoint
    assert not ({e["id"] for e in INDEX["objects"]} & {e["id"] for e in INDEX["not_cached"]})


def test_catalog_order_puts_demo_window_objects_first():
    """objs[0] must be a demo-window object (CAPSTONE with the shipped cache), never a
    historical fallback table such as LADEE 2014."""
    flags = [o.contemporaneous for o in OBJECTS]
    assert flags[0], f"first object {OBJECTS[0].name} is historical"
    assert flags == sorted(flags, reverse=True), "contemporaneous objects must precede historical ones"
    for group in (True, False):
        ids = [o.id for o in OBJECTS if o.contemporaneous is group]
        assert ids == sorted(ids), "within a group IDs must ascend numerically"
    if -1176 in BY_ID:
        assert OBJECTS[0].id == -1176
    assert [e["id"] for e in INDEX["objects"]] == [o.id for o in OBJECTS], "index order must match loader order"


@pytest.mark.parametrize("obj", OBJECTS, ids=[f"{o.id}:{o.name}" for o in OBJECTS])
def test_object_shapes_and_units(obj: H.HorizonsObject):
    assert obj.states.ndim == 2 and obj.states.shape[1] == 6
    assert obj.t_tdb_s.shape == (obj.states.shape[0],)
    assert obj.states.shape[0] >= 24
    assert np.all(np.diff(obj.t_tdb_s) > 0), "epochs must be strictly increasing"
    assert np.all(np.isfinite(obj.states))
    # span strings are UTC ISO and consistent with the TDB epochs (TDB-UTC ~ 69 s in 2026)
    for s_iso, t in ((obj.span[0], obj.t0), (obj.span[1], obj.t1)):
        datetime.fromisoformat(s_iso)
        assert abs(stime.seconds_since_j2000_tdb(s_iso) - t) < 1.5, (s_iso, t)
    # speed sanity: nothing in the Earth-Moon system moves faster than LEO speed (~11 km/s)
    v = np.linalg.norm(obj.states[:, 3:], axis=1)
    assert 0.0 < v.max() < 11.5, v.max()
    assert obj.metadata["source"] == "JPL Horizons"
    assert "REAL DATA" in " ".join(obj.notes)


@pytest.mark.parametrize("obj", OBJECTS, ids=[f"{o.id}:{o.name}" for o in OBJECTS])
def test_no_table_discontinuities(obj: H.HorizonsObject):
    """Consecutive samples must never imply a chord speed above the physical limit (a merged
    Horizons file can splice segments hundreds of thousands of km apart, as SLIM_merged did
    for 2023-12-26..2024-01-08). Such rows must have been dropped at build time."""
    chord = np.linalg.norm(np.diff(obj.states[:, :3], axis=0), axis=1) / np.diff(obj.t_tdb_s)
    assert chord.max() <= H.MAX_PLAUSIBLE_SPEED_KMS, (obj.name, chord.max())
    # chord speed of a smooth arc never exceeds the peak reported speed (+ rounding slack)
    assert chord.max() <= np.linalg.norm(obj.states[:, 3:], axis=1).max() + 1e-3
    assert abs(obj.metadata["max_chord_speed_kms"] - chord.max()) < 2e-3
    for seg in obj.metadata["dropped_segments"]:
        assert seg["implied_speed_kms"] > H.MAX_PLAUSIBLE_SPEED_KMS and seg["n_samples"] >= 1
        assert any("DROPPED" in n for n in obj.notes), "dropped segments must be disclosed in the notes"


def test_slim_merged_artifact_removed_if_cached():
    """Regression for the verifier finding: SLIM's Horizons table contained a 386 348 km jump
    (2023-12-25T22:59:59 -> 23:59:59 UTC) into a mis-centred segment. The committed file must
    keep only the consistent lunar-orbit / landed segment and document what was dropped."""
    if -240 not in BY_ID:
        pytest.skip("SLIM not cached")
    o = BY_ID[-240]
    assert o.regime == "lunar_orbit"
    assert o.metadata["selenocentric_range_km"][1] < H.LUNAR_ORBIT_MAX_SELENO_KM
    dropped = o.metadata["dropped_segments"]
    assert dropped and any(s["moon_offset_signature"] for s in dropped)
    assert sum(s["n_samples"] for s in dropped) + len(o) == 1081  # the raw table had 1081 rows
    assert o.span[0] >= "2024-01-09"


def test_split_at_discontinuities_synthetic():
    """Two smooth arcs separated by an impossible jump: the longer arc is kept, the shorter
    reported; a clean table passes through untouched."""
    dt = 600.0
    n1, n2 = 20, 50
    v = np.array([1.0, 0.5, 0.0])
    t = np.arange(n1 + n2) * dt
    pos1 = 300_000.0 + np.outer(np.arange(n1) * dt, v)
    pos2 = np.array([900_000.0, 0.0, 0.0]) + np.outer(np.arange(n2) * dt, v)   # ~600 000 km away
    s = np.hstack([np.vstack([pos1, pos2]), np.tile(v, (n1 + n2, 1))])
    tk, sk, dropped = H.split_at_discontinuities(t, s)
    assert tk.shape == (n2,) and sk.shape == (n2, 6) and np.allclose(sk[:, :3], pos2)
    assert len(dropped) == 1 and dropped[0]["n_samples"] == n1
    assert dropped[0]["implied_speed_kms"] > H.MAX_PLAUSIBLE_SPEED_KMS
    assert dropped[0]["jump_km"] > 500_000
    # clean table: nothing dropped, arrays returned as-is
    tc, sc, d0 = H.split_at_discontinuities(t[:n1], s[:n1])
    assert d0 == [] and tc.shape == (n1,) and np.shares_memory(sc, s)
    # a real 11 km/s LEO arc sampled at 1 h (chord speed ~ 0) is never cut
    assert H.split_at_discontinuities(np.array([0.0, 3600.0]), np.array([[7000, 0, 0, 0, 7.5, 0], [-7000, 0, 0, 0, -7.5, 0]], float))[2] == []
    # a heliocentric context object moving 25 km/s relative to Earth (chord == reported speed)
    # is fast but smooth: must not be cut (regression: Lunar Trailblazer / BioSentinel 1 d tables)
    vh = np.array([25.0, 0.0, 0.0])
    th = np.arange(10) * 86400.0
    sh = np.hstack([5.0e6 + np.outer(th, vh), np.tile(vh, (10, 1))])
    assert H.split_at_discontinuities(th, sh)[2] == []


@pytest.mark.parametrize("obj", OBJECTS, ids=[f"{o.id}:{o.name}" for o in OBJECTS])
def test_geocentric_distance_plausible(obj: H.HorizonsObject):
    r = obj.geocentric_range_km
    lo, hi = GEO_BOUNDS[obj.regime]  # KeyError => an unexpected regime was cached
    rmin, rmax = float(r.min()), float(r.max())
    print(f"\n{obj.id} {obj.name}: regime={obj.regime} geocentric r in [{rmin:.0f}, {rmax:.0f}] km")
    assert lo <= rmin, (obj.name, rmin, lo)
    assert rmax <= hi, (obj.name, rmax, hi)
    assert rmin > R_EARTH, "a cached state lies inside the Earth (past re-entry): should have been trimmed"
    assert rmax > H.CISLUNAR_MIN_APOGEE_KM, "cached object never gets beyond GEO: not xGEO"
    smin = obj.metadata["selenocentric_range_km"][0]
    assert smin > R_MOON, f"{obj.name}: selenocentric range {smin} km inside the Moon (past impact)"
    # metadata ranges were computed from the same samples
    mlo, mhi = obj.metadata["geocentric_range_km"]
    assert abs(mlo - rmin) < 1.0 and abs(mhi - rmax) < 1.0


# ---------------------------------------------------------------------------
# interpolation
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("method", ["hermite", "spline"])
def test_interpolation_reproduces_samples(method):
    for obj in OBJECTS:
        idx = np.linspace(0, len(obj) - 1, 15).astype(int)
        for k in idx:
            s = obj.state_at(float(obj.t_tdb_s[k]), method=method)
            assert s is not None
            ref = obj.states[k]
            assert np.linalg.norm(s[:3] - ref[:3]) <= 1e-6 * np.linalg.norm(ref[:3]), (obj.name, k)
            assert np.linalg.norm(s[3:] - ref[3:]) <= 1e-6 * np.linalg.norm(ref[3:]) + 1e-9, (obj.name, k)


def test_interpolation_between_samples_is_smooth():
    """The stored leave-one-out metric must be reproducible, honest (label matches value), and
    small for every object active in the demo window."""
    for obj in OBJECTS:
        loo = obj.leave_one_out_error_km("hermite")
        stored = obj.metadata["interp_leave_one_out_max_km"]["hermite"]
        assert abs(loo - stored) < 1e-2 * max(1.0, stored), (obj.name, loo, stored)
        assert obj.metadata["interp_quality"] == H.interp_quality(loo)
        if obj.contemporaneous:
            # every object active in the demo window was sampled finely enough for ~km-level
            # Hermite interpolation (< 20 km at 2x step  =>  ~1 km at the nominal step)
            assert loo < 20.0, f"{obj.name}: 2x-step Hermite error {loo:.2f} km too large for sampling step"
            assert obj.metadata["interp_quality"] == "good"
        # Where the sampling resolves the orbit, Hermite (uses velocity) must not be worse than a
        # position-only spline (for 1 h samples of a 2 h lunar orbit both are meaningless).
        if obj.metadata["interp_quality"] in ("good", "coarse"):
            assert loo <= obj.leave_one_out_error_km("spline") * 1.05 + 1e-9, obj.name


def test_state_at_outside_span():
    obj = OBJECTS[0]
    assert obj.state_at(obj.t0 - 1.0) is None
    assert obj.state_at(obj.t1 + 1.0) is None
    with pytest.raises(ValueError):
        obj.state_at(obj.t1 + 1.0, strict=True)
    out = obj.states_at(np.array([obj.t0 - 10.0, obj.t0, obj.t1 + 10.0]))
    assert np.isnan(out[0]).all() and np.isnan(out[2]).all() and np.isfinite(out[1]).all()
    assert obj.state_at_utc(obj.span[0]) is not None or obj.state_at_utc(obj.span[1]) is not None


def test_loader_filters():
    active = H.load_horizons_objects(contemporaneous_only=True)
    assert active and all(o.contemporaneous for o in active)
    at_demo = H.load_horizons_objects(at_utc=H.DEMO_EPOCH_UTC)
    assert at_demo, "nothing cached covers the demo epoch"
    t = stime.seconds_since_j2000_tdb(H.DEMO_EPOCH_UTC)
    assert all(o.in_span(t) for o in at_demo)
    assert H.get_horizons_object(OBJECTS[0].id).id == OBJECTS[0].id
    assert H.get_horizons_object(123456789) is None


# ---------------------------------------------------------------------------
# physics cross-checks against DE440s
# ---------------------------------------------------------------------------
@pytest.mark.skipif(not DE440S_PATH.exists(), reason="de440s.bsp not cached")
def test_lunar_orbiters_are_near_de440s_moon():
    """Lunar orbiters must sit within 10 000 km of the DE440s Moon (jplephem directly on the
    kernel: Moon(301) and Earth(399) are both given relative to the EMB(3))."""
    from jplephem.spk import SPK

    spk = SPK.open(str(DE440S_PATH))
    checked = []
    for obj in OBJECTS:
        if obj.regime != "lunar_orbit" or not obj.contemporaneous:
            continue
        jd = stime.J2000_JD + obj.t_tdb_s / stime.DAY_S
        r_moon = (spk[3, 301].compute(jd) - spk[3, 399].compute(jd)).T
        d = np.linalg.norm(obj.states[:, :3] - r_moon, axis=1)
        print(f"\n{obj.id} {obj.name}: selenocentric r in [{d.min():.1f}, {d.max():.1f}] km")
        assert d.max() < 10_000.0 or obj.id in (-192, -193), (obj.name, d.max())  # ARTEMIS apolune ~19 000 km
        assert d.max() < 25_000.0, (obj.name, d.max())
        assert d.min() > R_MOON, (obj.name, d.min())  # above the lunar surface
        checked.append(obj.id)
    assert checked, "no contemporaneous lunar orbiter cached (LRO expected)"
    if -85 in BY_ID:  # LRO specifically, if present
        assert -85 in checked


@pytest.mark.skipif(not DE440S_PATH.exists(), reason="de440s.bsp not cached")
def test_module_moon_helper_matches_jplephem():
    from jplephem.spk import SPK

    spk = SPK.open(str(DE440S_PATH))
    t = np.array([stime.seconds_since_j2000_tdb(H.DEMO_EPOCH_UTC)])
    jd = stime.J2000_JD + t / stime.DAY_S
    ref = (spk[3, 301].compute(jd) - spk[3, 399].compute(jd)).T
    got = H.moon_position_gcrf_km(t)
    assert np.allclose(got, ref, atol=1e-6)
    assert 356_000 < np.linalg.norm(ref) < 407_000


def test_capstone_is_nrho_if_cached():
    """CAPSTONE flies the Gateway 9:2 L2 southern NRHO: perilune ~1 500-1 700 km altitude and
    apolune ~70 000 km (Lee, "White Paper: Gateway Destination Orbit Model", NASA 2019;
    Zimovan-Spreen, Howell, Davis, Celest. Mech. Dyn. Astron. 132:28, 2020)."""
    if -1176 not in BY_ID:
        pytest.skip("CAPSTONE not cached")
    obj = BY_ID[-1176]
    lo, hi = obj.metadata["selenocentric_range_km"]
    assert obj.regime == "nrho"
    assert 2_500 < lo < 5_500, lo        # perilune radius (CAPSTONE's NRHO is slightly off-nominal)
    assert 60_000 < hi < 80_000, hi      # apolune radius


# ---------------------------------------------------------------------------
# pure parsers (offline)
# ---------------------------------------------------------------------------
_SAMPLE = """*******************************************************************************
Target body name: CAPSTONE (spacecraft) (-1176)   {source: CAPSTONE_merged}
Center body name: Earth (399)                     {source: DE441}
Reference frame : ICRF
*******************************************************************************
            JDTDB,            Calendar Date (TDB),                      X,                      Y,                      Z,                     VX,                     VY,                     VZ,
**************************************
$$SOE
2461086.500000000, A.D. 2026-Feb-15 00:00:00.0000,  1.921324174413011E+05, -2.907959571817073E+05, -2.136561355636021E+05,  8.429704453763153E-01,  4.741033269718353E-01,  5.844482075230870E-02,
2461086.541666667, A.D. 2026-Feb-15 01:00:00.0000,  1.951572016702563E+05, -2.890817775604124E+05, -2.134296101856498E+05,  8.374650193766431E-01,  4.782199899698782E-01,  6.738649377648930E-02,
$$EOE
"""


def test_parse_vectors_csv():
    t, s, info = H.parse_vectors_csv(_SAMPLE)
    assert t.shape == (2,) and s.shape == (2, 6)
    assert abs(t[0] - (2461086.5 - 2451545.0) * 86400.0) < 1e-6
    assert abs(t[1] - t[0] - 3600.0) < 1e-3
    assert s[0, 0] == pytest.approx(192132.4174413011) and s[1, 5] == pytest.approx(0.0673864937764893)
    assert info == {"target_name": "CAPSTONE (spacecraft)", "target_id": -1176,
                    "trajectory_source": "CAPSTONE_merged", "center": "Earth (399)", "frame": "ICRF"}
    with pytest.raises(ValueError):
        H.parse_vectors_csv("no table here")


def test_parse_span_error_and_time_roundtrip():
    which, when = H.parse_span_error('No ephemeris for target "Artemis II (spacecraft)" prior to A.D. 2026-APR-02 01:58:32.3050 TDB')
    assert which == "prior to" and when == datetime(2026, 4, 2, 1, 58, 32, 305000)
    which, when = H.parse_span_error('No ephemeris for target "LADEE (spacecraft)" after A.D. 2014-APR-18 04:33:07.1850 TDB')
    assert which == "after" and when.year == 2014
    assert H.parse_span_error("$$SOE ... $$EOE") is None
    assert H.format_horizons_time(datetime(2026, 4, 2, 1, 58, 32, 305000)) == "2026-04-02 01:58:32.305"


def test_parse_name_search():
    multi = (" Multiple major-bodies match string \"ARTEMIS*\"\n\n  ID#      Name                               Designation  IAU/aliases/other\n"
             "  -------  ---------------------------------- -----------  -------------------\n"
             "     -192  THEMIS-B (spacecraft)              2007-004B    ARTEMIS-P1\n"
             "    -1024  Artemis II (spacecraft)            2026-069A    Integrity Orion EM-2\n")
    r = H.parse_name_search(multi)
    assert r["status"] == "multiple" and [m["id"] for m in r["matches"]] == [-192, -1024]
    assert H.parse_name_search("JPL/DASTCOM Small-body Index Search Results\n Matching small-bodies:\n    No matches found.")["status"] == "unknown"
    single = "*****\n Revised: Sep 30, 2026          LRO Spacecraft / (Moon)                     -85\n"
    r = H.parse_name_search(single)
    assert r["status"] == "single" and r["matches"][0]["id"] == -85
    sb = "*****\nJPL/HORIZONS               1143 Odysseus (1930 BH)         2026-Oct-02 03:34:59\nRec #:    1143"
    assert H.parse_name_search(sb)["status"] == "small_body"


def test_json_roundtrip(tmp_path):
    obj = OBJECTS[0]
    p = tmp_path / f"{obj.id}.json"
    n = obj.save_json(p)
    assert n == p.stat().st_size
    back = H.HorizonsObject.from_json_dict(json.loads(p.read_text()))
    assert back.id == obj.id and back.name == obj.name and len(back) == len(obj)
    assert np.allclose(back.states, obj.states, atol=1e-6)
    assert H.load_horizons_objects(tmp_path)[0].id == obj.id


def test_refresh_offline_keeps_cache(tmp_path, monkeypatch):
    """With no network the refresh must log and return the existing index untouched."""
    import httpx

    def boom(*a, **k):
        raise httpx.ConnectError("offline")

    monkeypatch.setattr(httpx, "get", boom)
    monkeypatch.setattr(H, "REQUEST_INTERVAL_S", 0.0)
    d = tmp_path / "horizons"
    d.mkdir()
    (d / "index.json").write_text(json.dumps({"objects": [{"id": 1}], "not_cached": [], "name_searches": []}))
    idx = H.refresh(d, tmp_path / "cache")
    assert idx["objects"] == [{"id": 1}] and "last_refresh_error" in idx
    assert json.loads((d / "index.json").read_text())["objects"] == [{"id": 1}]


def test_rebuild_from_cache_is_offline_and_merges_index(tmp_path, monkeypatch):
    """Rebuilding the compact JSON from an archived raw response must not touch the network
    and must merge (not clobber) rows for objects it did not rebuild."""
    import httpx

    def boom(*a, **k):
        raise AssertionError("network must not be used by rebuild_from_cache")

    monkeypatch.setattr(httpx, "get", boom)
    d, c = tmp_path / "horizons", tmp_path / "cache"
    d.mkdir(), c.mkdir()
    (c / "-1176.txt").write_text(_SAMPLE)
    (c / "-1176.meta.json").write_text(json.dumps({"span_note": "", "contemporaneous": True, "fetched_utc": "2026-10-02T00:00:00Z"}))
    (d / "index.json").write_text(json.dumps({
        "generated_utc": "2026-10-01T00:00:00Z", "requests_made": 7, "objects": [],
        "not_cached": [{"id": -999, "name": "Phantom", "status": "no_data", "reason": "kept from a previous run"}],
        "name_searches": [{"query": "GATEWAY", "status": "unknown"}]}))
    cands = [next(cd for cd in H.CANDIDATES if cd.id == -1176), H.Candidate(-85, "LRO", "lunar_orbit", "10 m")]
    idx = H.rebuild_from_cache(d, c, candidates=cands)
    assert (d / "-1176.json").exists() and not (d / "-85.json").exists()   # -85 has no raw cache: skipped
    assert [e["id"] for e in idx["objects"]] == [-1176]
    assert idx["objects"][0]["n_samples"] == 2 and idx["objects"][0]["regime"] == "nrho"
    assert [e["id"] for e in idx["not_cached"]] == [-999], "untouched rows must survive a partial rebuild"
    assert idx["name_searches"] == [{"query": "GATEWAY", "status": "unknown"}]
    assert idx["requests_made_this_run"] == 0 and idx["requests_made_total"] == 7
    assert idx["refresh_history"][-1]["kind"] == "rebuild_from_cache" and idx["refresh_history"][0]["requests"] == 7
    assert idx["first_generated_utc"] == "2026-10-01T00:00:00Z"
    obj = H.load_horizons_objects(d)[0]
    assert obj.metadata["fetched_utc"] == "2026-10-02T00:00:00Z" and "rebuilt_utc" in obj.metadata
    assert obj.state_at(obj.t0 + 1800.0) is not None


# ---------------------------------------------------------------------------
# network (opt-in)
# ---------------------------------------------------------------------------
@pytest.mark.network
@pytest.mark.skipif(not os.environ.get("SELENE_NETWORK_TESTS"), reason="set SELENE_NETWORK_TESTS=1 to hit JPL Horizons")
def test_live_capstone_matches_cache():
    client = H.HorizonsClient()
    cand = next(c for c in H.CANDIDATES if c.id == -1176)
    live = H.resolve_and_fetch(client, cand)
    assert live["status"] == "ok"
    if -1176 in BY_ID:
        cached = BY_ID[-1176]
        k = 1  # interior sample: the first epoch may fall 1e-4 s outside the rounded cached span
        s = cached.state_at(float(live["t"][k]))
        assert s is not None and np.linalg.norm(s[:3] - live["states"][k, :3]) < 1.0
