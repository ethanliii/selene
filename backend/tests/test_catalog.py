"""Tests for the notional object catalogue and the unified Catalog."""
from __future__ import annotations

import numpy as np
import pytest

from selene.constants import GM_MOON, R_MOON
from selene.dynamics import frames
from selene.dynamics.ephemeris import get_ephemeris
from selene.objects import notional
from selene.objects.catalog import Catalog, get_catalog

DAY = 86400.0
REQUIRED = [
    "SIM-DRO-01", "SIM-NRHO-RELAY-01", "SIM-L1-HALO-01", "SIM-L2-HALO-01",
    "SIM-L2-LYAP-01", "SIM-ELFO-01", "SIM-RES-31-01", "SIM-XGEO-DEBRIS-01",
]
LIBRATION_IDS = ["SIM-DRO-01", "SIM-DRO-02", "SIM-NRHO-RELAY-01", "SIM-NRHO-02", "SIM-L1-HALO-01",
                 "SIM-L2-HALO-01", "SIM-L2-HALO-LOG-01", "SIM-L2-LYAP-01"]


@pytest.fixture(scope="module")
def cat():
    return get_catalog()


# ---------------------------------------------------------------------------
# notional definitions
def test_required_members_and_labels():
    ids = notional.notional_ids()
    for rid in REQUIRED:
        assert rid in ids
    assert len(ids) >= 8
    for o in notional.NOTIONAL_OBJECTS:
        assert o.kind == "simulated"
        assert "notional" in o.actor
        assert o.as_dict()["label"] == "SIMULATED"
        assert o.physical.cr_area_mass > 0


def test_demo_protagonist_is_a_dro_with_10_to_15_day_period():
    o = notional.get_notional("SIM-DRO-01")
    rec = o.record()
    assert rec.family == "DRO"
    assert 10.0 <= rec.period_days <= 15.0
    assert 1.5 <= o.physical.radius_m <= 2.5


def test_relay_is_on_the_9_2_nrho():
    o = notional.get_notional("SIM-NRHO-RELAY-01")
    rec = o.record()
    assert "NRHO_9:2" in rec.tags
    assert 6.4 <= rec.period_days <= 6.7
    assert "allied" in o.actor


def test_debris_is_faint():
    o = notional.get_notional("SIM-XGEO-DEBRIS-01")
    assert o.physical.radius_m == pytest.approx(0.3)
    assert o.physical.albedo == pytest.approx(0.1)


def test_elfo_elements_match_ely_2005():
    # Ely (2005) frozen ELFO: a ≈ 6541 km, e = 0.6, i = 56.2°, ω = 90°
    k = notional.get_notional("SIM-ELFO-01").kepler
    assert k is not None
    assert k.a_km == pytest.approx(6541.4, abs=1.0)
    assert k.e == pytest.approx(0.6)
    assert k.i_deg == pytest.approx(56.2)
    assert k.argp_deg == pytest.approx(90.0)
    s = notional.keplerian_moon_state_gcrf(k, frames.DEMO_EPOCH_TDB_S)
    sm = frames.gcrf_to_moon_centered(s, frames.DEMO_EPOCH_TDB_S)
    r = np.linalg.norm(sm[:3])
    v = np.linalg.norm(sm[3:])
    assert r == pytest.approx(k.a_km * (1 - k.e), rel=1e-9)          # perilune at ν = 0
    assert v == pytest.approx(np.sqrt(GM_MOON * (2 / r - 1 / k.a_km)), rel=1e-9)  # vis-viva
    assert abs(np.dot(sm[:3], sm[3:])) < 1e-6                         # perpendicular at perilune


def test_initial_state_round_trips_to_the_cr3bp_phase_state():
    o = notional.get_notional("SIM-DRO-01")
    s_rot = notional.cr3bp_phase_state(o.record(), o.phase)
    s_g = notional.initial_state_gcrf(o)
    back = frames.gcrf_to_rot(s_g, frames.DEMO_EPOCH_TDB_S)
    assert np.allclose(back, s_rot, atol=1e-9)
    # phase 0 is the stored IC itself
    assert np.allclose(notional.cr3bp_phase_state(o.record(), 0.0), o.record().ic_array)
    # phase 1.0 wraps to the IC (periodic orbit) to the closure tolerance
    assert np.allclose(notional.cr3bp_phase_state(o.record(), 1.0), o.record().ic_array, atol=1e-8)


# ---------------------------------------------------------------------------
# catalog composition
def test_catalog_composition(cat):
    sim = cat.objects("simulated")
    real = cat.objects("horizons")
    assert len(sim) >= 8
    assert len(real) >= 1
    assert all(e.is_real for e in real) and not any(e.is_real for e in sim)
    assert all(e.label == "SIMULATED" for e in sim)
    assert any("CAPSTONE" in e.name for e in real)
    for e in real:
        assert e.horizons is not None and e.horizons.in_span(cat.epoch_s)
    assert cat.epoch_utc.startswith("2026-03-01T00:00:00")


def test_epoch_states_are_plausible(cat):
    t0 = cat.epoch_s
    for oid in LIBRATION_IDS:
        s = cat.epoch_state(oid)
        assert s.shape == (6,)
        g = np.linalg.norm(s[:3])
        assert 300_000 <= g <= 480_000, (oid, g)
        assert 0.3 <= np.linalg.norm(s[3:]) <= 2.5, (oid, s[3:])
    s = cat.epoch_state("SIM-ELFO-01")
    dm = np.linalg.norm(frames.gcrf_to_moon_centered(s, t0)[:3])
    assert 1800 <= dm <= 12_000
    for oid in ("SIM-RES-31-01", "SIM-XGEO-DEBRIS-01"):
        g = np.linalg.norm(cat.epoch_state(oid)[:3])
        assert 50_000 <= g <= 600_000, (oid, g)


def test_nrho_relay_stays_bound_to_the_moon_for_a_week(cat):
    t0 = cat.epoch_s
    ts = np.linspace(t0, t0 + 7 * DAY, 7 * 24 * 6 + 1)  # 10-min grid
    st = cat.state_at("SIM-NRHO-RELAY-01", ts)
    assert st.shape == (len(ts), 6)
    moon = get_ephemeris().moon_state(ts)[:, :3]
    d = np.linalg.norm(st[:, :3] - moon, axis=1)
    assert d.min() >= 100.0 + R_MOON - R_MOON  # never below 100 km from the centre (and see next line)
    assert d.min() > R_MOON + 100.0, "relay would impact the Moon"
    assert d.max() <= 75_000.0
    # it goes through at least one perilune (< 5000 km) and one apolune (> 60 000 km) in a week (T = 6.56 d)
    assert d.min() < 5000.0 and d.max() > 60_000.0


def test_truth_window_is_cached_and_no_object_impacts(cat):
    eph = get_ephemeris()
    for e in cat.objects("simulated"):
        tr = cat.truth(e.id)
        assert tr.t_min <= cat.epoch_s - 1 * DAY + 1e-6
        assert tr.t_max >= cat.epoch_s + 14 * DAY - 1e-6
        assert tr.build_time_s < 2.0, (e.id, tr.build_time_s)
        g = tr.grid()
        assert g.frame == "gcrf_km" and len(g) == 15 * 24 * 6 + 1
        moon = eph.moon_state(g.t_s)[:, :3]
        dm = np.linalg.norm(g.positions - moon, axis=1)
        assert dm.min() > R_MOON + 100.0, (e.id, dm.min())
        assert np.linalg.norm(g.positions, axis=1).min() > 20_000.0
        if tr.fit_info is not None:
            assert tr.fit_info["dv_mps"] < 50.0, (e.id, tr.fit_info)
            assert tr.fit_info["max_after_km"] < 20_000.0, (e.id, tr.fit_info)
        # the same object is returned from the cache
        assert cat.truth(e.id) is tr


def test_fitted_halos_shadow_their_reference_orbit(cat):
    """The ephemeris truth of the (fitted) halo objects stays within ~10 000 km of the CR3BP
    reference over the full 14 days, whereas the raw mapping departs or impacts."""
    for oid in ("SIM-L1-HALO-01", "SIM-L2-HALO-01"):
        o = notional.get_notional(oid)
        assert o.ephem_fit
        tr = cat.truth(oid)
        ts = np.linspace(cat.epoch_s, cat.epoch_s + 14 * DAY, 57)
        rot = frames.gcrf_to_rot(tr.at(ts), ts)
        ref = notional.reference_rot_state(o, ts - cat.epoch_s)
        dev = np.linalg.norm(rot[:, :3] - ref[:, :3], axis=1) * 384_400.0
        assert dev.max() < 20_000.0, (oid, dev.max())
        assert np.median(dev) < 6_000.0


def test_state_at_is_deterministic_across_instances():
    a = Catalog(include_horizons=False)
    b = Catalog(include_horizons=False)
    ts = frames.DEMO_EPOCH_TDB_S + np.array([0.0, 3 * DAY, 10 * DAY])
    for oid in ("SIM-DRO-02", "SIM-L2-LYAP-01"):
        assert np.array_equal(a.state_at(oid, ts), b.state_at(oid, ts))


def test_state_at_vectorised_matches_scalar_and_extends_outside_window(cat):
    t0 = cat.epoch_s
    ts = t0 + np.array([-0.5 * DAY, 2.0 * DAY, 13.9 * DAY])
    v = cat.state_at("SIM-DRO-02", ts)
    for i, t in enumerate(ts):
        assert np.allclose(v[i], cat.state_at("SIM-DRO-02", float(t)), rtol=0, atol=1e-9)
    # outside the cached window: deterministic extension, continuous with the arc
    s20 = cat.state_at("SIM-DRO-02", t0 + 20 * DAY)
    assert 250_000 < np.linalg.norm(s20[:3]) < 500_000
    s_edge = cat.state_at("SIM-DRO-02", t0 + 14 * DAY)
    s_edge2 = cat.state_at("SIM-DRO-02", t0 + 14 * DAY + 1.0)
    assert np.linalg.norm(s_edge2[:3] - s_edge[:3]) < 2.0  # < 2 km in 1 s


def test_trajectory_frames(cat):
    t0 = cat.epoch_s
    g = cat.trajectory("SIM-NRHO-RELAY-01", t0, t0 + 3 * DAY, 50, "gcrf_km")
    m = cat.trajectory("SIM-NRHO-RELAY-01", t0, t0 + 3 * DAY, 50, "moon_km")
    r = cat.trajectory("SIM-NRHO-RELAY-01", t0, t0 + 3 * DAY, 50, "rot_nd")
    assert g.frame == "gcrf_km" and m.frame == "moon_km" and r.frame == "rot_nd"
    moon = get_ephemeris().moon_state(g.t_s)
    assert np.allclose(g.states - moon, m.states, atol=1e-6)
    # rotating-frame positions are near the Moon (x ≈ 1-μ) and z is negative at apolune for a southern NRHO
    assert np.all(np.abs(r.positions[:, 0] - 0.9879) < 0.08)
    assert np.allclose(frames.rot_to_gcrf(r.states, r.t_s), g.states, rtol=1e-9, atol=1e-6)
    with pytest.raises(ValueError):
        cat.trajectory("SIM-NRHO-RELAY-01", t0, t0 + DAY, 10, "bogus")


def test_horizons_states(cat):
    real = cat.objects("horizons")
    e = real[0]
    s = cat.state_at(e.id, cat.epoch_s)
    assert s.shape == (6,) and np.isfinite(s).all()
    assert 1e5 < np.linalg.norm(s[:3]) < 1e6
    ts = np.array([cat.epoch_s, e.horizons.t1 + 10 * DAY])
    v = cat.state_at(e.id, ts)
    assert np.isfinite(v[0]).all() and np.isnan(v[1]).all()
    with pytest.raises(ValueError):
        cat.state_at(e.id, e.horizons.t1 + 10 * DAY)
    with pytest.raises(ValueError):
        cat.truth(e.id)
    tr = cat.trajectory(e.id, cat.epoch_s, e.horizons.t1 + 5 * DAY, 20)
    assert 0 < len(tr) < 20  # out-of-span epochs dropped


def test_summary_is_json_friendly(cat):
    import json

    for oid in ("SIM-DRO-01", "SIM-ELFO-01", cat.objects("horizons")[0].id):
        d = cat.summary(oid)
        json.dumps(d)
        assert len(d["state_gcrf_km"]) == 6 and len(d["state_rot_nd"]) == 6
        assert d["epoch_utc"].startswith("2026-03-01")
    d = cat.summary("SIM-DRO-01")
    assert d["label"] == "SIMULATED" and d["actor"] == "notional actor"
    assert d["orbit_record"]["family"] == "DRO"
    with pytest.raises(KeyError):
        cat.get("NOPE")
