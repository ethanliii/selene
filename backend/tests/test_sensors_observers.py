import numpy as np
import pytest

from selene.constants import GEO_RADIUS_KM, L_STAR, MU, T_STAR
from selene.dynamics.cr3bp import jacobi, propagate_cr3bp
from selene.dynamics.frames import DEMO_EPOCH_TDB_S
from selene.sensors import observers as obs_mod
from selene.sensors.observers import (
    DEFAULT_OBSERVERS,
    FALLBACK_ORBITS,
    LIBRARY_QUERIES,
    SpaceObserver,
    geo_state,
    get_observer,
    observer_state,
    orbit_summary,
    resolve_orbit,
)


def _library_available() -> bool:
    try:
        from selene.orbits.library import get_library

        return len(get_library()) > 0
    except Exception:
        return False


@pytest.fixture
def no_library(monkeypatch):
    """Force the fallback path (as if selene.orbits.library were not importable)."""
    monkeypatch.setattr(obs_mod, "_library_or_none", lambda: None)
    monkeypatch.setattr(obs_mod, "_ORBIT_CACHE", {})
    yield


def _closure_and_jacobi(orb):
    sol = propagate_cr3bp(orb.ic, tf=orb.period, dense_output=True)
    err = np.linalg.norm(sol.y[:, -1] - orb.ic)
    y = sol.sol(np.linspace(0, orb.period, 200))
    return err, np.ptp(jacobi(y.T))


@pytest.mark.parametrize("platform", sorted(FALLBACK_ORBITS))
def test_fallback_orbits_close(platform, no_library):
    orb = resolve_orbit(SpaceObserver(f"t_{platform}", "t", platform))
    assert orb.source == "fallback"
    err, dC = _closure_and_jacobi(orb)
    assert err < 1e-8, f"{platform}: closure error {err}"
    assert dC < 1e-9


@pytest.mark.parametrize("platform", sorted(FALLBACK_ORBITS))
def test_resolved_orbits_close(platform):
    """Whatever source wins (library when present, else fallback), the orbit must be periodic."""
    orb = resolve_orbit(SpaceObserver(f"r_{platform}", "r", platform))
    assert orb.source == "fallback" or orb.source.startswith("library:")
    err, dC = _closure_and_jacobi(orb)
    assert err < 1e-8, f"{platform}: closure error {err} (source {orb.source})"
    assert dC < 1e-9


@pytest.mark.skipif(not _library_available(), reason="orbit library not built")
def test_library_integration_when_present():
    """Spec item 2: positions come from selene.orbits.library when it has a matching record."""
    from selene.orbits.library import get_library

    lib = get_library()
    for o in DEFAULT_OBSERVERS:
        if o.platform_orbit == "geo":
            continue
        orb = resolve_orbit(o)
        assert orb.source.startswith("library:"), f"{o.id} resolved to {orb.source}"
        rid = orb.source.split(":", 1)[1]
        rec = lib[rid]
        assert np.allclose(orb.ic, rec.ic) and orb.period == pytest.approx(rec.period_nd)
        assert orb.closure_error < 1e-8
    # family / branch expectations of the default queries
    assert resolve_orbit(get_observer("l1_halo_obs")).source.startswith("library:L1_halo_N")
    assert resolve_orbit(get_observer("l2_halo_obs")).source.startswith("library:L2_halo_S")
    assert resolve_orbit(get_observer("dro_obs")).source.startswith("library:DRO")
    nrho = resolve_orbit(get_observer("nrho_obs"))
    assert "NRHO_9:2" in lib[nrho.source.split(":", 1)[1]].tags
    # resonant platforms resolve only through the library (both families)
    r31 = resolve_orbit(SpaceObserver("r31", "r", "resonant"))
    assert r31.source.startswith("library:RES_31")
    r21 = resolve_orbit(SpaceObserver("r21", "r", "resonant", orbit_ref="resonant_2:1"))
    assert r21.source.startswith("library:RES_21")
    assert 20.0 <= r21.period * T_STAR / 86400.0 <= 30.0
    # orbit_ref by record id, and a bad orbit_ref falls back to the platform default
    some_dro = lib.members("DRO")[5]
    o = SpaceObserver("byid", "byid", "dro", orbit_ref=some_dro.id)
    assert resolve_orbit(o).source == f"library:{some_dro.id}"
    bad = resolve_orbit(SpaceObserver("bad", "bad", "dro", orbit_ref="no_such_record"))
    assert bad.source.startswith("library:DRO")
    assert set(LIBRARY_QUERIES) == {"l1_halo", "l2_halo", "nrho", "dro", "resonant"}


def test_record_adapter_accepts_library_and_generic_shapes():
    ic = list(FALLBACK_ORBITS["dro"][0])
    P = FALLBACK_ORBITS["dro"][1]
    assert obs_mod._record_ic_period({"id": "a", "ic": ic, "period_nd": P})[1] == pytest.approx(P)
    assert obs_mod._record_ic_period({"id": "b", "x0": ic, "period": P})[1] == pytest.approx(P)
    got = obs_mod._record_ic_period({"id": "c", "ic": ic, "period_days": P * T_STAR / 86400.0})
    assert got[1] == pytest.approx(P, rel=1e-12)
    assert obs_mod._record_ic_period({"id": "d", "ic": ic}) is None
    assert obs_mod._record_ic_period({"id": "e", "ic": ic[:3], "period_nd": P}) is None


def test_nrho_matches_literature_ranges():
    # Zimovan-Spreen, Howell & Davis (2020); Lee (2019): 9:2 synodic NRHO, P ~ 6.56 d,
    # perilune radius ~3200-3400 km (PLAN acceptance 6.4-6.7 d, 3000-3600 km).  Checked for
    # whichever source wins (library record or polished fallback).
    orb = resolve_orbit(SpaceObserver("n", "n", "nrho"))
    P_days = orb.period * T_STAR / 86400.0
    assert 6.4 <= P_days <= 6.7
    sol = propagate_cr3bp(orb.ic, tf=orb.period, dense_output=True)
    y = sol.sol(np.linspace(0, orb.period, 5000))
    r_moon = np.sqrt((y[0] - (1 - MU)) ** 2 + y[1] ** 2 + y[2] ** 2) * L_STAR
    assert 3000.0 <= r_moon.min() <= 3600.0
    assert 65_000.0 <= r_moon.max() <= 75_000.0


def test_dro_and_halo_periods():
    dro = resolve_orbit(SpaceObserver("d", "d", "dro"))
    assert 9.0 <= dro.period * T_STAR / 86400.0 <= 16.0        # ~70 000 km DRO: ~14 d
    assert abs(dro.ic[2]) < 1e-12 and abs(dro.ic[5]) < 1e-12  # planar
    l2 = resolve_orbit(SpaceObserver("h", "h", "l2_halo"))
    assert 13.0 <= l2.period * T_STAR / 86400.0 <= 15.5
    assert l2.ic[2] < 0                                         # southern
    l1 = resolve_orbit(SpaceObserver("h1", "h1", "l1_halo"))
    assert 11.0 <= l1.period * T_STAR / 86400.0 <= 12.5
    assert l1.ic[2] > 0                                         # northern


def test_geo_state_geometry_and_era_cross_check():
    """GEO point: exact radius, circular speed, equatorial, and the analytic ERA-only model agrees
    with the full astropy transform to the precession-nutation level (~0.13 deg in 2026)."""
    from selene.sensors.observers import geo_state_era

    t = DEMO_EPOCH_TDB_S + 12345.0
    s = geo_state(0.0, t)[0]
    assert np.linalg.norm(s[:3]) == pytest.approx(GEO_RADIUS_KM, rel=1e-6)
    assert np.linalg.norm(s[3:]) == pytest.approx(3.0747, rel=1e-3)     # km/s circular GEO
    assert abs(s[2]) / GEO_RADIUS_KM < 5e-3                             # within 0.3 deg of the GCRS equator
    assert abs(np.dot(s[:3], s[3:])) / (GEO_RADIUS_KM * 3.0747) < 1e-4  # circular
    e = geo_state_era(0.0, t)[0]
    ang = np.degrees(np.arccos(np.dot(e[:3], s[:3]) / (np.linalg.norm(e[:3]) * np.linalg.norm(s[:3]))))
    assert ang < 0.2
    # two longitudes 90 deg apart are 90 deg apart in inertial space
    s90 = geo_state(90.0, t)[0]
    assert np.degrees(np.arccos(np.dot(s[:3], s90[:3]) / GEO_RADIUS_KM**2)) == pytest.approx(90.0, abs=0.01)


def test_observer_state_vectorised_and_continuous():
    ts = DEMO_EPOCH_TDB_S + np.linspace(0, 3 * 86400.0, 49)
    for o in DEFAULT_OBSERVERS:
        st = observer_state(o, ts)
        assert st.shape == (49, 6)
        assert np.all(np.isfinite(st))
        single = observer_state(o, float(ts[3]))
        assert single.shape == (6,)
        assert np.allclose(single, st[3])
        # finite-difference velocity consistency (loose: CR3BP frame mapping is approximate)
        dt = 60.0
        p0 = observer_state(o, float(ts[5]) - dt)[:3]
        p1 = observer_state(o, float(ts[5]) + dt)[:3]
        v_fd = (p1 - p0) / (2 * dt)
        assert np.linalg.norm(v_fd - st[5, 3:]) < 0.02 * max(np.linalg.norm(st[5, 3:]), 0.1)
        assert observer_state(o, np.array([])).shape == (0, 6)
    # platforms sit where they should
    r_dro = np.linalg.norm(observer_state(get_observer("dro_obs"), ts)[:, :3], axis=1)
    assert r_dro.min() > 2.5e5 and r_dro.max() < 5.2e5
    r_geo = np.linalg.norm(observer_state(get_observer("geo_west"), ts)[:, :3], axis=1)
    assert np.allclose(r_geo, GEO_RADIUS_KM)


def test_custom_and_invalid_platforms():
    ic, P = FALLBACK_ORBITS["dro"]
    o = SpaceObserver("c", "c", "custom", custom_ic=ic, custom_period=P)
    assert observer_state(o, DEMO_EPOCH_TDB_S).shape == (6,)
    with pytest.raises(ValueError):
        SpaceObserver("x", "x", "not_an_orbit")
    with pytest.raises(ValueError):
        observer_state(SpaceObserver("c2", "c2", "custom"), DEMO_EPOCH_TDB_S)
    summ = orbit_summary(get_observer("nrho_obs"))
    assert summ["source"] in ("fallback",) or summ["source"].startswith("library")
    assert orbit_summary(get_observer("geo_west"))["source"] == "geostationary_itrs"


def test_resonant_without_library_has_clear_error(no_library):
    with pytest.raises(ValueError, match="resonant"):
        resolve_orbit(SpaceObserver("r", "r", "resonant"))
    assert orbit_summary(SpaceObserver("r2", "r2", "resonant"))["source"] == "unavailable"
