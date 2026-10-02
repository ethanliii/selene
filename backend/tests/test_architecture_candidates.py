"""Unit tests for selene.architecture.candidates (platforms, sensor specs, presets)."""
from __future__ import annotations

import math

import numpy as np
import pytest

from selene.architecture.candidates import (
    PLATFORMS,
    PRESET_ARCHITECTURES,
    Architecture,
    SensorSpec,
    aperture_from_limiting_mag,
    architecture_from_dict,
    limiting_mag_from_aperture,
    preset_architectures,
)
from selene.sensors.observers import PLATFORM_ORBITS, SpaceObserver, observer_state, resolve_orbit
from selene.sensors.sites import DEFAULT_SITES, GroundSite


def test_platforms_map_to_known_observer_orbits():
    assert set(PLATFORMS) == {"geo", "l1_halo", "l2_halo_S", "dro", "nrho_9_2", "resonant_3_1"}
    for v in PLATFORMS.values():
        assert v in PLATFORM_ORBITS


def test_limiting_mag_rule_matches_ground_network_rule_of_thumb():
    # sites.py: 0.5 m ~ 18.5 mag, 1.0 m ~ 19.5-20 mag, 1.5 m ~ 20.5 mag
    assert limiting_mag_from_aperture(0.5) == pytest.approx(18.5)
    assert 19.5 <= limiting_mag_from_aperture(1.0) <= 20.1   # formula gives 20.005
    assert 20.3 <= limiting_mag_from_aperture(1.5) <= 21.0
    assert limiting_mag_from_aperture(1e-3) == 14.0 and limiting_mag_from_aperture(50.0) == 23.0
    for d in (0.2, 0.4, 1.0):
        assert aperture_from_limiting_mag(limiting_mag_from_aperture(d)) == pytest.approx(d)
    with pytest.raises(ValueError):
        limiting_mag_from_aperture(0.0)


def test_sensor_spec_resolution_and_validation():
    s = SensorSpec("dro", aperture_m=1.0)
    assert s.resolved_aperture_m == 1.0 and s.resolved_limiting_mag == pytest.approx(limiting_mag_from_aperture(1.0))
    s2 = SensorSpec("geo", limiting_mag=18.0)
    assert s2.resolved_limiting_mag == 18.0 and s2.resolved_aperture_m == pytest.approx(aperture_from_limiting_mag(18.0))
    s3 = SensorSpec("l2_halo_S")
    assert (s3.resolved_aperture_m, s3.resolved_limiting_mag) == (0.4, 18.0)
    with pytest.raises(ValueError):
        SensorSpec("l3_halo")
    with pytest.raises(ValueError):
        SensorSpec("dro", aperture_m=-1.0)
    with pytest.raises(ValueError):
        SensorSpec("dro", fov_deg=0.0)
    with pytest.raises(ValueError):
        SensorSpec("dro", slew_rate_deg_s=0.0)


@pytest.mark.parametrize("platform", sorted(PLATFORMS))
def test_every_platform_builds_an_observer_with_a_state(platform):
    obs = SensorSpec(platform, aperture_m=0.4, fov_deg=2.5, slew_rate_deg_s=0.5, lon_deg=30.0).to_observer(3, "t")
    assert isinstance(obs, SpaceObserver) and obs.id == f"t_3_{platform}"
    assert obs.platform_orbit == PLATFORMS[platform] and obs.fov_deg == 2.5 and obs.slew_rate_deg_s == 0.5
    if platform != "geo":
        orb = resolve_orbit(obs)
        assert orb.source.startswith("library:")
    from selene.dynamics.frames import DEMO_EPOCH_TDB_S
    x = observer_state(obs, DEMO_EPOCH_TDB_S)
    assert x.shape == (6,) and np.all(np.isfinite(x))
    r = np.linalg.norm(x[:3])
    assert (42_000 < r < 42_300) if platform == "geo" else (50_000 < r < 600_000)


def test_presets_are_four_distinct_architectures_with_rationale():
    names = [a.name for a in PRESET_ARCHITECTURES]
    assert len(names) == 4 and len(set(names)) == 4
    assert names[0] == "Ground only"
    for a in PRESET_ARCHITECTURES:
        assert a.ground is True and len(a.notes) > 40
        sensors = a.all_sensors()
        assert len(sensors) == len(DEFAULT_SITES) + len(a.sensors)
        assert all(isinstance(s, (GroundSite, SpaceObserver)) for s in sensors)
        ids = [s.id for s in sensors]
        assert len(ids) == len(set(ids))
    by = {a.name: a for a in PRESET_ARCHITECTURES}
    assert by["Ground only"].n_space_sensors == 0
    assert [s.platform for s in by["Ground + 2 GEO"].sensors] == ["geo", "geo"]
    assert [s.platform for s in by["Ground + L2 halo"].sensors] == ["l2_halo_S"]
    assert sorted(s.platform for s in by["Ground + DRO + L1 halo"].sensors) == ["dro", "l1_halo"]
    fresh = preset_architectures()
    fresh[0].sensors.append(SensorSpec("dro"))
    assert PRESET_ARCHITECTURES[0].n_space_sensors == 0   # presets are not mutated by callers


def test_cost_proxies_and_dict_round_trip():
    a = Architecture("Two DROs", [SensorSpec("dro", aperture_m=0.5), SensorSpec("dro", aperture_m=1.0, phase=0.5)], ground=False)
    cp = a.cost_proxies()
    assert cp["n_space_sensors"] == 2 and cp["n_ground_sites"] == 0
    assert cp["total_space_aperture_m"] == pytest.approx(1.5)
    assert cp["total_space_collecting_area_m2"] == pytest.approx(math.pi * (0.25**2 + 0.5**2))
    d = a.as_dict()
    b = architecture_from_dict({"name": d["name"], "ground": d["ground"], "notes": d["notes"],
                                "sensors": [{k: v for k, v in s.items() if k in ("platform", "aperture_m", "limiting_mag", "fov_deg",
                                                                                 "slew_rate_deg_s", "lon_deg", "phase", "label")}
                                            for s in d["sensors"]]})
    assert b.name == a.name and b.n_space_sensors == 2 and b.total_aperture_m == pytest.approx(1.5)
    assert [o.id for o in b.observers()] == ["two_dros_0_dro", "two_dros_1_dro"]


def test_architecture_from_dict_rejects_bad_input():
    with pytest.raises(ValueError):
        architecture_from_dict({"name": "x", "sensors": [{"platform": "mars_orbit"}]})
    with pytest.raises(ValueError):
        architecture_from_dict({"name": "x", "sensors": [{"platform": "dro", "aperture_cm": 40}]})
    with pytest.raises(ValueError):
        architecture_from_dict({"name": "  ", "sensors": []})
    with pytest.raises(ValueError):
        Architecture("empty", [], ground=False).all_sensors()
