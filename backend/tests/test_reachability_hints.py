"""Tasking hints: which sensor could see the reachable set, where to point, angular spread."""
from __future__ import annotations

import numpy as np
import pytest

from selene.reachability.sampling import ReachabilityConfig, reachability_for_object
from selene.reachability.tasking_hint import default_sensor_list, resolve_sensors, sensor_hints
from selene.sensors.reasons import REASON_NAMES


@pytest.fixture(scope="module")
def rs():
    return reachability_for_object("SIM-DRO-01", cfg=ReachabilityConfig(dv_budget_mps=50.0, horizon_h=72.0, n_dirs=30))


def test_resolve_sensors():
    allS = default_sensor_list()
    assert len(allS) == 15
    assert [s.id for s in resolve_sensors(["geo_west", "haleakala"])] == ["geo_west", "haleakala"]
    with pytest.raises(KeyError):
        resolve_sensors(["nope"])


def test_hint_structure_and_bounds(rs):
    h = sensor_hints(rs, radius_m=2.0, albedo=0.3)
    steps = h["steps"]
    assert [s.t_h for s in steps] == [0.0, 6.0, 12.0, 18.0, 24.0, 30.0, 36.0, 42.0, 48.0, 54.0, 60.0, 66.0, 72.0]
    for s in steps:
        assert s.n_active == rs.n_samples
        assert len(s.sensors) == 15
        fr = [x.fov_capture_fraction for x in s.sensors]
        assert fr == sorted(fr, reverse=True)
        for x in s.sensors:
            assert 0.0 <= x.fov_capture_fraction <= x.visible_fraction <= 1.0
            if x.n_visible:
                assert np.isclose(np.linalg.norm(x.pointing_gcrf), 1.0)
                assert 0.0 <= x.ra_deg < 360.0 and -90.0 <= x.dec_deg <= 90.0
                assert 0.0 <= x.spread_p50_deg <= x.spread_p90_deg <= x.spread_max_deg
                assert x.n_fields_p90 >= 1
                assert x.range_min_km <= x.range_median_km <= x.range_max_km
            else:
                assert x.dominant_block_reason in REASON_NAMES.values()
    assert set(h["summary"]) == {s.id for s in default_sensor_list()}
    assert h["best_overall"] is not None


def test_space_observers_see_the_cloud_and_ground_is_glare_limited(rs):
    """Demo epoch (2026-03-01) is two days before full Moon and the DRO sits ~10 deg from the Moon as seen from
    Earth: ground sites are blocked (lunar glare / daylight) while space observers see the whole cloud."""
    h = sensor_hints(rs, radius_m=2.0, albedo=0.3)
    s0 = h["steps"][0]
    by = {x.sensor_id: x for x in s0.sensors}
    for sid in ("geo_west", "geo_east", "l1_halo_obs", "dro_obs"):
        assert by[sid].visible_fraction > 0.9, sid
    ground = [x for x in s0.sensors if x.kind == "ground"]
    assert all(x.visible_fraction < 0.5 for x in ground)
    reasons = {x.dominant_block_reason for x in ground if x.visible_fraction == 0.0}
    assert reasons <= {"moon_exclusion", "daylight", "low_elevation", "too_faint"} and reasons
    # spread grows with time as the cloud balloons
    sp = [max(x.spread_p90_deg or 0.0 for x in st.sensors) for st in h["steps"]]
    assert sp[-1] > sp[1] >= 0.0
    assert h["summary"]["geo_west"]["first_opportunity_h"] == 0.0


def test_sensor_subset_and_thinning(rs):
    h = sensor_hints(rs, sensors=["dro_obs", "nrho_obs"], radius_m=2.0, albedo=0.3, max_samples=50)
    assert len(h["steps"][0].sensors) == 2
    # n_active is the true active count (matches envelope[].n_active); the thinned subset is n_evaluated
    assert h["steps"][0].n_active == rs.n_samples == rs.envelope[0]["n_active"]
    assert h["steps"][0].n_evaluated == 50
    assert h["steps"][0].as_dict()["n_evaluated"] == 50
