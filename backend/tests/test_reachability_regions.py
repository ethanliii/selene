"""Region membership tests with constructed points (rotating-frame nd / GCRF km)."""
from __future__ import annotations

import numpy as np
import pytest

from selene.constants import GEO_RADIUS_KM, L_STAR, MU, R_MOON
from selene.dynamics.cr3bp import jacobi, lagrange_points
from selene.reachability.regions import (
    C_L1,
    C_L2,
    MOON_ROT,
    RegionInputs,
    classify,
    default_regions,
    jacobi_rot,
    nrho_reference_samples,
)

L = lagrange_points(MU)
REG = {r.key: r for r in default_regions()}


def _inputs(pos_rot=None, pos_gcrf=None, pos_moon=None, vel_rot=None, vel_gcrf=None):
    """Build RegionInputs with harmless defaults for the frames a given test does not care about."""
    n = None
    for a in (pos_rot, pos_gcrf, pos_moon):
        if a is not None:
            n = np.asarray(a, float).reshape(-1, 3).shape[0]
            break
    far_rot = np.tile([0.5, 0.5, 0.0], (n, 1))            # mid-way, far from every landmark
    far_gcrf = np.tile([200_000.0, 200_000.0, 0.0], (n, 1))
    far_moon = np.tile([150_000.0, 150_000.0, 0.0], (n, 1))
    return RegionInputs(
        pos_rot=np.asarray(pos_rot, float).reshape(-1, 3) if pos_rot is not None else far_rot,
        pos_gcrf=np.asarray(pos_gcrf, float).reshape(-1, 3) if pos_gcrf is not None else far_gcrf,
        pos_moon=np.asarray(pos_moon, float).reshape(-1, 3) if pos_moon is not None else far_moon,
        vel_rot=None if vel_rot is None else np.asarray(vel_rot, float).reshape(-1, 3),
        vel_gcrf=None if vel_gcrf is None else np.asarray(vel_gcrf, float).reshape(-1, 3),
    )


def test_jacobi_constants_match_literature():
    # Koon-Lo-Marsden-Ross (2011) Table 2.1 / Szebehely: Earth-Moon C1 ~ 3.1883, C2 ~ 3.1722
    assert abs(C_L1 - 3.1883) < 2e-3
    assert abs(C_L2 - 3.1722) < 2e-3
    assert C_L1 > C_L2
    s = np.array([0.5, 0.2, 0.1, 0.3, -0.1, 0.05])
    assert np.isclose(jacobi_rot(s[:3], s[3:]), jacobi(s))


def test_gateway_pointwise_fallback_is_the_proximity_ball():
    """Without a time series (series=False) the gateways fall back to the neck ball."""
    inside = L[0] + np.array([0.03, 0.02, 0.01])
    outside = L[0] + np.array([0.05, 0.02, 0.0])
    inp = _inputs(pos_rot=[inside, outside, L[1], L[1] + [0.0, 0.0, 0.06]])
    inp.series = False
    assert REG["l1_gateway"].contains(inp).tolist() == [True, False, False, False]
    assert REG["l2_gateway"].contains(inp).tolist() == [False, False, True, False]
    # configurable radius
    big = {r.key: r for r in default_regions(gateway_radius_nd=0.08)}
    assert big["l1_gateway"].contains(inp).tolist() == [True, True, False, False]


def _series(xs, y=0.0):
    """(T, 3) rotating-frame path along x through the L1 neighbourhood (y, z offsets fixed)."""
    return np.array([[x, y, 0.0] for x in xs])


def test_gateway_is_a_neck_transit_not_a_graze():
    """The gateway test is a realm change through the neck ball: a path that crosses the plane x = x_L1 and
    leaves the ball on the other side transits; a DRO-like graze that dips past L1 and returns to the lunar
    side does not, however close it comes (the demo DRO comes within 1 400 km of L1 and 4 000 km past the plane)."""
    xl = L[0][0]
    r = 0.05
    transit = _series([xl + 0.10, xl + 0.04, xl + 0.01, xl - 0.01, xl - 0.04, xl - 0.10])          # Moon side -> Earth side
    graze = _series([xl + 0.10, xl + 0.04, xl + 0.01, xl - 0.011, xl + 0.01, xl + 0.04, xl + 0.10])  # dips 4 200 km past, returns
    far_cross = _series([xl + 0.10, xl + 0.01, xl - 0.01, xl - 0.10], y=0.08)   # crosses the plane OUTSIDE the ball
    for path, expect in ((transit, True), (graze, False), (far_cross, False)):
        inp = _inputs(pos_rot=path)
        m = REG["l1_gateway"].contains(inp)
        assert m.shape == (len(path),)
        assert bool(m.any()) is expect, (path[:, 0] - xl, m)
    m = REG["l1_gateway"].contains(_inputs(pos_rot=transit))
    assert m.tolist() == [False, True, True, True, True, False]        # the epochs inside the ball during the transit
    assert np.argmax(m) == 1                                           # first hit = neck entry
    # unresolved at the end of the horizon: counts once the plane has been crossed
    half = _series([xl + 0.10, xl + 0.04, xl + 0.01, xl - 0.01])
    assert REG["l1_gateway"].contains(_inputs(pos_rot=half)).tolist() == [False, True, True, True]
    half_no_cross = _series([xl + 0.10, xl + 0.04, xl + 0.01])
    assert not REG["l1_gateway"].contains(_inputs(pos_rot=half_no_cross)).any()
    # (N, T, 3) batches: rows are independent trajectories (equal length 7; the transit row coasts on in the Earth realm)
    transit7 = np.vstack([transit, [[xl - 0.20, 0.0, 0.0]]])
    batch = np.stack([transit7, graze, np.full((7, 3), np.nan)])
    out = REG["l1_gateway"].contains(RegionInputs(batch, np.zeros((3, 7, 3)) + 1e5, np.zeros((3, 7, 3)) + 1e5))
    assert out.shape == (3, 7) and out[0].any() and not out[1].any() and not out[2].any()
    assert out[0].tolist() == [False, True, True, True, True, False, False]
    # the same test at L2 (lunar realm -> exterior realm)
    xl2 = L[1][0]
    out_l2 = REG["l2_gateway"].contains(_inputs(pos_rot=_series([xl2 - 0.10, xl2 - 0.02, xl2 + 0.02, xl2 + 0.10])))
    assert out_l2.tolist() == [False, True, True, False]
    assert r == REG["l1_gateway"].params["radius_nd"] and REG["l1_gateway"].params["membership"] == "neck_transit"


def test_nrho_corridor_tube():
    pts = nrho_reference_samples(600)
    assert pts.shape == (600, 3)
    perilune_km = np.linalg.norm(pts - MOON_ROT, axis=1).min() * L_STAR
    assert 3000.0 < perilune_km < 3600.0          # 9:2 NRHO perilune (Zimovan-Spreen 2020 / Lee 2019)
    # pts[0] is the library IC = apolune (farthest point from the Moon, below it); moving further
    # down (-z) moves away from the whole curve, unlike an offset along the orbit's own tangent.
    on = pts[0]
    assert np.argmax(np.linalg.norm(pts - MOON_ROT, axis=1)) == 0
    near = on + np.array([0.0, 0.0, -5_000.0 / L_STAR])     # 5 000 km off the orbit
    far = on + np.array([0.0, 0.0, -15_000.0 / L_STAR])     # 15 000 km off: outside a 10 000 km tube
    inp = _inputs(pos_rot=[on, near, far, [0.3, 0.3, 0.0]])
    assert REG["nrho_corridor"].contains(inp).tolist() == [True, True, False, False]
    narrow = {r.key: r for r in default_regions(nrho_tube_km=2_000.0)}
    assert narrow["nrho_corridor"].contains(inp).tolist() == [True, False, False, False]


def test_south_pole_cone():
    d = 10_000.0
    below = np.array([0.0, 0.0, -d])                      # straight below the Moon (lat -90)
    lat70 = d * np.array([np.cos(np.radians(-70)), 0.0, np.sin(np.radians(-70))])
    lat50 = d * np.array([np.cos(np.radians(-50)), 0.0, np.sin(np.radians(-50))])
    too_far = np.array([0.0, 0.0, -25_000.0])
    pos_moon = np.vstack([below, lat70, lat50, too_far])
    inp = _inputs(pos_rot=MOON_ROT + pos_moon / L_STAR, pos_moon=pos_moon)
    assert REG["south_pole_approach"].contains(inp).tolist() == [True, True, False, False]


def test_llo_shells_and_impact():
    alts = np.array([300.0, 3_000.0, 6_000.0, -10.0])
    pos_moon = np.stack([(R_MOON + a) * np.array([1.0, 0.0, 0.0]) for a in alts])
    inp = _inputs(pos_moon=pos_moon, pos_rot=MOON_ROT + pos_moon / L_STAR)
    assert REG["llo_shell"].contains(inp).tolist() == [True, True, False, False]
    assert REG["llo_inner"].contains(inp).tolist() == [True, False, False, False]
    assert REG["lunar_impact"].contains(inp).tolist() == [False, False, False, True]


def test_geo_belt_return():
    r_belt = np.array([GEO_RADIUS_KM, 0.0, 0.0])
    r_in = np.array([50_000.0, 0.0, 0.0])
    r_far = np.array([100_000.0, 0.0, 0.0])
    pos = np.vstack([r_belt, r_in, r_in, r_far, r_belt + [4_000.0, 0, 0]])
    vel = np.array([[0.0, 3.0, 0.0], [-1.0, 1.0, 0.0], [1.0, 1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 3.0, 0.0]])
    inp = _inputs(pos_gcrf=pos, vel_gcrf=vel)
    assert REG["geo_belt_return"].contains(inp).tolist() == [True, True, False, False, False]
    # without velocities only the belt band counts
    inp2 = _inputs(pos_gcrf=pos)
    assert REG["geo_belt_return"].contains(inp2).tolist() == [True, False, False, False, False]


def test_escape_region_energy_and_distance():
    p_out = np.array([1.4, 0.0, 0.0])          # |r| = 1.4 > 1.3
    p_in = np.array([1.0, 0.5, 0.0])           # |r| = 1.12 < 1.3
    v_slow = np.zeros(3)                       # C = 2U > C_L2 : energetically closed
    v_fast = np.array([0.0, 0.8, 0.0])         # C = 2U - 0.64 < C_L2
    inp = _inputs(pos_rot=[p_out, p_out, p_in], vel_rot=[v_slow, v_fast, v_fast])
    C = inp.jacobi
    assert C[0] > C_L2 > C[1]
    assert REG["earth_return_escape"].contains(inp).tolist() == [False, True, False]


def test_classify_broadcasts_over_grid_shapes():
    n, t = 4, 7
    pos_rot = np.tile(L[0], (n, t, 1))
    inp = RegionInputs(pos_rot, np.zeros((n, t, 3)) + 1e5, np.zeros((n, t, 3)) + 1e5, np.zeros((n, t, 3)), np.zeros((n, t, 3)))
    out = classify(inp)
    assert set(out) == set(REG)
    for k, v in out.items():
        assert v.shape == (n, t)
    # a path sitting AT L1 never changes realm: not a transit (series semantics) ...
    assert not out["l1_gateway"].any() and not out["l2_gateway"].any()
    # ... but it is inside the proximity ball point-wise
    inp.series = False
    assert classify(inp)["l1_gateway"].all()


def test_region_as_dict_is_jsonable():
    import json

    for r in default_regions():
        json.dumps(r.as_dict())
        assert r.why_it_matters and r.description
