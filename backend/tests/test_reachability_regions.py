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
    """The gateway test is a realm change through the neck ball: a path that comes from the lunar realm proper
    (x > x_L1 + delta), crosses the plane x = x_L1 inside the ball and goes on into the Earth realm proper
    (x < x_L1 - delta) transits; a DRO-like graze that dips past L1 and returns to the lunar side does not,
    however deep (the quiet demo DRO dips 4 200 km past the plane); nor does a passage left unresolved in the neck
    zone at the end of the horizon, or one that starts inside the neck zone."""
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
    # unresolved at the end of the horizon (still in the neck zone |x - x_L1| <= delta): NOT a transit ...
    half = _series([xl + 0.10, xl + 0.04, xl + 0.01, xl - 0.01])
    assert not REG["l1_gateway"].contains(_inputs(pos_rot=half)).any()
    # ... until the path reaches the Earth realm proper
    half_resolved = _series([xl + 0.10, xl + 0.04, xl + 0.01, xl - 0.01, xl - 0.06])
    assert REG["l1_gateway"].contains(_inputs(pos_rot=half_resolved)).tolist() == [False, True, True, True, False]
    # a path that starts inside the neck zone has no entry realm: not a transit
    start_inside = _series([xl + 0.01, xl - 0.01, xl - 0.04, xl - 0.10])
    assert not REG["l1_gateway"].contains(_inputs(pos_rot=start_inside)).any()
    # hysteresis: leaving the ball in the neck zone and wandering back to the lunar realm is a graze, not a transit
    wander = _series([xl + 0.10, xl + 0.02, xl - 0.02, xl - 0.04, xl - 0.03, xl + 0.02, xl + 0.10])
    assert not REG["l1_gateway"].contains(_inputs(pos_rot=wander)).any()
    # the plane must be crossed INSIDE the ball: grazing the ball on the lunar side and later crossing the plane far
    # from L1 (|y| = 0.2) is a realm change, but not through the L1 neck
    around = np.array([[xl + 0.10, 0.0, 0.0], [xl + 0.02, 0.0, 0.0], [xl + 0.01, 0.06, 0.0], [xl - 0.02, 0.20, 0.0], [xl - 0.10, 0.20, 0.0]])
    assert not REG["l1_gateway"].contains(_inputs(pos_rot=around)).any()
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
    assert REG["l1_gateway"].params["realm_margin_nd"] == r and REG["l1_gateway"].params["energy_gate"]["C_open"] == C_L1
    # a wider realm margin makes the same path unresolved at both ends (it starts and ends inside |x - x_L1| <= 0.15)
    wide = {r_.key: r_ for r_ in default_regions(realm_margin_nd=0.15)}
    assert not wide["l1_gateway"].contains(_inputs(pos_rot=transit)).any()
    long_path = np.vstack([[[xl + 0.20, 0.0, 0.0]], transit, [[xl - 0.20, 0.0, 0.0]]])
    assert wide["l1_gateway"].contains(_inputs(pos_rot=long_path)).tolist() == [False, False, True, True, True, True, False, False]


def test_gateway_energy_gate_requires_an_open_neck():
    """With velocities supplied, a geometric transit is only counted when the CR3BP-equivalent Jacobi constant on
    the passage is below C(L1) (+ the 5e-3 slack): along the x axis Omega has a minimum at L1, so zero-velocity
    points 0.03-0.04 nd from L1 sit 0.01-0.02 above C(L1) (neck closed), while a 0.2 nd speed opens it."""
    xl = L[0][0]
    path = _series([xl + 0.10, xl + 0.04, xl + 0.03, xl - 0.03, xl - 0.04, xl - 0.10])
    still = _inputs(pos_rot=path, vel_rot=np.zeros_like(path))
    C = still.jacobi
    assert (C[1:5] > C_L1 + 0.006).all(), C - C_L1
    assert not REG["l1_gateway"].contains(still).any()
    moving = _inputs(pos_rot=path, vel_rot=np.tile([-0.2, 0.0, 0.0], (len(path), 1)))
    assert (moving.jacobi[1:5] < C_L1).all()
    assert REG["l1_gateway"].contains(moving).tolist() == [False, True, True, True, True, False]
    # without velocities the gate is skipped (constructed position-only paths)
    assert REG["l1_gateway"].contains(_inputs(pos_rot=path)).any()
    # the omega_t_star rescaling is applied before the gate
    scaled = _inputs(pos_rot=path, vel_rot=np.tile([-0.2, 0.0, 0.0], (len(path), 1)))
    scaled.omega_t_star = np.full(len(path), 1.0)
    assert np.allclose(scaled.jacobi, moving.jacobi)


# ---------------------------------------------------------------------------
# the three cases the classifier is graded on (measured, not constructed)
def _ephem_inputs(states_gcrf, grid):
    from selene.constants import T_STAR
    from selene.dynamics import frames
    from selene.dynamics.ephemeris import get_ephemeris

    fr = frames.rotating_frame(grid)
    rot = frames.gcrf_to_rot(states_gcrf, grid)
    moon = get_ephemeris().position("moon", grid)
    inp = RegionInputs(rot[:, :3], states_gcrf[:, :3], states_gcrf[:, :3] - moon, rot[:, 3:6], states_gcrf[:, 3:6],
                       omega_t_star=np.asarray(fr.omega) * T_STAR)
    return inp, rot


def test_quiet_dro_truth_never_transits_a_gateway():
    """(a) The unperturbed SIM-DRO-01 truth over 2026-02-15 .. 2026-03-30 (43 days, four passes through the L1 ball):
    it comes within ~1 400 km of L1 and dips ~4 200 km past the x = x_L1 plane on one pass, with C ~ 2.95 < C(L1),
    and is still never a gateway transit — it never leaves the lunar realm proper (measured: min x - x_L1 = -4 186 km,
    max x - x_L2 = -1 214 km, 0 transits)."""
    from selene.objects.catalog import get_catalog
    from selene.time import seconds_since_j2000_tdb

    cat = get_catalog()
    grid = np.arange(float(seconds_since_j2000_tdb("2026-02-15T00:00:00")), float(seconds_since_j2000_tdb("2026-03-30T00:00:00")) + 1, 3600.0)
    inp, rot = _ephem_inputs(cat.state_at("SIM-DRO-01", grid), grid)
    d1 = np.linalg.norm(rot[:, :3] - L[0], axis=1) * L_STAR
    x_rel_km = (rot[:, 0] - L[0][0]) * L_STAR
    print(f"quiet DRO: min d(L1) {d1.min():,.0f} km, min x - x_L1 {x_rel_km.min():,.0f} km, C {inp.jacobi.min():.4f}..{inp.jacobi.max():.4f}")
    assert d1.min() < 5_000.0                                  # it really does graze L1 ...
    assert -10_000.0 < x_rel_km.min() < -1_000.0               # ... and really does dip past the plane ...
    assert (inp.jacobi < C_L1).all()                           # ... with the neck energetically open ...
    assert rot[:, 0].max() > L[0][0] + 0.05                    # ... but always returns to the lunar realm proper
    out = classify(inp)
    assert not out["l1_gateway"].any() and not out["l2_gateway"].any()   # 0 transits: routine DRO geometry
    from selene.reachability.regions import neck_passage_diagnostics
    dg = neck_passage_diagnostics(rot[None, :, :3], L[0], 0.05, 0.05)
    assert dg["n_enter_ball"] == 1 and dg["n_cross_plane_in_ball"] == 1 and dg["n_reach_far_realm"] == 0
    assert 1_000.0 < dg["max_depth_past_plane_km"] < 10_000.0 and abs(dg["closest_km"] - d1.min()) < 1.0


def test_demo_burned_truth_grazes_l1_without_transiting():
    """(b) The demo's burned truth (30 m/s at 2026-02-25T08:30Z) passes ~280 km from L1 on the hourly grid, dips
    ~200 km past the x = x_L1 plane with C ~ 2.945 < C(L1) and returns to x > x_L1 + 0.05: a graze, not a transit."""
    from selene.dynamics.ephemeris import EphemParams
    from selene.maneuver.synthetic import Burn, truth_with_burn
    from selene.objects.catalog import get_catalog
    from selene.scenario.demo import PROTAGONIST, DemoConfig
    from selene.time import seconds_since_j2000_tdb

    cfg = DemoConfig.make(0, True)
    cat = get_catalog()
    t0 = float(seconds_since_j2000_tdb(cfg.t0_utc)); t1 = float(seconds_since_j2000_tdb(cfg.t1_utc)); tb = float(seconds_since_j2000_tdb(cfg.burn_utc))
    phys = cat.get(PROTAGONIST).physical or {}
    params = EphemParams(srp=True, cr_area_mass=float(phys.get("cr_area_mass_m2_kg", 0.0)))
    truth = truth_with_burn(cat.state_at(PROTAGONIST, t0), t0, t1 + 2 * 86400.0, Burn(tb, cfg.burn_mps * 1e-3 * cfg.burn_dir_gcrf), params)
    grid = np.arange(t0, t1 + 2 * 86400.0 + 1, 3600.0)
    inp, rot = _ephem_inputs(truth.at(grid), grid)
    d1 = np.linalg.norm(rot[:, :3] - L[0], axis=1) * L_STAR
    x_rel_km = (rot[:, 0] - L[0][0]) * L_STAR
    after = grid > tb
    print(f"burned truth: min d(L1) {d1.min():,.0f} km at +{(grid[np.argmin(d1)] - tb) / 3600:.1f} h, min x - x_L1 {x_rel_km.min():,.0f} km, "
          f"C after burn {inp.jacobi[after].min():.4f}..{inp.jacobi[after].max():.4f}")
    assert 100.0 < d1.min() < 1_000.0                                  # a genuine graze of L1
    assert -1_000.0 < x_rel_km.min() < 0.0                             # it crosses the plane, by a few hundred km only
    assert (inp.jacobi[after] < C_L1).all()                            # neck open: the energy gate is not what saves us
    k_min = int(np.argmin(d1))
    assert rot[k_min:, 0].max() > L[0][0] + 0.05                       # and it returns to the lunar realm proper
    out = classify(inp)
    assert not out["l1_gateway"].any() and not out["l2_gateway"].any()


def _lyapunov_transit(rec, su: int, ss: int, n: int = 1200):
    """Stitched CR3BP time series through an L1 Lyapunov orbit: perturb its IC by 1e-6 along (su * unstable + ss *
    stable) eigenvectors of the monodromy matrix, propagate 2.5 periods backward and forward.  su = -ss gives the
    textbook transit orbit (one realm -> the neck -> the other realm); su = ss gives a non-transit orbit that
    approaches the neck and bounces back into the realm it came from (Koon-Lo-Marsden-Ross ch. 2)."""
    from selene.dynamics.cr3bp import propagate_cr3bp

    T = rec.period_nd
    Phi = propagate_cr3bp(rec.ic_array, tf=T, stm=True).y[6:, -1].reshape(6, 6)
    w, V = np.linalg.eig(Phi)
    vu = np.real(V[:, int(np.argmax(np.abs(w)))]); vu /= np.linalg.norm(vu)
    vs = np.real(V[:, int(np.argmin(np.abs(w)))]); vs /= np.linalg.norm(vs)
    s0 = rec.ic_array + 1e-6 * (su * vu + ss * vs)
    fwd = propagate_cr3bp(s0, t_eval=np.linspace(0.0, 2.5 * T, n))
    bwd = propagate_cr3bp(s0, t_eval=np.linspace(0.0, -2.5 * T, n))
    y = np.hstack([bwd.y[:, ::-1], fwd.y[:, 1:]])
    return y[:3].T.copy(), y[3:6].T.copy()


def test_l1_lyapunov_manifold_transit_is_classified_as_a_transit():
    """(c) Constructed transit orbits seeded from the stable/unstable manifolds of a small L1 Lyapunov orbit
    (C = 3.1873, just below C(L1): the neck is a few thousand km wide) MUST be classified as L1 gateway transits,
    in both directions, with the first flagged epoch being the neck entry; the companion non-transit orbits (same
    energy, same neck approach, bounce back) must not."""
    from selene.orbits.library import get_library

    rec = get_library().find(family="L1_lyapunov", jacobi=3.1873)
    assert rec is not None and rec.jacobi < C_L1
    xl = L[0][0]
    for su, ss, expect in ((+1, -1, True), (-1, +1, True), (+1, +1, False), (-1, -1, False)):
        P, V = _lyapunov_transit(rec, su, ss)
        inp = RegionInputs(P, np.zeros_like(P) + 1e5, np.zeros_like(P) + 1e5, V, None)
        m = REG["l1_gateway"].contains(inp)
        realm_start, realm_end = np.sign(P[0, 0] - xl), np.sign(P[-1, 0] - xl)
        assert abs(P[0, 0] - xl) > 0.2 and abs(P[-1, 0] - xl) > 0.2          # both ends deep in a realm proper
        assert bool(realm_start != realm_end) is expect
        assert bool(m.any()) is expect, (su, ss)
        assert np.linalg.norm(P - L[0], axis=1).min() * L_STAR < 5_000.0   # every case passes through the neck ball
        if expect:
            k = int(np.argmax(m))
            assert np.linalg.norm(P[k] - L[0]) < 0.05 and np.linalg.norm(P[k - 1] - L[0]) >= 0.05   # first hit = ball entry
            assert np.sign(P[k - 1, 0] - xl) == realm_start
            assert inp.jacobi[m].max() < C_L1
        assert not REG["l2_gateway"].contains(inp).any()


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
