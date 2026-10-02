"""Reachable-set sampler: propagation fidelity, 0-Δv invariance, monotonicity, L1 case, timing."""
from __future__ import annotations

import time

import numpy as np
import pytest

from selene.constants import GM_MOON, R_MOON
from selene.dynamics import frames
from selene.dynamics.ephemeris import EphemParams, get_ephemeris, propagate_ephemeris
from selene.objects.catalog import get_catalog
from selene.reachability.sampling import (
    LADDER_MPS,
    ReachabilityConfig,
    batch_gcrf_to_rot,
    build_samples,
    compute_reachability,
    fibonacci_sphere,
    magnitude_ladder,
    propagate_stacked,
    propagate_with_burns,
    reachability_for_object,
    refine_min_dv,
)

H = 3600.0


@pytest.fixture(scope="module")
def dro():
    cat = get_catalog()
    return cat.epoch_s, cat.epoch_state("SIM-DRO-01")


# ---------------------------------------------------------------------------
def test_fibonacci_sphere_uniformity():
    u = fibonacci_sphere(200)
    assert u.shape == (200, 3)
    assert np.allclose(np.linalg.norm(u, axis=1), 1.0)
    assert np.linalg.norm(u.mean(axis=0)) < 0.02          # balanced
    # nearest-neighbour angles roughly uniform: spread of NN angle small
    cos = np.clip(u @ u.T, -1, 1)
    np.fill_diagonal(cos, -1)
    nn = np.degrees(np.arccos(cos.max(axis=1)))
    assert nn.std() / nn.mean() < 0.2
    assert fibonacci_sphere(1).shape == (1, 3)


def test_magnitude_ladder_is_nested():
    s10, s50, s150 = (set(magnitude_ladder(b).tolist()) for b in (10, 50, 150))
    assert s10 < s50 < s150
    assert max(s150) == 150.0 and 150.0 in s150
    assert magnitude_ladder(0).tolist() == [0.0]
    assert magnitude_ladder(37.0).tolist() == [2.0, 5.0, 10.0, 20.0, 37.0]
    assert all(a < b for a, b in zip(LADDER_MPS, LADDER_MPS[1:]))


def test_build_samples_shapes():
    cfg = ReachabilityConfig(dv_budget_mps=50, horizon_h=72, n_dirs=10, burn_epochs_h=(0, 12))
    s = build_samples(cfg)
    n_m = len(magnitude_ladder(50))
    assert s["dirs"].shape == (10 * n_m * 2, 3)
    assert s["n_rays"] == 20
    assert set(np.unique(s["burn_h"])) == {0.0, 12.0}
    assert len(np.unique(s["ray"])) == 20


# ---------------------------------------------------------------------------
def test_stacked_matches_per_sample_propagation(dro):
    """Stacked 6N system vs. independent per-sample DOP853 (same force model, tolerances, max_step) < 1 m over 72 h."""
    t0, s0 = dro
    rng = np.random.default_rng(1)
    rows = np.repeat(s0[None], 6, axis=0)
    rows[:, 3:] += rng.normal(0.0, 0.02, (6, 3))             # ~20 m/s random kicks
    tg = t0 + np.arange(0, 72 * H + 1, H)
    params = EphemParams(srp=True, cr_area_mass=0.0104)
    cfg = ReachabilityConfig()
    st, info = propagate_stacked(rows, t0, tg, params, rtol=cfg.rtol, atol=cfg.atol, max_step_s=cfg.max_step_s)
    assert st.shape == (6, len(tg), 6) and info["nfev"] > 0
    worst = 0.0
    for i in range(6):
        sol = propagate_ephemeris(rows[i], t0, t_eval_s=tg, params=params, rtol=cfg.rtol, atol=cfg.atol, max_step=cfg.max_step_s)
        err = np.linalg.norm(sol.y[:3].T - st[i, :, :3], axis=1).max()
        worst = max(worst, err)
    print(f"stacked vs per-sample max position difference: {worst * 1e3:.3f} m")
    assert worst < 1e-3


def test_batch_frame_transform_matches_frames_module(dro):
    t0, s0 = dro
    tg = t0 + np.arange(0, 24 * H + 1, 6 * H)
    rows = np.repeat(s0[None], 3, axis=0)
    rows[1, 3:] += 0.01
    rows[2, :3] += 500.0
    st, _ = propagate_stacked(rows, t0, tg)
    fr = frames.rotating_frame(tg)
    rot = batch_gcrf_to_rot(st, fr)
    for i in range(3):
        ref = frames.gcrf_to_rot(st[i], tg)
        assert np.abs(rot[i] - ref).max() < 1e-12


def test_propagate_with_burns_semantics(dro):
    t0, s0 = dro
    tg = t0 + np.arange(0, 24 * H + 1, H)
    dv = np.array([[0.0, 0.0, 0.0], [0.03, 0.0, 0.0], [0.0, 0.03, 0.0]])
    tb = np.array([t0, t0, t0 + 6 * H])
    st, info = propagate_with_burns(s0, t0, dv, tb, tg)
    assert info["n_segments"] == 2
    # row 2 (burn at +6 h) equals the nominal row up to and including the burn epoch sample
    assert np.array_equal(st[2, :7], st[0, :7])
    assert np.linalg.norm(st[2, 7, :3] - st[0, 7, :3]) > 1.0            # then departs (30 m/s * 1 h ~ 100 km)
    # row 1 (burn at t0) equals a direct propagation of the kicked state
    kicked = s0.copy()
    kicked[3:] += dv[1]
    sol = propagate_ephemeris(kicked, t0, t_eval_s=tg, rtol=1e-11, atol=1e-8, max_step=1800.0)
    assert np.linalg.norm(sol.y[:3].T - st[1, :, :3], axis=1).max() < 1e-3
    with pytest.raises(ValueError):
        propagate_with_burns(s0, t0, dv, np.array([t0, t0, t0 + 48 * H]), tg)


# ---------------------------------------------------------------------------
def test_zero_budget_coincides_with_nominal():
    cfg = ReachabilityConfig(dv_budget_mps=0.0, horizon_h=72, n_dirs=12)
    rs = reachability_for_object("SIM-DRO-01", cfg=cfg)
    dev = np.linalg.norm(rs.states_gcrf[:, :, :3] - rs.nominal_gcrf[None, :, :3], axis=2)
    assert dev.max() < 1.0, dev.max()                        # km (0 by construction: identical rows of one ODE)
    # the independent check: the stacked nominal agrees with the catalog truth arc (separate
    # per-object DOP853 integration of the same force model) to < 1 km over 72 h
    truth = get_catalog().state_at("SIM-DRO-01", rs.t_grid_s)
    assert np.linalg.norm(truth[:, :3] - rs.nominal_gcrf[:, :3], axis=1).max() < 1.0
    assert rs.meta["n_terminated"] == 0
    hit_by_samples = {k for k, v in rs.hits.items() if v.any()}
    hit_by_nominal = {k for k, v in rs.nominal_hits.items() if v.any()}
    assert hit_by_samples == hit_by_nominal
    assert not any(s.newly_reachable for s in rs.region_stats)
    for s in rs.region_stats:
        assert s.fraction in (0.0, 1.0)
        if s.nominal_hits:
            assert s.earliest_h == s.earliest_nominal_h


def _stats(rs):
    return {s.key: s for s in rs.region_stats}


@pytest.mark.parametrize("horizon_h", [72.0, 168.0])
def test_monotone_in_budget(horizon_h):
    runs = {}
    for b in (10.0, 50.0, 150.0):
        cfg = ReachabilityConfig(dv_budget_mps=b, horizon_h=horizon_h, n_dirs=40)
        runs[b] = _stats(reachability_for_object("SIM-DRO-01", cfg=cfg))
    keys = list(runs[10.0])
    counts = []
    for b in (10.0, 50.0, 150.0):
        counts.append(sum(1 for k in keys if runs[b][k].n_hit > 0))
        print(f"{horizon_h:.0f} h, budget {b:5.0f} m/s: " + ", ".join(f"{k}={runs[b][k].fraction:.3f}" for k in keys if runs[b][k].n_hit))
    assert counts == sorted(counts)
    if horizon_h == 168.0:
        assert counts[-1] > counts[0]                  # with a week, 150 m/s opens regions 10 m/s cannot reach
    for k in keys:
        f = [runs[b][k].fraction for b in (10.0, 50.0, 150.0)]
        assert f == sorted(f), (k, f)
        dv = [runs[b][k].min_dv_mps for b in (10.0, 50.0, 150.0)]
        dv_def = [v for v in dv if v is not None]
        assert dv_def == sorted(dv_def, reverse=True), (k, dv)
        e = [runs[b][k].earliest_h for b in (10.0, 50.0, 150.0)]
        e_def = [v for v in e if v is not None]
        assert e_def == sorted(e_def, reverse=True), (k, e)


def test_monotone_in_horizon():
    runs = {}
    for h in (24.0, 72.0, 168.0):
        cfg = ReachabilityConfig(dv_budget_mps=50.0, horizon_h=h, n_dirs=40)
        runs[h] = _stats(reachability_for_object("SIM-DRO-01", cfg=cfg))
    keys = list(runs[24.0])
    counts = [sum(1 for k in keys if runs[h][k].n_hit > 0) for h in (24.0, 72.0, 168.0)]
    print("regions reached vs horizon 24/72/168 h:", counts)
    assert counts == sorted(counts)
    assert counts[-1] > counts[0]                      # the DRO reaches more with a week than with a day
    for k in keys:
        f = [runs[h][k].fraction for h in (24.0, 72.0, 168.0)]
        assert f == sorted(f), (k, f)


def test_l1_gateway_reached_by_an_explicitly_constructed_burn():
    """Find the cheapest sampled burn to L1 from the DRO over 168 h, refine it, then rebuild that single
    burn explicitly and verify it enters the L1 gateway sphere."""
    cfg = ReachabilityConfig(dv_budget_mps=150.0, horizon_h=168.0, n_dirs=100)
    rs = reachability_for_object("SIM-DRO-01", cfg=cfg)
    st = rs.stats_by_key()["l1_gateway"]
    assert st.n_hit > 0 and st.newly_reachable, "L1 gateway should be reachable from DRO_019 within 168 h and 150 m/s"
    ref = refine_min_dv(rs, "l1_gateway")
    assert ref is not None and ref["dv_mps"] <= st.min_dv_mps
    print(f"Δv to L1 gateway from SIM-DRO-01 (DRO_019): ladder {st.min_dv_mps:.0f} m/s, refined {ref['dv_mps']:.1f} m/s "
          f"(misses at {ref['dv_mps_lower']:.1f}), burn at +{ref['burn_h']:.0f} h, arrival +{ref['arrival_h']:.1f} h, "
          f"dir GCRF {np.round(ref['dir_gcrf'], 3).tolist()}")
    # honesty note: the cheapest hit arrives at the horizon edge, so this figure is horizon-limited;
    # also report the cheapest sample that arrives with >= 24 h margin
    fh = rs.first_hit_h["l1_gateway"]
    margin = np.isfinite(fh) & (fh <= 144.0)
    if margin.any():
        j = int(np.argmin(np.where(margin, rs.dv_mps, np.inf)))
        print(f"  cheapest L1 arrival before +144 h: {rs.dv_mps[j]:.0f} m/s (ladder) arriving +{fh[j]:.1f} h")
    u = np.asarray(ref["dir_gcrf"])
    one = {"dirs": u[None], "dv_mps": np.array([ref["dv_mps"]]), "burn_h": np.array([ref["burn_h"]]),
           "ray": np.array([0]), "n_rays": 1, "magnitudes": np.array([ref["dv_mps"]]), "unit_dirs": u[None]}
    rs1 = compute_reachability(rs.nominal_gcrf[0], rs.t0_s, cfg, "SIM-DRO-01", rs.regions, samples=one)
    assert rs1.stats_by_key()["l1_gateway"].n_hit == 1
    assert np.isclose(rs1.first_hit_h["l1_gateway"][0], ref["arrival_h"])
    # the 0-Δv nominal never enters L1 over this horizon
    assert not rs1.nominal_hits["l1_gateway"].any()
    miss = dict(one, dv_mps=np.array([ref["dv_mps_lower"]]), magnitudes=np.array([ref["dv_mps_lower"]]))
    rs0 = compute_reachability(rs.nominal_gcrf[0], rs.t0_s, cfg, "SIM-DRO-01", rs.regions, samples=miss)
    assert rs0.stats_by_key()["l1_gateway"].n_hit == 0


def test_lunar_impact_is_flagged_and_masked(dro):
    """A state 10 000 km from the Moon falling straight in must be terminated as a lunar impact."""
    t0, _ = dro
    eph = get_ephemeris()
    sm = eph.state("moon", t0)
    r_rel = np.array([10_000.0, 0.0, 0.0])
    v_rel = np.array([-1.0, 0.0, 0.0])             # radial infall (~1.0 km/s + gravity)
    s0 = np.concatenate([sm[:3] + r_rel, sm[3:] + v_rel])
    cfg = ReachabilityConfig(dv_budget_mps=0.0, horizon_h=24.0, n_dirs=1, burn_epochs_h=(0.0,))
    rs = compute_reachability(s0, t0, cfg, "impactor")
    assert rs.term_reason[0] == "lunar_impact"
    assert rs.stats_by_key()["lunar_impact"].n_hit == 1
    assert rs.meta["nominal_terminated"]["reason"] == "lunar_impact"
    # exact impact epoch: independent single-row propagation with a terminal surface event
    t_imp = _event_impact_time(s0, t0, t0 + 24 * H, EphemParams())
    t_hit = rs.first_hit_h["lunar_impact"][0]
    print(f"radial infall from 10 000 km: impact at +{t_hit:.4f} h (event reference +{(t_imp - t0) / H:.4f} h)")
    assert abs(rs.term_t_s[0] - t_imp) < 1.0                                   # s
    assert abs(t_hit - (t_imp - t0) / H) < 1.0 / H
    assert np.isclose(rs.meta["nominal_terminated"]["t_h"], t_hit)
    assert 1.5 < t_hit < 2.0                                                   # ~1.7 h, not the 0 h detection epoch
    # rows are valid up to the crossing (credited with the LLO shell it falls through) and masked after
    k = rs.term_idx[0]
    assert rs.t_grid_s[k] >= rs.term_t_s[0] > rs.t_grid_s[k - 1]
    assert rs.first_hit_h["llo_shell"][0] <= t_hit
    assert not rs.active[0, k:].any()
    assert rs.envelope[-1]["n_active"] == 0


def _event_impact_time(s0, t0, t_end, params, radius=R_MOON):
    eph = get_ephemeris()

    def ev(t, s, _p):
        return np.linalg.norm(s[:3] - eph.position("moon", t)) - radius
    ev.terminal, ev.direction = True, -1
    sol = propagate_ephemeris(s0, t0, tf_s=t_end, params=params, rtol=1e-11, atol=1e-10, max_step=1800.0, events=ev)
    return float(sol.t_events[0][0]) if len(sol.t_events[0]) else np.nan


def test_dro_impactors_terminate_at_the_true_surface_crossing():
    """Impacting samples from the DRO (150 m/s, 168 h) must (a) carry the exact impact epoch from
    the event propagation, (b) stay active (and be credited to the lunar regions they fall through)
    until that epoch, and (c) be masked from the first grid epoch at/after it."""
    cfg = ReachabilityConfig(dv_budget_mps=150.0, horizon_h=168.0, n_dirs=60)
    rs = reachability_for_object("SIM-DRO-01", cfg=cfg)
    imp = [i for i, r in enumerate(rs.term_reason) if r == "lunar_impact"]
    assert len(imp) >= 3, "expected some lunar impactors at 150 m/s over a week"
    params = EphemParams(srp=True, cr_area_mass=cfg.cr_area_mass)
    late_h = []
    for i in imp[:6]:
        k = rs.term_idx[i]
        t_ref = _event_impact_time(rs.states_gcrf[i, k - 1], rs.t_grid_s[k - 1], rs.t_grid_s[-1], params)
        assert np.isfinite(t_ref)
        assert abs(rs.term_t_s[i] - t_ref) < 1.0                                # exact to < 1 s
        assert rs.t_grid_s[k - 1] < rs.term_t_s[i] <= rs.t_grid_s[k]            # masked from the first epoch at/after impact
        late_h.append((rs.t_grid_s[k] - rs.term_t_s[i]) / H)
        hit = rs.regions_hit_by_sample()[i]
        assert "llo_shell" in hit and hit[-1] == "lunar_impact", hit            # falls through the LLO shell first
        assert rs.first_hit_h["llo_shell"][i] < rs.first_hit_h["lunar_impact"][i]
        assert np.isclose(rs.first_hit_h["lunar_impact"][i], rs.term_t_h[i])
        # the last valid grid state is still above the surface
        d = np.linalg.norm(rs.states_gcrf[i, k - 1, :3] - rs.moon_gcrf[k - 1])
        assert d > R_MOON
    print(f"{len(imp)} impactors; masking lag after the true impact {min(late_h):.2f}-{max(late_h):.2f} h (grid 1 h)")
    st = rs.stats_by_key()
    assert st["lunar_impact"].earliest_h == pytest.approx(np.nanmin(rs.term_t_h[imp]))
    assert st["llo_shell"].n_hit >= st["lunar_impact"].n_hit


def test_min_dv_is_zero_for_regions_on_the_nominal_path():
    cfg = ReachabilityConfig(dv_budget_mps=50.0, horizon_h=168.0, n_dirs=12)
    rs = reachability_for_object("SIM-DRO-01", cfg=cfg)
    st = rs.stats_by_key()["l2_gateway"]
    assert st.nominal_hits and st.fraction == 1.0
    assert st.min_dv_mps == 0.0 and not st.newly_reachable
    assert st.earliest_h <= st.earliest_nominal_h
    ref = refine_min_dv(rs, "l2_gateway")
    assert ref["dv_mps"] == 0.0 and ref["sample_index"] is None


def test_jacobi_classifier_uses_instantaneous_frame_rate():
    """C along the 0-Δv DRO drifts ~0.04 with T*-scaled velocities (ω·T* wobble) but only ~1e-3
    once rescaled to the instantaneous rate; the regions code must use the latter."""
    from selene.constants import T_STAR
    from selene.reachability.regions import RegionInputs

    cfg = ReachabilityConfig(dv_budget_mps=0.0, horizon_h=168.0, n_dirs=1, burn_epochs_h=(0.0,))
    rs = reachability_for_object("SIM-DRO-01", cfg=cfg)
    pm = rs.nominal_gcrf[:, :3] - rs.moon_gcrf
    raw = RegionInputs(rs.nominal_rot[:, :3], rs.nominal_gcrf[:, :3], pm, rs.nominal_rot[:, 3:], rs.nominal_gcrf[:, 3:])
    w = np.asarray(rs.frame.omega) * T_STAR
    scaled = RegionInputs(rs.nominal_rot[:, :3], rs.nominal_gcrf[:, :3], pm, rs.nominal_rot[:, 3:], rs.nominal_gcrf[:, 3:],
                          omega_t_star=w)
    c_raw, c_sc = raw.jacobi, scaled.jacobi
    print(f"Jacobi drift over 168 h: T*-scaled {np.ptp(c_raw):.4f}, instantaneous-omega {np.ptp(c_sc):.4f}; "
          f"omega*T* in [{w.min():.3f}, {w.max():.3f}]")
    assert 0.9 < w.min() < w.max() < 1.1 and np.ptp(w) > 0.02
    assert np.ptp(c_sc) < 0.005
    assert np.ptp(c_sc) < 0.2 * np.ptp(c_raw)


def test_envelope_grows_with_time():
    cfg = ReachabilityConfig(dv_budget_mps=50.0, horizon_h=72.0, n_dirs=30)
    rs = reachability_for_object("SIM-DRO-01", cfg=cfg)
    env = rs.envelope
    assert env[0]["t_h"] == 0.0 and env[-1]["t_h"] == 72.0
    assert env[0]["max_radius_km"] < 1e-6                # all samples start at the nominal
    radii = [e["max_radius_km"] for e in env]
    assert radii[-1] > 1000.0 and radii[-1] > radii[1]
    assert env[-1]["hull_volume_km3"] > 0 and len(env[-1]["semi_axes_km"]) == 3
    # a 50 m/s burn applied at t0 displaces ~ dv * t to first order (upper bound with lunar focusing)
    assert radii[-1] < 0.05 * 72 * 3600 * 5


def test_timing_2000_samples_72h():
    cfg = ReachabilityConfig(dv_budget_mps=50.0, horizon_h=72.0, n_dirs=100)   # 100 x 5 mags x 4 epochs = 2000
    t = time.perf_counter()
    rs = reachability_for_object("SIM-DRO-01", cfg=cfg)
    dt = time.perf_counter() - t
    print(f"{rs.n_samples} samples x 72 h: {dt:.2f} s total, propagate {rs.timing['propagate_s']:.2f} s, nfev {rs.timing['nfev']}")
    assert rs.n_samples == 2000
    assert dt < 20.0


def test_config_validation():
    with pytest.raises(ValueError):
        ReachabilityConfig(horizon_h=0)
    with pytest.raises(ValueError):
        ReachabilityConfig(horizon_h=24, burn_epochs_h=(0, 48))
    with pytest.raises(ValueError):
        ReachabilityConfig(dv_budget_mps=-1)
    with pytest.raises(ValueError):
        ReachabilityConfig(dv_budget_mps=50, magnitudes_mps=[-5])
    with pytest.raises(ValueError):
        ReachabilityConfig(dv_budget_mps=50, magnitudes_mps=[])
    with pytest.raises(ValueError):
        ReachabilityConfig(dv_budget_mps=10, magnitudes_mps=[500])
    with pytest.raises(ValueError):
        ReachabilityConfig(dv_budget_mps=10, magnitudes_mps=["x"])
    assert ReachabilityConfig(dv_budget_mps=10, magnitudes_mps=[10, 5, 5]).magnitudes().tolist() == [5.0, 10.0]
