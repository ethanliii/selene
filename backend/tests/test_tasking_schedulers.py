"""Scheduler tests: greedy vs baselines, MILP vs greedy surrogate, slew feasibility, lunar-glare honesty.

Numbers are printed (``-s``) so the milestone report can quote them; assertions are the robust
ordering claims only.
"""
from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from selene.dynamics.frames import DEMO_EPOCH_TDB_S
from selene.sensors.constraints import angular_separation
from selene.sensors.observers import SpaceObserver, get_observer
from selene.tasking.greedy import SlotContext, run_greedy, run_schedule
from selene.tasking.metrics import comparison_row, custody_metrics, null_custody_pct, run_null, run_random, run_round_robin
from selene.tasking.optimize import HAVE_MILP, build_window, greedy_surrogate_plan, run_milp, solve_window_local_search, solve_window_milp, surrogate_objective
from selene.tasking.scenario import DEFAULT_OBJECT_IDS, TASKING_PRESETS, make_scenario


def _fmt(row):
    return {k: (round(v, 3) if isinstance(v, float) else v) for k, v in row.items()
            if k in ("method", "custody_pct", "custody_pct_min_object", "mean_tslo_h", "n_observations",
                     "summed_trace_final_km2", "summed_trace_mean_km2", "objective_milp_total",
                     "objective_greedy_surrogate_total", "n_filled_idle", "n_greedy_kept", "runtime_s")}


@pytest.fixture(scope="module")
def default_scn():
    """The API default: 8 objects, 9-sensor mixed network, 48 h, 20-min slots."""
    return make_scenario()


@pytest.fixture(scope="module")
def scarce_scn():
    """Sensor-scarce custody problem: 8 objects, DRO + GEO observer only, stale 200 km prior."""
    return make_scenario(sensor_ids=["dro_obs", "geo_west"], horizon_h=36, slot_min=30, initial_sigma_km=200.0, q_psd=1e-12)


# ---------------------------------------------------------------------------
def test_default_scenario_shape(default_scn):
    scn = default_scn
    assert scn.n_objects == 8 and scn.n_sensors == 9 and scn.n_slots == 144
    assert [tr.object_id for tr in scn.tracks] == list(DEFAULT_OBJECT_IDS)
    assert scn.timing["total_s"] < 20.0
    frac = scn.table.fraction_visible()
    # ground sites are glare-blind at the demo epoch; the L1/L2/DRO observers see most objects
    ground = [i for i, sid in enumerate(scn.table.sensor_ids) if sid in TASKING_PRESETS["ground_only"][0]]
    assert np.all(frac[ground] == 0.0)
    assert frac[scn.table.sensor_ids.index("l1_halo_obs")].mean() > 0.9


def test_greedy_beats_random_and_round_robin_on_default_scenario(default_scn):
    """On the API default (10 km prior, q = 1e-18, 4 space observers) custody at 100 km is 100 % for
    *every* policy -- including the null policy that never observes anything (propagated sigma stays
    below ~56 km over 48 h).  The custody % comparison is therefore vacuous here and is reported as
    such; the discriminating metrics are the time-mean / final summed position trace and TSLO.
    The custody % ordering is asserted in the sensor-scarce test below."""
    scn = default_scn
    rows = {}
    rows["null"] = comparison_row(run_null(scn))
    for kind in ("trace", "logdet"):
        s = dataclasses.replace(scn, gain_kind=kind)
        rows[f"greedy_{kind}"] = comparison_row(run_greedy(s))
    rows["round_robin"] = comparison_row(run_round_robin(scn))
    rand = [comparison_row(run_random(scn, seed)) for seed in range(5)]
    rows["random_mean"] = {k: (float(np.mean([r[k] for r in rand])) if isinstance(rand[0][k], (int, float)) else "random") for k in rand[0]}
    rows["random_best"] = {k: (float(np.min([r[k] for r in rand])) if isinstance(rand[0][k], (int, float)) else "random") for k in rand[0]}
    print("\nDEFAULT SCENARIO (8 obj, mixed_9, 48 h, 20 min):")
    for k, r in rows.items():
        print("  ", k, _fmt(r))
    null = rows["null"]
    assert null["custody_pct"] == 100.0 and null["n_observations"] == 0      # documents the saturation honestly
    g = rows["greedy_trace"]
    for base in ("round_robin", "random_mean", "random_best"):
        b = rows[base]
        assert g["summed_trace_final_km2"] < b["summed_trace_final_km2"], base
        assert g["summed_trace_mean_km2"] < b["summed_trace_mean_km2"], base
        assert g["mean_tslo_h"] <= b["mean_tslo_h"] + 1e-9, base
    assert g["summed_trace_mean_km2"] < 0.6 * rows["round_robin"]["summed_trace_mean_km2"]
    assert g["summed_trace_mean_km2"] < 0.01 * null["summed_trace_mean_km2"]
    # all methods make the same number of observations: the comparison is about *which* objects
    assert g["n_observations"] == rows["round_robin"]["n_observations"]
    # the log-det variant also beats the baselines on the summed trace here
    assert rows["greedy_logdet"]["summed_trace_mean_km2"] < rows["round_robin"]["summed_trace_mean_km2"]


def test_greedy_custody_beats_baselines_when_sensors_are_scarce(scarce_scn):
    scn = scarce_scn
    null = null_custody_pct(scn)
    g = comparison_row(run_greedy(scn))
    rr = comparison_row(run_round_robin(scn))
    rand = [comparison_row(run_random(scn, seed)) for seed in range(8)]
    rc = np.array([r["custody_pct"] for r in rand])
    print("\nSCARCE SCENARIO (8 obj, dro_obs + geo_west, 36 h, 30 min, sigma0 = 200 km): null-policy custody", null)
    print("   greedy      ", _fmt(g))
    print("   round_robin ", _fmt(rr))
    print(f"   random x{len(rand)}: custody mean {rc.mean():.2f} %, min {rc.min():.2f}, max {rc.max():.2f}; "
          f"mean trace mean {np.mean([r['summed_trace_mean_km2'] for r in rand]):.0f} km2")
    assert null == 0.0                                 # the prior is outside the threshold: custody only via observation
    assert 0.0 < g["custody_pct"] < 100.0              # honest: not everything can be held
    assert g["custody_pct"] > rr["custody_pct"]
    # greedy beats random *in expectation*; a lucky seed can beat it (1 of 20 did in review), so the
    # claim is made against the mean over seeds, not the max
    assert g["custody_pct"] > rc.mean() + 2.0
    assert g["summed_trace_mean_km2"] < rr["summed_trace_mean_km2"]
    assert g["summed_trace_mean_km2"] < min(r["summed_trace_mean_km2"] for r in rand)
    assert g["summed_trace_final_km2"] < rr["summed_trace_final_km2"]
    assert g["summed_trace_final_km2"] < np.mean([r["summed_trace_final_km2"] for r in rand])


def test_metrics_consistency(default_scn):
    res = run_greedy(default_scn)
    m = custody_metrics(res, 100.0)
    K1 = default_scn.n_slots + 1
    assert len(m["per_object"]) == 8 and len(m["sensors"]) == 9
    for o in m["per_object"]:
        assert len(o["sigma_series_km"]) == K1 and 0.0 <= o["custody_pct"] <= 100.0
        assert o["custody_pct"] == pytest.approx(100.0 * np.mean(np.array(o["sigma_series_km"]) < 100.0))
    assert sum(o["n_obs"] for o in m["per_object"]) == len(res.assignments) == sum(s["n_obs"] for s in m["sensors"])
    assert m["overall"]["custody_pct"] == pytest.approx(np.mean([o["custody_pct"] for o in m["per_object"]]))
    assert len(m["overall"]["summed_trace_series_km2"]) == K1
    # stricter threshold -> custody can only drop
    m2 = custody_metrics(res, 1.0)
    assert m2["overall"]["custody_pct"] <= m["overall"]["custody_pct"]
    # every assignment was visible and recorded with a non-negative gain
    assert all(a.reasons == [] and a.gain >= 0.0 and a.sigma_after_km <= a.sigma_before_km + 1e-9 for a in res.assignments)


def test_engine_rejects_infeasible_policy(default_scn):
    def bad_policy(k, ctx):
        s = ctx.scn.table.sensor_ids.index("haleakala")   # glare-blind -> infeasible
        return {s: 0}

    with pytest.raises(RuntimeError):
        run_schedule(default_scn, bad_policy, method="bad")


# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def short_scn():
    return make_scenario(horizon_h=12, slot_min=20, sensor_ids=["geo_west", "l2_halo_obs", "dro_obs"], initial_sigma_km=50.0, q_psd=1e-13)


@pytest.mark.skipif(not HAVE_MILP, reason="scipy.optimize.milp unavailable")
def test_milp_objective_at_least_greedy_surrogate_on_every_window(short_scn, default_scn, scarce_scn):
    """objective_milp is the *raw* solver value (the policy's greedy-kept guard does not rewrite it),
    so this is a genuine check; n_greedy_kept == 0 confirms the guard never fired and
    n_filled_idle == 0 confirms the MILP itself fills every usable sensor-slot (the previous
    count-indexed penalty left ~30 % idle and relied on a greedy fill pass -- review finding)."""
    res = run_milp(short_scn, horizon=4, fill_idle=False)
    g = run_greedy(short_scn)
    assert res.meta["fallbacks"] == 0 and res.meta["n_windows"] == short_scn.n_slots
    assert res.meta["n_greedy_kept"] == 0 and res.meta["n_filled_idle"] == 0 and not res.meta["budget_exhausted"]
    n_strict = 0
    for w in res.meta["objective_windows"]:
        assert w["status"] == "optimal", w
        assert w["objective_milp"] >= w["objective_greedy_surrogate"] - 1e-9, w
        n_strict += w["objective_milp"] > w["objective_greedy_surrogate"] + 1e-9
    rm, rg = comparison_row(res), comparison_row(g)
    print(f"\nMILP vs GREEDY (12 h, 3 space sensors): MILP strictly better surrogate on {n_strict}/{len(res.meta['objective_windows'])} windows")
    print("   milp   ", _fmt(rm))
    print("   greedy ", _fmt(rg))
    assert rm["objective_milp_total"] >= rm["objective_greedy_surrogate_total"]
    assert rm["n_observations"] == rg["n_observations"]          # no idle sensor-slots left by the MILP alone
    # realised custody must not be worse by more than noise (one slot of one object)
    assert rm["custody_pct"] >= rg["custody_pct"] - 100.0 / (short_scn.n_objects * (short_scn.n_slots + 1))
    # sensor-scarce case: MILP alone vs greedy vs round-robin on realised custody (reported, not asserted
    # beyond 'not worse than round-robin by more than one object-slot')
    ms = comparison_row(run_milp(scarce_scn, horizon=4, fill_idle=False))
    gs, rs = comparison_row(run_greedy(scarce_scn)), comparison_row(run_round_robin(scarce_scn))
    print(f"   scarce realised custody: milp {ms['custody_pct']:.2f} %, greedy {gs['custody_pct']:.2f} %, round_robin {rs['custody_pct']:.2f} %; "
          f"mean trace milp {ms['summed_trace_mean_km2']:.0f} / greedy {gs['summed_trace_mean_km2']:.0f} / rr {rs['summed_trace_mean_km2']:.0f} km2")
    assert ms["n_filled_idle"] == 0 and ms["n_greedy_kept"] == 0
    assert ms["custody_pct"] >= rs["custody_pct"] - 100.0 / (scarce_scn.n_objects * (scarce_scn.n_slots + 1))
    # the same window, solved directly: MILP >= greedy surrogate >= 0 and feasible per sensor-slot
    ctx = SlotContext(default_scn)
    ctx.k = 5
    wp = build_window(ctx, 4)
    chosen, obj, status = solve_window_milp(wp)
    gset = greedy_surrogate_plan(wp)
    assert status == "optimal" and obj >= surrogate_objective(wp, gset) - 1e-9 >= -1e-9
    ss = [(wp.triples[i][0], wp.triples[i][1]) for i in chosen]
    assert len(ss) == len(set(ss))
    ls_set, ls_obj = solve_window_local_search(wp, start=gset)
    assert ls_obj >= surrogate_objective(wp, gset) - 1e-9 and ls_obj <= obj + 1e-9


def test_local_search_fallback_runs(short_scn):
    res = run_milp(short_scn, horizon=3, use_milp=False)
    assert res.method == "local_search" and res.meta["fallbacks"] == res.meta["n_windows"]
    for w in res.meta["objective_windows"]:
        assert w["objective_milp"] >= w["objective_greedy_surrogate"] - 1e-9


@pytest.mark.skipif(not HAVE_MILP, reason="scipy.optimize.milp unavailable")
def test_milp_time_budget_falls_back_to_greedy(default_scn):
    """A wall-clock budget bounds the MILP path: once exhausted the remaining slots are scheduled
    by the myopic greedy rule and the result says so."""
    import time

    t = time.perf_counter()
    res = run_milp(default_scn, horizon=8, time_budget_s=1.0)
    dt = time.perf_counter() - t
    print(f"\nMILP BUDGET: horizon 8, budget 1 s -> {dt:.2f} s wall, {res.meta['n_windows_solved']} windows solved, "
          f"{res.meta['n_budget_exhausted']} greedy slots")
    assert res.meta["budget_exhausted"] and res.meta["n_budget_exhausted"] > 0
    assert res.meta["n_windows_solved"] + res.meta["n_budget_exhausted"] == default_scn.n_slots
    assert dt < 6.0
    assert len(res.assignments) == len(run_greedy(default_scn).assignments)   # still a full schedule
    # per-window objectives are only summed over solved windows (nan-free)
    assert np.isfinite(res.meta["objective_milp_total"]) and np.isfinite(res.meta["objective_greedy_surrogate_total"])


# ---------------------------------------------------------------------------
def test_slew_feasibility_respected_and_binding():
    slow = SpaceObserver("slow_dro", "slow-slewing DRO observer (test)", "dro", limiting_mag=18.5, fov_deg=2.0,
                         slew_rate_deg_s=0.01)      # 0.01 deg/s x 300 s = 3 deg per 5-min slot
    fast = SpaceObserver("fast_dro", "fast-slewing DRO observer (test)", "dro", limiting_mag=18.5, fov_deg=2.0,
                         slew_rate_deg_s=1.0)
    kw = dict(sensor_ids=[], horizon_h=3, slot_min=5, initial_sigma_km=50.0, q_psd=1e-12)
    res_slow = run_greedy(make_scenario(extra_sensors=[slow], **kw))
    res_fast = run_greedy(make_scenario(extra_sensors=[fast], **kw))
    for res, sensor in ((res_slow, slow), (res_fast, fast)):
        tab = res.scenario.table
        rate = np.deg2rad(sensor.slew_rate_deg_s)
        prev, prev_t = None, None
        for a in res.assignments:
            j, k = tab.object_ids.index(a.object_id), a.slot
            los = tab.los[0, j, k]
            if prev is not None:
                # consecutive pointings within slew_rate x (time since the previous pointing);
                # back-to-back slots -> slew_rate x slot
                assert angular_separation(prev, los) <= rate * (a.t_s - prev_t) + 1e-9
            prev, prev_t = los, a.t_s
        assert all(a.t_s - b.t_s == res.scenario.slot_s for a, b in zip(res.assignments[1:], res.assignments[:-1]))
    n_slow = len({a.object_id for a in res_slow.assignments})
    n_fast = len({a.object_id for a in res_fast.assignments})
    print(f"\nSLEW: slow observer touched {n_slow} objects, fast observer {n_fast}; "
          f"{len(res_slow.assignments)} / {len(res_fast.assignments)} assignments")
    assert len(res_slow.assignments) > 0 and n_slow < n_fast      # the constraint binds


def test_idle_sensor_accumulates_slew_budget():
    """An idle sensor is credited with slew_rate x (time since its last pointing): after enough
    idle slots a slow sensor can re-point to a far-away object (previously it could never move
    more than one slot's worth of slew, however long it had been idle -- review finding)."""
    slow = SpaceObserver("slow_dro", "slow-slewing DRO observer (test)", "dro", limiting_mag=18.5, fov_deg=2.0,
                         slew_rate_deg_s=0.01)
    scn = make_scenario(sensor_ids=[], extra_sensors=[slow], horizon_h=3, slot_min=5, initial_sigma_km=50.0)
    ctx = SlotContext(scn)
    tab = scn.table
    # point at object 0 at slot 0, then look for the farthest visible object
    ctx.boresight[0] = tab.los[0, 0, 0].copy()
    ctx.last_point_k[0] = 0
    seps = np.array([angular_separation(tab.los[0, 0, 0], tab.los[0, j, 0]) for j in range(scn.n_objects)])
    j_far = int(np.argmax(seps))
    assert np.rad2deg(seps[j_far]) > 10.0 and tab.visible[0, j_far].all()
    limit_per_slot = np.deg2rad(0.01) * scn.slot_s                 # 3 deg
    n_expected = int(np.ceil(seps[j_far] / limit_per_slot))        # from the slot-0 geometry (the LOS drifts a little)
    assert n_expected > 1
    ctx.k = 1
    assert not ctx.feasible(0, j_far)                              # one slot: too far
    feas = []
    for k in range(1, scn.n_slots + 1):
        ctx.k = k
        feas.append(ctx.feasible(0, j_far))
    first = feas.index(True) + 1
    assert abs(first - n_expected) <= 1 and all(feas[first - 1:])  # reachable once enough idle slots have passed, and stays so
    assert ctx.slew_time_s(0, first) == pytest.approx(first * scn.slot_s)


# ---------------------------------------------------------------------------
def test_lost_custody_is_sticky_under_fov_acquisition_and_recoverable_by_search():
    """Review finding: the 'irf' model recovered a 1e5 km prior in 2-3 pointed looks at a detection
    probability of ~1e-4.  With the Bernoulli-mixture expected covariance ('fov', the default) it
    stays lost; a mosaic search (search_tiles) or the optimistic models recover it."""
    kw = dict(object_ids=["SIM-DRO-01"], sensor_ids=["dro_obs"], horizon_h=6, slot_min=20)
    out = {}
    for acq, tiles, s0 in (("fov", 1, 1e5), ("irf", 1, 1e5), ("none", 1, 1e5), ("fov", 25, 1e4)):
        res = run_greedy(make_scenario(initial_sigma_km=s0, acquisition=acq, search_tiles=tiles, **kw))
        m = custody_metrics(res)["overall"]
        out[(acq, tiles, s0)] = (m["custody_pct"], res.sigma_km[0], res.assignments[0].p_acq if res.assignments else None)
        print(f"\nSTICKY sigma0={s0:g} acq={acq} tiles={tiles}: p_acq(first)={out[(acq, tiles, s0)][2]}, custody {m['custody_pct']:.1f} %, "
              f"sigma[0:4] = {np.round(res.sigma_km[0][:4], 0)}")
    c_fov, sig_fov, p_fov = out[("fov", 1, 1e5)]
    assert p_fov < 1e-3                                    # the 2-deg field covers almost none of a 1e5 km cloud
    assert c_fov == 0.0 and sig_fov[-1] > 0.99 * sig_fov[0]
    assert out[("irf", 1, 1e5)][0] > 50.0                  # the optimistic model 'recovers' (documented as such)
    assert out[("none", 1, 1e5)][0] > 50.0
    c_search, sig_search, p_search = out[("fov", 25, 1e4)]
    assert p_search > 0.1 and sig_search[-1] < 100.0 < sig_search[0] and c_search > 0.0


# ---------------------------------------------------------------------------
def test_ground_only_custody_is_honestly_lower_than_with_dro_observer_at_bright_moon():
    """Window chosen with the sensors' lunar-glare logic: the demo epoch is ~2 days before full
    Moon (illuminated fraction > 0.9), so the glare exclusion is ~14-15 deg."""
    from selene.dynamics.ephemeris import get_ephemeris
    from selene.sensors.constraints import illuminated_fraction

    eph = get_ephemeris()
    t0 = DEMO_EPOCH_TDB_S + 1.0 * 86400.0
    k = float(illuminated_fraction(eph.position("sun", t0), eph.position("moon", t0)))
    assert k > 0.9
    kw = dict(t0_s=t0, horizon_h=24, slot_min=30, initial_sigma_km=300.0)   # stale prior: custody only via observation
    ground = run_greedy(make_scenario(preset="ground_only", **kw))
    with_dro = run_greedy(make_scenario(preset="ground_plus_dro", **kw))
    mg, md = custody_metrics(ground)["overall"], custody_metrics(with_dro)["overall"]
    print(f"\nBRIGHT MOON (illum {k:.2f}): ground-only custody {mg['custody_pct']:.1f} % ({mg['n_observations']} obs), "
          f"ground + DRO observer {md['custody_pct']:.1f} % ({md['n_observations']} obs)")
    assert mg["custody_pct"] < 100.0
    assert mg["custody_pct"] < md["custody_pct"]
    assert mg["n_observations"] == 0 and md["n_observations"] > 0
    blocked = sum(s["blocked_reason_counts"]["moon_exclusion"] for s in custody_metrics(ground)["sensors"])
    assert blocked > 0
