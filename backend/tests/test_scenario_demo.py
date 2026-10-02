"""M9 demo scenario builder: engines run end to end, story order, custody decay/recovery, dv characterisation, brief."""
from __future__ import annotations

import json
import time

import numpy as np
import pytest

from selene.scenario.brief import FOOTER_PREFIX, generate_brief
from selene.scenario.demo import EXTRA_EVENT_KINDS, NO_SITE_AVAILABLE, PROTAGONIST, build_scenario

SPEC_KINDS = {"custody_nominal", "maneuver_detected", "custody_degraded", "custody_lost", "entered_region", "reachability_alert",
              "tasking_update", "observation", "custody_regained", "maneuver_characterised", "brief_ready"}
# 'entered_region' is emitted only when the SIMULATED truth really enters a region (gateways are neck transits); the demo
# truth grazes L1 without transiting, which the bundle reports as a 'closest_approach' event instead of inventing an entry
REQUIRED_KINDS = tuple(sorted(SPEC_KINDS - {"entered_region"}))


@pytest.fixture(scope="module")
def bundle():
    tic = time.perf_counter()
    b = build_scenario(seed=0, fast=True)
    b["_build_s"] = time.perf_counter() - tic
    return b


def _first(events, kind, **flt):
    for e in events:
        if e["kind"] == kind and all(e.get(k) == v for k, v in flt.items()):
            return e
    raise AssertionError(f"no event of kind {kind!r}")


def test_fast_build_time_and_shape(bundle):
    print(f"fast build {bundle['_build_s']:.1f} s; frames={len(bundle['frames'])} events={len(bundle['events'])}")
    assert bundle["_build_s"] < 90.0
    assert len(bundle["frames"]) >= 20
    meta = bundle["meta"]
    for k in ("title", "t0_utc", "t1_utc", "duration_s", "playback_speed", "epoch_choice_rationale", "disclaimer"):
        assert k in meta, k
    assert "SIMULATED" in meta["disclaimer"] and "notional actor" in meta["disclaimer"]
    # the window is what the frames span, and the playback compresses it into the scripted 2 minutes
    assert meta["duration_s"] == pytest.approx((len(bundle["frames"]) - 1) * meta["frame_dt_s"])
    assert meta["duration_s"] == pytest.approx(bundle["frames"][-1]["t_rel_s"])
    assert meta["playback_s"] == 120.0 and meta["playback_speed"] == pytest.approx(meta["duration_s"] / 120.0)
    t_prev = -1.0
    for fr in bundle["frames"]:
        assert fr["t_rel_s"] > t_prev and fr["t"] == fr["t_rel_s"]
        t_prev = fr["t_rel_s"]
        assert fr["t_utc"].endswith("Z")
        ids = {o["id"] for o in fr["objects"]}
        assert PROTAGONIST in ids and "SIM-NRHO-RELAY-01" in ids
        for o in fr["objects"]:
            assert o["custody"] in ("CUSTODY", "DEGRADED", "LOST") and o["custody_ui"] in ("held", "degraded", "lost")
            assert len(o["pos_rot"]) == 3 and len(o["pos_gcrf_km"]) == 3 and o["sigma_km"] >= 0
            assert 0.3 < np.linalg.norm(o["pos_rot"]) < 3.0          # nondimensional, inside the cislunar volume
        assert fr["clouds"] and fr["clouds"][0]["object_id"] == PROTAGONIST
        pts = fr["clouds"][0]["points_rot"]
        assert len(pts) % 3 == 0 and 0 < len(pts) // 3 <= 3000
        assert len(fr["clouds"][0]["points"]) == len(pts)            # frontend alias (flat xyz)
        kinds = {s["kind"] for s in fr["sensors"]}
        assert kinds == {"ground", "space"}
        for s in fr["sensors"]:
            if s["active"]:
                assert s["tasked_object"] and len(s["pointing_rot"]) == 3
                assert s["pointing_basis"] in ("reachable_set_centroid", "filter_estimate", "reference_trajectory")
                assert abs(np.linalg.norm(s["pointing_rot"]) - 1.0) < 1e-3
        if "reachable" in fr:
            assert len(fr["reachable"]["points_rot"]) // 3 <= 1500 and fr["reachable"]["regions"]
        assert fr["ground_blind_reason"] is None or isinstance(fr["ground_sites_available"], int)
    assert any("reachable" in fr for fr in bundle["frames"])
    assert any(s["active"] and s["kind"] == "ground" for fr in bundle["frames"] for s in fr["sensors"])


def test_event_kinds_and_story_order(bundle):
    ev = bundle["events"]
    kinds = {e["kind"] for e in ev}
    missing = [k for k in REQUIRED_KINDS if k not in kinds]
    assert not missing, missing
    # every kind outside the spec list is documented in meta.extra_event_kinds
    extra = kinds - SPEC_KINDS
    assert extra <= set(EXTRA_EVENT_KINDS) and extra <= set(bundle["meta"]["extra_event_kinds"]), extra
    assert all(e["severity"] in ("info", "warn", "alert") for e in ev)
    assert [e["t_rel_s"] for e in ev] == sorted(e["t_rel_s"] for e in ev)
    t_burn = _first(ev, "maneuver")["t_rel_s"]
    t_det = _first(ev, "maneuver_detected")["t_rel_s"]
    t_lost = _first(ev, "custody_lost")["t_rel_s"]
    t_task = _first(ev, "tasking_update")["t_rel_s"]
    t_reg = _first(ev, "custody_regained")["t_rel_s"]
    t_char = _first(ev, "maneuver_characterised")["t_rel_s"]
    t_brief = _first(ev, "brief_ready")["t_rel_s"]
    assert t_burn < t_det <= t_lost < t_task <= t_reg < t_char < t_brief
    assert t_det <= _first(ev, "reachability_alert")["t_rel_s"] <= t_task
    m = bundle["metrics"]
    assert m["detection_latency_h"] == pytest.approx((t_det - t_burn) / 3600.0, abs=1e-6)
    # the DEGRADED event sits on the status change itself (metrics agree), not on the first ground-blind frame
    assert _first(ev, "custody_degraded")["t_rel_s"] == pytest.approx(m["t_degraded_rel_s"])
    # same-epoch ordering in the feed is narrative: the tasking update precedes any tracklet it produced, the
    # re-acquisition tracklet precedes the custody-regained verdict
    at_task = [(i, e) for i, e in enumerate(ev) if e["t_rel_s"] == t_task]
    i_task = next(i for i, e in at_task if e["kind"] == "tasking_update")
    i_reacq = next(i for i, e in at_task if e["kind"] == "observation" and e["data"].get("reacquired"))
    assert all(i > i_task for i, e in at_task if e["kind"] == "observation"), "no tracklet before the tasking update at the same epoch"
    if t_reg == t_task:
        i_reg = next(i for i, e in at_task if e["kind"] == "custody_regained")
        assert i_task < i_reacq < i_reg
    # numbers in the texts are not flattened to zero by formatting
    assert f"{m['filter']['prior_vel_mps']:.1f} m/s" in _first(ev, "custody_nominal")["text"]
    txt_char = _first(ev, "maneuver_characterised")["text"]
    assert "+/- 0.0 m/s" not in txt_char and f"+/- {m['dv_estimate']['dv_sigma_mps']:.2f} m/s" in txt_char


def test_ground_loss_attribution_is_per_site_honest(bundle):
    """The blind attribution only asks the sites that could have observed (night, object above the elevation limit);
    hours when no site had the object up are counted separately and never as glare."""
    m = bundle["metrics"]
    loss = m["loss_reason"]
    assert loss["dominant"] == "moon_exclusion"                      # this epoch: the bright Moon within ~10 deg
    assert loss["at_loss"] == "moon_exclusion" and loss["available_sites_at_loss"], loss
    n_blind = loss["n_blind_frames_after_detection"]
    assert n_blind == loss["n_frames_no_site_available_after_detection"] + loss["n_frames_available_site_blocked_after_detection"]
    assert loss["n_frames_available_site_blocked_after_detection"] > loss["n_frames_no_site_available_after_detection"]
    assert sum(loss["counts_after_detection"].values()) == n_blind
    assert NO_SITE_AVAILABLE in loss["counts_between_detection_and_loss"], "the daylight / low-elevation hours right after the burn are not glare"
    # frames: whenever no site is available the frame says so, and glare frames name at least one available site
    for fr in bundle["frames"]:
        if fr["ground_blind_reason"] == NO_SITE_AVAILABLE:
            assert fr["ground_sites_available"] == 0
        elif fr["ground_blind_reason"] is not None:
            assert fr["ground_sites_available"] >= 1
    ev = bundle["events"]
    lost_txt = _first(ev, "custody_lost")["text"]
    assert "lunar glare" in lost_txt and all(site in lost_txt for site in loss["available_sites_at_loss"])
    assert "elevation" in _first(ev, "custody_degraded")["text"] or "glare" in _first(ev, "custody_degraded")["text"]


def test_sigma_grows_after_loss_and_drops_after_regain(bundle):
    m = bundle["metrics"]
    sig = np.array([s["sigma_km"] for s in m["sigma_timeline"]])
    t = np.array([s["t_rel_s"] for s in m["sigma_timeline"]])
    t_det, t_lost, t_reg = m["t_detect_rel_s"], m["t_lost_rel_s"], m["t_regained_rel_s"]
    assert t_lost is not None and t_reg is not None
    pre = sig[t < m["t_burn_rel_s"]]
    between = sig[(t >= t_det) & (t <= t_lost)]
    after = sig[t > t_reg]
    assert pre.max() < m["filter"]["custody_km"]
    assert between.max() > m["filter"]["lost_km"] > 10 * pre.max()
    assert np.all(np.diff(between) > 0), "uncertainty must grow monotonically while unobserved"
    assert sig[np.argmin(np.abs(t - t_reg))] < m["filter"]["custody_km"]
    assert after.max() < m["filter"]["custody_km"]
    statuses = [c["status"] for c in m["custody_timeline"]]
    assert "LOST" in statuses and statuses[-1] == "CUSTODY" and statuses[0] == "CUSTODY"
    assert m["regained_after_h"] is not None and m["regained_after_h"] > 0
    assert m["custody_pct"] < 100.0   # honest: custody was not perfect


def test_tasking_pointing_and_search_budget_are_consistent(bundle):
    """The scheduler's acquisition model and the truth verification share one mosaic budget, and the rendered
    boresight in the re-acquisition slot is the tasked centroid (offset from the truth by the reported amount)."""
    m = bundle["metrics"]
    tk = m["tasking"]
    budget = tk["search_tiles_used_by_scheduler"]
    assert budget == bundle["meta"]["search_fields_per_slot"] >= 1
    assert tk["sensor_ids"], "re-acquisition must have happened"
    k = int(round(tk["t_task_rel_s"] / bundle["meta"]["frame_dt_s"]))
    fr = bundle["frames"][k]
    truth = {o["id"]: np.array(o["pos_rot"]) for o in fr["objects"]}[PROTAGONIST]
    sens = {s["id"]: s for s in fr["sensors"]}
    for r in tk["sensors"]:
        assert r["acquired"] and r["n_fields_used"] <= budget and r["n_fields_used"] <= r["n_fields_p90"]
        fov = sens[r["sensor_id"]]["fov_deg"]
        assert r["mosaic_radius_deg"] <= max(0.5 * fov, 0.5 * fov * np.sqrt(budget)) + 1e-6
        assert r["offset_deg"] <= r["mosaic_radius_deg"]
        s = sens[r["sensor_id"]]
        assert s["active"] and s["tasked_object"] == PROTAGONIST and s["pointing_basis"] == "reachable_set_centroid"
        los = truth - np.array(s["pos_rot"])
        los /= np.linalg.norm(los)
        ang = np.degrees(np.arccos(np.clip(float(los @ np.array(s["pointing_rot"])), -1, 1)))
        assert ang == pytest.approx(r["offset_deg"], abs=0.05), (r["sensor_id"], ang, r["offset_deg"])


def test_dv_estimate_within_30pct_and_consistent(bundle):
    m = bundle["metrics"]
    assert m["dv_true_mps"] == pytest.approx(30.0)
    assert m["dv_est_mps"] is not None, m["dv_estimate_error"]
    print(f"dv est {m['dv_est_mps']} +- {m['dv_est_sigma_mps']} m/s (truth {m['dv_true_mps']}), err {m['dv_est_err_pct']} %, "
          f"dir err {m['dir_err_deg']} deg (+-{m['dir_sigma_deg']}), epoch err {m['t_burn_est_err_s']} s (+-{m['t_burn_est_sigma_s']})")
    assert abs(m["dv_est_err_pct"]) < 30.0
    assert m["dir_err_deg"] < 20.0
    # the quoted uncertainty must not be absurdly optimistic: error within 5 sigma
    assert abs(m["dv_est_mps"] - m["dv_true_mps"]) < 5.0 * max(m["dv_est_sigma_mps"], 0.01)
    assert m["detection"]["status"] == "maneuver_declared"
    # covariance realism: the filter must not be over-confident (mean NEES above the 95 % band); a conservative
    # (below-band) covariance is reported honestly in the brief rather than hidden
    assert m["detection"]["nees_mean"] <= m["detection"]["nees_bounds"][1]
    assert m["detection"]["nees_baseline_mean"] <= m["detection"]["nees_baseline_bound"]
    est = m["dv_estimate"]
    assert est["converged"] and est["reduced_chi2"] < 5.0
    assert len(est["dv_rtn_mps"]) == 3 and len(est["dv_vnb_mps"]) == 3


def test_reachability_block_is_honest(bundle):
    r = bundle["metrics"]["reachability"]
    keys = {x["key"] for x in r["regions"]}
    assert {"l1_gateway", "l2_gateway", "nrho_corridor", "south_pole_approach", "geo_belt_return", "lunar_impact"} <= keys
    l1 = next(x for x in r["regions"] if x["key"] == "l1_gateway")
    l2 = next(x for x in r["regions"] if x["key"] == "l2_gateway")
    # gateways are neck TRANSITS (regions.py): the quiet DRO grazes both L1 and L2 every revolution without changing
    # realm, so neither is on the nominal path; what the burn opens is reported with a positive minimum delta-v
    assert l1["nominal_hits"] is False and l2["nominal_hits"] is False
    assert l1["n_hit"] > 0 and l1["newly_reachable"] and l1["min_dv_mps"] > 0 and l1["earliest_h"] is not None
    tg = bundle["metrics"]["truth_geometry"]
    # the SIMULATED truth grazes L1 (closest approach refined on the dense solution, well below the hourly-grid value)
    # but stays in the lunar realm: no gateway entry is claimed for it, and the bundle says so explicitly
    assert tg["l1_transit"] is False
    assert 100.0 < tg["min_dist_to_L1_km"] < 1000.0 and tg["min_dist_to_L1_km"] <= tg["min_dist_to_L1_grid_km"]
    assert tg["min_dist_to_L1_unperturbed_km"] > tg["min_dist_to_L1_km"]
    assert not any(en["key"] in ("l1_gateway", "l2_gateway") for en in tg["region_entries"])
    ca = _first(bundle["events"], "closest_approach")
    assert ca["data"]["transit"] is False and "does NOT transit" in ca["text"] and f"{tg['min_dist_to_L1_km']:,.0f} km" in ca["text"]
    assert r["dv_budget_mps"] == 100.0 and r["horizon_h"] == 168.0
    alert = _first(bundle["events"], "reachability_alert")
    txt = alert["text"]
    # the headline lists only regions the burn OPENS; routine geometry of the unperturbed orbit is never flagged as new
    assert "could newly enter" in txt and "L1 gateway" in txt and f">= {l1['min_dv_mps']:.0f} m/s" in txt
    assert "l1_gateway" in alert["data"]["newly_reachable"] and alert["data"]["on_unperturbed_path"] == []
    assert "awareness" in txt.lower() and "intent" not in txt.lower().replace("no intent", "")
    # the relay-corridor sentence never flips meaning with the sampling: both 0 and a few rays read as marginal
    assert "sampling resolution" in txt and "no closer approach is implied" in txt
    # one definition of sigma at detection everywhere: metrics == frame == degraded event text
    m = bundle["metrics"]
    k_det = int(round(m["t_detect_rel_s"] / bundle["meta"]["frame_dt_s"]))
    prot = next(o for o in bundle["frames"][k_det]["objects"] if o["id"] == PROTAGONIST)
    assert m["sigma_at_detection_km"] == pytest.approx(prot["sigma_km"], abs=0.1)
    deg = _first(bundle["events"], "custody_degraded")
    assert f"sigma_pos {m['sigma_at_detection_km']:.0f} km" in deg["text"] or deg["t_rel_s"] != m["t_detect_rel_s"]
    assert m["ukf_sigma_at_detection_km"] > 0 and "particle-cloud" in m["sigma_at_detection_definition"]
    # the characterised direction is frame-tagged and in GCRF like the truth direction
    ch = _first(bundle["events"], "maneuver_characterised")
    assert ch["data"]["direction_frame"] == "gcrf" and len(ch["data"]["direction_rot"]) == 3
    cosang = float(np.dot(ch["data"]["direction"], m["dv_true_dir_gcrf"]))
    assert np.degrees(np.arccos(np.clip(cosang, -1, 1))) < 5.0
    assert bundle["meta"]["time_fields"]["t"].startswith("seconds since")


def test_brief_contents(bundle):
    b = bundle["brief"]
    m = bundle["metrics"]
    assert b.startswith("# Analyst brief")
    assert FOOTER_PREFIX in b and b.rstrip().endswith("Z")
    for sec in ("Bottom line up front", "Timeline (UTC)", "Maneuver characterisation", "Custody status", "Reachability", "Sensor tasking", "Confidence and caveats"):
        assert sec in b, sec
    assert f"{m['dv_est_mps']:.2f}" in b and f"{m['dv_true_mps']:.2f}" in b
    assert m["t_detect_utc"] in b and m["t_lost_utc"] in b and m["t_regained_utc"] in b
    assert "lunar glare" in b and "alpha = 0.01" in b and "NEES" in b
    loss = m["loss_reason"]
    assert f"{loss['n_frames_available_site_blocked_after_detection']} of the {loss['n_blind_frames_after_detection']} blind hours" in b
    assert "not a glare effect" in b and "sampling resolution" in b
    l1 = next(x for x in m["reachability"]["regions"] if x["key"] == "l1_gateway")
    assert f"minimum Δv to reach ≈ {l1['min_dv_mps']:.0f} m/s" in b
    tg = m["truth_geometry"]
    assert f"passes {tg['min_dist_to_L1_km']:,.0f} km from L1" in b and "does NOT transit the L1 neck" in b
    assert "## Terms used below" in b and "**tracklet**" in b and b.index("**RTN**") < b.index("## Timeline (UTC)")
    for bad in ("target", "engage", "strike", "intercept"):
        assert bad not in b.lower()
    # deterministic from the metrics
    again = generate_brief(bundle["meta"], m, bundle["events"], generated_utc="2026-10-02T00:00:00Z")
    assert again == generate_brief(bundle["meta"], m, bundle["events"], generated_utc="2026-10-02T00:00:00Z")


def test_bundle_json_and_size(bundle):
    txt = json.dumps({k: v for k, v in bundle.items() if not k.startswith("_")}, separators=(",", ":"), allow_nan=False)
    mb = len(txt.encode()) / 1e6
    print(f"fast bundle {mb:.2f} MB")
    assert mb < 15.0


def test_determinism():
    a = build_scenario(seed=0, fast=True)["metrics"]
    b = build_scenario(seed=0, fast=True)["metrics"]
    assert a["dv_est_mps"] == b["dv_est_mps"] and a["t_lost_utc"] == b["t_lost_utc"] and a["sigma_timeline"] == b["sigma_timeline"]
