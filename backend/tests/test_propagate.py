"""M1 acceptance tests for the high-level propagator and the CR3BP-vs-ephemeris agreement."""
import json

import numpy as np

from selene.constants import L_STAR, T_STAR
from selene.dynamics import frames as f
from selene.dynamics import propagate as p
from selene.dynamics.cr3bp import propagate_cr3bp
from selene.dynamics.ephemeris import EphemParams, propagate_ephemeris

DAY = 86400.0
T0 = f.DEMO_EPOCH_TDB_S
# L1 Lyapunov IC, Koon-Lo-Marsden-Ross (2011) ch. 4 (see test_cr3bp.py)
KLMR_IC = np.array([0.8234, 0.0, 0.0, 0.0, 0.1263, 0.0])


def test_cr3bp_vs_ephemeris_two_days_l1_lyapunov():
    """Physics agreement check (PLAN §3, M1): the same rotating-frame IC propagated with the
    CR3BP and with the DE440s n-body model must stay within 1 % of L* (3844 km) over 2 days
    when compared in the rotating frame.  Residual sources: lunar eccentricity (ω·T* ≠ 1 at
    the epoch, instantaneous length scale), solar tide, and the L1 instability (~4× growth in
    2 days)."""
    t_eval = T0 + np.linspace(0.0, 2 * DAY, 97)
    tr_c = p.propagate(KLMR_IC, T0, t_eval, model="cr3bp", frame_in="rot_nd")
    tr_e = p.propagate(KLMR_IC, T0, t_eval, model="ephemeris", frame_in="rot_nd")
    assert tr_c.frame == tr_e.frame == "gcrf_km"
    rc = tr_c.to_frame("rot_nd").positions
    re = tr_e.to_frame("rot_nd").positions
    err_km = np.linalg.norm(rc - re, axis=1) * L_STAR
    assert err_km[0] < 1e-6  # identical start
    assert err_km[-1] < 0.01 * L_STAR, f"CR3BP vs ephemeris rotating-frame error after 2 d = {err_km[-1]:.1f} km"
    assert err_km.max() < 0.01 * L_STAR, f"max rotating-frame error {err_km.max():.1f} km"
    # the error should be a smooth, growing divergence, not a frame glitch
    assert np.all(np.diff(err_km[10:]) > -1.0)


def test_cr3bp_model_matches_direct_propagation_in_rotating_frame():
    t_eval = T0 + np.linspace(0.0, 1.5 * DAY, 20)
    tr = p.propagate(KLMR_IC, T0, t_eval, model="cr3bp", frame_in="rot_nd")
    rot = tr.to_frame("rot_nd").states
    direct = propagate_cr3bp(KLMR_IC, t_eval=(t_eval - T0) / T_STAR).y.T
    assert np.max(np.abs(rot - direct)) < 1e-9
    assert tr.meta["model"] == "cr3bp" and "approximate" in tr.meta["note"]


def test_ephemeris_model_matches_low_level_and_frame_inputs():
    t_eval = T0 + np.linspace(0.0, 3 * DAY, 25)
    s0_gcrf = f.rot_to_gcrf(KLMR_IC, T0)
    tr_g = p.propagate(s0_gcrf, T0, t_eval, model="ephemeris", frame_in="gcrf_km")
    tr_r = p.propagate(KLMR_IC, T0, t_eval, model="ephemeris", frame_in="rot_nd")
    tr_m = p.propagate(f.gcrf_to_moon_centered(s0_gcrf, T0), T0, t_eval, model="ephemeris", frame_in="moon_km")
    sol = propagate_ephemeris(s0_gcrf, T0, t_eval_s=t_eval)
    assert np.allclose(tr_g.states, sol.y.T, rtol=0, atol=1e-6)
    assert np.allclose(tr_r.states, tr_g.states, rtol=0, atol=1e-6)
    assert np.allclose(tr_m.states, tr_g.states, rtol=0, atol=1e-6)


def test_t_eval_not_starting_at_t0_and_backward():
    t_eval = T0 + np.array([0.5, 1.0, 1.5]) * DAY
    tr = p.propagate(KLMR_IC, T0, t_eval, model="ephemeris", frame_in="rot_nd")
    assert tr.t_s.shape == (3,) and tr.states.shape == (3, 6)
    ref = p.propagate(KLMR_IC, T0, T0 + np.linspace(0, 1.5 * DAY, 4), model="ephemeris", frame_in="rot_nd")
    assert np.allclose(tr.states, ref.states[1:], atol=1e-6)
    # backward in time
    back = p.propagate(tr.states[-1], tr.t_s[-1], np.array([T0]), model="ephemeris")
    assert np.linalg.norm(back.states[0, :3] - ref.states[0, :3]) < 1e-3


def test_trajectory_interpolation_and_serialisation():
    t_eval = T0 + np.linspace(0.0, 2 * DAY, 49)  # 1-hour samples
    tr = p.propagate(KLMR_IC, T0, t_eval, model="ephemeris", frame_in="rot_nd", keep_dense=True)
    tq = T0 + np.array([0.37, 1.21]) * DAY
    dense = tr.at(tq)  # dense output path
    tr_nodense = p.Trajectory(tr.t_s, tr.states, tr.frame, {"model": "ephemeris"})
    spline = tr_nodense.at(tq)  # cubic spline path
    assert dense.shape == (2, 6) and spline.shape == (2, 6)
    assert np.max(np.abs(dense[:, :3] - spline[:, :3])) < 1.0  # km, slow L1 dynamics on 1-h samples
    assert tr.at(float(tq[0])).shape == (6,)
    assert tr.positions.shape == (49, 3) and len(tr) == 49
    # frame conversions round-trip
    rt = tr.to_frame("rot_nd").to_frame("moon_km").to_frame("gcrf_km")
    assert np.max(np.abs(rt.states - tr.states)) < 1e-6
    assert tr.to_frame("gcrf_km") is tr
    # JSON-serialisable, dense output stripped
    d = tr.as_dict()
    txt = json.dumps(d)
    assert "sol" not in d["meta"] and d["frame"] == "gcrf_km" and len(json.loads(txt)["states"]) == 49
