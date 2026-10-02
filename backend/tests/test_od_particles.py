import time

import numpy as np

from selene.dynamics.ephemeris import EphemParams
from selene.dynamics.frames import DEMO_EPOCH_TDB_S as T0
from selene.objects.catalog import get_catalog
from selene.od.particles import propagate_cloud, sample_particles
from selene.od.realism import default_P0
from selene.od.stacked import propagate_many


def _cloud(oid, n=2000, days=7.0, step_h=6.0, sig_pos=100.0, sig_vel=1.0, seed=0):
    cat = get_catalog()
    tr = cat.truth(oid)
    params = EphemParams(srp=True, cr_area_mass=tr.params.cr_area_mass)
    tg = T0 + np.arange(0.0, days * 86400.0 + 1e-6, step_h * 3600.0)
    t = time.perf_counter()
    pc = propagate_cloud(tr.at(T0), default_P0(sig_pos, sig_vel), T0, tg, n, np.random.default_rng(seed), params, object_id=oid)
    return pc, time.perf_counter() - t, tr


def test_dro_cloud_grows_monotonically_without_observations():
    pc, dt, tr = _cloud("SIM-DRO-01")
    assert dt < 15.0, f"2000 particles over 7 days took {dt:.1f} s"
    m = pc.metrics()
    sig = m["sigma_pos_km"]
    assert np.all(np.diff(sig) > 0), sig
    assert sig[-1] > 10 * sig[0]
    # stretching along the flow: strongly anisotropic after a week
    axes = m["axes_km"][-1]
    assert axes[0] > 5 * axes[2]
    # the mean particle (index 0) is the mean state itself -> tracks the truth
    assert np.linalg.norm(pc.states[-1, 0, :3] - tr.at(pc.t_s[-1])[:3]) < 0.1
    # linear (STM) and sample covariances agree early and diverge late (non-linearity)
    assert abs(m["linear_sigma_pos_km"][1] - sig[1]) / sig[1] < 0.05
    print(f"\nDRO sigma_pos km per 24 h: {np.round(sig[::4], 1).tolist()}  axes@7d {np.round(axes, 1).tolist()}")


def test_nrho_cloud_reported_not_assumed():
    """NRHO uncertainty is phase-dependent: it does NOT grow monotonically (it compresses after
    perilune), so we only check that it is finite, larger than the start, and record the numbers."""
    pc, dt, tr = _cloud("SIM-NRHO-RELAY-01", n=1000)
    m = pc.metrics()
    sig = m["sigma_pos_km"]
    assert np.all(np.isfinite(sig)) and sig[-1] > sig[0]
    dro, _, _ = _cloud("SIM-DRO-01", n=1000)
    ds = dro.metrics()["sigma_pos_km"]
    print(f"\nNRHO sigma_pos km per 24 h: {np.round(sig[::4], 1).tolist()} (monotone={bool(np.all(np.diff(sig) > 0))}); "
          f"DRO: {np.round(ds[::4], 1).tolist()}")


def test_banana_shape_with_iod_sized_prior():
    pc, _, _ = _cloud("SIM-DRO-01", n=1500, sig_pos=500.0, sig_vel=5.0, step_h=12.0)
    m = pc.metrics()
    # early: straight ellipsoid; after a week: visibly bent (quadratic deviation relative to the minor axis)
    assert m["curvature"][1] < 0.1
    assert m["curvature"][-1] > 0.3, m["curvature"].round(3).tolist()
    assert m["mean_offset_km"][-1] > 10.0 * m["mean_offset_km"][1]    # sample mean departs from the linear mean
    print(f"\nbanana metrics (12 h steps): curvature {m['curvature'].round(3).tolist()} "
          f"skew {m['skew_major'].round(2).tolist()} offset_km {m['mean_offset_km'].round(1).tolist()}")


def test_stacked_propagator_matches_truth_and_export():
    cat = get_catalog()
    tr = cat.truth("SIM-L2-HALO-01")
    params = EphemParams(srp=True, cr_area_mass=tr.params.cr_area_mass)
    tg = T0 + np.array([0.0, 3600.0, 86400.0, 3 * 86400.0])
    Y = propagate_many(np.tile(tr.at(T0), (3, 1)), T0, tg, params)
    assert Y.shape == (4, 3, 6)
    assert np.allclose(Y[:, 0, :3], tr.at(tg)[:, :3], atol=0.05)   # km-level agreement over 3 days
    assert np.allclose(Y[:, 0], Y[:, 2])
    # backward propagation and duplicate output epochs
    Yb = propagate_many(tr.at(T0), T0, np.array([T0 - 3600.0, T0 - 3600.0, T0 - 7200.0]), params)
    assert np.allclose(Yb[0], Yb[1]) and np.allclose(Yb[2, :3], tr.at(T0 - 7200.0)[:3], atol=0.05)
    pc, _, _ = _cloud("SIM-DRO-01", n=200, days=1.0, step_h=12.0)
    d = pc.to_dict(max_particles=50)
    assert len(d["frames"]) == 3 and len(d["frames"][0]["positions_km"]) == 50 and len(d["frames"][0]["positions_rot"][0]) == 3
    assert abs(np.linalg.norm(d["frames"][0]["positions_rot"][0]) - np.linalg.norm(d["frames"][0]["positions_rot"][1])) < 0.01
    s = sample_particles(np.zeros(6), np.eye(6), 5000, np.random.default_rng(1))
    assert np.allclose(s.std(axis=0), 1.0, atol=0.05) and np.all(s[0] == 0)
