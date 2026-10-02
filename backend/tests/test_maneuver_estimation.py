"""M5 impulsive-Δv estimation: 10 m/s burn recovered within 20 % / 20° from 10 post-burn obs."""
from __future__ import annotations

import json

import numpy as np
import pytest

from selene.dynamics import frames
from selene.dynamics.ephemeris import EphemParams
from selene.dynamics.frames import DEMO_EPOCH_TDB_S as T0
from selene.maneuver._simple_ekf import run_ekf
from selene.maneuver.config import DetectorConfig, EstimatorConfig
from selene.maneuver.detection import detect
from selene.maneuver.estimation import classify_maneuver, estimate_impulsive_dv
from selene.maneuver.synthetic import (
    DIRECTION_NAMES,
    Burn,
    direction_unit,
    generate_measurements,
    initial_covariance,
    rtn_basis,
    sample_initial_state,
    truth_with_burn,
    vnb_basis,
)
from selene.objects.catalog import get_catalog
from selene.sensors.observers import get_observer

D = 86400.0
CADENCE = 6 * 3600.0


@pytest.fixture(scope="module")
def burn_case():
    cat = get_catalog()
    e = cat.get("SIM-DRO-01")
    ph = e.physical
    params = EphemParams(srp=True, cr_area_mass=ph["cr_area_mass_m2_kg"])
    x0 = cat.state_at("SIM-DRO-01", T0)
    sensors = [get_observer(i) for i in ("dro_obs", "l1_halo_obs", "geo_west")]
    tb = T0 + 3.3 * D
    dv_true = np.array([5.0, -7.0, 4.0]); dv_true *= 10e-3 / np.linalg.norm(dv_true)  # 10 m/s oblique
    burn = Burn(tb, dv_true)
    truth = truth_with_burn(x0, T0, T0 + 7 * D, burn, params)
    rng = np.random.default_rng(123)
    epochs = T0 + CADENCE * np.arange(1, 29)
    meas = generate_measurements(truth.at, epochs, sensors, 1.0, rng, ph["radius_m"], ph["albedo"])
    P0 = initial_covariance(20.0, 2.0)
    xs = sample_initial_state(truth.at(T0), P0, rng)
    run = run_ekf(xs, P0, T0, meas, params=params, q_psd=1e-18, object_id="SIM-DRO-01")
    rep = detect(run, DetectorConfig(alpha=0.01), truth=truth.at, params=params)
    assert rep.declared
    i = int(np.searchsorted(run.t_s, rep.first_declared_t_s))
    assert i >= 1 and run.t_s[i - 1] < tb < run.t_s[i]
    return {"params": params, "truth": truth, "burn": burn, "run": run, "i": i, "post": list(run.meas[i:]),
            "x_pre": run.x[i - 1], "P_pre": run.P[i - 1], "t_pre": float(run.t_s[i - 1]),
            "window": (float(run.t_s[i - 1]), float(run.t_s[i]))}


def _errors(est, burn):
    dv_t = burn.dv_kms
    mag_err_pct = 100 * abs(np.linalg.norm(est.dv_gcrf_kms) - np.linalg.norm(dv_t)) / np.linalg.norm(dv_t)
    cosang = est.direction_gcrf @ dv_t / np.linalg.norm(dv_t)
    return mag_err_pct, float(np.degrees(np.arccos(np.clip(cosang, -1, 1)))), est.t_burn_s - burn.t_s


def test_dv_estimate_10mps_with_prior(burn_case):
    c = burn_case
    est = estimate_impulsive_dv(c["x_pre"], c["t_pre"], c["post"], c["window"], P_pre=c["P_pre"], params=c["params"],
                                cfg=EstimatorConfig(n_post_obs=10))
    mag_err, dir_err, dt = _errors(est, c["burn"])
    print(f"\n|dv| est {est.magnitude_mps:.3f}±{est.magnitude_sigma_mps:.3f} m/s (true 10), dir err {dir_err:.2f}° "
          f"(σ {est.direction_sigma_deg:.2f}°), t_b err {dt:+.0f} s (σ {est.t_burn_sigma_s:.0f}), rms {est.residual_rms_arcsec:.2f}″, "
          f"red χ² {est.reduced_chi2:.2f}, {est.meta['elapsed_s']:.2f}s")
    assert est.n_obs == 10 and est.converged
    assert mag_err < 20.0 and dir_err < 20.0
    assert abs(dt) < CADENCE
    assert est.reduced_chi2 < 5.0 and est.residual_rms_arcsec < 3.0
    assert est.magnitude_sigma_mps > 0 and np.isfinite(est.direction_sigma_deg)
    assert est.frame_center in ("earth", "moon")
    assert np.linalg.norm(est.dv_rtn_mps) == pytest.approx(est.magnitude_mps, rel=1e-9)
    assert np.linalg.norm(est.dv_vnb_mps) == pytest.approx(est.magnitude_mps, rel=1e-9)
    assert est.classification["heuristic"] is True and est.classification["primary"]
    assert c["window"][0] <= est.t_burn_s <= c["window"][1]
    assert len(est.grid) == 9
    json.dumps(est.as_dict())


def test_dv_estimate_without_prior_fixed_pre_state(burn_case):
    c = burn_case
    est = estimate_impulsive_dv(c["x_pre"], c["t_pre"], c["post"], c["window"], P_pre=None, params=c["params"],
                                cfg=EstimatorConfig(n_post_obs=10, refine=False, n_grid=5))
    mag_err, dir_err, dt = _errors(est, c["burn"])
    assert est.dx_pre_correction is None and est.meta["with_prior"] is False
    assert mag_err < 20.0 and dir_err < 20.0
    assert len(est.grid) == 5 and abs(dt) < CADENCE


def test_estimate_needs_two_observations(burn_case):
    c = burn_case
    with pytest.raises(ValueError):
        estimate_impulsive_dv(c["x_pre"], c["t_pre"], c["post"][:1], c["window"], params=c["params"])
    with pytest.raises(ValueError):
        estimate_impulsive_dv(c["x_pre"], c["t_pre"], c["post"], (c["t_pre"] - 10.0, c["t_pre"] - 5.0), params=c["params"])


def test_local_frames_and_named_directions(burn_case):
    x = burn_case["truth"].at(T0 + 1 * D)
    t = T0 + 1 * D
    for center in ("earth", "moon"):
        for B in (rtn_basis(x, t, center), vnb_basis(x, t, center)):
            assert np.allclose(B @ B.T, np.eye(3), atol=1e-12)
            assert np.linalg.det(B) == pytest.approx(1.0, abs=1e-12)  # right-handed
    for name in DIRECTION_NAMES:
        u = direction_unit(name, x, t)
        assert np.linalg.norm(u) == pytest.approx(1.0)
    assert direction_unit("prograde", x, t, "earth") @ x[3:] > 0
    assert np.allclose(direction_unit("retrograde", x, t, "earth"), -direction_unit("prograde", x, t, "earth"))
    assert direction_unit("toward_earth", x, t) @ x[:3] < 0
    with pytest.raises(ValueError):
        direction_unit("sideways", x, t)


def test_classification_labels_are_heuristic_and_consistent(burn_case):
    x_pre = burn_case["truth"].at(burn_case["burn"].t_s - 1.0)
    t = burn_case["burn"].t_s
    # big Earth-centred prograde burn: energy up, Jacobi down, 'in-track raise'
    u = direction_unit("prograde", x_pre, t, "earth")
    x_post = x_pre.copy(); x_post[3:] += 0.1 * u
    dv_rtn = rtn_basis(x_pre, t, "earth") @ (0.1 * u) * 1e3
    cls = classify_maneuver(x_pre, x_post, t, dv_rtn, "earth", sigma_mag_mps=1.0)
    assert cls["heuristic"] is True and cls["primary"] == "in-track raise"
    assert cls["energy_earth_post_km2_s2"] > cls["energy_earth_pre_km2_s2"]
    # Jacobi C = 2U − v_rot² drops iff Δv has a component along the *rotating-frame* velocity; on a
    # DRO the inertial-prograde direction is anti-aligned with v_rot, so C can legitimately rise.
    s_pre, s_post = frames.gcrf_to_rot(x_pre, t), frames.gcrf_to_rot(x_post, t)
    assert (cls["jacobi_post"] < cls["jacobi_pre"]) == ((s_post[3:] - s_pre[3:]) @ s_pre[3:] > 0)
    assert any("Jacobi constant" in s for s in cls["labels"])
    assert cls["jacobi_L1"] > cls["jacobi_L2"]  # C_L1 ≈ 3.1883 > C_L2 ≈ 3.1722 (Earth-Moon)
    assert cls["jacobi_L1"] == pytest.approx(3.1883, abs=2e-3) and cls["jacobi_L2"] == pytest.approx(3.1722, abs=2e-3)
    assert any("Earth-relative energy increased" in s for s in cls["labels"])
    # negligible burn
    cls0 = classify_maneuver(x_pre, x_pre, t, np.zeros(3), "earth", sigma_mag_mps=1.0)
    assert cls0["primary"] == "no significant maneuver"
    # normal burn -> plane change
    n = rtn_basis(x_pre, t, "moon")[2]
    x_post = x_pre.copy(); x_post[3:] += 0.01 * n
    cls2 = classify_maneuver(x_pre, x_post, t, rtn_basis(x_pre, t, "moon") @ (0.01 * n) * 1e3, "moon")
    assert cls2["primary"] == "plane change"
