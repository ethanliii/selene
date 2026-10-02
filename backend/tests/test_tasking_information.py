"""Unit tests for selene.tasking.information (covariance bookkeeping, gains, acquisition, STM chain)."""
from __future__ import annotations

import numpy as np
import pytest

from selene.dynamics.frames import DEMO_EPOCH_TDB_S
from selene.tasking import information as I

RNG = np.random.default_rng(42)


def _spd(scale_pos=100.0, scale_vel=1e-3):
    A = RNG.standard_normal((6, 6))
    P = A @ A.T + np.eye(6)
    D = np.diag([scale_pos] * 3 + [scale_vel] * 3)
    return D @ P @ D


def _obs_geometry():
    obs = np.array([42_164.0, 0.0, 0.0])
    tgt = np.array([3.2e5, 1.1e5, 4.0e4])
    return obs, tgt


# ---------------------------------------------------------------------------
def test_process_noise_structure_and_scaling():
    Q = I.process_noise(600.0, 1e-18)
    assert Q.shape == (6, 6) and np.allclose(Q, Q.T)
    assert np.allclose(np.diag(Q)[:3], 1e-18 * 600.0**3 / 3.0)
    assert np.allclose(np.diag(Q)[3:], 1e-18 * 600.0)
    assert np.allclose(Q[0, 3], 1e-18 * 600.0**2 / 2.0)
    assert np.all(np.linalg.eigvalsh(Q) >= -1e-30)
    # doubling the step multiplies the position variance by 8
    Q2 = I.process_noise(1200.0, 1e-18)
    assert Q2[0, 0] / Q[0, 0] == pytest.approx(8.0)


def test_predict_covariance_symmetric_and_grows_with_Q():
    P = _spd()
    phi = np.eye(6) + 1e-3 * RNG.standard_normal((6, 6))
    Q = I.process_noise(1200.0, 1e-12)
    Pn = I.predict_covariance(P, phi, Q)
    assert np.allclose(Pn, Pn.T)
    assert np.trace(Pn[:3, :3]) > np.trace((phi @ P @ phi.T)[:3, :3])


def test_angles_jacobian_matches_numerical_on_sky_derivatives():
    obs, tgt = _obs_geometry()
    H = I.angles_jacobian(obs, tgt)
    assert H.shape == (2, 6) and np.all(H[:, 3:] == 0.0)
    from selene.sensors.visibility import measurement_model

    ra0, dec0 = measurement_model(obs, tgt)
    h = 1e-2
    for k in range(3):
        d = np.zeros(3)
        d[k] = h
        rap, decp = measurement_model(obs, tgt + d)
        ram, decm = measurement_model(obs, tgt - d)
        assert H[0, k] == pytest.approx(np.cos(dec0) * (rap - ram) / (2 * h), rel=1e-5, abs=1e-14)
        assert H[1, k] == pytest.approx((decp - decm) / (2 * h), rel=1e-5, abs=1e-14)


def test_update_reduces_uncertainty_and_all_gain_kinds_nonnegative():
    obs, tgt = _obs_geometry()
    H = I.angles_jacobian(obs, tgt)
    for _ in range(20):
        P = _spd()
        Pp = I.update_covariance(P, H, I.ARCSEC)
        assert np.allclose(Pp, Pp.T)
        # P+ <= P- in the Loewner order
        assert np.all(np.linalg.eigvalsh(P - Pp) >= -1e-9 * np.max(np.abs(P)))
        for kind in ("logdet", "trace", "maxeig"):
            assert I.gain(P, Pp, kind) >= 0.0
        assert I.gain(P, Pp, "logdet") > 0.0 and I.gain(P, Pp, "trace") > 0.0
    # no information: p_acq -> 0 leaves P unchanged; more frames -> more reduction
    P = _spd()
    assert np.allclose(I.update_covariance(P, H, I.ARCSEC, p_acq=0.0), P)
    assert I.gain(P, I.update_covariance(P, H, I.ARCSEC, n_obs=4), "trace") > I.gain(P, I.update_covariance(P, H, I.ARCSEC), "trace")


def test_raw_gain_is_nonnegative_without_the_clamp():
    """gain() clamps tiny negative round-off; check the *unclamped* quantities are >= 0 as well."""
    obs, tgt = _obs_geometry()
    H = I.angles_jacobian(obs, tgt)
    for _ in range(20):
        P = _spd()
        Pp = I.update_covariance(P, H, I.ARCSEC)
        assert I._logdet(P) - I._logdet(Pp) >= -1e-9
        assert np.trace(P[:3, :3]) - np.trace(Pp[:3, :3]) >= -1e-9 * np.trace(P[:3, :3])
        assert np.linalg.eigvalsh(P[:3, :3])[-1] - np.linalg.eigvalsh(Pp[:3, :3])[-1] >= -1e-9 * np.trace(P[:3, :3])


def test_acquisition_models_fov_is_bernoulli_mixture_and_irf_is_optimistic():
    """'fov' = p P_det + (1-p) P_prior (expected covariance); 'irf' (R/p) gives the expected
    *information* and is far more optimistic for small p -- the review finding that motivated the
    switch of the default."""
    obs, tgt = _obs_geometry()
    H = I.angles_jacobian(obs, tgt)
    P = np.diag([1e10] * 3 + [1e-6] * 3)                 # 1e5 km prior: custody lost
    P_det = I.update_covariance(P, H, I.ARCSEC, model="none")
    for p in (1e-4, 0.3, 0.9):
        P_fov = I.update_covariance(P, H, I.ARCSEC, p_acq=p, model="fov")
        assert np.allclose(P_fov, p * P_det + (1.0 - p) * P)
        # expected trace reduction is exactly p times that of a certain detection
        assert I.gain(P, P_fov, "trace") == pytest.approx(p * I.gain(P, P_det, "trace"), rel=1e-9)
        P_irf = I.update_covariance(P, H, I.ARCSEC, p_acq=p, model="irf")
        assert I.sigma_pos_km(P_irf) <= I.sigma_pos_km(P_fov) + 1e-9
    p = 8.7e-5
    sig_fov = I.sigma_pos_km(I.update_covariance(P, H, I.ARCSEC, p_acq=p, model="fov"))
    sig_irf = I.sigma_pos_km(I.update_covariance(P, H, I.ARCSEC, p_acq=p, model="irf"))
    print(f"\nACQUISITION p={p:g}: prior RSS sigma {I.sigma_pos_km(P):.0f} km -> fov {sig_fov:.0f} km, irf {sig_irf:.0f} km")
    assert sig_fov > 0.999 * I.sigma_pos_km(P)           # sticky: essentially unchanged
    assert sig_irf < 0.7 * I.sigma_pos_km(P)             # irf 'recovers' most of it in one look
    # p = 1 -> all three models agree
    assert np.allclose(I.update_covariance(P, H, I.ARCSEC, p_acq=1.0, model="fov"), P_det)
    assert np.allclose(I.update_covariance(P, H, I.ARCSEC, p_acq=1.0, model="irf"), P_det)


def test_logdet_gain_matches_closed_form_for_scalar_like_case():
    # P = sigma^2 I on position, measurement of two on-sky axes with noise r: det ratio known
    obs = np.array([0.0, 0.0, 0.0])
    tgt = np.array([4.0e5, 0.0, 0.0])          # LOS along x: on-sky axes are y and z
    H = I.angles_jacobian(obs, tgt)
    sig = 50.0
    P = np.diag([sig**2] * 3 + [1e-8] * 3)
    r = I.ARCSEC
    Pp = I.update_covariance(P, H, r)
    # each on-sky axis: variance sig^2 -> (1/sig^2 + 1/(r*rho)^2)^-1
    rho = 4.0e5
    post = 1.0 / (1.0 / sig**2 + 1.0 / (r * rho) ** 2)
    expected = 0.5 * 2.0 * np.log(sig**2 / post)
    assert I.gain(P, Pp, "logdet") == pytest.approx(expected, rel=1e-6)
    assert Pp[1, 1] == pytest.approx(post, rel=1e-6) and Pp[0, 0] == pytest.approx(sig**2, rel=1e-9)


def test_acquisition_probability_limits():
    los = np.array([1.0, 0.0, 0.0])
    fov_half = np.deg2rad(1.0)
    small = np.diag([1.0] * 3 + [1e-8] * 3)           # 1 km at 4e5 km -> 0.5 arcsec: certain
    huge = np.diag([1e10] * 3 + [1e-8] * 3)           # 1e5 km -> 14 deg >> 1 deg field
    assert I.acquisition_probability(small, los, 4.0e5, fov_half) == pytest.approx(1.0)
    assert I.acquisition_probability(huge, los, 4.0e5, fov_half) < 0.02
    # only the perpendicular components matter: uncertainty along the LOS is harmless
    along = np.diag([1e10, 1.0, 1.0] + [1e-8] * 3)
    assert I.acquisition_probability(along, los, 4.0e5, fov_half) == pytest.approx(1.0)
    # a mosaic of n fields scales the covered solid angle: p(n) = 1 - (1 - p(1))^n
    p1 = I.acquisition_probability(huge, los, 4.0e5, fov_half)
    p9 = I.acquisition_probability(huge, los, 4.0e5, fov_half, n_tiles=9)
    assert p9 == pytest.approx(1.0 - (1.0 - p1) ** 9, rel=1e-9) and p9 > p1


def test_symplectic_inverse_is_exact_for_symplectic_matrix():
    # build a random symplectic matrix exp(J S) with S symmetric
    from scipy.linalg import expm

    S = RNG.standard_normal((6, 6))
    S = 0.1 * (S + S.T)
    M = expm(I._J_SYMPLECTIC @ S)
    assert np.allclose(I.symplectic_inverse(M) @ M, np.eye(6), atol=1e-10)


def test_slew_feasible():
    a = np.array([1.0, 0.0, 0.0])
    b = np.array([np.cos(np.deg2rad(30)), np.sin(np.deg2rad(30)), 0.0])
    assert I.slew_feasible(None, b, 0.0, 1.0)
    assert I.slew_feasible(a, b, np.deg2rad(1.0), 30.0)         # 30 deg in 30 s at 1 deg/s
    assert not I.slew_feasible(a, b, np.deg2rad(1.0), 29.0)
    assert I.slew_feasible(a, b, np.deg2rad(1.0), 60.0, fraction=0.5)
    # the third argument is the *elapsed* time: an idle sensor accumulates slew budget
    assert not I.slew_feasible(a, b, np.deg2rad(0.01), 300.0)   # one 5-min slot: 3 deg
    assert I.slew_feasible(a, b, np.deg2rad(0.01), 10 * 300.0)  # ten idle slots: 30 deg


# ---------------------------------------------------------------------------
# with the catalog: STM chain and visibility table
@pytest.fixture(scope="module")
def short_tracks():
    t_nodes = DEMO_EPOCH_TDB_S + 1200.0 * np.arange(19)      # 6 h, 20-min slots
    return I.build_tracks(["SIM-DRO-01", "SIM-L2-HALO-01"], t_nodes, q_psd=1e-18, sigma0_pos_km=10.0)


def test_stm_chain_matches_segmentwise_propagation(short_tracks):
    from selene.dynamics.ephemeris import EphemParams, propagate_ephemeris
    from selene.objects.catalog import get_catalog

    cat = get_catalog()
    D = np.diag([1, 1, 1, 1e4, 1e4, 1e4])
    Di = np.linalg.inv(D)
    for tr in short_tracks:
        assert tr.phi.shape == (18, 6, 6) and tr.Q.shape == (18, 6, 6)
        assert tr.meta["stm"]["max_abs_det_minus_1"] < 1e-8
        tp = cat.truth(tr.object_id).params
        params = EphemParams(srp=tp.srp, cr_area_mass=tp.cr_area_mass)
        for k in (0, 9, 17):
            sol = propagate_ephemeris(tr.x_nodes[k], tr.t_nodes[k], tf_s=tr.t_nodes[k + 1], params=params, stm=True)
            phi_seg = sol.y[6:, -1].reshape(6, 6)
            rel = np.linalg.norm(Di @ (tr.phi[k] - phi_seg) @ D) / np.linalg.norm(Di @ phi_seg @ D)
            assert rel < 1e-7, (tr.object_id, k, rel)


def test_visibility_table_and_zero_gain_when_invisible(short_tracks):
    from selene.sensors.observers import get_observer
    from selene.sensors.sites import get_site

    sensors = [get_site("haleakala"), get_observer("dro_obs")]
    tab = I.build_visibility_table(sensors, short_tracks)
    assert tab.visible.shape == (2, 2, 19) and tab.los.shape == (2, 2, 19, 3)
    assert np.allclose(np.linalg.norm(tab.los, axis=-1), 1.0)
    assert tab.sigma_rad[0] == pytest.approx(I.ARCSEC)
    # at the demo epoch (2 d before full Moon) the ground site is glare-blinded, the DRO observer is not
    assert not tab.visible[0].any() and tab.visible[1].any()
    assert "moon_exclusion" in tab.reasons_at(0, 0, 0)
    P = short_tracks[0].P0
    g0, p0, Pp0 = I.expected_gain(tab, 0, 0, 0, P, short_tracks[0].x_nodes[0])
    assert g0 == 0.0 and p0 == 0.0 and Pp0 is None
    k = int(np.flatnonzero(tab.visible[1, 0])[0])
    for kind in ("logdet", "trace", "maxeig"):
        g, p, Pp = I.expected_gain(tab, 1, 0, k, P, short_tracks[0].x_nodes[k], kind=kind)
        assert g >= 0.0 and 0.0 < p <= 1.0 and Pp is not None
        if kind != "maxeig":
            assert g > 0.0
        else:
            # isotropic prior: one angles-only look leaves the line-of-sight eigenvalue untouched,
            # so the max-eigenvalue criterion correctly reports no reduction
            assert g == pytest.approx(0.0, abs=1e-9)
    # the 'none' acquisition model gives p = 1 and at least as much gain
    g_none, p_none, _ = I.expected_gain(tab, 1, 0, k, P, short_tracks[0].x_nodes[k], acquisition="none")
    assert p_none == 1.0 and g_none >= g - 1e-12
