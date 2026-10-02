"""M1 acceptance tests for the rotating <-> GCRF <-> Moon-centered frame transforms (PLAN §2.3)."""
import numpy as np
import pytest

from selene.constants import MU, T_STAR
from selene.dynamics import frames as f
from selene.dynamics.ephemeris import get_ephemeris

DAY = 86400.0
T0 = f.DEMO_EPOCH_TDB_S


def test_demo_epoch_is_2026_03_01():
    from selene.time import tdb_jd_to_utc_iso, jd_tdb_from_et

    assert tdb_jd_to_utc_iso(jd_tdb_from_et(T0)).startswith("2026-03-01T00:00:00")


def test_frame_basis_is_orthonormal_right_handed():
    for t in (T0, T0 + 7 * DAY, T0 + 100 * DAY):
        fi = f.rotating_frame(t)
        assert np.allclose(fi.R.T @ fi.R, np.eye(3), atol=1e-14)
        assert abs(np.linalg.det(fi.R) - 1.0) < 1e-13
        # ẑ along the Earth-Moon angular momentum, ω mostly along ẑ
        sm = get_ephemeris().moon_state(t)
        h = np.cross(sm[:3], sm[3:])
        assert np.allclose(fi.R[:, 2], h / np.linalg.norm(h))
        assert abs(np.dot(fi.omega_vec, fi.R[:, 2]) / fi.omega - 1.0) < 1e-6
        assert abs(np.dot(fi.omega_vec, fi.R[:, 1])) < 1e-20


def test_omega_mean_sidereal_rate_and_eccentricity_band():
    ts = T0 + np.linspace(0, 27.321661 * DAY, 400)  # one sidereal month
    fi = f.rotating_frame(ts)
    w = fi.omega
    mean_rate = 2 * np.pi / (27.321661 * DAY)  # 2.6617e-6 rad/s
    assert abs(w.mean() - 2.66e-6) / 2.66e-6 < 0.03, f"mean |ω| = {w.mean():.4e} rad/s"
    assert abs(w.mean() - mean_rate) / mean_rate < 0.02
    # instantaneous |ω| ∝ 1/d² swings ~±12 % with the lunar eccentricity (e ≈ 0.055)
    assert 2.3e-6 < w.min() and w.max() < 3.1e-6
    assert 356_000 < fi.d_km.min() and fi.d_km.max() < 407_000


def test_moon_maps_to_1_minus_mu_at_rest():
    for t in (T0, T0 + 3.3 * DAY, T0 + 200 * DAY):
        sm = get_ephemeris().moon_state(t)
        s_rot = f.gcrf_to_rot(sm, t)
        assert np.max(np.abs(s_rot[:3] - np.array([1.0 - MU, 0.0, 0.0]))) < 1e-12
        assert np.max(np.abs(s_rot[3:])) < 1e-10
        # Earth (origin of GCRF) at (-mu, 0, 0), at rest
        s_e = f.gcrf_to_rot(np.zeros(6), t)
        assert np.max(np.abs(s_e[:3] - np.array([-MU, 0.0, 0.0]))) < 1e-12
        assert np.max(np.abs(s_e[3:])) < 1e-10


def test_round_trip_random_states():
    rng = np.random.default_rng(42)
    N = 50
    S = np.concatenate([rng.uniform(-4.5e5, 4.5e5, (N, 3)), rng.uniform(-1.5, 1.5, (N, 3))], axis=1)
    ts = T0 + rng.uniform(0, 60 * DAY, N)
    back = f.rot_to_gcrf(f.gcrf_to_rot(S, ts), ts)
    rel_pos = np.max(np.abs(back[:, :3] - S[:, :3])) / 4.5e5
    rel_vel = np.max(np.abs(back[:, 3:] - S[:, 3:])) / 1.5
    assert rel_pos < 1e-9 and rel_vel < 1e-9, f"round trip rel err pos {rel_pos:.1e} vel {rel_vel:.1e}"
    # the other direction, nondimensional inputs
    Snd = np.concatenate([rng.uniform(-1.2, 1.2, (N, 3)), rng.uniform(-1.5, 1.5, (N, 3))], axis=1)
    back_nd = f.gcrf_to_rot(f.rot_to_gcrf(Snd, ts), ts)
    assert np.max(np.abs(back_nd - Snd)) < 1e-9
    # scalar API agrees with the batched one
    s1 = f.gcrf_to_rot(S[0], ts[0])
    assert np.allclose(s1, f.gcrf_to_rot(S, ts)[0], rtol=0, atol=1e-13)
    assert np.allclose(f.rot_pos_to_gcrf(Snd[:3, :3], ts[:3]), f.rot_to_gcrf(Snd[:3], ts[:3])[:, :3])
    assert np.allclose(f.gcrf_pos_to_rot(S[0, :3], ts[0]), s1[:3])


def test_velocity_transform_is_time_derivative_of_position_transform():
    """For a free particle moving on a straight line in GCRF, the nondimensional rotating-frame
    velocity must equal T* d/dt of the rotating-frame position: this checks the ω×r, ḋ and
    precession terms together, independently of how they were derived."""
    s = np.array([3.0e5, 1.0e5, 5.0e4, -0.3, 0.8, 0.1])
    for t in (T0, T0 + 11.1 * DAY):
        h = 1.0

        def rpos(tq):
            sq = s.copy()
            sq[:3] = s[:3] + s[3:] * (tq - t)
            return f.gcrf_to_rot(sq, tq)[:3]

        num = (rpos(t + h) - rpos(t - h)) / (2 * h) * T_STAR
        ana = f.gcrf_to_rot(s, t)[3:]
        err = np.max(np.abs(num - ana))
        assert err < 1e-8, f"velocity transform mismatch {err:.2e} nd"


def test_nd_velocity_scaling_definition():
    """v_nd = v_km_s / (d(t)/T*): a velocity purely along the (non-rotating) z-axis of the frame
    picks up no ω×r term when r is on the rotation axis... simpler: check the scale factor
    through a state at the barycenter with zero position offset."""
    fi = f.rotating_frame(T0)
    v_gcrf = 0.5 * fi.R[:, 2]  # 0.5 km/s along ẑ
    s = np.concatenate([fi.r_bary_gcrf, fi.v_bary_gcrf + v_gcrf])
    s_rot = f.gcrf_to_rot(s, T0)
    assert np.allclose(s_rot[:3], 0.0, atol=1e-15)
    assert np.allclose(s_rot[3:], [0.0, 0.0, 0.5 / (fi.d_km / T_STAR)], atol=1e-14)


def test_moon_centered_transforms():
    s = np.array([-1.0e5, 2.0e5, 3.0e4, 0.5, -0.2, 0.1])
    sm = get_ephemeris().moon_state(T0)
    s_mc = f.gcrf_to_moon_centered(s, T0)
    assert np.allclose(s_mc, s - sm)
    assert np.allclose(f.moon_centered_to_gcrf(s_mc, T0), s)
    assert np.allclose(f.gcrf_to_moon_centered(sm, T0), 0.0)
    ts = np.array([T0, T0 + DAY])
    S = np.vstack([s, s])
    assert f.gcrf_to_moon_centered(S, ts).shape == (2, 6)


def test_lagrange_points_gcrf_geometry():
    L = f.lagrange_points_gcrf(T0)
    fi = f.rotating_frame(T0)
    assert L.shape == (5, 3)
    # L1 lies on the Earth-Moon line at (x_L1 + mu) d from the Earth
    from selene.dynamics.cr3bp import lagrange_points

    x = lagrange_points()[:, 0]
    assert abs(np.linalg.norm(L[0]) - (x[0] + MU) * fi.d_km) < 1e-6
    assert abs(np.linalg.norm(L[1]) - (x[1] + MU) * fi.d_km) < 1e-6
    assert np.allclose(np.cross(L[0], fi.R[:, 0]), 0.0, atol=1e-6)
    # L4 is equidistant from Earth and Moon
    rm = get_ephemeris().position("moon", T0)
    assert abs(np.linalg.norm(L[3]) - np.linalg.norm(L[3] - rm)) < 1e-6
    assert f.lagrange_points_gcrf(np.array([T0, T0 + DAY])).shape == (2, 5, 3)


def test_nd_time_helpers():
    tau = np.array([0.0, 1.0, 2.5])
    ts = f.nd_time_to_tdb_s(tau, T0)
    assert np.allclose(ts, T0 + tau * T_STAR)
    assert np.allclose(f.tdb_s_to_nd_time(ts, T0), tau)


def test_shape_errors():
    with pytest.raises(ValueError):
        f.gcrf_to_rot(np.zeros(6), np.array([T0, T0 + 1]))
    with pytest.raises(ValueError):
        f.gcrf_to_rot(np.zeros((3, 6)), np.array([T0, T0 + 1]))
