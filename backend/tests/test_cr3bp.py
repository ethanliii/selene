"""M1 acceptance tests for the CR3BP engine (PLAN §2.1, §3)."""
import math

import numpy as np
import pytest

from selene.constants import MU
from selene.dynamics import cr3bp

# L1 Lyapunov orbit example, Koon-Lo-Marsden-Ross (2011), *Dynamical Systems, the Three-Body
# Problem and Space Mission Design*, ch. 4 (Earth-Moon, x0 = 0.8234, vy0 = 0.1263, T ≈ 2.7430).
KLMR_IC = np.array([0.8234, 0.0, 0.0, 0.0, 0.1263, 0.0])
KLMR_T = 2.7430


def test_lagrange_points_match_literature():
    # KLMR (2011) Table 2.1 / Szebehely (1967) for mu = 0.01215
    L = cr3bp.lagrange_points(MU)
    assert L.shape == (5, 3)
    assert abs(L[0, 0] - 0.836915) < 1e-5
    assert abs(L[1, 0] - 1.155682) < 1e-5
    assert abs(L[2, 0] - (-1.005063)) < 1e-5
    assert np.allclose(L[3], [0.5 - MU, math.sqrt(3) / 2, 0.0], atol=1e-15)
    assert np.allclose(L[4], [0.5 - MU, -math.sqrt(3) / 2, 0.0], atol=1e-15)
    # all five are equilibria of the rotating-frame EOM
    for p in L:
        acc = cr3bp.cr3bp_eom(0.0, np.concatenate([p, np.zeros(3)]))
        assert np.max(np.abs(acc)) < 1e-12


def test_jacobi_constant_conserved_over_ten_periods():
    t_eval = np.linspace(0.0, 10 * KLMR_T, 2001)
    sol = cr3bp.propagate_cr3bp(KLMR_IC, t_eval=t_eval, rtol=1e-12, atol=1e-12, method="DOP853")
    assert sol.success
    C = cr3bp.jacobi(sol.y.T)
    drift = float(np.max(np.abs(C - C[0])))
    assert drift < 1e-10, f"Jacobi drift {drift:.3e} over 10 periods (limit 1e-10)"
    # sanity: C of the L1 Lyapunov is slightly below C(L1) ≈ 3.1883 (orbit exists beyond the L1 neck)
    C_L1 = cr3bp.jacobi(np.concatenate([cr3bp.lagrange_points()[0], np.zeros(3)]))
    assert 3.0 < C[0] < C_L1


def test_jacobi_accepts_single_and_batched_states():
    c1 = cr3bp.jacobi(KLMR_IC)
    cN = cr3bp.jacobi(np.vstack([KLMR_IC, KLMR_IC]))
    assert isinstance(c1, float)
    assert cN.shape == (2,) and np.allclose(cN, c1)


def test_lyapunov_near_periodic_and_symmetric_crossing():
    """The literature IC is quoted to 4 digits: the orbit should return close to itself after
    one period and cross y = 0 at ≈ T/2 with |vx| small (perpendicular crossing)."""
    ev = cr3bp.y_zero_crossing_event(direction=-1)
    sol = cr3bp.propagate_cr3bp(KLMR_IC, tf=2 * KLMR_T, events=ev)
    assert sol.t_events[0].size >= 1
    t_half = sol.t_events[0][0]
    s_half = sol.y_events[0][0]
    assert abs(t_half - KLMR_T / 2) < 0.02
    assert abs(s_half[3]) < 5e-3  # vx at the crossing
    sol1 = cr3bp.propagate_cr3bp(KLMR_IC, tf=KLMR_T)
    assert np.linalg.norm(sol1.y[:3, -1] - KLMR_IC[:3]) < 0.05


def test_uxx_symmetric_and_matches_gradient_fd():
    x, y, z = 0.7, 0.25, -0.1
    U = cr3bp.uxx_matrix(x, y, z, MU)
    assert np.allclose(U, U.T)
    h = 1e-6
    fd = np.zeros((3, 3))
    base = np.array([x, y, z])
    for j in range(3):
        dp, dm = base.copy(), base.copy()
        dp[j] += h
        dm[j] -= h
        gp = np.array(cr3bp.cr3bp_accel(*dp, MU))
        gm = np.array(cr3bp.cr3bp_accel(*dm, MU))
        fd[:, j] = (gp - gm) / (2 * h)
    assert np.max(np.abs(U - fd)) / np.max(np.abs(U)) < 1e-7


def test_stm_derivative_matches_finite_differences():
    s = np.array([0.5, 0.2, 0.1, 0.1, -0.3, 0.05])
    s42 = np.concatenate([s, np.eye(6).ravel()])
    A = cr3bp.cr3bp_eom_stm(0.0, s42)[6:].reshape(6, 6)
    h = 1e-6
    A_fd = np.zeros((6, 6))
    for j in range(6):
        dp, dm = s.copy(), s.copy()
        dp[j] += h
        dm[j] -= h
        A_fd[:, j] = (cr3bp.cr3bp_eom(0.0, dp) - cr3bp.cr3bp_eom(0.0, dm)) / (2 * h)
    rel = np.max(np.abs(A - A_fd)) / np.max(np.abs(A))
    assert rel < 1e-6, f"A-matrix rel err {rel:.2e}"


def test_propagated_stm_matches_trajectory_sensitivity():
    tf = 1.0
    sol = cr3bp.propagate_cr3bp(KLMR_IC, tf=tf, stm=True)
    phi = sol.y[6:, -1].reshape(6, 6)
    h = 1e-7
    phi_fd = np.zeros((6, 6))
    for j in range(6):
        dp, dm = KLMR_IC.copy(), KLMR_IC.copy()
        dp[j] += h
        dm[j] -= h
        yp = cr3bp.propagate_cr3bp(dp, tf=tf).y[:, -1]
        ym = cr3bp.propagate_cr3bp(dm, tf=tf).y[:, -1]
        phi_fd[:, j] = (yp - ym) / (2 * h)
    rel = np.max(np.abs(phi - phi_fd)) / np.max(np.abs(phi))
    assert rel < 1e-6, f"STM vs FD rel err {rel:.2e}"
    # symplectic: det Φ = 1
    assert abs(np.linalg.det(phi) - 1.0) < 1e-8


def test_zero_velocity_curve_forbidden_region():
    L = cr3bp.lagrange_points()
    C_L1 = cr3bp.jacobi(np.concatenate([L[0], np.zeros(3)]))
    X, Y, forbidden = cr3bp.zero_velocity_curve(C_L1 + 0.01, MU, n=301)
    assert forbidden.shape == (301, 301)
    # with C slightly above C(L1) the L1 neck is closed: the point L1 itself is forbidden
    i = np.argmin(np.abs(Y[:, 0]))
    j = np.argmin(np.abs(X[0] - L[0, 0]))
    assert forbidden[i, j]
    # Earth's and Moon's neighbourhoods are always allowed
    assert not forbidden[i, np.argmin(np.abs(X[0] - (-MU + 0.05)))]
    assert not forbidden[i, np.argmin(np.abs(X[0] - (1 - MU + 0.02)))]
    _, _, open_neck = cr3bp.zero_velocity_curve(C_L1 - 0.01, MU, n=301)
    assert not open_neck[i, j]


@pytest.mark.parametrize("method", ["DOP853", "Radau"])
def test_propagator_methods_agree(method):
    sol = cr3bp.propagate_cr3bp(KLMR_IC, tf=1.0, method=method, rtol=1e-10, atol=1e-10)
    ref = cr3bp.propagate_cr3bp(KLMR_IC, tf=1.0)
    assert np.linalg.norm(sol.y[:3, -1] - ref.y[:3, -1]) < 1e-7
