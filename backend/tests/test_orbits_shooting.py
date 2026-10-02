"""Unit tests for the periodic-orbit correctors, continuation and stability (no library files)."""
import numpy as np
import pytest

from selene.constants import L_STAR, MU, T_STAR
from selene.dynamics.cr3bp import jacobi, lagrange_points, propagate_cr3bp
from selene.orbits.continuation import natural_parameter, pseudo_arclength
from selene.orbits.richardson import lyapunov_linear_ic, richardson_constants, richardson_halo_ic
from selene.orbits.shooting import (
    closure_error,
    integrate_to_crossing,
    multiple_shoot,
    shoot_symmetric,
    shoot_symmetric_period,
)
from selene.orbits.stability import stability_from_monodromy, stability_index

# Koon, Lo, Marsden & Ross (2011) §4.2, Earth-Moon L1 Lyapunov example (μ = 0.0121505):
KLMR_X0, KLMR_VY0, KLMR_T = 0.8234, 0.1263, 2.7430
# Zimovan-Spreen, Howell & Davis (2020), L2 southern 9:2 NRHO apolune IC (CR3BP, rounded):
NRHO_X0, NRHO_Z0, NRHO_VY0 = 1.0221, -0.1821, -0.1030


def test_integrate_to_crossing_skips_initial_point_and_counts():
    X0 = np.array([KLMR_X0, 0, 0, 0, KLMR_VY0, 0])
    t1, X1, Phi = integrate_to_crossing(X0, n_cross=1)
    assert t1 > 0.5 and abs(X1[1]) < 1e-12 and X1[4] < 0          # return crossing, ẏ < 0
    t2, X2, _ = integrate_to_crossing(X0, n_cross=2)
    assert t2 > t1 and X2[4] > 0
    assert Phi.shape == (6, 6) and abs(np.linalg.det(Phi) - 1) < 1e-8   # symplectic


def test_planar_shooting_recovers_klmr_lyapunov_from_perturbed_seed():
    r = shoot_symmetric([KLMR_X0, 0, 0, 0, KLMR_VY0 * 1.05, 0], mode="planar")
    assert r.converged and r.iterations <= 10
    assert abs(r.ic[0] - KLMR_X0) == 0.0                     # x0 is the fixed parameter
    assert abs(r.ic[4] - KLMR_VY0) < 1e-3
    assert abs(r.period - KLMR_T) < 1e-3
    assert r.closure_error < 1e-10
    # the corrector really closes the orbit: independent propagation
    assert closure_error(r.ic, r.period) < 1e-10


def test_fix_x0_and_fix_z0_modes_converge_to_the_same_nrho():
    X = [NRHO_X0, 0, NRHO_Z0, 0, NRHO_VY0, 0]
    a = shoot_symmetric(X, mode="fix_z0")
    b = shoot_symmetric(X, mode="fix_x0")
    assert a.converged and b.converged
    assert abs(a.period - b.period) < 2e-3
    assert abs(a.period * T_STAR / 86400 - 6.56) < 0.05
    assert a.closure_error < 1e-10 and b.closure_error < 1e-10


def test_explicit_time_formulation_matches_event_formulation():
    X = [NRHO_X0, 0, NRHO_Z0, 0, NRHO_VY0, 0]
    a = shoot_symmetric(X, mode="fix_z0")
    b = shoot_symmetric(X, mode="fix_z0", t_half=0.5 * a.period * 1.02)
    assert b.converged
    assert abs(a.period - b.period) < 1e-10
    assert np.max(np.abs(a.ic - b.ic)) < 1e-10


def test_fixed_period_shooting_hits_the_requested_period():
    T = 1.5112  # 2/9 synodic month in T* units
    r = shoot_symmetric_period([NRHO_X0, 0, NRHO_Z0, 0, NRHO_VY0, 0], T)
    assert r.converged and r.period == T
    assert r.closure_error < 1e-10
    sol = propagate_cr3bp(r.ic, tf=T, rtol=2.3e-14, atol=1e-14)
    assert np.max(np.abs(sol.y[:6, -1] - r.ic)) < 1e-10


def test_multiple_shooting_reconverges_perturbed_nrho():
    a = shoot_symmetric([NRHO_X0, 0, NRHO_Z0, 0, NRHO_VY0, 0], mode="fix_z0")
    X = a.ic.copy()
    X[4] += 2e-5                                   # ~2 cm/s kick: not periodic any more
    assert closure_error(X, a.period) > 1e-6
    ms = multiple_shoot(X, a.period * 1.001, n_patch=10, fix={2: X[2]})   # stay on this member
    assert ms.converged and ms.method == "multiple_shooting" and ms.iterations <= 12
    assert ms.closure_error < 1e-9
    assert abs(ms.period - a.period) < 1e-8
    assert ms.extra["patches"].shape == (10, 6)
    assert abs(ms.ic[1]) < 1e-14                   # phase constraint y0 = 0 enforced
    assert abs(ms.ic[2] - X[2]) < 1e-14            # pinned family parameter
    # free period: with fix_period the period is kept and the state adjusts instead
    ms2 = multiple_shoot(X, a.period, n_patch=10, fix_period=True)
    assert ms2.converged and ms2.period == a.period and ms2.closure_error < 1e-9


def test_pseudo_arclength_constraint_is_satisfied():
    X0, _, _ = richardson_halo_ic(2, 0.02, "S")
    a = shoot_symmetric(X0, mode="fix_z0")
    free = [0, 2, 4]
    tau = np.array([0.0, -1.0, 0.0])
    ds = 0.01
    v_pred = a.ic[free] + ds * tau
    Xp = a.ic.copy()
    Xp[free] = v_pred
    b = shoot_symmetric(Xp, mode="free3", arclength=(v_pred, tau, ds))
    assert b.converged
    assert abs(np.dot(b.ic[free] - v_pred, tau)) < 1e-11
    assert b.closure_error < 1e-10


# ---------------------------------------------------------------------------
def test_richardson_constants_and_seed_quality():
    rc1 = richardson_constants(1)
    rc2 = richardson_constants(2)
    L = lagrange_points()
    assert abs(rc1.gamma - ((1 - MU) - L[0, 0])) < 1e-15
    assert abs(rc2.gamma - (L[1, 0] - (1 - MU))) < 1e-15
    # Earth-Moon values quoted in the literature for these constants (e.g. Thurman & Worfolk
    # 1996 Table 1 / Koon et al. §6.4 with μ≈0.01215): c2(L1) ≈ 5.148, c2(L2) ≈ 3.190.
    assert abs(rc1.c2 - 5.1476) < 2e-3 and abs(rc2.c2 - 3.1904) < 2e-3
    assert 2.3 < rc1.lam < 2.4 and 1.85 < rc2.lam < 1.87
    # the third-order seed must be close enough for Newton to converge in a few iterations
    for libr, br in ((1, "N"), (1, "S"), (2, "N"), (2, "S")):
        X0, T_est, info = richardson_halo_ic(libr, 0.02, br)
        r = shoot_symmetric(X0, mode="fix_z0")
        assert r.converged and r.iterations <= 8, (libr, br)
        assert abs(r.period - T_est) / T_est < 0.02
        assert np.sign(r.ic[2]) == (1 if br == "N" else -1)
        # seed error small relative to the amplitude
        assert abs(r.ic[0] - X0[0]) < 0.01 and abs(r.ic[4] - X0[4]) < 0.01


def test_lyapunov_linear_seed_period_matches_linear_frequency():
    X0, T_lin, rc = lyapunov_linear_ic(1, 0.002)
    r = shoot_symmetric(X0, mode="planar")
    assert r.converged
    assert abs(r.period - T_lin) / T_lin < 5e-3  # tiny amplitude -> linear period
    # JPL catalogue: L1 Lyapunov period at vanishing amplitude 2.69158 TU
    assert abs(T_lin - 2.69158) < 1e-3


# ---------------------------------------------------------------------------
def test_stability_dro_stable_and_lyapunov_unstable():
    d = 0.1
    dro = shoot_symmetric([1 - MU - d, 0, 0, 0, np.sqrt(MU / d) + d, 0], mode="planar")
    st = stability_index(dro.ic, dro.period)
    assert st.stable and abs(st.nu - 1.0) < 1e-6
    assert abs(st.det - 1.0) < 1e-8
    assert np.all(np.abs(np.abs(st.eigenvalues) - 1.0) < 1e-5)
    lyap = shoot_symmetric([KLMR_X0, 0, 0, 0, KLMR_VY0, 0], mode="planar")
    st2 = stability_index(lyap.ic, lyap.period)
    # JPL catalogue stability index for the member at x0=0.823398: 1180.75
    assert not st2.stable and abs(st2.nu - 1180.0) / 1180.0 < 0.02
    assert st2.unit_pair_error < 1e-4   # (1,1) is a defective double eigenvalue: O(sqrt(eps*cond)) error
    assert abs(st2.det - 1.0) < 1e-6
    assert abs(np.abs(st2.eigenvalues[0]) * np.abs(st2.eigenvalues[-1]) - 1.0) < 1e-6  # reciprocal pair


def test_stability_from_monodromy_identity():
    st = stability_from_monodromy(np.eye(6))
    assert st.nu == 1.0 and st.stable


# ---------------------------------------------------------------------------
def test_natural_parameter_continuation_dro_monotone():
    d = 0.08
    X0 = np.array([1 - MU - d, 0, 0, 0, np.sqrt(MU / d) + d, 0])
    log = natural_parameter(X0, lambda X, th: shoot_symmetric(X, mode="planar"), 0, 0.80, 0.02, [4], step_max=0.02)
    xs = np.array([m.ic[0] for m in log.members])
    Ts = np.array([m.period for m in log.members])
    assert len(log.members) >= 6
    assert np.all(np.diff(xs) < 0) and np.all(np.diff(Ts) > 0)
    assert max(m.closure_error for m in log.members) < 1e-10
    assert np.ptp([jacobi(m.ic) for m in log.members]) > 0.01  # genuinely different orbits


def test_pseudo_arclength_halo_grows_amplitude():
    seeds = []
    for Az in (0.01, 0.016):
        X0, _, _ = richardson_halo_ic(2, Az, "S")
        seeds.append(shoot_symmetric(X0, mode="fix_z0"))
    log = pseudo_arclength(
        seeds,
        lambda Xp, vp, tau, ds: shoot_symmetric(Xp, mode="free3", arclength=(vp, tau, ds)),
        [0, 2, 4], ds=0.01, ds_max=0.02, n_max=8,
    )
    assert len(log.members) == 8
    zs = np.array([m.ic[2] for m in log.members])
    assert np.all(np.diff(zs) < 0)                              # southern: z0 more negative
    assert max(m.closure_error for m in log.members) < 1e-10
