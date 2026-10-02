"""Analytic approximations of libration-point orbits used as corrector seeds.

* :func:`lyapunov_linear_ic` -- the linearised in-plane periodic solution about a collinear
  point (first-order term of the Lindstedt-Poincaré expansion).
* :func:`richardson_halo_ic` -- Richardson's (1980) third-order halo-orbit approximation.

Richardson, D. L. (1980), "Analytic construction of periodic orbits about the collinear
points", Celestial Mechanics 22, 241-253.  The expansion is written in a frame centred at the
libration point L_i with lengths scaled by γ (distance from L_i to the smaller primary) and
the same time unit as the CR3BP.  The coefficients below are Richardson's eqs. (10)-(24)
(also reproduced in Koon, Lo, Marsden & Ross 2011 §6.4 and Thurman & Worfolk 1996).

Conventions in this module
--------------------------
* Inputs/outputs are in the standard CR3BP nondimensional rotating frame (L* = 384 400 km):
  the amplitude ``Az`` is given in L* units and converted to Richardson units internally.
* The rotating x-axis points from Earth to Moon for both L1 and L2; Richardson's
  ``c_n`` formulas differ between L1 and L2 to account for that (eq. 8 of the paper).
* Branch labelling follows the geometric rule used throughout the SELENE library (and by
  the JPL catalogue data we validate against): ``'N'`` = z > 0 at the perpendicular
  xz-plane crossing that is *farther from the Moon*; ``'S'`` = z < 0 there.  For the L2
  family this corresponds to Richardson's class with δ_n = -1 (his "southern"), because
  his τ₁ = 0 point lies on the Moon side of L2.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from selene.constants import MU
from selene.dynamics.cr3bp import lagrange_points

__all__ = ["RichardsonConstants", "richardson_constants", "richardson_halo_ic", "lyapunov_linear_ic"]


@dataclass
class RichardsonConstants:
    libr: int
    gamma: float          # L_i - Moon distance [L*]
    x_L: float            # libration point x [L*]
    c2: float
    c3: float
    c4: float
    lam: float            # in-plane linear frequency λ (Richardson's λ)
    k: float
    delta: float          # Δ = λ² - c2
    coef: dict            # a21..d32, s1, s2, l1, l2


def _cn(n: int, libr: int, gamma: float, mu: float) -> float:
    """Richardson (1980) eq. (8)."""
    if libr == 1:
        return (mu + (-1) ** n * (1 - mu) * gamma ** (n + 1) / (1 - gamma) ** (n + 1)) / gamma**3
    if libr == 2:
        return ((-1) ** n * mu + (-1) ** n * (1 - mu) * gamma ** (n + 1) / (1 + gamma) ** (n + 1)) / gamma**3
    raise ValueError("Richardson expansion implemented for L1 and L2 only")


def richardson_constants(libr: int, mu: float = MU) -> RichardsonConstants:
    L = lagrange_points(mu)
    xL = float(L[libr - 1, 0])
    xm = 1.0 - mu
    gamma = abs(xL - xm)
    c2, c3, c4 = (_cn(n, libr, gamma, mu) for n in (2, 3, 4))
    # λ: positive real root of λ⁴ + (c2 - 2) λ² - (c2 - 1)(1 + 2 c2) = 0
    lam2 = ((2.0 - c2) + np.sqrt((c2 - 2.0) ** 2 + 4.0 * (c2 - 1.0) * (1.0 + 2.0 * c2))) / 2.0
    lam = np.sqrt(lam2)
    k = 2.0 * lam / (lam2 + 1.0 - c2)
    delta = lam2 - c2
    d1 = (3.0 * lam2 / k) * (k * (6.0 * lam2 - 1.0) - 2.0 * lam)
    d2 = (8.0 * lam2 / k) * (k * (11.0 * lam2 - 1.0) - 2.0 * lam)
    a21 = 3.0 * c3 * (k**2 - 2.0) / (4.0 * (1.0 + 2.0 * c2))
    a22 = 3.0 * c3 / (4.0 * (1.0 + 2.0 * c2))
    a23 = -(3.0 * c3 * lam / (4.0 * k * d1)) * (3.0 * k**3 * lam - 6.0 * k * (k - lam) + 4.0)
    a24 = -(3.0 * c3 * lam / (4.0 * k * d1)) * (2.0 + 3.0 * k * lam)
    b21 = -(3.0 * c3 * lam / (2.0 * d1)) * (3.0 * k * lam - 4.0)
    b22 = 3.0 * c3 * lam / d1
    d21 = -c3 / (2.0 * lam2)
    a31 = -(9.0 * lam / (4.0 * d2)) * (4.0 * c3 * (k * a23 - b21) + k * c4 * (4.0 + k**2)) + (
        (9.0 * lam2 + 1.0 - c2) / (2.0 * d2)
    ) * (3.0 * c3 * (2.0 * a23 - k * b21) + c4 * (2.0 + 3.0 * k**2))
    a32 = -(1.0 / d2) * (
        (9.0 * lam / 4.0) * (4.0 * c3 * (k * a24 - b22) + k * c4)
        + 1.5 * (9.0 * lam2 + 1.0 - c2) * (c3 * (k * b22 + d21 - 2.0 * a24) - c4)
    )
    b31 = (3.0 / (8.0 * d2)) * (
        8.0 * lam * (3.0 * c3 * (k * b21 - 2.0 * a23) - c4 * (2.0 + 3.0 * k**2))
        + (9.0 * lam2 + 1.0 + 2.0 * c2) * (4.0 * c3 * (k * a23 - b21) + k * c4 * (4.0 + k**2))
    )
    b32 = (1.0 / d2) * (
        9.0 * lam * (c3 * (k * b22 + d21 - 2.0 * a24) - c4)
        + 0.375 * (9.0 * lam2 + 1.0 + 2.0 * c2) * (4.0 * c3 * (k * a24 - b22) + k * c4)
    )
    d31 = (3.0 / (64.0 * lam2)) * (4.0 * c3 * a24 + c4)
    d32 = (3.0 / (64.0 * lam2)) * (4.0 * c3 * (a23 - d21) + c4 * (4.0 + k**2))
    s1 = (1.0 / (2.0 * lam * (lam * (1.0 + k**2) - 2.0 * k))) * (
        1.5 * c3 * (2.0 * a21 * (k**2 - 2.0) - a23 * (k**2 + 2.0) - 2.0 * k * b21)
        - 0.375 * c4 * (3.0 * k**4 - 8.0 * k**2 + 8.0)
    )
    s2 = (1.0 / (2.0 * lam * (lam * (1.0 + k**2) - 2.0 * k))) * (
        1.5 * c3 * (2.0 * a22 * (k**2 - 2.0) + a24 * (k**2 + 2.0) + 2.0 * k * b22 + 5.0 * d21)
        + 0.375 * c4 * (12.0 - k**2)
    )
    a1 = -1.5 * c3 * (2.0 * a21 + a23 + 5.0 * d21) - 0.375 * c4 * (12.0 - k**2)
    a2 = 1.5 * c3 * (a24 - 2.0 * a22) + 1.125 * c4
    l1 = a1 + 2.0 * lam2 * s1
    l2 = a2 + 2.0 * lam2 * s2
    coef = dict(
        a21=a21, a22=a22, a23=a23, a24=a24, b21=b21, b22=b22, d21=d21, a31=a31, a32=a32,
        b31=b31, b32=b32, d31=d31, d32=d32, s1=s1, s2=s2, l1=l1, l2=l2, a1=a1, a2=a2,
    )
    return RichardsonConstants(libr, gamma, xL, c2, c3, c4, float(lam), float(k), float(delta), coef)


def _third_order_state(rc: RichardsonConstants, Ax: float, Az: float, dn: float, tau1: float):
    """Position and velocity of the third-order solution at phase τ₁ (Richardson units,
    origin at L_i).  Returns (x, y, z, vx, vy, vz)."""
    c = rc.coef
    lam, k = rc.lam, rc.k
    om = 1.0 + c["s1"] * Ax**2 + c["s2"] * Az**2
    w = lam * om
    ct, st = np.cos(tau1), np.sin(tau1)
    c2t, s2t = np.cos(2 * tau1), np.sin(2 * tau1)
    c3t, s3t = np.cos(3 * tau1), np.sin(3 * tau1)
    x = c["a21"] * Ax**2 + c["a22"] * Az**2 - Ax * ct + (c["a23"] * Ax**2 - c["a24"] * Az**2) * c2t + (
        c["a31"] * Ax**3 - c["a32"] * Ax * Az**2
    ) * c3t
    y = k * Ax * st + (c["b21"] * Ax**2 - c["b22"] * Az**2) * s2t + (c["b31"] * Ax**3 - c["b32"] * Ax * Az**2) * s3t
    z = dn * Az * ct + dn * c["d21"] * Ax * Az * (c2t - 3.0) + dn * (c["d32"] * Az * Ax**2 - c["d31"] * Az**3) * c3t
    vx = w * (Ax * st - 2.0 * (c["a23"] * Ax**2 - c["a24"] * Az**2) * s2t - 3.0 * (c["a31"] * Ax**3 - c["a32"] * Ax * Az**2) * s3t)
    vy = w * (k * Ax * ct + 2.0 * (c["b21"] * Ax**2 - c["b22"] * Az**2) * c2t + 3.0 * (c["b31"] * Ax**3 - c["b32"] * Ax * Az**2) * c3t)
    vz = w * (-dn * Az * st - 2.0 * dn * c["d21"] * Ax * Az * s2t - 3.0 * dn * (c["d32"] * Az * Ax**2 - c["d31"] * Az**3) * s3t)
    return np.array([x, y, z, vx, vy, vz]), 2.0 * np.pi / w


def richardson_halo_ic(libr: int, Az_nd: float, branch: str = "S", mu: float = MU):
    """Third-order halo seed about L1 or L2.

    Parameters
    ----------
    libr : 1 or 2
    Az_nd : out-of-plane amplitude in L* units (e.g. 0.02 ≈ 7 700 km)
    branch : 'N' or 'S' (geometric convention, see module docstring)

    Returns
    -------
    ``(X0, period, info)`` with ``X0 = (x0, 0, z0, 0, vy0, 0)`` at the perpendicular crossing
    farther from the Moon (rotating frame, nondimensional), the third-order period estimate,
    and a dict with Ax, Az (Richardson units), ω and the constants.
    """
    rc = richardson_constants(libr, mu)
    c = rc.coef
    Az = Az_nd / rc.gamma
    Ax2 = -(c["l2"] * Az**2 + rc.delta) / c["l1"]
    if Ax2 <= 0:
        raise ValueError("Richardson amplitude constraint has no real Ax for this Az")
    Ax = np.sqrt(Ax2)
    xm = 1.0 - mu
    best = None
    for dn in (+1.0, -1.0):
        for tau1 in (0.0, np.pi):
            s, T = _third_order_state(rc, Ax, Az, dn, tau1)
            X = np.array([rc.x_L + rc.gamma * s[0], 0.0, rc.gamma * s[2], 0.0, rc.gamma * s[4], 0.0])
            r2 = abs(X[0] - xm)
            want = (X[2] > 0) if branch.upper() == "N" else (X[2] < 0)
            if want and (best is None or r2 > best[0]):
                best = (r2, X, T, dn, tau1)
    _, X0, T, dn, tau1 = best
    info = dict(Ax_R=float(Ax), Az_R=float(Az), delta_n=dn, tau1=float(tau1), gamma=rc.gamma, lam=rc.lam, k=rc.k,
                Ax_km=float(Ax * rc.gamma * 384400.0))
    return X0, float(T), info


def lyapunov_linear_ic(libr: int, Ax_nd: float, mu: float = MU):
    """Linearised planar Lyapunov seed ``X0 = (x_L - Ax, 0, 0, 0, k λ Ax, 0)`` at the crossing on
    the Earth side of L_i (Richardson's first-order in-plane solution x = -Ax cos λt,
    y = k Ax sin λt), and the linear period 2π/λ.  For L2 this is the inner (Moon-side)
    crossing; the caller may re-express the orbit at the far crossing after correction."""
    rc = richardson_constants(libr, mu)
    X0 = np.array([rc.x_L - Ax_nd, 0.0, 0.0, 0.0, rc.k * rc.lam * Ax_nd, 0.0])
    return X0, 2.0 * np.pi / rc.lam, rc
