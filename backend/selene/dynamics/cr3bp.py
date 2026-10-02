"""Earth-Moon circular restricted three-body problem (CR3BP).

Frame and units
---------------
Rotating (synodic) barycentric frame, nondimensional units (PLAN §2.1):

* length unit  L* = 384 400 km  (``constants.L_STAR``)
* time unit    T* = sqrt(L*^3 / (GM_E + GM_M)) ≈ 3.7519e5 s  (``constants.T_STAR``);
  the primaries complete one revolution in 2π time units
* mass ratio   μ = GM_M / (GM_E + GM_M) ≈ 0.0121506  (``constants.MU``)

Earth sits at (-μ, 0, 0), Moon at (1-μ, 0, 0).  The x-axis points from Earth to Moon, the
z-axis is along the orbital angular momentum, y completes the right-handed triad.

Equations of motion (pseudo-potential U = ½(x²+y²) + (1-μ)/r₁ + μ/r₂)::

    ẍ - 2ẏ = ∂U/∂x,   ÿ + 2ẋ = ∂U/∂y,   z̈ = ∂U/∂z

Jacobi constant  C = 2U - (ẋ² + ẏ² + ż²)  is the single integral of motion and is used as the
primary numerical-accuracy diagnostic.

State transition matrix: Φ̇ = A(t) Φ with A = [[0, I], [U_xx, 2Ω]], Ω = [[0,1,0],[-1,0,0],[0,0,0]].

References
----------
* Szebehely, V. (1967), *Theory of Orbits*, Academic Press, ch. 1-4.
* Koon, Lo, Marsden, Ross (2011), *Dynamical Systems, the Three-Body Problem and Space Mission
  Design*, ch. 2 (equations, Jacobi constant, Lagrange points) and ch. 4 (L1 Lyapunov example).
"""
from __future__ import annotations

from typing import Callable

import numpy as np
from numba import njit
from scipy.integrate import solve_ivp
from scipy.optimize import brentq

from selene.constants import MU

__all__ = [
    "cr3bp_accel",
    "cr3bp_eom",
    "uxx_matrix",
    "cr3bp_eom_stm",
    "jacobi",
    "pseudo_potential",
    "lagrange_points",
    "propagate_cr3bp",
    "y_zero_crossing_event",
    "zero_velocity_curve",
]


# ---------------------------------------------------------------------------
# numba kernels
# ---------------------------------------------------------------------------
@njit(cache=True)
def cr3bp_accel(x: float, y: float, z: float, mu: float):
    """Rotating-frame acceleration (ẍ, ÿ, z̈) at position (x, y, z) for zero velocity terms
    excluded, i.e. the gradient of the pseudo-potential U (Coriolis added in :func:`cr3bp_eom`).

    Returns ``(ax, ay, az)`` = (∂U/∂x, ∂U/∂y, ∂U/∂z), nondimensional.
    """
    dx1 = x + mu
    dx2 = x - 1.0 + mu
    r1_sq = dx1 * dx1 + y * y + z * z
    r2_sq = dx2 * dx2 + y * y + z * z
    r1_3 = r1_sq * np.sqrt(r1_sq)
    r2_3 = r2_sq * np.sqrt(r2_sq)
    k1 = (1.0 - mu) / r1_3
    k2 = mu / r2_3
    ax = x - k1 * dx1 - k2 * dx2
    ay = y - k1 * y - k2 * y
    az = -k1 * z - k2 * z
    return ax, ay, az


@njit(cache=True)
def _cr3bp_eom_core(s, mu, out):
    ax, ay, az = cr3bp_accel(s[0], s[1], s[2], mu)
    out[0] = s[3]
    out[1] = s[4]
    out[2] = s[5]
    out[3] = ax + 2.0 * s[4]
    out[4] = ay - 2.0 * s[3]
    out[5] = az


@njit(cache=True)
def _uxx_core(x, y, z, mu, out):
    dx1 = x + mu
    dx2 = x - 1.0 + mu
    r1_sq = dx1 * dx1 + y * y + z * z
    r2_sq = dx2 * dx2 + y * y + z * z
    r1 = np.sqrt(r1_sq)
    r2 = np.sqrt(r2_sq)
    r1_3 = r1_sq * r1
    r2_3 = r2_sq * r2
    r1_5 = r1_3 * r1_sq
    r2_5 = r2_3 * r2_sq
    m1 = 1.0 - mu
    m2 = mu
    c = -m1 / r1_3 - m2 / r2_3
    a1 = 3.0 * m1 / r1_5
    a2 = 3.0 * m2 / r2_5
    out[0, 0] = 1.0 + c + a1 * dx1 * dx1 + a2 * dx2 * dx2
    out[1, 1] = 1.0 + c + a1 * y * y + a2 * y * y
    out[2, 2] = c + a1 * z * z + a2 * z * z
    uxy = a1 * dx1 * y + a2 * dx2 * y
    uxz = a1 * dx1 * z + a2 * dx2 * z
    uyz = a1 * y * z + a2 * y * z
    out[0, 1] = uxy
    out[1, 0] = uxy
    out[0, 2] = uxz
    out[2, 0] = uxz
    out[1, 2] = uyz
    out[2, 1] = uyz


@njit(cache=True)
def _cr3bp_eom_stm_core(s42, mu, out):
    # state derivative
    _cr3bp_eom_core(s42[:6], mu, out[:6])
    # A matrix
    uxx = np.zeros((3, 3))
    _uxx_core(s42[0], s42[1], s42[2], mu, uxx)
    A = np.zeros((6, 6))
    for i in range(3):
        A[i, i + 3] = 1.0
        for j in range(3):
            A[i + 3, j] = uxx[i, j]
    A[3, 4] = 2.0
    A[4, 3] = -2.0
    # Phi_dot = A Phi, row-major
    for i in range(6):
        for j in range(6):
            acc = 0.0
            for k in range(6):
                acc += A[i, k] * s42[6 + 6 * k + j]
            out[6 + 6 * i + j] = acc


# ---------------------------------------------------------------------------
# Python-facing API
# ---------------------------------------------------------------------------
def cr3bp_eom(t: float, s: np.ndarray, mu: float = MU) -> np.ndarray:
    """CR3BP equations of motion, ``solve_ivp``-compatible.  ``s = (x, y, z, vx, vy, vz)`` nondim."""
    out = np.empty(6)
    _cr3bp_eom_core(np.asarray(s, dtype=np.float64), mu, out)
    return out


def uxx_matrix(x: float, y: float, z: float, mu: float = MU) -> np.ndarray:
    """Hessian of the pseudo-potential U at (x, y, z); symmetric 3×3."""
    out = np.zeros((3, 3))
    _uxx_core(float(x), float(y), float(z), mu, out)
    return out


def cr3bp_eom_stm(t: float, s42: np.ndarray, mu: float = MU) -> np.ndarray:
    """State + STM derivative.  ``s42 = [state(6), Φ.ravel() (36, row-major)]``."""
    out = np.empty(42)
    _cr3bp_eom_stm_core(np.asarray(s42, dtype=np.float64), mu, out)
    return out


def pseudo_potential(s: np.ndarray, mu: float = MU) -> np.ndarray | float:
    """U = ½(x²+y²) + (1-μ)/r₁ + μ/r₂ for (6,), (N,6), (3,) or (N,3) inputs."""
    s = np.asarray(s, dtype=np.float64)
    x, y, z = s[..., 0], s[..., 1], s[..., 2]
    r1 = np.sqrt((x + mu) ** 2 + y**2 + z**2)
    r2 = np.sqrt((x - 1.0 + mu) ** 2 + y**2 + z**2)
    return 0.5 * (x**2 + y**2) + (1.0 - mu) / r1 + mu / r2


def jacobi(s: np.ndarray, mu: float = MU) -> np.ndarray | float:
    """Jacobi constant C = 2U - v².  Accepts (6,) or (N,6); returns float or (N,)."""
    s = np.asarray(s, dtype=np.float64)
    v2 = s[..., 3] ** 2 + s[..., 4] ** 2 + s[..., 5] ** 2
    C = 2.0 * pseudo_potential(s, mu) - v2
    return float(C) if C.ndim == 0 else C


def _collinear_f(x: float, mu: float) -> float:
    """∂U/∂x on the x-axis (y = z = 0)."""
    d1 = x + mu
    d2 = x - 1.0 + mu
    return x - (1.0 - mu) * d1 / abs(d1) ** 3 - mu * d2 / abs(d2) ** 3


def lagrange_points(mu: float = MU) -> np.ndarray:
    """L1..L5 positions in the rotating frame, shape (5, 3), nondimensional.

    Collinear points: roots of ∂U/∂x = 0 on the x-axis (equivalent to the classical quintic),
    bracketed by the primaries and solved with Brent's method to machine precision.
    Equilateral points: L4/L5 = (½ - μ, ±√3/2, 0).

    For μ = 0.012150 (Earth-Moon): L1 ≈ 0.836915, L2 ≈ 1.155682, L3 ≈ -1.005063
    (Koon-Lo-Marsden-Ross 2011, Table 2.1 / Szebehely 1967).
    """
    eps = 1e-9
    L1 = brentq(_collinear_f, -mu + eps, 1.0 - mu - eps, args=(mu,), xtol=1e-15, rtol=4 * np.finfo(float).eps)
    L2 = brentq(_collinear_f, 1.0 - mu + eps, 2.0, args=(mu,), xtol=1e-15, rtol=4 * np.finfo(float).eps)
    L3 = brentq(_collinear_f, -2.0, -mu - eps, args=(mu,), xtol=1e-15, rtol=4 * np.finfo(float).eps)
    h = np.sqrt(3.0) / 2.0
    return np.array(
        [
            [L1, 0.0, 0.0],
            [L2, 0.0, 0.0],
            [L3, 0.0, 0.0],
            [0.5 - mu, h, 0.0],
            [0.5 - mu, -h, 0.0],
        ]
    )


def y_zero_crossing_event(direction: int = 0) -> Callable:
    """Terminal ``solve_ivp`` event firing when y = 0 (xz-plane crossing).

    ``direction`` = 0 (any), +1 (y increasing), -1 (y decreasing).  Used by symmetric
    single-shooting (PLAN §2.4): integrate from the x-axis to the next crossing.
    """

    def event(t, s, *args):
        return s[1]

    event.terminal = True
    event.direction = direction
    return event


def propagate_cr3bp(
    s0: np.ndarray,
    tf: float | None = None,
    t_eval: np.ndarray | None = None,
    mu: float = MU,
    stm: bool = False,
    rtol: float = 1e-12,
    atol: float = 1e-12,
    method: str = "DOP853",
    events=None,
    dense_output: bool = False,
    max_step: float = np.inf,
):
    """Integrate the CR3BP from nondimensional state ``s0`` (6,) for time ``tf`` (nondim).

    If ``stm=True`` the 6×6 STM (initialised to the identity unless ``s0`` already has 42
    entries) is integrated alongside the state and ``sol.y`` has 42 rows.  ``t_eval`` may be
    given instead of ``tf`` (then ``tf = t_eval[-1]``).  Returns the scipy ``OdeResult``.
    """
    s0 = np.asarray(s0, dtype=np.float64).ravel()
    if t_eval is not None:
        t_eval = np.asarray(t_eval, dtype=np.float64)
        if tf is None:
            tf = float(t_eval[-1])
    if tf is None:
        raise ValueError("propagate_cr3bp needs tf or t_eval")
    t0 = 0.0 if t_eval is None else float(t_eval[0])
    if stm:
        if s0.size == 6:
            s0 = np.concatenate([s0, np.eye(6).ravel()])
        fun = cr3bp_eom_stm
    else:
        s0 = s0[:6]
        fun = cr3bp_eom
    return solve_ivp(
        fun,
        (t0, float(tf)),
        s0,
        method=method,
        t_eval=t_eval,
        args=(mu,),
        rtol=rtol,
        atol=atol,
        events=events,
        dense_output=dense_output,
        max_step=max_step,
    )


def zero_velocity_curve(
    C: float,
    mu: float = MU,
    n: int = 400,
    xlim: tuple[float, float] = (-1.5, 1.5),
    ylim: tuple[float, float] = (-1.5, 1.5),
):
    """Forbidden region in the xy-plane for Jacobi constant ``C`` (visualisation helper).

    Returns ``(X, Y, forbidden)`` where ``forbidden = 2U(x, y, 0) < C`` (motion impossible
    because v² = 2U - C would be negative).  Shapes (n, n).
    """
    xs = np.linspace(*xlim, n)
    ys = np.linspace(*ylim, n)
    X, Y = np.meshgrid(xs, ys)
    pts = np.stack([X.ravel(), Y.ravel(), np.zeros(X.size)], axis=1)
    U = pseudo_potential(pts, mu).reshape(X.shape)
    return X, Y, (2.0 * U) < C
