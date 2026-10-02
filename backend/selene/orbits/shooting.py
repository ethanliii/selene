"""Differential correction (shooting) for periodic orbits of the Earth-Moon CR3BP.

All quantities are nondimensional rotating-frame CR3BP units (``selene.dynamics.cr3bp``):
lengths in L* = 384 400 km, time in T* ≈ 3.7519e5 s, velocities in L*/T*.

Two correctors are provided.

**Symmetric single shooting** (Howell 1984; Pavlak 2013 ch. 2).  A periodic orbit that is
symmetric about the xz-plane crosses that plane perpendicularly twice.  Starting from
``X0 = (x0, 0, z0, 0, vy0, 0)`` the trajectory is integrated to a later y = 0 crossing and
Newton's method on the state transition matrix Φ drives ``vx = vz = 0`` there.  With the
crossing located by an event, the time of flight is eliminated and the sensitivity of the
crossing state to the free initial variables is

    d X(t_c) / d v  =  Φ[:, v] - (1/ẏ(t_c)) f(X(t_c)) Φ[1, v]            (1)

(``f`` = equations of motion), which is what :func:`shoot_symmetric` uses.  Modes:

* ``'fix_x0'``  free (z0, vy0)      -- halo continuation in x0
* ``'fix_z0'``  free (x0, vy0)      -- halo continuation in z0
* ``'planar'``  free vy0            -- Lyapunov / DRO / resonant (x0 fixed, z ≡ 0)
* ``'free3'``   free (x0, z0, vy0)  -- used with a pseudo-arclength constraint (square 3×3)
* ``'free2'``   free (x0, vy0), planar, used with a pseudo-arclength constraint

When ``t_half`` is given the time of flight is kept as an explicit unknown instead of an
event (constraints y = vx [= vz] = 0 at t_half); this variant converges to the perpendicular
crossing *nearest the supplied time*, which is what multi-loop resonant orbits need.

**Multiple shooting** (Pavlak 2013 §2.4; Marchand, Howell & Wilson 2007).  N patch states and
the period are free; continuity between consecutive arcs, periodicity, and an optional phase
constraint are imposed; the (6N+1)-dimensional update is the minimum-norm Newton step.  Used
for sensitive members (near-rectilinear halos) where a single half-period arc through a
~3000 km perilune is poorly conditioned.

References
----------
* Howell, K. C. (1984), "Three-dimensional, periodic, 'halo' orbits", Celest. Mech. 32, 53-71.
* Pavlak, T. A. (2013), PhD thesis, Purdue (differential corrections, multiple shooting).
* Marchand, Howell, Wilson (2007), "Improved corrections process for constrained trajectory
  design in the n-body problem", J. Spacecraft & Rockets 44(4).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np

from selene.constants import MU
from selene.dynamics.cr3bp import cr3bp_eom, propagate_cr3bp

__all__ = [
    "ShootingResult",
    "integrate_to_crossing",
    "shoot_symmetric",
    "shoot_symmetric_period",
    "multiple_shoot",
    "closure_error",
]

Mode = Literal["fix_x0", "fix_z0", "planar", "free3", "free2"]

# free-variable column indices in the 6-state for each mode
_FREE = {
    "fix_x0": [2, 4],
    "fix_z0": [0, 4],
    "planar": [4],
    "free3": [0, 2, 4],
    "free2": [0, 4],
}
# constrained rows at the crossing (vx, vz) or (vx,) for planar
_ROWS = {
    "fix_x0": [3, 5],
    "fix_z0": [3, 5],
    "planar": [3],
    "free3": [3, 5],
    "free2": [3],
}

RTOL = 1e-12          # Newton-iteration integrations (PLAN §2.4)
ATOL = 1e-12
CLOSURE_RTOL = 2.3e-14  # independent closure check: the full-period error of a strongly unstable
CLOSURE_ATOL = 1e-14    # orbit (|λ_max| ~ 2e3) is dominated by integrator error amplification, so
                        # the check integrates at the tightest tolerance DOP853 accepts in double
                        # precision (scipy clamps rtol at 100 eps ≈ 2.2e-14).


@dataclass
class ShootingResult:
    """Outcome of a corrector run.  ``ic`` is the corrected 6-state on the xz-plane (symmetric
    shooting) or the first patch point (multiple shooting); ``period`` in nondimensional time;
    ``closure_error`` = |X(T) - X0|_inf from an independent full-period propagation."""

    ic: np.ndarray
    period: float
    converged: bool
    closure_error: float
    iterations: int
    half_state: np.ndarray | None = None
    message: str = ""
    method: str = "single_shooting"
    extra: dict = field(default_factory=dict)

    def as_tuple(self):
        return self.ic, self.period, self.converged, self.closure_error, self.iterations


# ---------------------------------------------------------------------------
# crossing integration
# ---------------------------------------------------------------------------
def _crossing_event(sign0: float, t_min: float):
    """y = 0 event that cannot fire before ``t_min`` (the start point itself lies on y = 0;
    scipy would otherwise report an event at t = 0)."""

    def ev(t, s, *args):
        if t <= t_min:
            return sign0 * 1e-30
        return s[1]

    ev.terminal = False
    ev.direction = 0
    return ev


def integrate_to_crossing(
    X0,
    mu: float = MU,
    n_cross: int = 1,
    direction: int = 0,
    t_max: float = 12.0,
    rtol: float = RTOL,
    atol: float = ATOL,
):
    """Integrate state+STM from ``X0`` to the ``n_cross``-th y = 0 crossing.

    ``direction`` = 0 (count any crossing), +1 / -1 (count only crossings with ẏ > 0 / < 0).
    Returns ``(t_c, X_c (6,), Phi_c (6,6))`` or raises ``RuntimeError`` if fewer crossings
    occur before ``t_max``.
    """
    X0 = np.asarray(X0, dtype=np.float64).ravel()[:6]
    sign0 = 1.0 if X0[4] >= 0 else -1.0
    ev = _crossing_event(sign0, 1e-6)
    sol = propagate_cr3bp(X0, tf=t_max, mu=mu, stm=True, rtol=rtol, atol=atol, events=[ev], dense_output=False)
    te = sol.t_events[0]
    ye = sol.y_events[0]
    if len(te) == 0:
        raise RuntimeError("no y=0 crossing before t_max")
    if direction != 0:
        keep = [k for k in range(len(te)) if np.sign(ye[k][4]) == np.sign(direction)]
        te = te[keep]
        ye = ye[keep]
    if len(te) < n_cross:
        raise RuntimeError(f"only {len(te)} crossings before t_max={t_max}")
    k = n_cross - 1
    Xc = ye[k][:6].copy()
    Phi = ye[k][6:].reshape(6, 6).copy()
    return float(te[k]), Xc, Phi


def _half_period_guess_tmax(X0, t_half_guess):
    return 3.0 * t_half_guess if t_half_guess else 12.0


# ---------------------------------------------------------------------------
# symmetric single shooting
# ---------------------------------------------------------------------------
def shoot_symmetric(
    X0,
    mode: Mode = "fix_x0",
    mu: float = MU,
    tol: float = 1e-12,
    max_iter: int = 25,
    n_cross: int = 1,
    direction: int | None = None,
    t_half: float | None = None,
    t_max: float | None = None,
    arclength: tuple[np.ndarray, np.ndarray, float] | None = None,
    check_closure: bool = True,
    damping: float = 1.0,
    max_step_norm: float = 0.05,
    stall_tol: float = 1e-10,
    rtol: float = RTOL,
    atol: float = ATOL,
) -> ShootingResult:
    """Symmetric single-shooting corrector (see module docstring).

    Parameters
    ----------
    X0 : initial guess ``(x0, 0, z0, 0, vy0, 0)``.
    mode : which initial variables are free (see module docstring).
    n_cross, direction : which y = 0 crossing is the half-period point (event formulation).
        ``direction=None`` selects the crossing with ẏ opposite in sign to ``vy0`` (the
        return crossing of a single loop) when ``n_cross == 1``, else any crossing.
    t_half : if given, use the explicit-time formulation and converge to the crossing nearest
        this time (needed for multi-loop resonant orbits).
    arclength : ``(v_pred, tangent, ds)`` pseudo-arclength constraint (v - v_pred)·tangent = 0
        in the free-variable space (requires ``mode`` 'free3' or 'free2').
    tol : convergence tolerance on |constraints|_inf (nondimensional velocity).
    rtol, atol : integrator tolerances for the Newton iterations (default 1e-12).  A final
        "polish" pass at the tightest tolerance (``CLOSURE_RTOL``/``CLOSURE_ATOL``) is used for
        orbits with fast, low perigee passages (resonant families): the 1e-12 integration
        error committed there is otherwise baked into the converged IC and shows up as
        ~1e-10..1e-9 of apparent closure error.

    Returns
    -------
    ShootingResult with ``ic`` (6,), ``period`` = 2 t_half, ``half_state`` the state at the
    crossing, ``closure_error`` from an independent full-period propagation (or nan if
    ``check_closure`` is False).
    """
    X = np.asarray(X0, dtype=np.float64).ravel()[:6].copy()
    X[1] = 0.0
    X[3] = 0.0
    X[5] = 0.0
    if mode in ("planar", "free2"):
        X[2] = 0.0
    free = _FREE[mode]
    rows = _ROWS[mode]
    if direction is None:
        direction = -1 if (n_cross == 1 and X[4] > 0) else (1 if n_cross == 1 else 0)
    if t_max is None:
        t_max = _half_period_guess_tmax(X, t_half)

    converged = False
    msg = ""
    t_c = np.nan
    Xc = None
    it = 0
    th = t_half
    prev_err = None
    for it in range(1, max_iter + 1):
        try:
            if th is None:
                t_c, Xc, Phi = integrate_to_crossing(X, mu, n_cross, direction, t_max, rtol=rtol, atol=atol)
            else:
                sol = propagate_cr3bp(X, tf=th, mu=mu, stm=True, rtol=rtol, atol=atol)
                if not sol.success:
                    raise RuntimeError(sol.message)
                t_c = th
                Xc = sol.y[:6, -1].copy()
                Phi = sol.y[6:, -1].reshape(6, 6)
        except RuntimeError as e:  # pragma: no cover - defensive
            msg = f"integration failed: {e}"
            break
        f = cr3bp_eom(t_c, Xc, mu)
        if th is None:
            F = Xc[rows]
            J = Phi[np.ix_(rows, free)] - np.outer(f[rows], Phi[1, free]) / Xc[4]
        else:
            # explicit time: constraints (y, vx[, vz]) ; unknowns (free..., t_half)
            crows = [1] + rows
            F = Xc[crows]
            J = np.hstack([Phi[np.ix_(crows, free)], f[crows][:, None]])
        if arclength is not None:
            v_pred, tau, ds = arclength
            v = X[free]
            F = np.append(F, np.dot(v - v_pred, tau))
            row = np.zeros(J.shape[1])
            row[: len(free)] = tau
            J = np.vstack([J, row])
        err = np.max(np.abs(F))
        if err < tol:
            converged = True
            break
        if err < stall_tol and prev_err is not None and err > 0.5 * prev_err:
            # residual is already at the level of integrator noise for this (very sensitive)
            # orbit and no longer improving: accept, the independent closure check reports quality
            converged = True
            msg = f"stalled at integrator noise (|F|={err:.1e})"
            break
        prev_err = err
        if J.shape[0] == J.shape[1]:
            try:
                dv = np.linalg.solve(J, -F)
            except np.linalg.LinAlgError:
                dv = np.linalg.lstsq(J, -F, rcond=None)[0]
        else:
            # under-determined -> minimum norm; over-determined -> least squares
            dv = np.linalg.lstsq(J, -F, rcond=None)[0]
        nrm = np.max(np.abs(dv))
        if nrm > max_step_norm:
            dv *= max_step_norm / nrm
        dv *= damping
        X[free] += dv[: len(free)]
        if th is not None:
            th += dv[-1]
            if th <= 0:
                msg = "half period became non-positive"
                break
        if not np.all(np.isfinite(X)):
            msg = "diverged"
            break
    period = 2.0 * t_c if np.isfinite(t_c) else np.nan
    ce = closure_error(X, period, mu) if (converged and check_closure) else np.nan
    return ShootingResult(
        ic=X,
        period=float(period),
        converged=converged,
        closure_error=float(ce),
        iterations=it,
        half_state=None if Xc is None else Xc,
        message=msg if msg else ("converged" if converged else "max_iter"),
        method="single_shooting",
    )



def shoot_symmetric_period(
    X0,
    period: float,
    mu: float = MU,
    tol: float = 1e-12,
    max_iter: int = 25,
    max_step_norm: float = 0.02,
) -> ShootingResult:
    """Correct a symmetric 3-D orbit to a *prescribed period*.

    Unknowns (x0, z0, vy0); constraints y = vx = vz = 0 at t = period/2 (square 3×3 Newton).
    Used to pin a family member with a given period (e.g. the 9:2 synodic-resonant NRHO).
    """
    X = np.asarray(X0, dtype=np.float64).ravel()[:6].copy()
    X[1] = X[3] = X[5] = 0.0
    free = [0, 2, 4]
    crows = [1, 3, 5]
    th = 0.5 * period
    converged = False
    it = 0
    for it in range(1, max_iter + 1):
        sol = propagate_cr3bp(X, tf=th, mu=mu, stm=True, rtol=RTOL, atol=ATOL)
        Xc = sol.y[:6, -1]
        Phi = sol.y[6:, -1].reshape(6, 6)
        F = Xc[crows]
        if np.max(np.abs(F)) < tol:
            converged = True
            break
        J = Phi[np.ix_(crows, free)]
        dv = np.linalg.solve(J, -F)
        nrm = np.max(np.abs(dv))
        if nrm > max_step_norm:
            dv *= max_step_norm / nrm
        X[free] += dv
    ce = closure_error(X, period, mu) if converged else np.nan
    return ShootingResult(X, float(period), converged, float(ce), it, method="single_shooting_fixed_period",
                          message="converged" if converged else "max_iter")


# ---------------------------------------------------------------------------
# multiple shooting
# ---------------------------------------------------------------------------
def multiple_shoot(
    X0,
    period: float,
    n_patch: int = 8,
    mu: float = MU,
    tol: float = 1e-12,
    max_iter: int = 30,
    phase: Literal["y0", "none"] = "y0",
    patches: np.ndarray | None = None,
    max_step_norm: float = 0.05,
    fix: dict | None = None,
    fix_period: bool = False,
) -> ShootingResult:
    """General multiple-shooting periodic-orbit corrector with free period.

    Unknowns: patch states ``X_0..X_{N-1}`` and the period ``T``; each arc is integrated for
    ``T/N``.  Constraints: continuity ``X_i(T/N) - X_{i+1} = 0``, periodicity
    ``X_{N-1}(T/N) - X_0 = 0`` and (``phase='y0'``) ``y_0 = 0`` which removes the phase
    freedom along the orbit.  Without further constraints the system is under-determined by
    one (the family direction) and the minimum-norm Newton step ``δ = -J⁺ F`` is used; that
    can drift along the family from a poor guess, so ``fix={index: value}`` pins components
    of the first patch state (e.g. ``{2: z0}`` for a halo) and ``fix_period=True`` pins the
    period, each making the system square.  Returns the corrected first patch state as
    ``ic``; ``extra['patches']`` holds all corrected patch states.
    """
    X0 = np.asarray(X0, dtype=np.float64).ravel()[:6]
    N = int(n_patch)
    T = float(period)
    if patches is None:
        sol = propagate_cr3bp(X0, t_eval=np.linspace(0.0, T, N + 1), mu=mu, rtol=RTOL, atol=ATOL)
        P = sol.y[:6, :N].T.copy()
    else:
        P = np.asarray(patches, dtype=np.float64).reshape(N, 6).copy()
    converged = False
    it = 0
    fix = dict(fix or {})
    T0 = T
    n_extra = (1 if phase == "y0" else 0) + len(fix) + (1 if fix_period else 0)
    nF = 6 * N + n_extra
    nV = 6 * N + 1
    for it in range(1, max_iter + 1):
        F = np.zeros(nF)
        J = np.zeros((nF, nV))
        dt = T / N
        for i in range(N):
            sol = propagate_cr3bp(P[i], tf=dt, mu=mu, stm=True, rtol=RTOL, atol=ATOL)
            Xf = sol.y[:6, -1]
            Phi = sol.y[6:, -1].reshape(6, 6)
            j = (i + 1) % N
            F[6 * i : 6 * i + 6] = Xf - P[j]
            J[6 * i : 6 * i + 6, 6 * i : 6 * i + 6] = Phi
            J[6 * i : 6 * i + 6, 6 * j : 6 * j + 6] -= np.eye(6)
            J[6 * i : 6 * i + 6, 6 * N] = cr3bp_eom(dt, Xf, mu) / N
        row = 6 * N
        if phase == "y0":
            F[row] = P[0, 1]
            J[row, 1] = 1.0
            row += 1
        for k, val in fix.items():
            F[row] = P[0, k] - val
            J[row, k] = 1.0
            row += 1
        if fix_period:
            F[row] = T - T0
            J[row, 6 * N] = 1.0
        err = np.max(np.abs(F))
        if err < tol:
            converged = True
            break
        dv = np.linalg.lstsq(J, -F, rcond=None)[0]
        nrm = np.max(np.abs(dv))
        if nrm > max_step_norm:
            dv *= max_step_norm / nrm
        P += dv[: 6 * N].reshape(N, 6)
        T += dv[-1]
        if not (np.all(np.isfinite(P)) and T > 0):
            break
    ic = P[0].copy()
    ce = closure_error(ic, T, mu) if converged else np.nan
    return ShootingResult(
        ic=ic,
        period=float(T),
        converged=converged,
        closure_error=float(ce),
        iterations=it,
        message="converged" if converged else "max_iter",
        method="multiple_shooting",
        extra={"patches": P, "n_patch": N},
    )


# ---------------------------------------------------------------------------
def closure_error(ic, period: float, mu: float = MU, rtol: float = CLOSURE_RTOL, atol: float = CLOSURE_ATOL) -> float:
    """|X(T) - X(0)|_inf from a full-period propagation (independent of the corrector).

    Integrated at ``CLOSURE_RTOL`` (2.3e-14): for the most unstable Lyapunov/halo members
    (|λ_max| ≈ 1500-2400) a 1e-12 integration alone contributes ~2e-10 of apparent closure
    error, masking the quality of the initial conditions (measured for the same converged
    L2 halo IC: 2e-10 / 2e-11 / 5e-13 at rtol 1e-12 / 1e-13 / 1e-14; for the most unstable
    L2 Lyapunov member 1.4e-10 / 5e-11 at 1e-13 / 1e-14)."""
    ic = np.asarray(ic, dtype=np.float64).ravel()[:6]
    sol = propagate_cr3bp(ic, tf=float(period), mu=mu, rtol=rtol, atol=atol)
    if not sol.success:
        return float("nan")
    return float(np.max(np.abs(sol.y[:6, -1] - ic)))
