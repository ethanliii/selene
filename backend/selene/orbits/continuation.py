"""Family continuation for CR3BP periodic orbits.

Two schemes (Keller 1977; Doedel et al. 2007; Pavlak 2013 §2.5):

* :func:`natural_parameter` -- one initial-state component (e.g. x0) is the family
  parameter and is stepped explicitly; the remaining free variables are predicted by linear
  extrapolation from the previous two members and corrected by a user-supplied corrector.
  The step is adaptive: halved on failure or slow convergence, grown on fast convergence.
  Fails at folds of the family with respect to the parameter.

* :func:`pseudo_arclength` -- the free-variable vector v (e.g. (x0, z0, vy0)) is stepped
  along the unit tangent τ estimated from the previous two converged members,
  v_pred = v_k + ds τ, and the corrector solves the shooting constraints together with the
  orthogonality constraint (v - v_pred)·τ = 0 (Keller's pseudo-arclength condition).  This
  follows the family through folds in any single coordinate (needed for halo → NRHO).
  Step control on the Newton iteration count.

Both return the list of converged :class:`~selene.orbits.shooting.ShootingResult` members in
order.  All units nondimensional (see ``selene.dynamics.cr3bp``).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Sequence

import numpy as np

from selene.orbits.shooting import ShootingResult

__all__ = ["ContinuationLog", "natural_parameter", "pseudo_arclength"]


@dataclass
class ContinuationLog:
    members: list = field(default_factory=list)
    steps: list = field(default_factory=list)
    failures: int = 0
    stop_reason: str = ""


def _accept_adapt(step, iters, fast_iters, slow_iters, grow, step_max):
    if iters <= fast_iters:
        return min(step * grow, step_max)
    if iters >= slow_iters:
        return step / grow
    return step


def natural_parameter(
    seed_ic,
    corrector: Callable[[np.ndarray, float | None], ShootingResult],
    param_index: int,
    p_end: float,
    step: float,
    free_index: Sequence[int],
    step_min: float | None = None,
    step_max: float | None = None,
    n_max: int = 200,
    include: Sequence[float] = (),
    stop: Callable[[ShootingResult], bool] | None = None,
    fast_iters: int = 5,
    slow_iters: int = 9,
    grow: float = 1.5,
    max_retries: int = 12,
    use_period_guess: bool = False,
    max_period_jump: float = 0.15,
) -> ContinuationLog:
    """Natural-parameter continuation of ``seed_ic[param_index]`` towards ``p_end``.

    ``corrector(X_guess, t_half_guess)`` must return a :class:`ShootingResult`; the first call
    corrects the seed itself.  ``free_index`` are the state components predicted by linear
    extrapolation (the corrector's free variables).  Parameter values listed in ``include``
    are landed on exactly when the sweep passes them.  ``stop(result)`` may terminate the
    family (the member for which it returns True is discarded).  A corrected member whose
    period differs from the previous one by more than ``max_period_jump`` (relative) is
    treated as a jump onto a different solution branch and rejected (step halved).
    """
    log = ContinuationLog()
    X = np.asarray(seed_ic, dtype=np.float64).copy()
    direction = 1.0 if p_end >= X[param_index] else -1.0
    step = abs(step)
    step_min = step_min if step_min is not None else step / 64.0
    step_max = step_max if step_max is not None else step * 4.0
    include = sorted([float(v) for v in include], reverse=(direction < 0))

    r = corrector(X, None)
    if not r.converged:
        log.stop_reason = "seed did not converge"
        return log
    if stop is not None and stop(r):
        log.stop_reason = "stop at seed"
        return log
    log.members.append(r)
    retries = 0
    while len(log.members) < n_max:
        last = log.members[-1]
        p_last = last.ic[param_index]
        if direction * (p_end - p_last) <= 1e-15:
            log.stop_reason = "reached p_end"
            break
        h = min(step, abs(p_end - p_last))
        if len(log.members) == 1:
            # no tangent yet (only the seed): take a short first step so the second member
            # provides a usable linear predictor before full-size steps
            h = min(h, 0.25 * step)
        p_next = p_last + direction * h
        # land exactly on requested parameter values
        for v in include:
            if direction * (v - p_last) > 1e-12 and direction * (p_next - v) >= -1e-12:
                p_next = v
                h = abs(v - p_last)
                break
        X = last.ic.copy()
        t_half = 0.5 * last.period
        if len(log.members) >= 2:
            prev = log.members[-2]
            dp = p_last - prev.ic[param_index]
            if abs(dp) > 0:
                frac = (p_next - p_last) / dp
                for k in free_index:
                    X[k] = last.ic[k] + frac * (last.ic[k] - prev.ic[k])
                t_half = 0.5 * (last.period + frac * (last.period - prev.period))
        X[param_index] = p_next
        r = corrector(X, t_half if use_period_guess else None)
        ok = r.converged and np.isfinite(r.closure_error)
        if ok and abs(r.period - last.period) > max_period_jump * last.period:
            ok = False  # jumped onto another branch / crossing
        if ok:
            if stop is not None and stop(r):
                log.stop_reason = "stop criterion"
                break
            log.members.append(r)
            log.steps.append(h)
            step = _accept_adapt(step, r.iterations, fast_iters, slow_iters, grow, step_max)
            retries = 0
        else:
            log.failures += 1
            retries += 1
            step = h / 2.0
            if step < step_min or retries > max_retries:
                log.stop_reason = "step underflow"
                break
    else:
        log.stop_reason = "n_max"
    return log


def pseudo_arclength(
    seeds: Sequence[ShootingResult],
    corrector: Callable[[np.ndarray, np.ndarray, np.ndarray, float], ShootingResult],
    free_index: Sequence[int],
    ds: float,
    ds_min: float | None = None,
    ds_max: float | None = None,
    n_max: int = 200,
    stop: Callable[[ShootingResult], bool] | None = None,
    fast_iters: int = 4,
    slow_iters: int = 7,
    grow: float = 1.4,
    max_retries: int = 14,
    weights: Sequence[float] | None = None,
    max_period_jump: float = 0.25,
) -> ContinuationLog:
    """Pseudo-arclength continuation starting from two converged members ``seeds`` (ordered
    in the desired direction of travel).

    ``corrector(X_pred, v_pred, tau, ds)`` must solve the shooting problem with free
    variables ``free_index`` plus the constraint ``(v - v_pred)·tau = 0`` and return a
    :class:`ShootingResult`.  ``weights`` optionally scale the free variables when forming
    the tangent (default equal weights).
    """
    log = ContinuationLog()
    free_index = list(free_index)
    w = np.ones(len(free_index)) if weights is None else np.asarray(weights, dtype=np.float64)
    ds_min = ds_min if ds_min is not None else ds / 256.0
    ds_max = ds_max if ds_max is not None else ds * 4.0
    members = list(seeds)
    if len(members) < 2:
        raise ValueError("pseudo_arclength needs two seed members")
    log.members.extend(members)
    retries = 0
    while len(log.members) < n_max:
        a, b = log.members[-2], log.members[-1]
        va = a.ic[free_index] * w
        vb = b.ic[free_index] * w
        tau = vb - va
        nrm = np.linalg.norm(tau)
        if nrm == 0:
            log.stop_reason = "zero tangent"
            break
        tau /= nrm
        v_pred = vb + ds * tau
        X = b.ic.copy()
        X[free_index] = v_pred / w
        r = corrector(X, v_pred / w, tau * w / np.linalg.norm(tau * w), ds)
        ok = r.converged and np.isfinite(r.closure_error)
        if ok:
            # reject spurious jumps onto a different solution branch
            jump = np.linalg.norm(r.ic[free_index] * w - vb)
            if jump > 3.0 * ds + 1e-9 or abs(r.period - b.period) > max_period_jump * b.period:
                ok = False
        if ok:
            if stop is not None and stop(r):
                log.stop_reason = "stop criterion"
                break
            log.members.append(r)
            log.steps.append(ds)
            ds = _accept_adapt(ds, r.iterations, fast_iters, slow_iters, grow, ds_max)
            retries = 0
        else:
            log.failures += 1
            retries += 1
            ds /= 2.0
            if ds < ds_min or retries > max_retries:
                log.stop_reason = "step underflow"
                break
    else:
        log.stop_reason = "n_max"
    return log
