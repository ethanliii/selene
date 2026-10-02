"""Stacked propagation of many trajectories under the ephemeris model with ONE ``solve_ivp``.

Why
---
The UKF propagates 13 sigma points and the particle cloud propagates thousands of samples between
the same two epochs.  Calling :func:`selene.dynamics.ephemeris.propagate_ephemeris` once per
trajectory costs ≈ 1–3 ms of Python/scipy overhead each; stacking all of them into one
``6·n``-dimensional system evaluated by a single numba kernel removes that overhead and — more
importantly for the UKF — makes every sigma point see the *same* step sequence, so the (tiny)
integration errors are common-mode and cancel in the weighted sigma-point covariance.  With the
scaled-UKF default α = 1e-3 the central weight is W₀ ≈ −10⁶, so uncorrelated per-trajectory
errors of 1e-6 km would otherwise be amplified to km-level covariance noise.

Force model
-----------
Identical to :mod:`selene.dynamics.ephemeris` (Earth point mass + Moon/Sun direct and indirect
terms from the :class:`~selene.dynamics.ephemeris.BodyCache` spline, optional cannonball SRP with
cylindrical Earth/Moon shadow), re-using its compiled kernels so the two code paths agree to
round-off (checked in ``tests/test_od_particles.py``).

Error control
-------------
``solve_ivp`` controls the RMS error over *all* ``6·n`` components, so an individual trajectory
may locally exceed ``rtol``; for sigma points (nearly identical trajectories) this is irrelevant.
For a dispersed particle cloud we use rtol = atol = 1e-9: measured against an individual DOP853
propagation at 1e-12, the per-particle 7-day position mismatch of a 2000-particle, 500 km / 5 m/s
cloud is 1–200 m on SIM-DRO-01 (worst outlying particle 0.19 km) and ≤ 1 m on SIM-NRHO-RELAY-01,
i.e. sub-km, three to four orders of magnitude below the km-scale cloud it describes (not
"metre-level" for every particle).  Tighten ``rtol`` if individual particle trajectories matter.
"""
from __future__ import annotations

from dataclasses import replace

import numpy as np
from numba import njit
from scipy.integrate import solve_ivp

from selene.constants import AU_KM, GM_EARTH, GM_MOON, GM_SUN, P_SRP_1AU, R_EARTH, R_MOON
from selene.dynamics.ephemeris import (
    BodyCache,
    EphemParams,
    _cyl_shadow,
    _point_mass_accel,
    _spline_eval,
)

__all__ = ["propagate_many", "ensure_cache", "StackedModel"]

_SRP_SCALE = P_SRP_1AU * 1e-3  # km/s² per (m²/kg)
_ZERO3 = np.zeros(3)


@njit(cache=True)
def _stacked_rhs(t, y, n, c_moon, c_sun, x0, dx, ng, use_moon, use_sun, use_srp, srp_coef):
    out = np.empty_like(y)
    r_b = np.empty((2, 3))
    gm_b = np.empty(2)
    k = 0
    r_moon = np.empty(3)
    r_sun = np.empty(3)
    _spline_eval(c_moon, x0, dx, ng, t, r_moon)
    _spline_eval(c_sun, x0, dx, ng, t, r_sun)
    if use_moon:
        r_b[k, :] = r_moon
        gm_b[k] = GM_MOON
        k += 1
    if use_sun:
        r_b[k, :] = r_sun
        gm_b[k] = GM_SUN
        k += 1
    rb = r_b[:k]
    gb = gm_b[:k]
    acc = np.empty(3)
    r = np.empty(3)
    zero = np.zeros(3)
    for i in range(n):
        b = 6 * i
        r[0] = y[b]
        r[1] = y[b + 1]
        r[2] = y[b + 2]
        out[b] = y[b + 3]
        out[b + 1] = y[b + 4]
        out[b + 2] = y[b + 5]
        _point_mass_accel(r, rb, gb, GM_EARTH, acc)
        if use_srp:
            nu = _cyl_shadow(r, zero, R_EARTH, r_sun) * _cyl_shadow(r, r_moon, R_MOON, r_sun)
            if nu > 0.0:
                d0 = r[0] - r_sun[0]
                d1 = r[1] - r_sun[1]
                d2 = r[2] - r_sun[2]
                dn = np.sqrt(d0 * d0 + d1 * d1 + d2 * d2)
                f = nu * srp_coef * (AU_KM / dn) ** 2 / dn
                acc[0] += f * d0
                acc[1] += f * d1
                acc[2] += f * d2
        out[b + 3] = acc[0]
        out[b + 4] = acc[1]
        out[b + 5] = acc[2]
    return out


def ensure_cache(params: EphemParams | None, t_lo: float, t_hi: float, margin_s: float = 3600.0) -> EphemParams:
    """Return ``params`` with a :class:`BodyCache` covering ``[t_lo, t_hi]`` (built if needed)."""
    params = params or EphemParams()
    lo, hi = min(t_lo, t_hi) - margin_s, max(t_lo, t_hi) + margin_s
    if params.cache is None or not (params.cache.covers(lo) and params.cache.covers(hi)):
        params = replace(params, cache=BodyCache(lo, hi))
    return params


class StackedModel:
    """Bound numba RHS for a given force-model configuration (cache coefficients captured once)."""

    def __init__(self, params: EphemParams):
        if params.cache is None:
            raise ValueError("StackedModel needs params.cache (use ensure_cache)")
        c = params.cache
        self.params = params
        self._args = (
            c._coef["moon"], c._coef["sun"], c.t0_s, c.dt_s, c.n,
            "moon" in params.bodies, "sun" in params.bodies,
            bool(params.srp and params.cr_area_mass > 0.0), _SRP_SCALE * float(params.cr_area_mass),
        )

    def rhs(self, t, y, n):
        return _stacked_rhs(t, y, n, *self._args)

    def accel_one(self, t_s: float, r: np.ndarray) -> np.ndarray:
        y = np.concatenate([np.asarray(r, dtype=np.float64), np.zeros(3)])
        return self.rhs(float(t_s), y, 1)[3:]


def propagate_many(
    states0: np.ndarray,
    t0_s: float,
    t_eval_s,
    params: EphemParams | None = None,
    rtol: float = 1e-10,
    atol: float = 1e-10,
    method: str = "DOP853",
    max_step: float = np.inf,
) -> np.ndarray:
    """Propagate ``states0`` (n,6) [GCRF km, km/s] from ``t0_s`` to each absolute TDB time in
    ``t_eval_s`` (monotone, all on one side of ``t0_s``) -> array (T, n, 6).

    Returns copies of the input when every requested time equals ``t0_s``.
    """
    s0 = np.asarray(states0, dtype=np.float64)
    squeeze = s0.ndim == 1
    s0 = s0.reshape(-1, 6)
    n = s0.shape[0]
    t_eval = np.atleast_1d(np.asarray(t_eval_s, dtype=np.float64))
    T = t_eval.size
    out = np.empty((T, n, 6))
    same = np.isclose(t_eval, t0_s, rtol=0.0, atol=1e-9)
    if same.all():
        out[:] = s0[None]
        return out[:, 0] if squeeze else out
    if T > 1:
        d = np.diff(t_eval)
        if not (np.all(d >= 0) or np.all(d <= 0)):
            raise ValueError("t_eval_s must be monotone")
    tf = float(t_eval[-1]) if abs(t_eval[-1] - t0_s) >= abs(t_eval[0] - t0_s) else float(t_eval[0])
    if np.any((t_eval - t0_s) * (tf - t0_s) < -1e-9):
        raise ValueError("t_eval_s must all lie on one side of t0_s")
    params = ensure_cache(params, t0_s, tf)
    model = StackedModel(params)
    ask, inverse = np.unique(t_eval[~same], return_inverse=True)   # strictly monotone, de-duplicated
    if tf < t0_s:
        ask, inverse = ask[::-1], (ask.size - 1 - inverse)
    ask_full = np.concatenate([[t0_s], ask])
    sol = solve_ivp(model.rhs, (float(t0_s), tf), s0.ravel(), method=method, t_eval=ask_full,
                    args=(n,), rtol=rtol, atol=atol, max_step=max_step)
    if not sol.success:
        raise RuntimeError(f"stacked propagation failed: {sol.message}")
    y = sol.y.T.reshape(len(ask_full), n, 6)[1:]
    out[same] = s0[None]
    out[~same] = y[inverse]
    return out[:, 0] if squeeze else out
