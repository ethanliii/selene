"""High-level propagation wrapper returning a frame-aware :class:`Trajectory`.

Two models:

* ``'ephemeris'`` -- Earth + Moon + Sun point masses from DE440s (:mod:`selene.dynamics.ephemeris`),
  integrated in Earth-centered GCRF [km, km/s].
* ``'cr3bp'`` -- circular restricted three-body problem (:mod:`selene.dynamics.cr3bp`),
  integrated in the nondimensional rotating frame and then mapped to GCRF with the
  *instantaneous* rotating frame of :mod:`selene.dynamics.frames`.

**Approximation note for ``model='cr3bp'``.**  A CR3BP solution lives in an idealised circular
Earth-Moon system.  Expressing it in GCRF with the real (eccentric, Sun-perturbed) instantaneous
frame means that at each output time the state is scaled by the true Earth-Moon distance d(t)
and rotated by the true orientation; the resulting GCRF trajectory is therefore not a solution
of any single force model and should be used for visualisation, seeding and coarse planning,
not as a truth reference.  Use the ephemeris model for truth and estimation.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np
from scipy.interpolate import CubicSpline

from selene.constants import T_STAR
from selene.dynamics import frames
from selene.dynamics.cr3bp import propagate_cr3bp
from selene.dynamics.ephemeris import EphemParams, propagate_ephemeris

__all__ = ["Trajectory", "propagate", "Frame", "Model"]

Frame = Literal["rot_nd", "gcrf_km", "moon_km"]
Model = Literal["ephemeris", "cr3bp"]


@dataclass
class Trajectory:
    """Sampled trajectory.  ``t_s`` TDB seconds past J2000 (N,), ``states`` (N,6) in ``frame``:

    * ``'gcrf_km'``  Earth-centered GCRF, km and km/s
    * ``'moon_km'``  Moon-centered, GCRF axes, km and km/s
    * ``'rot_nd'``   instantaneous Earth-Moon rotating frame, nondimensional (see frames.py)
    """

    t_s: np.ndarray
    states: np.ndarray
    frame: Frame = "gcrf_km"
    meta: dict = field(default_factory=dict)

    def __post_init__(self):
        self.t_s = np.asarray(self.t_s, dtype=np.float64).ravel()
        self.states = np.asarray(self.states, dtype=np.float64).reshape(len(self.t_s), 6)
        self._interp = None

    # -- accessors -------------------------------------------------------
    @property
    def positions(self) -> np.ndarray:
        return self.states[:, :3]

    @property
    def velocities(self) -> np.ndarray:
        return self.states[:, 3:6]

    def __len__(self) -> int:
        return len(self.t_s)

    def at(self, t_s) -> np.ndarray:
        """State(s) at arbitrary time(s) inside the sampled span.

        Uses the integrator's dense output if it was stored in ``meta['sol']`` (exact to the
        integration tolerance), otherwise a not-a-knot cubic spline through the samples
        (adequate when samples are dense relative to the dynamics; linear if N < 4).
        """
        t = np.asarray(t_s, dtype=np.float64)
        sol = self.meta.get("sol")
        if sol is not None and self.frame == self.meta.get("sol_frame", self.frame):
            y = sol(t)
            return y.T if t.ndim else y
        if self._interp is None:
            if len(self.t_s) >= 4:
                order = np.argsort(self.t_s)
                self._interp = CubicSpline(self.t_s[order], self.states[order], axis=0)
            else:
                self._interp = lambda tq: np.stack(
                    [np.interp(tq, self.t_s, self.states[:, k]) for k in range(6)], axis=-1
                )
        return self._interp(t)

    # -- conversions -----------------------------------------------------
    def to_frame(self, frame: Frame) -> "Trajectory":
        if frame == self.frame:
            return self
        s = self.states
        # go through GCRF
        if self.frame == "rot_nd":
            g = frames.rot_to_gcrf(s, self.t_s)
        elif self.frame == "moon_km":
            g = frames.moon_centered_to_gcrf(s, self.t_s)
        else:
            g = s
        if frame == "gcrf_km":
            out = g
        elif frame == "rot_nd":
            out = frames.gcrf_to_rot(g, self.t_s)
        elif frame == "moon_km":
            out = frames.gcrf_to_moon_centered(g, self.t_s)
        else:
            raise ValueError(f"unknown frame {frame!r}")
        meta = {k: v for k, v in self.meta.items() if k not in ("sol",)}
        return Trajectory(self.t_s.copy(), out, frame, meta)

    def as_dict(self) -> dict:
        """JSON-serialisable representation (no dense-output object)."""
        meta = {k: v for k, v in self.meta.items() if k != "sol"}
        return {
            "frame": self.frame,
            "t_s": self.t_s.tolist(),
            "states": self.states.tolist(),
            "meta": _jsonable(meta),
        }


def _jsonable(x):
    if isinstance(x, dict):
        return {str(k): _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, (np.floating, np.integer)):
        return x.item()
    if isinstance(x, (str, int, float, bool)) or x is None:
        return x
    return str(x)


def propagate(
    s0,
    t0_s: float,
    t_eval_s,
    model: Model = "ephemeris",
    frame_in: Frame = "gcrf_km",
    params: EphemParams | None = None,
    rtol: float | None = None,
    atol: float | None = None,
    keep_dense: bool = False,
) -> Trajectory:
    """Propagate ``s0`` (given in ``frame_in`` at TDB ``t0_s``) to ``t_eval_s`` and return a
    :class:`Trajectory` in ``'gcrf_km'``.

    ``t_eval_s`` are absolute TDB seconds; the first entry need not equal ``t0_s`` (the
    integrator starts at ``t0_s`` and reports at the requested times, which must be monotone
    and all on one side of ``t0_s``).  ``params`` configures the ephemeris force model
    (ignored by the CR3BP).  See module docstring for the CR3BP -> GCRF approximation.
    """
    s0 = np.asarray(s0, dtype=np.float64).ravel()[:6]
    t_eval_s = np.atleast_1d(np.asarray(t_eval_s, dtype=np.float64))
    t0_s = float(t0_s)

    if model == "ephemeris":
        if frame_in == "rot_nd":
            s0g = frames.rot_to_gcrf(s0, t0_s)
        elif frame_in == "moon_km":
            s0g = frames.moon_centered_to_gcrf(s0, t0_s)
        else:
            s0g = s0
        kw = {}
        if rtol is not None:
            kw["rtol"] = rtol
        if atol is not None:
            kw["atol"] = atol
        sol = propagate_ephemeris(
            s0g, t0_s, t_eval_s=_with_t0(t_eval_s, t0_s), params=params, dense_output=keep_dense, **kw
        )
        if not sol.success:
            raise RuntimeError(f"ephemeris propagation failed: {sol.message}")
        states = _pick(sol, t_eval_s)
        meta = {"model": "ephemeris", "t0_s": t0_s, "nfev": int(sol.nfev)}
        if keep_dense:
            meta["sol"] = sol.sol
            meta["sol_frame"] = "gcrf_km"
        return Trajectory(t_eval_s, states, "gcrf_km", meta)

    if model == "cr3bp":
        if frame_in == "gcrf_km":
            s0r = frames.gcrf_to_rot(s0, t0_s)
        elif frame_in == "moon_km":
            s0r = frames.gcrf_to_rot(frames.moon_centered_to_gcrf(s0, t0_s), t0_s)
        else:
            s0r = s0
        tau = (t_eval_s - t0_s) / T_STAR
        kw = {}
        if rtol is not None:
            kw["rtol"] = rtol
        if atol is not None:
            kw["atol"] = atol
        sol = propagate_cr3bp(s0r, t_eval=_with_t0(tau, 0.0), **kw)
        if not sol.success:
            raise RuntimeError(f"CR3BP propagation failed: {sol.message}")
        states_rot = _pick(sol, tau)
        states = frames.rot_to_gcrf(states_rot, t_eval_s)
        meta = {
            "model": "cr3bp",
            "t0_s": t0_s,
            "nfev": int(sol.nfev),
            "note": "CR3BP solution mapped to GCRF with the instantaneous rotating frame (approximate)",
        }
        return Trajectory(t_eval_s, states, "gcrf_km", meta)

    raise ValueError(f"unknown model {model!r}")


def _with_t0(t_eval, t0):
    """solve_ivp requires t_eval to start at t0 for the chosen direction; prepend if needed."""
    if t_eval[0] == t0:
        return t_eval
    return np.concatenate([[t0], t_eval])


def _pick(sol, t_eval):
    """Return (N,6) states at the requested t_eval (drop the prepended t0 sample if any)."""
    y = sol.y[:6].T
    if len(y) == len(t_eval):
        return y
    return y[1:]
