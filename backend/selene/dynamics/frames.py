"""Frame transforms: instantaneous Earth-Moon rotating frame <-> GCRF <-> Moon-centered.

Rotating frame definition (PLAN §2.3)
------------------------------------
Built at each instant from the DE440s Earth->Moon relative state (r, v) [km, km/s]:

    x̂ = r/|r|,   ẑ = (r×v)/|r×v|,   ŷ = ẑ×x̂,        d(t) = |r|,   ḋ = (r·v)/d

* Origin: Earth-Moon barycenter  r_b = μ r  (Earth-centered), v_b = μ v, with μ = ``constants.MU``
  so the Moon sits *exactly* at (1-μ, 0, 0) in nondimensional coordinates.  (DE440s' own EMB
  segment uses the same mass ratio to ~1e-10; we use μ for exact consistency with the CR3BP.)
* Lengths are scaled by the instantaneous distance d(t) (not the mean L*): x_nd = x_km / d(t).
* Velocities use the fixed CR3BP time unit T*:  **v_nd = v_km_s / (d(t) / T*)**, i.e. one
  nondimensional time unit is always T* = ``constants.T_STAR`` seconds, and the length unit is d(t).
  Consequently the Moon's nondimensional angular rate is ω(t)·T*, which oscillates by ~±10 %
  about 1 because the real lunar orbit is eccentric; this is the dominant CR3BP/ephemeris
  discrepancy seen in ``test_propagate.py``.
* Frame angular velocity (GCRF components):  ω = ω_z ẑ + ω_x x̂  with ω_z = |r×v|/d² and
  ω_x = d·a_z/|r×v| (a_z = out-of-plane component of the Earth-Moon relative acceleration,
  obtained by central-differencing the DE440s Moon velocity, i.e. purely kinematic).  The ω_x
  term is the instantaneous precession/nutation of the lunar orbit plane (~1e-10 rad/s here,
  ≈ 3e-5 of ω_z); including it makes the velocity transform the exact time derivative of the
  position transform (verified numerically in ``test_frames.py``).

Velocity transform (r_rel = r - r_b, v_rel = v - v_b, R = [x̂ ŷ ẑ] columns):

    r_rot = Rᵀ r_rel / d
    v_rot = T* · [ Rᵀ (v_rel - ω × r_rel)/d  -  r_rot · ḋ/d ]

and its exact inverse in :func:`rot_to_gcrf`.  Moon-centered frame = GCRF axes translated to the
Moon (no rotation; "MCI" for display and lunar-orbit bookkeeping).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from selene.constants import MU, T_STAR
from selene.dynamics.cr3bp import lagrange_points
from selene.dynamics.ephemeris import get_ephemeris
from selene.time import seconds_since_j2000_tdb

__all__ = [
    "DEMO_EPOCH_UTC",
    "DEMO_EPOCH_TDB_S",
    "FrameInfo",
    "rotating_frame",
    "rot_to_gcrf",
    "gcrf_to_rot",
    "rot_pos_to_gcrf",
    "gcrf_pos_to_rot",
    "gcrf_to_moon_centered",
    "moon_centered_to_gcrf",
    "lagrange_points_gcrf",
    "nd_time_to_tdb_s",
    "tdb_s_to_nd_time",
]

#: Demo reference epoch (DECISIONS.md): 2026-03-01T00:00:00 UTC, as TDB seconds past J2000.
DEMO_EPOCH_UTC = "2026-03-01T00:00:00"
DEMO_EPOCH_TDB_S: float = float(seconds_since_j2000_tdb(DEMO_EPOCH_UTC))

#: Central-difference step [s] for the Earth-Moon relative acceleration (frame precession rate).
_ACCEL_FD_STEP_S = 30.0


@dataclass
class FrameInfo:
    """Instantaneous rotating-frame description.  For array input every field gains a leading N."""

    t_s: float | np.ndarray
    R: np.ndarray            # (3,3) or (N,3,3): columns are x̂, ŷ, ẑ expressed in GCRF
    omega_vec: np.ndarray    # (3,) or (N,3) rad/s, GCRF components
    d_km: float | np.ndarray
    ddot_km_s: float | np.ndarray
    r_bary_gcrf: np.ndarray  # (3,) or (N,3) km, Earth-centered
    v_bary_gcrf: np.ndarray  # (3,) or (N,3) km/s

    @property
    def omega(self) -> float | np.ndarray:
        """|ω| in rad/s."""
        return np.linalg.norm(self.omega_vec, axis=-1)


def nd_time_to_tdb_s(tau, t0_s: float):
    """Nondimensional time τ (units of T*) measured from TDB epoch ``t0_s`` -> TDB seconds."""
    return t0_s + np.asarray(tau, dtype=np.float64) * T_STAR


def tdb_s_to_nd_time(t_s, t0_s: float):
    return (np.asarray(t_s, dtype=np.float64) - t0_s) / T_STAR


def rotating_frame(t_s) -> FrameInfo:
    """Rotating-frame basis, angular velocity and barycenter state at TDB time(s) ``t_s``."""
    t = np.asarray(t_s, dtype=np.float64)
    scalar = t.ndim == 0
    tt = np.atleast_1d(t)
    eph = get_ephemeris()
    sm = eph.moon_state(tt)                # (N,6) Earth->Moon
    rm, vm = sm[:, :3], sm[:, 3:]
    d = np.linalg.norm(rm, axis=1)
    xhat = rm / d[:, None]
    h = np.cross(rm, vm)
    hn = np.linalg.norm(h, axis=1)
    zhat = h / hn[:, None]
    yhat = np.cross(zhat, xhat)
    ddot = np.einsum("ij,ij->i", rm, vm) / d
    omega_z = hn / d**2
    # Out-of-plane Earth-Moon relative acceleration from the kernel itself (central difference
    # of the DE440s velocity; the Chebyshev velocity is smooth so h = 30 s gives ~1e-17 km/s²
    # truncation error).  This captures Sun, planets and Earth-J2 torques on the lunar orbit
    # without a force model, so the frame rate is exactly the kinematic one.
    h = _ACCEL_FD_STEP_S
    a_rel = (eph.moon_state(tt + h)[:, 3:] - eph.moon_state(tt - h)[:, 3:]) / (2.0 * h)
    a_z = np.einsum("ij,ij->i", a_rel, zhat)
    omega_x = d * a_z / hn
    omega_vec = omega_z[:, None] * zhat + omega_x[:, None] * xhat
    R = np.stack([xhat, yhat, zhat], axis=-1)  # (N,3,3) columns
    r_b = MU * rm
    v_b = MU * vm
    if scalar:
        return FrameInfo(float(t), R[0], omega_vec[0], float(d[0]), float(ddot[0]), r_b[0], v_b[0])
    return FrameInfo(tt, R, omega_vec, d, ddot, r_b, v_b)


def _prep(s, t_s):
    s = np.asarray(s, dtype=np.float64)
    t = np.asarray(t_s, dtype=np.float64)
    if s.ndim == 1:
        if t.ndim != 0:
            raise ValueError("scalar state requires scalar time")
        return s[None, :], np.atleast_1d(t), True
    if t.ndim == 0:
        t = np.full(s.shape[0], float(t))
    if t.shape[0] != s.shape[0]:
        raise ValueError("state and time arrays must have the same length")
    return s, t, False


def gcrf_to_rot(s_gcrf, t_s) -> np.ndarray:
    """Earth-centered GCRF state [km, km/s] -> nondimensional rotating-frame state.
    Supports (6,) with scalar t and (N,6) with (N,) t."""
    s, t, scalar = _prep(s_gcrf, t_s)
    f = rotating_frame(t)
    r_rel = s[:, :3] - f.r_bary_gcrf
    v_rel = s[:, 3:6] - f.v_bary_gcrf
    d = f.d_km[:, None]
    r_rot = np.einsum("nji,nj->ni", f.R, r_rel) / d  # Rᵀ r
    w_x_r = np.cross(f.omega_vec, r_rel)
    v_rot = T_STAR * (np.einsum("nji,nj->ni", f.R, v_rel - w_x_r) / d - r_rot * (f.ddot_km_s / f.d_km)[:, None])
    out = np.concatenate([r_rot, v_rot], axis=1)
    return out[0] if scalar else out


def rot_to_gcrf(s_rot_nd, t_s) -> np.ndarray:
    """Nondimensional rotating-frame state -> Earth-centered GCRF state [km, km/s]."""
    s, t, scalar = _prep(s_rot_nd, t_s)
    f = rotating_frame(t)
    d = f.d_km[:, None]
    r_rel = d * np.einsum("nij,nj->ni", f.R, s[:, :3])
    inner = d * s[:, 3:6] / T_STAR + f.ddot_km_s[:, None] * s[:, :3]
    v_rel = np.einsum("nij,nj->ni", f.R, inner) + np.cross(f.omega_vec, r_rel)
    out = np.concatenate([r_rel + f.r_bary_gcrf, v_rel + f.v_bary_gcrf], axis=1)
    return out[0] if scalar else out


def rot_pos_to_gcrf(r_rot_nd, t_s) -> np.ndarray:
    """Position-only rotating (nd) -> GCRF [km].  (3,)/(N,3) with scalar/(N,) t."""
    r = np.asarray(r_rot_nd, dtype=np.float64)
    s = np.concatenate([r, np.zeros_like(r)], axis=-1)
    return rot_to_gcrf(s, t_s)[..., :3]


def gcrf_pos_to_rot(r_gcrf_km, t_s) -> np.ndarray:
    """Position-only GCRF [km] -> rotating (nd)."""
    r = np.asarray(r_gcrf_km, dtype=np.float64)
    s = np.concatenate([r, np.zeros_like(r)], axis=-1)
    return gcrf_to_rot(s, t_s)[..., :3]


def gcrf_to_moon_centered(s, t_s) -> np.ndarray:
    """Earth-centered GCRF -> Moon-centered (GCRF axes) state; translation only. (6,) or (N,6)."""
    s = np.asarray(s, dtype=np.float64)
    sm = get_ephemeris().moon_state(t_s)
    return s - sm


def moon_centered_to_gcrf(s, t_s) -> np.ndarray:
    s = np.asarray(s, dtype=np.float64)
    sm = get_ephemeris().moon_state(t_s)
    return s + sm


def lagrange_points_gcrf(t_s) -> np.ndarray:
    """L1..L5 of the instantaneous rotating frame in Earth-centered GCRF [km], shape (5,3)
    (or (N,5,3) for array t).  Display-only: the true libration regions are not fixed points
    in the ephemeris model."""
    L = lagrange_points(MU)
    t = np.asarray(t_s, dtype=np.float64)
    if t.ndim == 0:
        return rot_pos_to_gcrf(L, float(t))
    return np.stack([rot_pos_to_gcrf(L, float(ti)) for ti in t], axis=0)
