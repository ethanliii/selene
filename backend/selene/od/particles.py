"""Particle-cloud uncertainty propagation: how custody decays when nobody is looking.

A Gaussian posterior (x, P) at t₀ is sampled (N particles, Cholesky of P) and every particle is
propagated with the ephemeris force model (Earth + Moon + Sun, optional SRP) through one stacked
``solve_ivp`` (:mod:`selene.od.stacked`).  The sample statistics on the output grid give the
honest, non-linear uncertainty: in three-body dynamics the cloud stretches along the flow and
bends (the "banana"), so the Gaussian covariance (also reported, from the STM) under-represents
the true spread after a few days.

Metrics per epoch
-----------------
* ``sigma_pos_km = √trace(Cov_rr)``, ``sigma_vel_km_s``  — sample covariance
* ``axes_km``       — principal 1σ semi-axes of the position cloud (eigenvalues of Cov_rr)
* ``volume_km3``    — 1σ ellipsoid volume (4/3)π·∏axes
* ``extent_km``     — max pairwise-free extent proxy: 2 × max distance from the sample mean
* ``linear_sigma_pos_km`` — the same quantity from the STM-propagated P (Gaussian/linear)
* ``skew_major, kurt_major`` — skewness / excess kurtosis of the projection on the major axis
  (0 for a Gaussian; a bent cloud shows |skew| ≫ 0)
* ``curvature`` — RMS orthogonal deviation of the cloud from its major axis divided by the second
  axis, after fitting a quadratic in the major coordinate: ≈ 0 when the cloud is a straight
  ellipsoid, > 0.3 when visibly banana-shaped.

Export: GCRF km positions and nondimensional rotating-frame positions
(:func:`selene.dynamics.frames.gcrf_to_rot`, vectorised) per epoch for the 3-D scene.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from selene.dynamics import frames
from selene.dynamics.ephemeris import EphemParams
from selene.od.batch import propagate_with_stm_to
from selene.od.stacked import ensure_cache, propagate_many
from selene.od.types import jsonable, tdb_s_to_utc_iso
from selene.od.ukf import sqrt_psd

__all__ = ["ParticleCloud", "sample_particles", "propagate_cloud", "linear_covariance"]


def sample_particles(x: np.ndarray, P: np.ndarray, n: int, rng=None) -> np.ndarray:
    """(n,6) Gaussian samples of N(x, P) (first particle is the mean itself)."""
    rng = np.random.default_rng(0) if rng is None else rng
    L = sqrt_psd(np.asarray(P, dtype=np.float64))
    z = rng.standard_normal((int(n), 6))
    z[0] = 0.0
    return np.asarray(x, dtype=np.float64)[None, :] + z @ L.T


def linear_covariance(x: np.ndarray, P: np.ndarray, t0_s: float, t_grid, params: EphemParams) -> tuple[np.ndarray, np.ndarray]:
    """STM-propagated (Gaussian) mean (T,6) and covariance (T,6,6) for comparison."""
    X, Phi = propagate_with_stm_to(np.asarray(x, float), float(t0_s), np.asarray(t_grid, float), params)
    return X, np.einsum("tij,jk,tlk->til", Phi, P, Phi)


@dataclass
class ParticleCloud:
    t0_s: float
    t_s: np.ndarray                 # (T,)
    states: np.ndarray              # (T,N,6) GCRF km, km/s
    object_id: str = ""
    linear_P: np.ndarray | None = None   # (T,6,6)
    linear_x: np.ndarray | None = None   # (T,6)
    meta: dict = field(default_factory=dict)

    # -- statistics -----------------------------------------------------------
    @property
    def n(self) -> int:
        return self.states.shape[1]

    def mean(self) -> np.ndarray:
        return self.states.mean(axis=1)

    def cov(self) -> np.ndarray:
        d = self.states - self.mean()[:, None, :]
        return np.einsum("tni,tnj->tij", d, d) / max(self.n - 1, 1)

    def metrics(self) -> dict:
        C = self.cov()
        mu = self.mean()
        T = len(self.t_s)
        sig_p = np.sqrt(np.trace(C[:, :3, :3], axis1=1, axis2=2))
        sig_v = np.sqrt(np.trace(C[:, 3:, 3:], axis1=1, axis2=2))
        axes = np.empty((T, 3))
        skew = np.empty(T)
        kurt = np.empty(T)
        curv = np.empty(T)
        extent = np.empty(T)
        for k in range(T):
            w, V = np.linalg.eigh(C[k, :3, :3])
            order = np.argsort(w)[::-1]
            w, V = np.clip(w[order], 0, None), V[:, order]
            axes[k] = np.sqrt(w)
            d = self.states[k, :, :3] - mu[k, :3]
            extent[k] = 2.0 * float(np.max(np.linalg.norm(d, axis=1)))
            u = d @ V[:, 0]
            s = float(np.std(u))
            if s > 0:
                z = u / s
                skew[k] = float(np.mean(z ** 3))
                kurt[k] = float(np.mean(z ** 4) - 3.0)
            else:
                skew[k] = kurt[k] = 0.0
            v = d @ V[:, 1]
            if axes[k, 1] > 0 and s > 0:
                A = np.stack([np.ones_like(u), u, u ** 2], axis=1)
                coef, *_ = np.linalg.lstsq(A, v, rcond=None)
                bend = coef[2] * u ** 2
                curv[k] = float(np.sqrt(np.mean(bend ** 2)) / axes[k, 1])
            else:
                curv[k] = 0.0
        out = {
            "t_s": self.t_s, "hours": (self.t_s - self.t0_s) / 3600.0,
            "sigma_pos_km": sig_p, "sigma_vel_km_s": sig_v, "axes_km": axes,
            "volume_km3": 4.0 / 3.0 * np.pi * np.prod(axes, axis=1), "extent_km": extent,
            "skew_major": skew, "kurt_major": kurt, "curvature": curv,
        }
        if self.linear_P is not None:
            out["linear_sigma_pos_km"] = np.sqrt(np.trace(self.linear_P[:, :3, :3], axis1=1, axis2=2))
            out["mean_offset_km"] = np.linalg.norm(mu[:, :3] - self.linear_x[:, :3], axis=1)
        return out

    def rot_positions(self) -> np.ndarray:
        """(T,N,3) nondimensional rotating-frame positions (vectorised per epoch)."""
        T, N = self.states.shape[:2]
        out = np.empty((T, N, 3))
        for k in range(T):
            out[k] = frames.gcrf_to_rot(self.states[k], np.full(N, self.t_s[k]))[:, :3]
        return out

    def to_dict(self, max_particles: int = 500, include_rot: bool = True, digits: int = 2) -> dict:
        """JSON-safe export: metrics + per-epoch frames (subsampled particle positions)."""
        m = self.metrics()
        idx = np.arange(self.n) if self.n <= max_particles else np.linspace(0, self.n - 1, max_particles).astype(int)
        rot = self.rot_positions()[:, idx] if include_rot else None
        epochs = tdb_s_to_utc_iso(self.t_s)
        fr = []
        for k in range(len(self.t_s)):
            f = {"epoch": epochs[k], "t_s": float(self.t_s[k]), "frame": "gcrf",
                 "positions_km": np.round(self.states[k, idx, :3], digits).tolist(),
                 "mean_km": np.round(m["t_s"][k] * 0 + self.mean()[k, :3], 3).tolist()}
            if rot is not None:
                f["positions_rot"] = np.round(rot[k], 6).tolist()
            fr.append(f)
        return jsonable({
            "object_id": self.object_id, "t0_s": self.t0_s, "n_particles": self.n, "n_exported": int(idx.size),
            "epochs": epochs, "metrics": m, "frames": fr, "meta": self.meta,
        })


def propagate_cloud(
    x: np.ndarray,
    P: np.ndarray,
    t0_s: float,
    t_grid,
    n: int = 2000,
    rng=None,
    params: EphemParams | None = None,
    rtol: float = 1e-9,
    atol: float = 1e-9,
    with_linear: bool = True,
    object_id: str = "",
) -> ParticleCloud:
    """Sample N(x,P) at ``t0_s`` and propagate to the absolute TDB epochs ``t_grid`` (monotone)."""
    t_start = time.perf_counter()
    t_grid = np.atleast_1d(np.asarray(t_grid, dtype=np.float64))
    params = ensure_cache(params, min(t0_s, t_grid.min()), max(t0_s, t_grid.max()))
    parts = sample_particles(x, P, n, rng)
    states = propagate_many(parts, float(t0_s), t_grid, params, rtol=rtol, atol=atol)
    lin_x = lin_P = None
    if with_linear:
        lin_x, lin_P = linear_covariance(x, P, t0_s, t_grid, params)
    meta = {"elapsed_s": time.perf_counter() - t_start, "rtol": rtol, "srp": bool(params.srp),
            "cr_area_mass": float(params.cr_area_mass), "force_model": "DE440s Earth+Moon+Sun point masses"
            + (" + cannonball SRP" if params.srp else "")}
    return ParticleCloud(float(t0_s), t_grid, states, object_id, lin_P, lin_x, meta)
