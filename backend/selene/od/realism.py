"""Covariance-realism (filter consistency) tests: NEES and NIS against χ² bounds.

Definitions (Bar-Shalom, Li & Kirubarajan, *Estimation with Applications to Tracking and
Navigation*, 2001, §5.4)
-------------------------------------------------------------------------------------------
* NEES  ε_k = e_kᵀ P_k⁻¹ e_k,  e_k = x̂_k − x_k^true    ~ χ²_{n_x}   (n_x = 6)
* NIS   d_k = ν_kᵀ S_k⁻¹ ν_k                           ~ χ²_{n_z}   (n_z = 2)

For M independent Monte-Carlo runs the run-averaged statistic ε̄_k = (1/M) Σ_m ε_k^(m) satisfies
M ε̄_k ~ χ²_{M n}, so the two-sided (1−α) acceptance interval is
``[χ²_{Mn}(α/2)/M, χ²_{Mn}(1−α/2)/M]``.  A consistent filter has ≈ (1−α) of its epochs inside;
values above the band mean the filter is **over-confident** (covariance too small — the
dangerous failure for custody), below means pessimistic.  The *time*-averaged NEES of a single
run is also reported, but successive errors are correlated so its nominal χ²_{nK}/K band is only
indicative (we do not assert on it).

:func:`run_realism_mc` wires the synthetic pipeline: truth from the catalog, measurements from
:func:`selene.od.measurements.simulate_observations` (new noise each run), an initial estimate
drawn from N(truth, P₀), and the UKF; it returns the fraction of epochs inside the band.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
from scipy.stats import chi2

from selene.dynamics.ephemeris import EphemParams
from selene.od.measurements import simulate_observations
from selene.od.types import FilterRun, jsonable
from selene.od.ukf import UKFConfig, ekf_run, ukf_run

__all__ = ["nees", "chi2_mean_bounds", "RealismResult", "run_realism_mc", "default_P0"]


def nees(err: np.ndarray, P: np.ndarray) -> np.ndarray:
    """(N,) NEES for errors (N,6) and covariances (N,6,6)."""
    err = np.asarray(err, dtype=np.float64).reshape(-1, 6)
    P = np.asarray(P, dtype=np.float64).reshape(-1, 6, 6)
    out = np.empty(err.shape[0])
    for k in range(err.shape[0]):
        out[k] = float(err[k] @ np.linalg.solve(P[k], err[k]))
    return out


def chi2_mean_bounds(dof: int, n_runs: int, conf: float = 0.95) -> tuple[float, float]:
    """Two-sided bounds for the mean of ``n_runs`` independent χ²_dof variables."""
    a = 1.0 - conf
    k = dof * n_runs
    return float(chi2.ppf(a / 2.0, k) / n_runs), float(chi2.ppf(1.0 - a / 2.0, k) / n_runs)


def default_P0(sigma_pos_km: float = 100.0, sigma_vel_m_s: float = 1.0) -> np.ndarray:
    return np.diag([sigma_pos_km ** 2] * 3 + [(sigma_vel_m_s * 1e-3) ** 2] * 3)


@dataclass
class RealismResult:
    n_runs: int
    conf: float
    t_s: np.ndarray                 # (K,) measurement epochs (common to all runs)
    nees_runs: np.ndarray           # (M,K)
    nis_runs: np.ndarray            # (M,K)
    pos_err_runs: np.ndarray        # (M,K) km
    nees_bounds: tuple[float, float]
    nis_bounds: tuple[float, float]
    runs: list = field(default_factory=list)   # FilterRun objects (optional)
    meta: dict = field(default_factory=dict)

    @property
    def nees_mean(self) -> np.ndarray:
        return self.nees_runs.mean(axis=0)

    @property
    def nis_mean(self) -> np.ndarray:
        return self.nis_runs.mean(axis=0)

    @property
    def nees_inside_fraction(self) -> float:
        lo, hi = self.nees_bounds
        m = self.nees_mean
        return float(np.mean((m >= lo) & (m <= hi))) if m.size else float("nan")

    @property
    def nis_inside_fraction(self) -> float:
        lo, hi = self.nis_bounds
        m = self.nis_mean
        return float(np.mean((m >= lo) & (m <= hi))) if m.size else float("nan")

    def summary(self) -> dict:
        return jsonable({
            "n_runs": self.n_runs, "n_epochs": int(self.t_s.size), "conf": self.conf,
            "nees_bounds": self.nees_bounds, "nis_bounds": self.nis_bounds,
            "nees_inside_fraction": self.nees_inside_fraction, "nis_inside_fraction": self.nis_inside_fraction,
            "nees_overall_mean": float(self.nees_runs.mean()) if self.nees_runs.size else None,
            "nis_overall_mean": float(self.nis_runs.mean()) if self.nis_runs.size else None,
            "nees_time_avg_per_run": self.nees_runs.mean(axis=1), "nis_time_avg_per_run": self.nis_runs.mean(axis=1),
            "final_pos_err_km_per_run": self.pos_err_runs[:, -1] if self.pos_err_runs.size else [],
            "rms_final_pos_err_km": float(np.sqrt(np.mean(self.pos_err_runs[:, -1] ** 2))) if self.pos_err_runs.size else None,
            "verdict": self._verdict(), **self.meta,
        })

    def _verdict(self) -> str:
        f = self.nees_inside_fraction
        if not np.isfinite(f):
            return "no data"
        m = float(self.nees_runs.mean()) if self.nees_runs.size else 0.0
        if f >= self.conf - 0.1:
            return "consistent"
        return "over-confident (covariance too small)" if m > 6 else "pessimistic (covariance too large)"

    def as_dict(self) -> dict:
        d = self.summary()
        d.update(jsonable({"t_s": self.t_s, "nees_mean": self.nees_mean, "nis_mean": self.nis_mean,
                           "nees_runs": self.nees_runs, "nis_runs": self.nis_runs, "pos_err_runs": self.pos_err_runs}))
        return d


def run_realism_mc(
    n_runs: int,
    object_id: str,
    t0_s: float,
    t1_s: float,
    sensors="space",
    cadence_s: float = 6 * 3600.0,
    sigma_arcsec: float = 1.0,
    P0: np.ndarray | None = None,
    config: UKFConfig | None = None,
    params: EphemParams | None = None,
    seed: int = 0,
    conf: float = 0.95,
    filter_kind: str = "ukf",
    respect_visibility: bool = True,
    keep_runs: bool = False,
    max_obs: int | None = None,
) -> RealismResult:
    """Monte-Carlo NEES/NIS consistency check of the UKF (or EKF) on a catalog object."""
    from selene.objects.catalog import get_catalog

    t_start = time.perf_counter()
    cat = get_catalog()
    truth = cat.truth(object_id)
    params = params or EphemParams(srp=True, cr_area_mass=truth.params.cr_area_mass)
    P0 = default_P0() if P0 is None else np.asarray(P0, dtype=np.float64)
    config = config or UKFConfig()
    grid = np.arange(float(t0_s), float(t1_s) + 1e-6, float(cadence_s))
    x_true0 = truth.at(float(t0_s))
    runner = ukf_run if filter_kind == "ukf" else ekf_run
    nees_runs, nis_runs, err_runs, runs = [], [], [], []
    t_common = None
    L0 = np.linalg.cholesky(P0)
    for m in range(int(n_runs)):
        rng = np.random.default_rng(seed + 1000 * m)
        obs = simulate_observations(object_id, sensors, grid, sigma_arcsec, rng, respect_visibility)
        if max_obs is not None:
            obs = obs[:max_obs]
        if len(obs) == 0:
            raise ValueError(f"{object_id}: no visible measurements for sensors={sensors!r} in the window")
        x0 = x_true0 + L0 @ rng.standard_normal(6)
        run: FilterRun = runner(obs, x0, P0, float(t0_s), params=params, config=config, object_id=object_id)
        e = run.truth_error(truth.at)
        nees_runs.append(nees(e, run.P))
        nis_runs.append(run.nis.copy())
        err_runs.append(np.linalg.norm(e[:, :3], axis=1))
        if t_common is None:
            t_common = run.t_s.copy()
        if keep_runs:
            runs.append(run)
    nees_runs = np.array(nees_runs)
    nis_runs = np.array(nis_runs)
    err_runs = np.array(err_runs)
    meta = {"object_id": object_id, "filter": filter_kind, "sensors": str(sensors), "cadence_s": cadence_s,
            "sigma_arcsec": sigma_arcsec, "q_psd_km2_s3": config.q_psd, "alpha": config.alpha,
            "P0_sigma_pos_km": float(np.sqrt(np.trace(P0[:3, :3]))), "P0_sigma_vel_m_s": float(np.sqrt(np.trace(P0[3:, 3:])) * 1e3),
            "elapsed_s": time.perf_counter() - t_start, "seed": seed}
    return RealismResult(int(n_runs), conf, t_common if t_common is not None else np.zeros(0), nees_runs, nis_runs,
                         err_runs, chi2_mean_bounds(6, int(n_runs), conf), chi2_mean_bounds(2, int(n_runs), conf),
                         runs, meta)
