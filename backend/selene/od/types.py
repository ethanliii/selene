"""Shared output container for the sequential filters (UKF / EKF) — used by the OD and the
maneuver-detection tracks.

All quantities are Earth-centred GCRF, km / km/s, TDB seconds past J2000.  The innovation is the
*on-sky* angular residual ``[Δra·cos(dec), Δdec]`` in radians, so that the measurement noise
covariance is the isotropic ``σ² I₂`` (σ = ``Measurement.sigma_rad``) and the NIS is dimensionless.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from selene.time import J2000_JD

__all__ = ["FilterRun", "jsonable", "tdb_s_to_utc_iso"]

_ARCSEC = np.pi / (180.0 * 3600.0)


def tdb_s_to_utc_iso(t_s) -> list[str]:
    """Vectorised TDB seconds past J2000 -> ISO UTC strings (millisecond precision)."""
    from astropy.time import Time

    t = np.atleast_1d(np.asarray(t_s, dtype=np.float64))
    if t.size == 0:
        return []
    iso = Time(J2000_JD, t / 86400.0, format="jd", scale="tdb").utc.isot
    return [str(x)[:23] for x in np.atleast_1d(iso)]


def jsonable(x: Any):
    """Recursively convert numpy / dataclass content into JSON-safe python (NaN/inf -> None)."""
    if isinstance(x, dict):
        return {str(k): jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [jsonable(v) for v in x]
    if isinstance(x, np.ndarray):
        return jsonable(x.tolist())
    if isinstance(x, (np.floating, float)):
        xf = float(x)
        return xf if np.isfinite(xf) else None
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.bool_,)):
        return bool(x)
    if isinstance(x, (str, int, bool)) or x is None:
        return x
    if hasattr(x, "as_dict"):
        return jsonable(x.as_dict())
    if hasattr(x, "to_dict"):
        return jsonable(x.to_dict())
    return str(x)


@dataclass
class FilterRun:
    """One pass of a sequential filter over a measurement set (one row per processed measurement).

    t_s      (N,)      TDB s past J2000 of each processed measurement
    x        (N,6)     posterior means (GCRF km, km/s)
    P        (N,6,6)   posterior covariances
    x_pred   (N,6)     prior (predicted) means at the measurement epochs
    P_pred   (N,6,6)   prior covariances (after any robustness inflation that was applied)
    innov    (N,2)     innovation [Δra·cos(dec), Δdec] rad (measured − predicted)
    S        (N,2,2)   innovation covariances
    nis      (N,)      normalised innovation squared νᵀ S⁻¹ ν (χ²₂ under H₀)
    meas     list[Measurement] in processing order
    object_id, meta    bookkeeping (meta is free-form; must stay JSON-safe via :func:`jsonable`)
    """

    t_s: np.ndarray
    x: np.ndarray
    P: np.ndarray
    x_pred: np.ndarray
    P_pred: np.ndarray
    innov: np.ndarray
    S: np.ndarray
    nis: np.ndarray
    meas: list
    object_id: str = ""
    meta: dict = field(default_factory=dict)

    def __post_init__(self):
        n = len(self.t_s)
        self.t_s = np.asarray(self.t_s, dtype=np.float64).reshape(n)
        self.x = np.asarray(self.x, dtype=np.float64).reshape(n, 6)
        self.P = np.asarray(self.P, dtype=np.float64).reshape(n, 6, 6)
        self.x_pred = np.asarray(self.x_pred, dtype=np.float64).reshape(n, 6)
        self.P_pred = np.asarray(self.P_pred, dtype=np.float64).reshape(n, 6, 6)
        self.innov = np.asarray(self.innov, dtype=np.float64).reshape(n, 2)
        self.S = np.asarray(self.S, dtype=np.float64).reshape(n, 2, 2)
        self.nis = np.asarray(self.nis, dtype=np.float64).reshape(n)

    # -- derived series --------------------------------------------------------
    def __len__(self) -> int:
        return len(self.t_s)

    @property
    def sigma_pos_km(self) -> np.ndarray:
        """sqrt(trace(P[:3,:3])) per epoch [km]: the 1σ position-uncertainty radius proxy."""
        return np.sqrt(np.maximum(np.trace(self.P[:, :3, :3], axis1=1, axis2=2), 0.0))

    @property
    def sigma_vel_km_s(self) -> np.ndarray:
        return np.sqrt(np.maximum(np.trace(self.P[:, 3:, 3:], axis1=1, axis2=2), 0.0))

    @property
    def sigma_pos_pred_km(self) -> np.ndarray:
        return np.sqrt(np.maximum(np.trace(self.P_pred[:, :3, :3], axis1=1, axis2=2), 0.0))

    def final_state(self) -> tuple[float, np.ndarray, np.ndarray]:
        """(t_s, x, P) of the last update (raises IndexError on an empty run)."""
        return float(self.t_s[-1]), self.x[-1].copy(), self.P[-1].copy()

    def truth_error(self, truth_fn) -> np.ndarray:
        """(N,6) posterior error x − truth(t) for a callable ``truth_fn(t_s) -> (6,)/(N,6)``."""
        if len(self) == 0:
            return np.zeros((0, 6))
        return self.x - np.asarray(truth_fn(self.t_s), dtype=np.float64).reshape(len(self), 6)

    def nees(self, truth_fn) -> np.ndarray:
        """Normalised estimation error squared eᵀP⁻¹e per epoch (χ²₆ for a consistent filter)."""
        e = self.truth_error(truth_fn)
        out = np.empty(len(self))
        for k in range(len(self)):
            out[k] = float(e[k] @ np.linalg.solve(self.P[k], e[k]))
        return out

    # -- export ----------------------------------------------------------------
    def to_dict(self, include_pred: bool = True, include_meas: bool = True, cov_digits: int | None = None) -> dict:
        """JSON-safe dictionary.  Keys follow the frontend ``UkfResult`` contract (``epochs``,
        ``states``, ``covs``, ``nis``, ``sigma_pos_km``) and add the raw ``t_s`` series,
        per-epoch ``sigma_vel_km_s``, innovations (arcsec), S, the prior series and the measurements.
        """
        n = len(self)
        d = {
            "object_id": self.object_id,
            "n": n,
            "t_s": self.t_s.tolist(),
            "epochs": tdb_s_to_utc_iso(self.t_s),
            "states": self.x.tolist(),
            "covs": self.P.tolist(),
            "nis": jsonable(self.nis),
            "sigma_pos_km": self.sigma_pos_km.tolist(),
            "sigma_vel_km_s": self.sigma_vel_km_s.tolist(),
            "innov_arcsec": (self.innov / _ARCSEC).tolist(),
            "S_arcsec2": (self.S / _ARCSEC**2).tolist(),
            "sensor_ids": [getattr(m, "sensor_id", None) for m in self.meas],
            "meta": jsonable(self.meta),
        }
        if include_pred:
            d["states_pred"] = self.x_pred.tolist()
            d["covs_pred"] = self.P_pred.tolist()
            d["sigma_pos_pred_km"] = self.sigma_pos_pred_km.tolist()
        if include_meas:
            d["measurements"] = [jsonable(m) for m in self.meas]
        return d
