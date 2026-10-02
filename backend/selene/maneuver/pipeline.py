"""End-to-end synthetic maneuver scenario: truth (+ injected burn) → measurements → filter →
detection → Δv estimation.  Used by ``POST /api/maneuver/detect`` and by the tests.

All burns are SIMULATED and attributed to a notional actor.  Real (JPL Horizons) objects are
**refused** by this pipeline: their cached ephemeris is an interpolated navigation product whose
interpolation error (km-level, see the catalog notes) far exceeds the filter's angular precision,
so any "detection" on them would be model mismatch attributed to a real spacecraft — which the
project rule forbids.  Track real objects with the OD routes instead.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np

from selene.constants import GM_MOON
from selene.dynamics.ephemeris import EphemParams, get_ephemeris
from selene.maneuver._simple_ekf import run_ekf
from selene.maneuver.config import DetectorConfig, EstimatorConfig, FilterConfig
from selene.maneuver.detection import DetectionReport, _jsonable, detect
from selene.maneuver.estimation import DvEstimate, estimate_impulsive_dv
from selene.maneuver.synthetic import (
    Burn,
    PiecewiseTruth,
    direction_unit,
    generate_measurements,
    initial_covariance,
    sample_initial_state,
    truth_with_burn,
)
from selene.maneuver.types import FilterRun
from selene.sensors.observers import DEFAULT_OBSERVERS
from selene.sensors.sites import DEFAULT_SITES
from selene.sensors.visibility import Sensor
from selene.time import J2000_JD

__all__ = ["ScenarioResult", "resolve_sensors", "make_burn", "run_filter", "run_scenario", "estimation_window", "resolve_prior"]

_ALL_SENSORS: dict[str, Sensor] = {s.id: s for s in DEFAULT_SITES}
_ALL_SENSORS.update({o.id: o for o in DEFAULT_OBSERVERS})
ARCSEC = np.pi / (180.0 * 3600.0)


def resolve_sensors(ids: Optional[Sequence[str]]) -> list[Sensor]:
    if not ids:
        return list(DEFAULT_SITES) + list(DEFAULT_OBSERVERS)
    out = []
    for i in ids:
        if i not in _ALL_SENSORS:
            raise KeyError(f"unknown sensor {i!r}; known: {sorted(_ALL_SENSORS)}")
        out.append(_ALL_SENSORS[i])
    return out


def make_burn(t_burn_s: float, x_at_burn: np.ndarray, dv_mps: Optional[Sequence[float]] = None,
              magnitude_mps: Optional[float] = None, direction: Optional[str] = None,
              center: str = "auto") -> Burn:
    """Build a :class:`Burn` from GCRF components [m/s] or (magnitude [m/s], named direction)."""
    if dv_mps is not None:
        return Burn(float(t_burn_s), np.asarray(dv_mps, dtype=np.float64) * 1e-3)
    if magnitude_mps is None or direction is None:
        raise ValueError("injected burn needs dv_mps or (magnitude_mps and direction)")
    u = direction_unit(direction, x_at_burn, t_burn_s, center)
    return Burn(float(t_burn_s), float(magnitude_mps) * 1e-3 * u)


def _try_ukf(x0, P0, t0_s, meas, params, q_psd, object_id) -> tuple[Optional[FilterRun], str]:
    """Best-effort adapter for the OD track's UKF (API discovered at run time)."""
    try:
        from selene.od import ukf as ukf_mod  # type: ignore
    except Exception as e:  # noqa: BLE001
        return None, f"selene.od.ukf not importable ({type(e).__name__})"
    fn = None
    for name in ("ukf_run", "run_ukf", "run"):
        fn = getattr(ukf_mod, name, None)
        if callable(fn):
            break
    if fn is None:
        return None, "selene.od.ukf has no ukf_run()/run_ukf() entry point"
    # selene.od.ukf.ukf_run(meas, x0, P0, t0_s, params=None, config=UKFConfig(q_psd=...), object_id="")
    kw: dict = {"params": params, "object_id": object_id}
    cfg_cls = getattr(ukf_mod, "UKFConfig", None)
    if cfg_cls is not None:
        try:
            kw["config"] = cfg_cls(q_psd=q_psd)
        except TypeError:
            pass
    fields = ("t_s", "x", "P", "x_pred", "P_pred", "innov", "S", "nis", "meas")
    try:
        out = fn(meas, x0, P0, t0_s, **kw)
    except TypeError as e:  # older/newer signature: minimal positional call
        try:
            out = fn(meas, x0, P0, t0_s)
        except Exception as e2:  # noqa: BLE001
            return None, f"UKF signature incompatible ({e}; {e2})"
    except Exception as e:  # noqa: BLE001
        return None, f"UKF raised {type(e).__name__}: {e}"
    if all(hasattr(out, a) for a in fields):
        return out, "ok"
    return None, "UKF returned an object without the FilterRun fields"


def resolve_prior(x_gcrf: np.ndarray, t_s: float, filt_cfg: FilterConfig) -> tuple[float, float, str]:
    """(σ_pos0 [km], σ_vel0 [m/s], regime label) for the initial covariance.

    Explicit values in ``filt_cfg`` win; otherwise the regime-dependent scenario default of
    :class:`FilterConfig` is used: ``moon_bound`` when the Moon-relative two-body energy is
    negative and the object is within ``moon_bound_radius_km`` of the Moon, else ``cislunar``.
    """
    x = np.asarray(x_gcrf, dtype=np.float64)
    sm = get_ephemeris().moon_state(float(t_s))
    rm, vm = x[:3] - sm[:3], x[3:] - sm[3:]
    r = float(np.linalg.norm(rm))
    eps = 0.5 * float(vm @ vm) - GM_MOON / r
    regime = "moon_bound" if (eps < 0.0 and r < filt_cfg.moon_bound_radius_km) else "cislunar"
    dflt = filt_cfg.moon_bound_prior if regime == "moon_bound" else filt_cfg.cislunar_prior
    sp = float(filt_cfg.sigma_pos0_km) if filt_cfg.sigma_pos0_km is not None else float(dflt[0])
    sv = float(filt_cfg.sigma_vel0_mps) if filt_cfg.sigma_vel0_mps is not None else float(dflt[1])
    label = regime + ("" if (filt_cfg.sigma_pos0_km is None and filt_cfg.sigma_vel0_mps is None) else " (explicit prior)")
    return sp, sv, label


def run_filter(x0, P0, t0_s, meas, params, filt_cfg: FilterConfig, object_id: str, prefer: str = "auto") -> tuple[FilterRun, str, str]:
    """Return (run, filter_used, note).  ``prefer`` ∈ {'auto', 'ukf', 'ekf'}."""
    note = ""
    if prefer in ("auto", "ukf"):
        run, note = _try_ukf(x0, P0, t0_s, meas, params, filt_cfg.q_psd_km2_s3, object_id)
        if run is not None:
            name = run.meta.get("filter", "ukf") if isinstance(run.meta, dict) else "ukf"
            return run, str(name), "selene.od.ukf"
        if prefer == "ukf":
            raise RuntimeError(f"UKF requested but unavailable: {note}")
    run = run_ekf(x0, P0, t0_s, meas, params=params, q_psd=filt_cfg.q_psd_km2_s3, object_id=object_id,
                  rtol=filt_cfg.rtol, atol=filt_cfg.atol)
    return run, "simple_ekf", (f"fallback: {note}" if note else "simple_ekf requested")


def estimation_window(run: FilterRun, report: DetectionReport, t0_s: float, window: int) -> Optional[tuple[int, float, float]]:
    """(index of first declaring update, t_lo, t_hi) for the Δv estimator, or None."""
    if not report.declared or report.first_declared_t_s is None:
        return None
    t_hi = float(report.first_declared_t_s)
    i = int(np.searchsorted(np.asarray(run.t_s), t_hi))
    tests = {d.test for d in report.detections if d.t_s == report.first_declared_t_s}
    back = window if ("nis_window" in tests and not ({"nis", "gap_refit"} & tests)) else 1
    j = i - back
    t_lo = float(run.t_s[j]) if j >= 0 else float(t0_s)
    if t_lo >= t_hi:
        t_lo = float(t0_s)
    return i, t_lo, t_hi


@dataclass
class ScenarioResult:
    object_id: str
    kind: str
    run: FilterRun
    report: DetectionReport
    estimate: Optional[DvEstimate]
    truth: Optional[PiecewiseTruth]
    burn: Optional[Burn]
    meas: list
    filter_used: str
    filter_note: str
    timing: dict
    config: dict
    estimate_error: Optional[str] = None
    meta: dict = field(default_factory=dict)

    def truth_block(self) -> Optional[dict]:
        if self.burn is None:
            return None
        d = self.burn.as_dict()
        d["t_burn_utc"] = _utc(self.burn.t_s)
        d["note"] = "SIMULATED burn injected into a notional object's truth (notional actor)"
        if self.estimate is not None:
            e = self.estimate
            dv_true = self.burn.dv_kms
            mag_t = np.linalg.norm(dv_true)
            err = e.dv_gcrf_kms - dv_true
            cosang = float(e.direction_gcrf @ (dv_true / mag_t)) if mag_t > 0 else float("nan")
            d["estimate_error"] = {
                "dv_error_mps": (err * 1e3).tolist(),
                "magnitude_error_mps": float((np.linalg.norm(e.dv_gcrf_kms) - mag_t) * 1e3),
                "magnitude_error_pct": float(100.0 * (np.linalg.norm(e.dv_gcrf_kms) - mag_t) / mag_t) if mag_t > 0 else None,
                "direction_error_deg": float(np.degrees(np.arccos(np.clip(cosang, -1, 1)))) if mag_t > 0 else None,
                "t_burn_error_s": float(e.t_burn_s - self.burn.t_s),
            }
        if self.report.first_declared_t_s is not None:
            lat = float(self.report.first_declared_t_s - self.burn.t_s)
            d["detection_latency_s"] = lat
            n_post = int(np.sum((np.asarray(self.run.t_s) > self.burn.t_s) & (np.asarray(self.run.t_s) <= self.report.first_declared_t_s)))
            d["detected_after_n_post_burn_obs"] = n_post
            # a declaration before the burn is a false alarm, not a detection
            d["premature_declaration"] = bool(lat < 0.0)
            d["detected"] = bool(lat >= 0.0 and self.burn.magnitude_mps > 0.0)
        else:
            d["detected"] = False
        return d

    def filter_block(self, max_points: int = 2000) -> dict:
        r = self.run
        n = len(r.t_s)
        P = np.asarray(r.P)
        return {
            "name": self.filter_used, "source_note": self.filter_note, "n_updates": n,
            "t_s": np.asarray(r.t_s).tolist(), "epochs_utc": [_utc(t) for t in np.asarray(r.t_s)],
            "states": np.asarray(r.x).tolist(),
            "sigma_pos_km": np.sqrt(np.maximum(np.trace(P[:, :3, :3], axis1=1, axis2=2), 0)).tolist() if n else [],
            "sigma_vel_mps": (np.sqrt(np.maximum(np.trace(P[:, 3:, 3:], axis1=1, axis2=2), 0)) * 1e3).tolist() if n else [],
            "sigma_pos_pred_km": np.sqrt(np.maximum(np.trace(np.asarray(r.P_pred)[:, :3, :3], axis1=1, axis2=2), 0)).tolist() if n else [],
            # per-axis 1-σ [km, km, km, m/s, m/s, m/s] (a raw P diagonal in km²/s² ≈ 1e-14 would be
            # destroyed by the route's 8-digit rounding)
            "sigma_diag_km_mps": (np.sqrt(np.maximum(np.diagonal(P, axis1=1, axis2=2), 0)) * np.array([1, 1, 1, 1e3, 1e3, 1e3])).tolist() if n else [],
            "nis": np.asarray(r.nis).tolist(),
            "innov_arcsec": (np.asarray(r.innov) / ARCSEC).tolist(),
            "sensor_ids": [getattr(m, "sensor_id", None) for m in r.meas],
            "measurements": [m.as_dict() for m in r.meas],
            "meta": _jsonable(r.meta) if isinstance(r.meta, dict) else {},
        }

    def as_dict(self) -> dict:
        rep = self.report.as_dict()
        for d in rep["detections"]:
            d["t_utc"] = _utc(d["t_s"])
        est = None
        if self.estimate is not None:
            est = self.estimate.as_dict()
            est["t_burn_utc"] = _utc(est["t_burn_s"])
        return _jsonable({
            "object_id": self.object_id, "kind": self.kind,
            "label": "SIMULATED" if self.kind == "simulated" else "REAL (JPL Horizons) — no burn injected",
            "filter": self.filter_block(), "detections": rep["detections"], "declared": rep["declared"],
            "status": rep["status"], "status_note": STATUS_NOTES.get(rep["status"], ""),
            "first_detection_utc": None if rep["first_detection_t_s"] is None else _utc(rep["first_detection_t_s"]),
            "first_declared_utc": None if rep["first_declared_t_s"] is None else _utc(rep["first_declared_t_s"]),
            "summary": rep["summary"], "dv_estimate": est, "estimate_error": self.estimate_error,
            "truth": self.truth_block(), "timing": self.timing, "config": self.config, "meta": self.meta,
        })


STATUS_NOTES = {
    "no_observations": "No sensor in the network ever saw the object in this span (blind, not quiet): custody was never established.",
    "insufficient_updates": "Too few updates to establish a tracking baseline; no maneuver test performed.",
    "filter_not_converged": "The filter never established a consistent NIS baseline (bad prior / strongly nonlinear regime / model mismatch); "
                            "innovations are not evidence of a maneuver and nothing is declared.",
    "filter_inconsistent": "Simulation diagnostic: the filter covariance is inconsistent with truth before any test; the verdict would be unreliable so nothing is declared.",
    "quiet": "Track established; no test exceeded its family-wise threshold.",
    "maneuver_declared": "Track established; at least one test exceeded its family-wise threshold after the baseline.",
}


def _utc(t_s: float) -> str:
    from astropy.time import Time
    return str(Time(J2000_JD, float(t_s) / 86400.0, format="jd", scale="tdb").utc.isot)[:23] + "Z"


def run_scenario(object_id: str, t0_s: float, t1_s: float, sensor_ids: Optional[Sequence[str]] = None,
                 cadence_s: float = 6 * 3600.0, sigma_arcsec: float = 1.0, burn: Optional[dict | Burn] = None,
                 seed: int = 0, det_cfg: Optional[DetectorConfig] = None, est_cfg: Optional[EstimatorConfig] = None,
                 filt_cfg: Optional[FilterConfig] = None, prefer_filter: str = "auto", estimate: bool = True,
                 max_obs_per_epoch: int = 1) -> ScenarioResult:
    """Run the full synthetic pipeline for a catalog object.

    ``burn`` is a :class:`Burn` or a dict ``{t_s, dv_mps | (magnitude_mps, direction)}``;
    ``None`` means a no-maneuver run (used for false-alarm calibration).
    """
    from selene.objects.catalog import get_catalog

    tic = time.perf_counter()
    timing: dict = {}
    det_cfg = det_cfg or DetectorConfig()
    est_cfg = est_cfg or EstimatorConfig()
    filt_cfg = filt_cfg or FilterConfig()
    rng = np.random.default_rng(seed)
    cat = get_catalog()
    entry = cat.get(object_id)  # KeyError for unknown ids
    sensors = resolve_sensors(sensor_ids)
    simulated = entry.kind == "simulated"
    if not simulated:
        raise ValueError(
            f"{object_id} is a real (JPL Horizons) object: maneuver detection and Δv estimation run only on SIMULATED "
            "objects. The cached Horizons ephemeris is an interpolated navigation product (km-level interpolation error, "
            "see the catalog notes), so residual tests against it would report model mismatch as a 'maneuver' of a real "
            "spacecraft; the project never attributes events to real objects. Use the OD routes to track it.")

    # -- truth -------------------------------------------------------------
    t_a = time.perf_counter()
    x_true0 = np.asarray(cat.state_at(object_id, t0_s), dtype=np.float64)
    phys = entry.physical or {}
    radius_m = float(phys.get("radius_m", 1.0)); albedo = float(phys.get("albedo", 0.2))
    params = EphemParams(srp=True, cr_area_mass=float(phys.get("cr_area_mass_m2_kg", 0.0)))
    burn_obj: Optional[Burn] = None
    if burn is not None:
        if isinstance(burn, Burn):
            burn_obj = burn
        else:
            tb = float(burn["t_s"])
            pre = truth_with_burn(x_true0, t0_s, max(tb, t0_s + 1.0), None, params)
            burn_obj = make_burn(tb, pre.at(tb), burn.get("dv_mps"), burn.get("magnitude_mps"), burn.get("direction"),
                                 est_cfg.center)
        if np.linalg.norm(burn_obj.dv_kms) == 0.0:
            burn_obj = Burn(burn_obj.t_s, np.zeros(3))
    truth = truth_with_burn(x_true0, t0_s, t1_s, burn_obj, params)
    truth_fn = truth.at
    timing["truth_s"] = time.perf_counter() - t_a

    # -- measurements --------------------------------------------------------
    t_a = time.perf_counter()
    n_ep = int(np.floor((t1_s - t0_s) / cadence_s + 1e-9))
    epochs = t0_s + cadence_s * np.arange(1, n_ep + 1)
    meas = generate_measurements(truth_fn, epochs, sensors, sigma_arcsec, rng, radius_m, albedo, max_obs_per_epoch)
    timing["measurements_s"] = time.perf_counter() - t_a

    # -- filter -------------------------------------------------------------
    t_a = time.perf_counter()
    sig_p0, sig_v0, prior_regime = resolve_prior(x_true0, t0_s, filt_cfg)
    P0 = initial_covariance(sig_p0, sig_v0)
    x0 = sample_initial_state(truth_fn(t0_s), P0, rng)
    run, filter_used, note = run_filter(x0, P0, t0_s, meas, params, filt_cfg, object_id, prefer_filter)
    timing["filter_s"] = time.perf_counter() - t_a

    # -- detection ----------------------------------------------------------
    t_a = time.perf_counter()
    report = detect(run, det_cfg, truth=truth_fn, params=params)
    timing["detection_s"] = time.perf_counter() - t_a

    # -- estimation ---------------------------------------------------------
    est: Optional[DvEstimate] = None
    est_err = None
    t_a = time.perf_counter()
    win = estimation_window(run, report, t0_s, det_cfg.window) if estimate else None
    if win is not None:
        i, t_lo, t_hi = win
        j = int(np.searchsorted(np.asarray(run.t_s), t_lo, side="right")) - 1
        if j >= 0:
            x_pre, P_pre, t_pre = np.asarray(run.x[j]), np.asarray(run.P[j]), float(run.t_s[j])
        else:
            x_pre, P_pre, t_pre = x0, P0, float(t0_s)
        try:
            est = estimate_impulsive_dv(x_pre, t_pre, list(run.meas[i:]), (t_lo, t_hi), P_pre=P_pre, params=params, cfg=est_cfg)
        except Exception as e:  # noqa: BLE001
            est_err = f"{type(e).__name__}: {e}"
    timing["estimation_s"] = time.perf_counter() - t_a
    timing["total_s"] = time.perf_counter() - tic

    config = {
        "t0_s": float(t0_s), "t1_s": float(t1_s), "cadence_s": float(cadence_s), "sigma_arcsec": float(sigma_arcsec),
        "sensors": [s.id for s in sensors], "seed": int(seed), "max_obs_per_epoch": int(max_obs_per_epoch),
        "detector": det_cfg.__dict__, "estimator": est_cfg.__dict__, "filter": filt_cfg.__dict__,
        "prior": {"sigma_pos0_km": sig_p0, "sigma_vel0_mps": sig_v0, "regime": prior_regime,
                  "note": "regime-dependent scenario default unless given explicitly (see FilterConfig)"},
        "force_model": {"bodies": list(params.bodies), "srp": params.srp, "cr_area_mass_m2_kg": params.cr_area_mass,
                        "note": "filter uses the object's catalogued C_R·A/m (assumption: physical parameters known)"},
    }
    meta = {"n_epochs": int(n_ep), "n_measurements": len(meas), "n_updates": len(run.t_s),
            "sensors_used": sorted({m.sensor_id for m in meas}),
            "custody_established": bool(report.summary["filter_health"]["baseline_established"]),
            "disclaimer": "All maneuvers are SIMULATED and attributed to a notional actor; no real spacecraft is implicated."}
    return ScenarioResult(object_id, entry.kind, run, report, est, truth, burn_obj, meas, filter_used, note,
                          timing, config, est_err, meta)
