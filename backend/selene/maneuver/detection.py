"""Maneuver detection tests on a :class:`~selene.od.types.FilterRun` (PLAN §2.7).

Tests (all return :class:`Detection` records with a χ² p-value and ``confidence = 1 − p``):

(a) **Per-update NIS** — d² = νᵀS⁻¹ν against χ²_m(1−α), m = 2 (RA/Dec).  Under H₀ (no
    maneuver, consistent filter) d² ~ χ²_m so the per-update false-alarm probability is exactly α.
(b) **Windowed NIS** — Σ of the last W NIS values against χ²_{mW}(1−α); more sensitive to small
    sustained residual growth than any single update (Bar-Shalom, Li & Kirubarajan 2001, §5.4).
(c) **Gap re-fit Mahalanobis** — after an observation gap (longer than ``gap_hours_for_refit``
    *and* ``gap_cadence_factor`` × the track's median cadence: a sensor blinded by lunar glare /
    daylight), the filter's prediction through the gap is compared in *state space* with a
    short-arc batch fit of the first k post-gap observations:
    d² = (x̂_fit − x̂_pred)ᵀ (P_fit + P_pred)⁻¹ (x̂_fit − x̂_pred) ~ χ²₆.  This catches a burn that
    a single, covariance-inflated innovation would not (the predicted S has grown with the gap).
    The arc must itself be consistent (its post-fit measurement χ² below χ²_{2k−6}(1−α)); an
    inconsistent arc means the anomaly lies *inside the arc*, not inside the gap, so the gap
    test is skipped and the per-update tests localise it.
(d) **NEES monitor** — with truth available (simulations) ε = eᵀP⁻¹e ~ χ²₆ per epoch and the
    average NEES over N epochs lies in [χ²_{6N}(α/2), χ²_{6N}(1−α/2)]/N for a consistent filter;
    a sudden NEES jump without a NIS jump means the filter *absorbed* a maneuver.
(e) **CUSUM** — one-sided Page CUSUM on (NIS − m − k): S_n = max(0, S_{n−1} + NIS_n − m − k),
    alarm when S_n > h; h is tuned by Monte Carlo for a target in-control average run length
    ARL₀ (Page 1954; Basseville & Nikiforov 1993, §2.2).

Filter-health gate.  A maneuver is a *departure from an established track*: the tests therefore
start only after a **baseline** of ``baseline_updates`` consecutive updates whose summed NIS is
consistent (χ²_{mW}(1−baseline_alpha)).  A filter that never reaches such a baseline (bad prior,
strongly nonlinear regime, model mismatch) gets the status ``filter_not_converged`` and no
maneuver is declared — a divergent filter produces large innovations that are not evidence of a
maneuver.  With truth (simulation) the mean NEES over the baseline is additionally checked
(``filter_inconsistent`` when the covariance is grossly optimistic).

Run-level **declaration** (``DetectionReport.declared``): a windowed, gap-refit or CUSUM alarm,
or a per-update NIS exceeding the Šidák family-wise threshold χ²_m(1 − (1−(1−α)^{1/N})) so
that a whole no-maneuver run of N tested updates has ≈ α probability of a false declaration
(the raw per-update list keeps the calibrated α rate for the Pfa test).
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Callable, Optional, Sequence

import numpy as np
from scipy.stats import chi2

from selene.dynamics.ephemeris import EphemParams, propagate_ephemeris
from selene.maneuver._simple_ekf import ensure_cache, innovation
from selene.maneuver.config import DetectorConfig
from selene.maneuver.types import FilterRun

__all__ = [
    "Detection", "DetectionReport", "STATUS_VALUES", "nis_test", "windowed_nis_test", "short_arc_fit",
    "gap_refit_test", "nees_monitor", "cusum_test", "cusum_threshold_for_arl", "find_gaps",
    "establish_baseline", "detect",
]

ARCSEC = np.pi / (180.0 * 3600.0)

#: Run-level status values (``DetectionReport.status``).
STATUS_VALUES = ("no_observations", "insufficient_updates", "filter_not_converged", "filter_inconsistent",
                 "quiet", "maneuver_declared")


@dataclass
class Detection:
    """One test exceedance.  ``statistic`` and ``threshold`` share the test's χ² units.

    ``p_value``/``confidence`` are the χ² tail probability and 1 − p for the χ² tests; the CUSUM
    has no per-alarm p-value (its in-control false-alarm *rate* is 1/ARL₀ per update), so both
    are ``None`` for CUSUM alarms and the rate is given in ``extra['false_alarm_rate_per_update']``.
    """

    t_s: float
    test: str
    statistic: float
    threshold: float
    p_value: Optional[float]
    confidence: Optional[float]
    index: int
    dof: int
    sensor_id: Optional[str] = None
    extra: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        d = {
            "t_s": float(self.t_s), "test": self.test, "statistic": float(self.statistic),
            "threshold": float(self.threshold),
            "p_value": None if self.p_value is None else float(self.p_value),
            "confidence": None if self.confidence is None else float(self.confidence),
            "index": int(self.index), "dof": int(self.dof), "sensor_id": self.sensor_id,
        }
        d.update({k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in self.extra.items()})
        return d


@dataclass
class DetectionReport:
    detections: list[Detection]
    summary: dict
    declared: bool
    first_detection_t_s: Optional[float]
    first_declared_t_s: Optional[float]
    status: str = "quiet"

    def by_test(self, name: str) -> list[Detection]:
        return [d for d in self.detections if d.test == name]

    def as_dict(self) -> dict:
        return {
            "declared": bool(self.declared),
            "status": self.status,
            "first_detection_t_s": self.first_detection_t_s,
            "first_declared_t_s": self.first_declared_t_s,
            "summary": _jsonable(self.summary),
            "detections": [d.as_dict() for d in self.detections],
        }


def _jsonable(x):
    if isinstance(x, dict):
        return {str(k): _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, (np.floating, float)):
        xf = float(x)
        return xf if np.isfinite(xf) else None
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, np.bool_):
        return bool(x)
    return x


def _p_conf(stat: float, dof: int) -> tuple[float, float]:
    p = float(chi2.sf(stat, dof))
    return p, 1.0 - p


# ---------------------------------------------------------------------------
# (a) per-update NIS
# ---------------------------------------------------------------------------
def nis_test(run: FilterRun, alpha: float = 0.01, m: int = 2, start: int = 0) -> list[Detection]:
    """Per-update NIS exceedances of χ²_m(1−α), skipping the first ``start`` updates."""
    thr = float(chi2.isf(alpha, m))
    out = []
    nis = np.asarray(run.nis, dtype=np.float64)
    for k in range(int(start), nis.size):
        if nis[k] > thr:
            p, c = _p_conf(nis[k], m)
            out.append(Detection(float(run.t_s[k]), "nis", float(nis[k]), thr, p, c, k, m,
                                 getattr(run.meas[k], "sensor_id", None) if k < len(run.meas) else None,
                                 {"innov_arcsec": (np.asarray(run.innov[k]) / ARCSEC).tolist()}))
    return out


# ---------------------------------------------------------------------------
# (b) windowed NIS
# ---------------------------------------------------------------------------
def windowed_nis_test(run: FilterRun, alpha: float = 0.01, window: int = 5, m: int = 2, start: int = 0) -> list[Detection]:
    """Sum of the last ``window`` NIS values against χ²_{m·window}(1−α) (only full windows that
    lie entirely at indices ≥ ``start``)."""
    W = int(window)
    dof = m * W
    thr = float(chi2.isf(alpha, dof))
    nis = np.asarray(run.nis, dtype=np.float64)
    out = []
    for k in range(max(int(start) + W - 1, W - 1), nis.size):
        s = float(nis[k - W + 1: k + 1].sum())
        if s > thr:
            p, c = _p_conf(s, dof)
            out.append(Detection(float(run.t_s[k]), "nis_window", s, thr, p, c, k, dof,
                                 getattr(run.meas[k], "sensor_id", None) if k < len(run.meas) else None,
                                 {"window": W}))
    return out


# ---------------------------------------------------------------------------
# filter-health baseline
# ---------------------------------------------------------------------------
def establish_baseline(nis: Sequence[float], start: int = 0, window: int = 5, alpha: float = 0.05, m: int = 2) -> Optional[int]:
    """Index of the last update of the first ``window`` consecutive updates (at indices ≥
    ``start``) whose summed NIS is below χ²_{m·window}(1−alpha), or ``None`` if the filter never
    establishes such a consistent baseline.  ``window == 0`` disables the gate (returns
    ``start − 1``)."""
    W = int(window)
    if W <= 0:
        return int(start) - 1
    nis = np.asarray(nis, dtype=np.float64)
    thr = float(chi2.isf(alpha, m * W))
    for k in range(int(start) + W - 1, nis.size):
        s = nis[k - W + 1: k + 1]
        if np.all(np.isfinite(s)) and float(s.sum()) <= thr:
            return k
    return None


# ---------------------------------------------------------------------------
# (c) gap re-fit
# ---------------------------------------------------------------------------
def find_gaps(t_s: Sequence[float], gap_s: float, cadence_factor: Optional[float] = None) -> list[int]:
    """Indices ``i`` such that t[i] − t[i−1] exceeds ``gap_s`` (the first post-gap measurement).

    With ``cadence_factor`` the interval must also exceed ``cadence_factor`` × the median
    inter-observation interval, so that a uniformly coarse cadence is not a run of "gaps".
    """
    t = np.asarray(t_s, dtype=np.float64)
    if t.size < 2:
        return []
    dt = np.diff(t)
    thr = float(gap_s)
    if cadence_factor is not None and cadence_factor > 0:
        thr = max(thr, float(cadence_factor) * float(np.median(dt)))
    return [int(i) for i in np.where(dt > thr)[0] + 1]


def short_arc_fit(x0: np.ndarray, t0_s: float, meas: Sequence, params: EphemParams | None = None,
                  prior_P: np.ndarray | None = None, prior_x: np.ndarray | None = None,
                  max_iter: int = 12, tol: float = 1e-9, rtol: float = 1e-10, atol: float = 1e-10) -> dict:
    """Gauss-Newton batch least squares of the epoch state ``x0`` (at ``t0_s``) to a short
    angles-only arc.  Measurements are whitened by their own σ; the STM maps the per-epoch
    Jacobian H_k back to the epoch: J_k = H_k Φ(t_k, t0).  An optional Gaussian prior
    (``prior_x``, ``prior_P``) regularises unobservable directions.

    Returns dict(x, P, rms_sigma, rms_arcsec, converged, n_iter, n_obs, cost, dof,
    chi2_threshold_arc) where ``cost`` is the post-fit *measurement* χ² (prior excluded) and
    ``dof = 2·n_obs − 6`` its degrees of freedom (0 for an exact 3-observation fit).
    """
    meas = sorted(meas, key=lambda mm: mm.t_s)
    x = np.array(x0, dtype=np.float64).ravel()[:6]
    times = np.array([float(mm.t_s) for mm in meas])
    if times.size == 0:
        raise ValueError("short_arc_fit needs at least one measurement")
    t_last = float(times.max())
    params = ensure_cache(params, min(t0_s, t_last), max(t0_s, t_last))
    uniq, inv = np.unique(times, return_inverse=True)
    Pinv = None
    if prior_P is not None:
        Pinv = np.linalg.inv(np.asarray(prior_P, dtype=np.float64))
        xp = np.array(prior_x if prior_x is not None else x0, dtype=np.float64).ravel()[:6]
    converged = False
    n_iter = 0
    cost = np.inf
    N = np.eye(6)
    r = np.zeros(2 * len(meas))
    for n_iter in range(1, max_iter + 1):
        states, phis = _propagate_with_stm(x, t0_s, uniq, params, rtol, atol)
        J = np.empty((2 * len(meas), 6))
        r = np.empty(2 * len(meas))
        for k, mm in enumerate(meas):
            j = inv[k]
            nu, H = innovation(mm, states[j])
            w = 1.0 / float(mm.sigma_rad)
            r[2 * k: 2 * k + 2] = nu * w
            J[2 * k: 2 * k + 2] = (H @ phis[j]) * w
        N = J.T @ J
        b = J.T @ r
        if Pinv is not None:
            N = N + Pinv
            b = b + Pinv @ (xp - x)
        try:
            dx = np.linalg.solve(N, b)
        except np.linalg.LinAlgError:
            dx = np.linalg.lstsq(N, b, rcond=None)[0]
        x = x + dx
        cost = float(r @ r)
        if np.linalg.norm(dx[:3]) < tol * max(1.0, np.linalg.norm(x[:3])) and np.linalg.norm(dx[3:]) < 1e-9:
            converged = True
            break
    # post-fit measurement residuals at the converged state
    try:
        states, _ = _propagate_with_stm(x, t0_s, uniq, params, rtol, atol)
        for k, mm in enumerate(meas):
            nu, _H = innovation(mm, states[inv[k]])
            r[2 * k: 2 * k + 2] = nu / float(mm.sigma_rad)
        cost = float(r @ r)
    except RuntimeError:
        pass
    try:
        P = np.linalg.inv(N)
    except np.linalg.LinAlgError:
        P = np.linalg.pinv(N)
    rms = float(np.sqrt(cost / max(1, r.size)))  # in units of σ
    sig = float(np.mean([mm.sigma_rad for mm in meas]))
    dof = int(max(0, 2 * len(meas) - 6))
    return {"x": x, "P": 0.5 * (P + P.T), "rms_sigma": rms, "rms_arcsec": rms * sig / ARCSEC,
            "converged": converged, "n_iter": n_iter, "n_obs": len(meas), "cost": cost, "dof": dof}


def _propagate_with_stm(x, t0, times, params, rtol, atol):
    """States (n,6) and STMs (n,6,6) at sorted ``times`` (≥ t0) from x at t0."""
    times = np.asarray(times, dtype=np.float64)
    out_x = np.empty((times.size, 6))
    out_phi = np.empty((times.size, 6, 6))
    mask0 = np.isclose(times, t0)
    for i in np.where(mask0)[0]:
        out_x[i] = x
        out_phi[i] = np.eye(6)
    rest = np.where(~mask0)[0]
    if rest.size:
        y0 = np.concatenate([x, np.eye(6).ravel()])
        sol = propagate_ephemeris(y0, float(t0), t_eval_s=times[rest], params=params, stm=True, rtol=rtol, atol=atol)
        if not sol.success:
            raise RuntimeError(f"propagation failed: {sol.message}")
        Y = sol.y.T
        if Y.shape[0] != rest.size:  # a prepended t0 sample
            Y = Y[-rest.size:]
        out_x[rest] = Y[:, :6]
        out_phi[rest] = Y[:, 6:].reshape(-1, 6, 6)
    return out_x, out_phi


def gap_refit_test(run: FilterRun, cfg: DetectorConfig | None = None, params: EphemParams | None = None,
                   start: int = 0, skipped: Optional[list] = None) -> list[Detection]:
    """Post-gap short-arc re-fit vs. the filter's through-gap prediction (Mahalanobis, χ²₆).

    For each gap (see :func:`find_gaps` with ``cfg.gap_hours_for_refit`` and
    ``cfg.gap_cadence_factor``) that starts at an index i > ``start``, fit the state at t_i to
    measurements i..i+k−1 with a weak prior N(x_pred[i], inflation·P_pred[i]) and test
    d² = Δxᵀ(P_fit + P_pred)⁻¹Δx > χ²₆(1−α).  Because the prior is centred on the prediction the
    test is slightly *conservative* (a maneuver can only be under-stated, never invented).

    Arc consistency: the fit's post-fit measurement χ² must be below χ²_{2k−6}(1−α) (k ≥ 4);
    otherwise the anomaly lies inside the arc (e.g. a burn between post-gap observations at a
    coarse cadence) and attributing it to the gap would misplace the burn epoch, so the gap is
    skipped (recorded in ``skipped`` when a list is passed) and left to the per-update tests.
    """
    cfg = cfg or DetectorConfig()
    out = []
    n = len(run.t_s)
    alpha = cfg.a_gap
    thr = float(chi2.isf(alpha, 6))
    for i in find_gaps(run.t_s, cfg.gap_hours_for_refit * 3600.0, cfg.gap_cadence_factor):
        if i <= int(start):
            continue
        k = min(cfg.refit_n_obs, n - i)
        if k < cfg.refit_min_obs:
            continue
        arc = list(run.meas[i: i + k])
        x_pred, P_pred = np.asarray(run.x_pred[i]), np.asarray(run.P_pred[i])
        try:
            fit = short_arc_fit(x_pred, float(run.t_s[i]), arc, params, prior_P=cfg.refit_prior_inflation * P_pred,
                                prior_x=x_pred)
        except (RuntimeError, np.linalg.LinAlgError, ValueError):
            continue
        dx = fit["x"] - x_pred
        Ssum = fit["P"] + P_pred
        try:
            d2 = float(dx @ np.linalg.solve(Ssum, dx))
        except np.linalg.LinAlgError:
            d2 = float(dx @ np.linalg.pinv(Ssum) @ dx)
        p, c = _p_conf(d2, 6)
        # Arc consistency must be judged *without* the prediction prior: after a real burn the
        # prior (even inflated 1e4×) still drags the poorly observed range direction and inflates
        # the angular residuals.  Re-fit from the solution with a regularisation-only prior.
        arc_cost, arc_dof = fit["cost"], fit["dof"]
        if arc_dof > 0:
            try:
                free = short_arc_fit(fit["x"], float(run.t_s[i]), arc, params,
                                     prior_P=cfg.refit_prior_inflation * 1e4 * P_pred, prior_x=x_pred, max_iter=8)
                arc_cost = min(arc_cost, free["cost"])
            except (RuntimeError, np.linalg.LinAlgError, ValueError):
                pass
        arc_thr = float(chi2.isf(alpha, arc_dof)) if arc_dof > 0 else None
        arc_consistent = None if arc_thr is None else bool(arc_cost <= arc_thr)
        extra = {"gap_hours": float((run.t_s[i] - run.t_s[i - 1]) / 3600.0), "n_fit_obs": k,
                 "arc_span_hours": float((run.t_s[i + k - 1] - run.t_s[i]) / 3600.0),
                 "dx_km": dx[:3].tolist(), "dv_kms": dx[3:].tolist(), "fit_rms_arcsec": fit["rms_arcsec"],
                 "fit_converged": fit["converged"], "arc_chi2": arc_cost, "arc_dof": arc_dof,
                 "arc_chi2_threshold": arc_thr, "arc_consistent": arc_consistent}
        if arc_consistent is False:
            if skipped is not None:
                skipped.append({"index": i, "t_s": float(run.t_s[i]), "reason": "post-gap arc internally inconsistent "
                                "(anomaly inside the arc, not inside the gap); left to the per-update tests", **extra,
                                "mahalanobis_d2": d2})
            continue
        if d2 > thr:
            out.append(Detection(float(run.t_s[i]), "gap_refit", d2, thr, p, c, i, 6,
                                 getattr(run.meas[i], "sensor_id", None), extra))
    return out


# ---------------------------------------------------------------------------
# (d) NEES monitor
# ---------------------------------------------------------------------------
def nees_monitor(run: FilterRun, truth: Callable | np.ndarray, alpha: float = 0.05, start: int = 0) -> tuple[list[Detection], dict]:
    """Filter-consistency monitor with truth.  ``truth`` is a callable t→(6,) or an (N,6) array.

    Returns (per-epoch exceedances of χ²₆(1−α), summary with the average NEES and its
    two-sided bounds [χ²_{6N}(α/2), χ²_{6N}(1−α/2)]/N over the tested epochs).
    """
    X = np.asarray(run.x, dtype=np.float64)
    if callable(truth):
        T = np.asarray(truth(np.asarray(run.t_s)), dtype=np.float64).reshape(X.shape)
    else:
        T = np.asarray(truth, dtype=np.float64).reshape(X.shape)
    e = X - T
    n = X.shape[0]
    nees = np.empty(n)
    for k in range(n):
        try:
            nees[k] = float(e[k] @ np.linalg.solve(run.P[k], e[k]))
        except np.linalg.LinAlgError:
            nees[k] = np.nan
    thr = float(chi2.isf(alpha, 6))
    out = []
    for k in range(int(start), n):
        if np.isfinite(nees[k]) and nees[k] > thr:
            p, c = _p_conf(nees[k], 6)
            out.append(Detection(float(run.t_s[k]), "nees", float(nees[k]), thr, p, c, k, 6,
                                 getattr(run.meas[k], "sensor_id", None) if k < len(run.meas) else None,
                                 {"pos_err_km": float(np.linalg.norm(e[k, :3]))}))
    sel = nees[int(start):]
    sel = sel[np.isfinite(sel)]
    N = sel.size
    summary = {"nees": nees, "mean_nees": float(sel.mean()) if N else float("nan"),
               "n_epochs": int(N), "dof": 6,
               "mean_bounds": [float(chi2.ppf(alpha / 2, 6 * N) / N), float(chi2.isf(alpha / 2, 6 * N) / N)] if N else [None, None],
               "pos_err_km": np.linalg.norm(e[:, :3], axis=1), "vel_err_mps": np.linalg.norm(e[:, 3:], axis=1) * 1e3}
    summary["consistent"] = bool(N and summary["mean_bounds"][0] <= summary["mean_nees"] <= summary["mean_bounds"][1])
    return out, summary


# ---------------------------------------------------------------------------
# (e) CUSUM
# ---------------------------------------------------------------------------
def _cusum_path(nis: np.ndarray, m: int, k: float, h: float, reset: bool = True) -> tuple[np.ndarray, list[int]]:
    S = np.empty(nis.size)
    s = 0.0
    alarms = []
    for i, v in enumerate(nis):
        s = max(0.0, s + float(v) - m - k)
        S[i] = s
        if s > h:
            alarms.append(i)
            if reset:
                s = 0.0
    return S, alarms


@lru_cache(maxsize=64)
def cusum_threshold_for_arl(arl0: float, k: float = 1.0, m: int = 2, n_mc: int = 2000, seed: int = 0) -> float:
    """Decision threshold h giving in-control average run length ≈ ``arl0`` for the CUSUM on
    (χ²_m − m − k), found by bisection on Monte Carlo run lengths (censored at 20·ARL₀)."""
    rng = np.random.default_rng(seed)
    n_max = int(max(50, 20 * arl0))
    draws = rng.chisquare(m, size=(n_mc, n_max)) - m - k

    def arl(h: float) -> float:
        s = np.zeros(n_mc)
        alive = np.ones(n_mc, dtype=bool)
        rl = np.full(n_mc, float(n_max))
        for i in range(n_max):
            s = np.maximum(0.0, s + draws[:, i])
            hit = alive & (s > h)
            rl[hit] = i + 1
            alive &= ~hit
            if not alive.any():
                break
        return float(rl.mean())

    lo, hi = 0.5, 5.0
    while arl(hi) < arl0 and hi < 1e3:
        hi *= 2.0
    for _ in range(25):
        mid = 0.5 * (lo + hi)
        if arl(mid) < arl0:
            lo = mid
        else:
            hi = mid
    return float(0.5 * (lo + hi))


def cusum_test(run: FilterRun, cfg: DetectorConfig | None = None, start: int = 0) -> list[Detection]:
    """Page CUSUM alarms (reset after each alarm).  No per-alarm p-value exists: ``p_value`` and
    ``confidence`` are ``None`` and the in-control false-alarm *rate per update* (≈ 1/ARL₀) is
    reported in ``extra['false_alarm_rate_per_update']``."""
    cfg = cfg or DetectorConfig()
    h = cfg.cusum_h if cfg.cusum_h is not None else cusum_threshold_for_arl(float(cfg.cusum_arl0), float(cfg.cusum_k), int(cfg.m))
    nis = np.asarray(run.nis, dtype=np.float64)
    S, alarms = _cusum_path(nis[int(start):], cfg.m, cfg.cusum_k, h)
    out = []
    for a in alarms:
        i = a + int(start)
        out.append(Detection(float(run.t_s[i]), "cusum", float(S[a]), float(h), None, None, i, 0,
                             getattr(run.meas[i], "sensor_id", None) if i < len(run.meas) else None,
                             {"arl0": float(cfg.cusum_arl0), "k": float(cfg.cusum_k),
                              "false_alarm_rate_per_update": 1.0 / float(cfg.cusum_arl0)}))
    return out


# ---------------------------------------------------------------------------
# orchestration
# ---------------------------------------------------------------------------
def detect(run: FilterRun, cfg: DetectorConfig | None = None, truth: Callable | np.ndarray | None = None,
           params: EphemParams | None = None) -> DetectionReport:
    """Run the filter-health gate, all configured tests and build the run-level report (see the
    module docstring).  ``report.status`` is one of :data:`STATUS_VALUES`."""
    cfg = cfg or DetectorConfig()
    tic = time.perf_counter()
    n = len(run.t_s)
    nis = np.asarray(run.nis, dtype=np.float64)
    start0 = min(int(cfg.min_updates_before_test), n)
    W = int(cfg.window)

    # -- filter-health gate ------------------------------------------------------------------
    # Baseline length: the configured value, shortened for short runs to max(3, n // 2) so that
    # a sparse track (e.g. 6 daily observations) can still establish a baseline from its first
    # updates; the search starts at the first update.
    Wb = int(min(cfg.baseline_updates, max(3, n // 2))) if cfg.baseline_updates > 0 else 0
    baseline_end = establish_baseline(nis, 0, Wb, cfg.baseline_alpha, cfg.m) if n else None
    gated = Wb > 0
    health: dict = {"baseline_updates": Wb, "baseline_alpha": cfg.baseline_alpha,
                    "baseline_established": baseline_end is not None,
                    "baseline_end_index": baseline_end, "baseline_end_t_s": None if baseline_end is None else float(run.t_s[baseline_end]),
                    "baseline_delay_updates": None if baseline_end is None else int(baseline_end - (Wb - 1)),
                    "nees_checked": False, "nees_consistent": None, "warnings": []}
    if gated and baseline_end is None:
        start = n            # nothing is tested
        health["warnings"].append("filter never established a consistent NIS baseline: innovations are not evidence of a maneuver")
    else:
        start = max(start0, baseline_end + 1) if gated else start0
        if gated and baseline_end - (Wb - 1) >= Wb:
            health["warnings"].append(f"baseline established late (after {baseline_end - (Wb - 1)} inconsistent updates): "
                                      "the filter was not tracking at first; treat the verdict with caution")

    # -- tests -------------------------------------------------------------------------------
    dets: list[Detection] = []
    skipped_gaps: list = []
    dets += nis_test(run, cfg.a_nis, cfg.m, start)
    dets += windowed_nis_test(run, cfg.a_window, W, cfg.m, start)
    dets += gap_refit_test(run, cfg, params, start=max(start - 1, 0), skipped=skipped_gaps)
    if cfg.cusum_enabled and n and start < n:
        dets += cusum_test(run, cfg, start)
    nees_summary = None
    nees_health_ok = True
    if truth is not None and n:
        nd, nees_summary = nees_monitor(run, truth, cfg.nees_alpha, start0)
        dets += nd
        # simulation-only health check over the baseline epochs (before any maneuver test)
        hi = (baseline_end + 1) if (gated and baseline_end is not None) else n
        sel = nees_summary["nees"][start0:hi]
        sel = sel[np.isfinite(sel)]
        if sel.size:
            bound = float(chi2.isf(cfg.nees_health_alpha, 6 * sel.size) / sel.size)
            gate_level = float(cfg.nees_health_max_ratio * 6)
            mean_b = float(sel.mean())
            health.update({"nees_checked": True, "nees_baseline_mean": mean_b, "nees_baseline_bound": bound,
                           "nees_baseline_gate_level": gate_level, "nees_baseline_epochs": int(sel.size),
                           "nees_consistent": bool(mean_b <= bound)})
            if mean_b > gate_level:
                nees_health_ok = False
                health["warnings"].append(f"covariance grossly inconsistent with truth over the baseline (mean NEES {mean_b:.1f} > "
                                          f"{gate_level:.0f} = {cfg.nees_health_max_ratio:g}× dof): verdict unreliable "
                                          "(simulation diagnostic; unavailable operationally)")
            elif mean_b > bound:
                health["warnings"].append(f"covariance optimistic over the baseline (mean NEES {mean_b:.1f} > {bound:.1f}, χ² bound at "
                                          f"α={cfg.nees_health_alpha:g}); verdict kept, treat covariances with caution")
    dets.sort(key=lambda d: (d.t_s, d.test))

    # -- declaration -------------------------------------------------------------------------
    n_tested = max(0, n - start)
    n_windows = max(0, n_tested - W + 1)
    gaps = [i for i in find_gaps(run.t_s, cfg.gap_hours_for_refit * 3600.0, cfg.gap_cadence_factor) if i > max(start - 1, 0)]
    n_family = n_tested + n_windows + len(gaps)
    thr_single = float(chi2.isf(cfg.a_nis, cfg.m))

    def fw(a: float) -> float:
        # One Šidák correction over the whole family of χ² tests performed in this run, so a
        # no-maneuver run is falsely *declared* with probability ≈ alpha (verified in the Pfa test).
        return 1.0 - (1.0 - a) ** (1.0 / n_family) if (n_family and cfg.declare_familywise) else a

    thr_fw = {"nis": float(chi2.isf(fw(cfg.a_nis), cfg.m)), "nis_window": float(chi2.isf(fw(cfg.a_window), cfg.m * W)),
              "gap_refit": float(chi2.isf(fw(cfg.a_gap), 6))}
    declaring = [d for d in dets if (d.test in thr_fw and d.statistic > thr_fw[d.test])
                 or (d.test == "cusum" and cfg.cusum_declares)]
    first_decl = min((d.t_s for d in declaring), default=None)
    maneuver_dets = [d for d in dets if d.test != "nees"]
    first_any = min((d.t_s for d in maneuver_dets), default=None)
    declared = bool(declaring)

    if n == 0:
        status = "no_observations"
    elif gated and baseline_end is None:
        status = "insufficient_updates" if n < Wb else "filter_not_converged"
        declared = False
    elif not nees_health_ok and cfg.nees_health_gates:
        status = "filter_inconsistent"
        if declared:
            health["warnings"].append("declaration suppressed: filter covariance inconsistent before any test")
        declared = False
    elif n_tested == 0:
        status = "insufficient_updates"
    else:
        status = "maneuver_declared" if declared else "quiet"
    if not declared:
        first_decl = None

    summary = {
        "status": status, "filter_health": health,
        "n_updates": n, "n_tested": n_tested, "test_start_index": int(start), "alpha": cfg.alpha,
        "alpha_per_test": {"nis": cfg.a_nis, "nis_window": cfg.a_window, "gap_refit": cfg.a_gap, "nees": cfg.nees_alpha},
        "window": W, "threshold_nis": thr_single, "threshold_nis_familywise": thr_fw["nis"], "thresholds_familywise": thr_fw,
        "alpha_familywise_per_test": {"nis": fw(cfg.a_nis), "nis_window": fw(cfg.a_window), "gap_refit": fw(cfg.a_gap)},
        "n_family": n_family, "cusum_declares": cfg.cusum_declares,
        "threshold_window": float(chi2.isf(cfg.a_window, cfg.m * W)), "threshold_gap_refit": float(chi2.isf(cfg.a_gap, 6)),
        "gap_hours_for_refit": cfg.gap_hours_for_refit, "gap_cadence_factor": cfg.gap_cadence_factor,
        "n_gaps": len(gaps), "gaps_skipped_arc_inconsistent": skipped_gaps,
        "counts": {t: sum(1 for d in dets if d.test == t) for t in ("nis", "nis_window", "gap_refit", "cusum", "nees")},
        "mean_nis": float(nis[start:].mean()) if n_tested else None,
        "max_nis": float(nis[start:].max()) if n_tested else None,
        "max_nis_t_s": float(run.t_s[start + int(np.argmax(nis[start:]))]) if n_tested else None,
        "nees": None if nees_summary is None else {k: v for k, v in nees_summary.items() if k not in ("nees", "pos_err_km", "vel_err_mps")},
        "elapsed_s": time.perf_counter() - tic,
        "filter": run.meta.get("filter") if isinstance(run.meta, dict) else None,
    }
    if nees_summary is not None:
        summary["nees_series"] = nees_summary["nees"]
        summary["pos_err_km"] = nees_summary["pos_err_km"]
        summary["vel_err_mps"] = nees_summary["vel_err_mps"]
    return DetectionReport(dets, summary, declared, first_any, first_decl, status)
