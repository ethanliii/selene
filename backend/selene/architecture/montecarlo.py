"""Monte Carlo scoring of candidate sensor architectures (PLAN §2.8, M8).

For every Monte Carlo *draw* the same object set, time window, prior covariance and injected
maneuvers are evaluated against every architecture, so architectures are compared on identical
conditions (common random numbers).  A draw randomises

* the **start epoch** ``t0`` uniformly inside the catalog's 14-day truth window (this phases
  every object along its orbit *and* the Moon along its synodic cycle, so lunar glare, Earth
  shadow and daylight conditions are sampled honestly; space platforms are phased with it),
* **unannounced impulsive maneuvers**: each SIMULATED object burns with Poisson probability
  ``1 − exp(−rate·horizon_days)`` (at most one burn per object is injected -- the first),
  at a slot node drawn uniformly in the first ``burn_window_fraction`` of the horizon, with
  |Δv| ~ U(dv_mps_range) and an isotropic direction (notional actor; no intent is assumed),
* the measurement-noise and acquisition realisations used by the detection surrogate.  These
  are drawn **once per draw as arrays indexed by (object, slot, i-th observation of that object
  in the slot)**, so two architectures that both observe object *j* at slot *k* use the very
  same acquisition uniform and innovation-noise vector; only the start epoch and burns are
  common where schedules differ (an architecture that observes at a slot the other skips simply
  consumes a cell the other never reads).

Metrics (all per architecture, then mean / p05 / p95 over draws)
-----------------------------------------------------------------
coverage %
    Fraction of (object, slot node) pairs at which **at least one** sensor of the architecture
    can see the object (geometry + photometry from :mod:`selene.sensors.visibility`, same table
    the tasker uses).  Says nothing about whether the sensor actually looked.
custody %
    From the greedy information-theoretic tasker (:mod:`selene.tasking.greedy`) run with the
    architecture's sensors from a *common stale prior* (defaults σ_pos = 25 km, σ_vel = 0.5 m/s:
    a catalog that is a few days old).  Custody at a node = RSS position sigma below
    ``custody_threshold_km`` (default 100 km); the percentage is a time fraction over all
    (object, node) cells (:func:`selene.tasking.metrics.custody_metrics`).  The zero-observation
    reference ``custody_pct_null`` is reported next to it.  Custody is evaluated on the *quiet*
    (no-burn) trajectories -- a linear covariance analysis cannot represent what a filter does
    with an un-modelled burn; that effect is captured by the detection latency instead.
revisit time
    Time between successive scheduled observations of an object (hours).  An object observed
    fewer than twice in the horizon has its revisit time **censored at the horizon length** (and
    is counted in ``n_objects_censored``) so an architecture that never revisits is penalised
    rather than ignored.  ``revisit_mean_h`` is the mean over objects of the per-object mean
    interval (censored objects at the horizon); ``revisit_max_h`` the longest interval;
    ``revisit_p95_h`` the 95th percentile of the **observed** intervals pooled over objects and
    draws (it discriminates architectures even when some objects are never revisited), and
    ``revisit_p95_censored_h`` the 95th percentile over (object, draw) cells of the per-object
    mean including censored cells -- at a 2-day horizon this is 48 h for every architecture
    that leaves at least 5 % of its (object, draw) cells unrevisited, so read it together with
    ``revisit_censored_objects_pct``.
maneuver-detection latency
    Time from the injected burn to the first scheduled post-burn observation that *detects*
    it, hours.  **Surrogate used** (the full UKF of :mod:`selene.od.ukf` + the detectors of
    :mod:`selene.maneuver.detection` cost ~1 s per object and run, far too slow for
    architectures × draws × objects):

    * the burn's effect on the predicted position is propagated **linearly** with the same
      ephemeris-model STM chain the tasker uses: δx(t_k) = Φ(t_k, t_b)·[0, Δv], i.e. the
      displacement is started at the burn node ``k_b`` and advanced with
      ``phi[k] = Φ(t_{k+1}, t_k)`` for ``k ≥ k_b`` only (verified against a nonlinear
      propagation of the burned truth in ``test_architecture_montecarlo.py``);
    * at a scheduled post-burn observation (sensor s, object j, node k) the filter's innovation
      is the on-sky angular displacement ν = H δx + w, with H the angles-only Jacobian
      (:func:`selene.tasking.information.angles_jacobian`) and w ~ N(0, S), S = H P⁻ Hᵀ + R
      from the **scheduled** pre-update covariance P⁻ (replayed sequentially within the slot
      exactly as the engine applies updates; the replay is checked against the engine's own
      post-update sigma and the mismatch reported as ``replay_sigma_max_rel_err``), R = σ²/n_obs;
    * detection = NIS νᵀ S⁻¹ ν > χ²₂(1 − α) (the per-update test of
      :func:`selene.maneuver.detection.nis_test`, same α and the same ``chi2.isf(α, 2)``
      threshold);
    * acquisition: the observation only yields an innovation when the sensor acquires the
      object (Bernoulli with the tasker's p_acq) **and** the displaced object is still inside
      the field of view when the sensor points at the prediction (|H δx| ≤ θ_fov/2).  A
      displaced object *outside* the field when the quiet-case acquisition probability was at
      least ``miss_p_acq_min`` (default 0.95) is an anomaly in its own right ("confidently
      expected, not there") and counts as a detection at that observation; a miss when the
      object was anyway uncertain does not.

    Burns that are not detected before the end of the horizon are *censored*:
    ``detections_pct`` is the detected fraction, ``detection_latency_*`` are over detected
    burns only, and ``detection_latency_censored_*`` count an undetected burn at the time left
    in the horizon (a lower bound, useful for ranking architectures that detect nothing).

    Quiet (no-burn / pre-burn) observations are sampled through the same test and their
    exceedances are reported as ``n_false_alarms`` / ``false_alarm_rate_per_obs``.  **This rate
    equals α by construction** -- the quiet innovation is drawn from the very S the test
    normalises by -- so it is bookkeeping (what fraction of this architecture's quiet
    observations would raise a per-update alarm at this α), not evidence that the surrogate
    reproduces the UKF.  The things that *are* checked are the linear displacement (nonlinear
    truth propagation), the replayed covariance (engine sigma series) and the threshold
    (``maneuver.detection.nis_test``).

    **Saturation.**  With σ = 1 arcsec, no model error and an exact displacement, a 1 m/s burn
    seen 20 min later from 400 000 km is already ~1 km ≈ 0.5 arcsec, and after a few hours tens
    of arcsec: for the default 1–20 m/s range the NIS test fires at essentially every acquired
    post-burn observation, so ``detection_latency_*`` reduces to *time to the first acquiring
    post-burn observation* and ``detections_pct`` to *was the object observed again at all*.
    The metric therefore measures the architecture's re-observation geometry (revisit, glare,
    daylight, FOV), not detector sensitivity; detector sensitivity only enters for burns well
    below 0.1 m/s or with much larger measurement noise.  The route's method notes say so.

Limitations.  Linear covariance analysis (no measurement realisations in the custody part; the
estimate mean stays on the truth); the tasker never re-plans after a detection; a burn's
nonlinear growth beyond the linear STM is ignored (fine for ≤ tens of m/s over days); no sensor
outages, weather or data latency; all sensor specifications are ASSUMED representative values.
"""
from __future__ import annotations

import atexit
import multiprocessing as mp
import os
import threading
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from concurrent.futures.process import BrokenProcessPool
from pickle import PicklingError
from dataclasses import dataclass, field, replace
from typing import Optional, Sequence

import numpy as np
from scipy.stats import chi2

from selene.architecture.candidates import Architecture
from selene.dynamics.frames import DEMO_EPOCH_TDB_S
from selene.objects.catalog import TRUTH_WINDOW_S
from selene.tasking.greedy import GreedyPolicy, Scenario, ScheduleResult, SlotContext, run_schedule
from selene.tasking.information import DEFAULT_Q_PSD, angles_jacobian, build_tracks, build_visibility_table, expected_gain
from selene.tasking.metrics import custody_metrics, null_custody_pct
from selene.tasking.scenario import resolve_objects, slot_grid

__all__ = [
    "ManeuverModel",
    "EvalConfig",
    "DrawScores",
    "ArchitectureScore",
    "EvaluationResult",
    "evaluate",
    "summarize",
    "burn_displacement",
    "METHOD_NOTES",
]

DAY_S = 86400.0


# ---------------------------------------------------------------------------
# configuration
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class ManeuverModel:
    """Random impulsive-maneuver model for the injected (SIMULATED, notional-actor) burns."""

    rate_per_object_per_day: float = 0.5
    dv_mps_range: tuple[float, float] = (1.0, 20.0)
    burn_window_fraction: float = 0.75     # burn nodes drawn from the first fraction of the horizon
    alpha: float = 0.01                    # per-update NIS false-alarm probability
    miss_p_acq_min: float = 0.95           # a miss counts as an anomaly only if quiet-case p_acq >= this

    def __post_init__(self):
        lo, hi = self.dv_mps_range
        if self.rate_per_object_per_day < 0:
            raise ValueError("rate_per_object_per_day must be >= 0")
        if not (0 <= lo <= hi):
            raise ValueError("dv_mps_range must satisfy 0 <= lo <= hi")
        if not (0 < self.burn_window_fraction <= 1):
            raise ValueError("burn_window_fraction must be in (0, 1]")
        if not (0 < self.alpha < 1):
            raise ValueError("alpha must be in (0, 1)")

    @property
    def nis_threshold(self) -> float:
        return float(chi2.ppf(1.0 - self.alpha, 2))

    def as_dict(self) -> dict:
        return {"rate_per_object_per_day": self.rate_per_object_per_day, "dv_mps_range": list(self.dv_mps_range),
                "burn_window_fraction": self.burn_window_fraction, "alpha": self.alpha,
                "nis_threshold_chi2_2": self.nis_threshold, "miss_p_acq_min": self.miss_p_acq_min,
                "direction": "isotropic (notional actor; no intent assumed)", "max_burns_per_object": 1}


@dataclass(frozen=True)
class EvalConfig:
    """Scenario numbers shared by every architecture and draw."""

    horizon_days: float = 2.0
    slot_min: float = 20.0
    custody_threshold_km: float = 100.0
    initial_sigma_km: float = 25.0
    initial_sigma_vel_kms: float = 5e-4
    q_psd: float = DEFAULT_Q_PSD
    sigma_arcsec: float = 1.0
    t0_s: float = DEMO_EPOCH_TDB_S
    phasing_span_days: Optional[float] = None    # None -> whole truth window minus the horizon
    gain_kind: str = "trace"
    acquisition: str = "fov"
    #: optional reference object population: overrides every object's catalog radius / albedo
    #: (what the studio page calls "target_radius_m" / "target_albedo"); None keeps catalog values
    target_radius_m: Optional[float] = None
    target_albedo: Optional[float] = None

    def __post_init__(self):
        if self.horizon_days <= 0:
            raise ValueError("horizon_days must be positive")
        if self.slot_min <= 0:
            raise ValueError("slot_min must be positive")
        if self.custody_threshold_km <= 0:
            raise ValueError("custody_threshold_km must be positive")
        if self.phasing_span_days is not None and self.phasing_span_days < 0:
            raise ValueError("phasing_span_days must be >= 0")
        if self.horizon_days * DAY_S > TRUTH_WINDOW_S[1] + 1e-6:
            raise ValueError(f"horizon_days={self.horizon_days} exceeds the {TRUTH_WINDOW_S[1] / DAY_S:.0f}-day truth window")
        # every draw window [t0 + offset, t0 + offset + horizon] must stay inside the catalog's cached
        # truth window (and the BodyCache span): the truth integrator clamps the Moon/Sun spline
        # outside it and would silently return wrong dynamics (verified: 12 000 km Moon error at +19 d)
        if self.phasing_span() + self.horizon_days * DAY_S > TRUTH_WINDOW_S[1] + 1e-6:
            raise ValueError(
                f"phasing_span_days ({self.phasing_span() / DAY_S:.2f}) + horizon_days ({self.horizon_days}) exceeds the "
                f"{TRUTH_WINDOW_S[1] / DAY_S:.0f}-day cached truth window; reduce one of them (default span = window - horizon)")
        if self.target_radius_m is not None and self.target_radius_m <= 0:
            raise ValueError("target_radius_m must be positive")
        if self.target_albedo is not None and not (0 < self.target_albedo <= 1):
            raise ValueError("target_albedo must be in (0, 1]")

    def phasing_span(self) -> float:
        """Span [s] of the uniform t0 offset (keeps every draw inside the cached truth window)."""
        if self.phasing_span_days is not None:
            return max(0.0, float(self.phasing_span_days) * DAY_S)
        return TRUTH_WINDOW_S[1] - self.horizon_days * DAY_S

    def as_dict(self) -> dict:
        return {"horizon_days": self.horizon_days, "slot_min": self.slot_min, "custody_threshold_km": self.custody_threshold_km,
                "initial_sigma_km": self.initial_sigma_km, "initial_sigma_vel_kms": self.initial_sigma_vel_kms,
                "q_psd_km2_s3": self.q_psd, "sigma_arcsec": self.sigma_arcsec, "t0_tdb_s": self.t0_s,
                "phasing_span_days": self.phasing_span() / DAY_S, "gain_kind": self.gain_kind, "acquisition": self.acquisition,
                "target_radius_m": self.target_radius_m, "target_albedo": self.target_albedo,
                "target_note": ("catalog per-object radius/albedo" if self.target_radius_m is None and self.target_albedo is None
                                else "reference object population overrides the catalog physical parameters")}


# ---------------------------------------------------------------------------
# per-draw results
# ---------------------------------------------------------------------------
@dataclass
class BurnRecord:
    object_id: str
    node: int
    t_burn_s: float
    dv_kms: np.ndarray
    detected: bool = False
    latency_h: float = float("nan")          # detected only
    censored_h: float = float("nan")         # time from burn to horizon end (used when undetected)
    detect_sensor: Optional[str] = None
    detect_how: Optional[str] = None         # 'nis' | 'miss'
    n_obs_after: int = 0

    def as_dict(self) -> dict:
        return {"object_id": self.object_id, "t_burn_s": self.t_burn_s, "dv_mps": float(np.linalg.norm(self.dv_kms) * 1e3),
                "dv_kms_gcrf": self.dv_kms.tolist(), "detected": self.detected,
                "latency_h": None if not self.detected else self.latency_h, "censored_h": self.censored_h,
                "detect_sensor": self.detect_sensor, "detect_how": self.detect_how, "n_obs_after_burn": self.n_obs_after}


@dataclass
class DrawScores:
    """Scores of one architecture on one draw."""

    draw: int
    t0_s: float
    coverage_pct: float
    custody_pct: float
    custody_pct_null: float
    mean_tslo_h: float
    revisit_mean_h: float
    revisit_max_h: float
    n_objects_censored: int
    n_observations: int
    n_quiet_obs_sampled: int
    n_false_alarms: int
    burns: list[BurnRecord]
    per_object: dict[str, dict]
    blocked_reasons: dict[str, int]
    runtime_s: float
    revisit_gaps_h: list[float] = field(default_factory=list)     # observed intervals pooled over objects
    replay_sigma_max_rel_err: float = 0.0                          # surrogate replay vs engine post-update sigma

    def as_dict(self) -> dict:
        return {
            "draw": self.draw, "t0_s": self.t0_s, "coverage_pct": self.coverage_pct, "custody_pct": self.custody_pct,
            "custody_pct_null": self.custody_pct_null, "mean_tslo_h": self.mean_tslo_h, "revisit_mean_h": self.revisit_mean_h,
            "revisit_max_h": self.revisit_max_h, "n_objects_censored": self.n_objects_censored,
            "n_observations": self.n_observations, "n_quiet_obs_sampled": self.n_quiet_obs_sampled,
            "n_false_alarms": self.n_false_alarms, "burns": [b.as_dict() for b in self.burns],
            "blocked_reasons": dict(self.blocked_reasons), "runtime_s": self.runtime_s,
            "n_revisit_gaps": len(self.revisit_gaps_h), "replay_sigma_max_rel_err": self.replay_sigma_max_rel_err,
        }


@dataclass
class ArchitectureScore:
    name: str
    architecture: dict
    draws: list[DrawScores]
    stats: dict = field(default_factory=dict)
    per_object: dict = field(default_factory=dict)
    notes: str = ""

    def as_dict(self, include_draws: bool = False) -> dict:
        d = {"name": self.name, "architecture": self.architecture, **self.stats, "per_object": self.per_object,
             "notes": self.notes}
        if include_draws:
            d["draws"] = [ds.as_dict() for ds in self.draws]
        return d


@dataclass
class EvaluationResult:
    scores: list[ArchitectureScore]
    object_ids: list[str]
    n_mc: int
    seed: int
    config: EvalConfig
    maneuver_model: ManeuverModel
    timing: dict
    draws_meta: list[dict]

    def as_dict(self, include_draws: bool = False) -> dict:
        return {
            "scores": [s.as_dict(include_draws) for s in self.scores],
            "per_object": {s.name: s.per_object for s in self.scores},
            "meta": {
                "n_mc": self.n_mc, "seed": self.seed, "object_ids": self.object_ids, "config": self.config.as_dict(),
                "maneuver_model": self.maneuver_model.as_dict(), "timing": self.timing, "draws": self.draws_meta,
                "method_notes": METHOD_NOTES,
            },
        }


METHOD_NOTES = [
    "Common random numbers: every architecture sees the same start epoch and injected burns per draw; acquisition and "
    "innovation-noise realisations are pre-drawn per (object, slot, i-th observation in the slot), so they coincide wherever "
    "two architectures observe the same object at the same slot.",
    "Start epoch per draw is uniform inside the catalog's 14-day truth window (phases objects, Moon and space platforms); "
    "phasing span + horizon is required to stay inside that window.",
    "coverage_pct = fraction of (object, slot node) pairs visible to >= 1 sensor (geometry + photometry), independent of tasking.",
    "custody_pct = greedy-tasker custody (RSS position sigma < threshold) from a common stale prior on quiet trajectories; "
    "custody_pct_null is the zero-observation reference.",
    "revisit = interval between successive scheduled observations of an object; revisit_mean_h censors objects observed < 2 times "
    "at the horizon; revisit_p95_h is the 95th percentile of the OBSERVED intervals pooled over objects and draws; "
    "revisit_p95_censored_h includes censored cells and saturates at the horizon when >= 5 % of cells are never revisited.",
    "detection latency SURROGATE: linear STM displacement of the burn dx(t_k) = Phi(t_k, t_b)[0, dv] started at the burn node, "
    "sampled innovation nu = H dx + w, w ~ N(0, H P- H^T + R) with the scheduled pre-update covariance replayed sequentially, "
    "per-update NIS test nu^T S^-1 nu > chi2_2(1-alpha) (same threshold as maneuver.detection.nis_test) at scheduled post-burn "
    "observations; a confidently expected object (quiet p_acq >= miss_p_acq_min) that is displaced out of the field of view counts "
    "as a detection ('miss'). The full UKF + maneuver.detection pipeline is not run here (too slow for architectures x draws x objects).",
    "SATURATION: with sigma = 1 arcsec and no model error the NIS test fires at essentially every acquired post-burn observation for "
    "burns >= ~0.1 m/s, so for the default 1-20 m/s range detection latency measures time-to-first-re-observation (revisit geometry, "
    "glare, daylight, FOV), not detector sensitivity. Do not present it as detector performance.",
    "Undetected burns are censored: detections_pct, latency over detected burns, and *_censored_* with the remaining horizon as a lower bound.",
    "false_alarm_rate_per_obs equals alpha by construction (quiet innovations are drawn from the S the test normalises by); it is "
    "bookkeeping of how many quiet observations would alarm at this alpha, NOT a calibration of the surrogate against the UKF. "
    "What is validated: the linear displacement against a nonlinear propagation of the burned truth, the replayed covariance "
    "against the engine's sigma series (replay_sigma_max_rel_err), and the threshold against maneuver.detection.nis_test.",
    "All simulated objects, burns and sensor specifications are notional (SIMULATED); no real spacecraft is ever given a burn.",
]


# ---------------------------------------------------------------------------
# recording greedy policy (captures the pre-update covariance at each slot)
# ---------------------------------------------------------------------------
class _RecordingGreedy(GreedyPolicy):
    def __init__(self):
        super().__init__()
        self.P_minus: dict[int, list[np.ndarray]] = {}

    def __call__(self, k: int, ctx: SlotContext) -> dict:
        self.P_minus[k] = [P.copy() for P in ctx.P]
        return super().__call__(k, ctx)


# ---------------------------------------------------------------------------
# metric helpers
# ---------------------------------------------------------------------------
def _coverage(table) -> tuple[float, np.ndarray]:
    vis = table.visible[:, :, :-1] if table.visible.shape[2] > 1 else table.visible
    any_vis = vis.any(axis=0)                     # (J, K)
    return float(any_vis.mean()) if any_vis.size else 0.0, any_vis.mean(axis=1)


def _revisit(res: ScheduleResult, horizon_h: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[float]]:
    """Per-object (mean interval h, max interval h, censored flag) and the pooled observed intervals."""
    J = res.scenario.n_objects
    times: list[list[float]] = [[] for _ in range(J)]
    idx = {oid: j for j, oid in enumerate(res.object_ids)}
    for a in res.assignments:
        times[idx[a.object_id]].append(a.t_s)
    mean_h = np.full(J, horizon_h)
    max_h = np.full(J, horizon_h)
    censored = np.ones(J, dtype=bool)
    pooled: list[float] = []
    for j in range(J):
        ts = np.unique(np.asarray(times[j], dtype=np.float64))
        if ts.size >= 2:
            gaps = np.diff(ts) / 3600.0
            mean_h[j], max_h[j], censored[j] = float(gaps.mean()), float(gaps.max()), False
            pooled.extend(float(g) for g in gaps)
    return mean_h, max_h, censored, pooled


@dataclass(frozen=True)
class _Noise:
    """Pre-drawn per-draw noise, indexed [object, slot, i-th observation of the object in the slot].

    Shared by every architecture of a draw (common random numbers); ``u`` acquisition uniforms,
    ``z`` standard-normal innovation vectors.  ``depth`` >= the largest number of sensors of any
    architecture, so an index never overflows."""

    u: np.ndarray      # (J, K, depth)
    z: np.ndarray      # (J, K, depth, 2)

    @staticmethod
    def draw(rng: np.random.Generator, n_objects: int, n_slots: int, depth: int) -> "_Noise":
        depth = max(1, int(depth))
        return _Noise(u=rng.uniform(size=(n_objects, n_slots, depth)), z=rng.standard_normal((n_objects, n_slots, depth, 2)))


def _sample_burns(rng: np.random.Generator, object_ids: Sequence[str], t_nodes: np.ndarray, model: ManeuverModel,
                  horizon_days: float) -> list[Optional[BurnRecord]]:
    K = t_nodes.size - 1
    k_max = max(1, int(np.floor(model.burn_window_fraction * K)))
    k_max = min(k_max, K - 1) if K >= 2 else 0
    out: list[Optional[BurnRecord]] = []
    for oid in object_ids:
        n_burn = rng.poisson(model.rate_per_object_per_day * horizon_days)
        # always consume the same number of random draws so that the burn pattern is reproducible per object
        kb = int(rng.integers(1, k_max + 1)) if k_max >= 1 else 0
        mag = rng.uniform(model.dv_mps_range[0], model.dv_mps_range[1]) * 1e-3
        u = rng.standard_normal(3)
        u /= max(np.linalg.norm(u), 1e-300)
        if n_burn < 1 or kb < 1:
            out.append(None)
            continue
        out.append(BurnRecord(oid, kb, float(t_nodes[kb]), mag * u, censored_h=float(t_nodes[-1] - t_nodes[kb]) / 3600.0))
    return out


def burn_displacement(track, k_burn: int, dv_kms: np.ndarray) -> np.ndarray:
    """Linear displacement δx(t_k) = Φ(t_k, t_b)·[0, Δv] on every node of ``track`` ((K+1, 6); zero for k <= k_burn).

    Uses the track's step STMs ``phi[k] = Φ(t_{k+1}, t_k)`` starting **at the burn node** (this is
    the quantity the detection surrogate replays; exposed so it can be checked against a
    nonlinear propagation of the burned truth)."""
    K = track.n_slots
    out = np.zeros((K + 1, 6))
    if k_burn >= K:
        return out
    d = np.concatenate([np.zeros(3), np.asarray(dv_kms, dtype=np.float64)])
    for k in range(int(k_burn), K):
        d = track.phi[k] @ d
        out[k + 1] = d
    return out


def _detect(res: ScheduleResult, policy: _RecordingGreedy, burns: list[Optional[BurnRecord]], model: ManeuverModel,
            noise: _Noise) -> tuple[int, int, float]:
    """Replay the scheduled observations with the burn displacement (see module docstring).

    Mutates the ``BurnRecord``s; returns (n_quiet_obs_sampled, n_false_alarms,
    replay_sigma_max_rel_err) where the last is the largest relative mismatch between the
    replayed post-update RSS position sigma and the engine's own ``sigma_km`` at observed nodes
    (a self-check that S is built from the covariance the tasker actually scheduled with)."""
    scn = res.scenario
    tab = scn.table
    thr = model.nis_threshold
    sidx = {sid: s for s, sid in enumerate(tab.sensor_ids)}
    oidx = {oid: j for j, oid in enumerate(tab.object_ids)}
    by_slot: dict[int, list] = {}
    for a in res.assignments:
        by_slot.setdefault(a.slot, []).append(a)
    burned_at = {oidx[b.object_id]: b for b in burns if b is not None}
    # linear burn displacement δx(t_k): created at the burn node and advanced with phi[k] for k >= k_b only
    delta: dict[int, np.ndarray] = {}
    n_quiet = 0
    n_fa = 0
    max_rel = 0.0
    for k in range(scn.n_slots):
        slots = by_slot.get(k, [])
        if slots:
            P_local = [P.copy() for P in policy.P_minus[k]]
            n_in_slot = np.zeros(len(P_local), dtype=int)
            for a in slots:
                s, j = sidx[a.sensor_id], oidx[a.object_id]
                i = int(n_in_slot[j])
                n_in_slot[j] += 1
                tr = scn.tracks[j]
                P = P_local[j]
                H = angles_jacobian(tab.obs_pos[s, k], tr.x_nodes[k, :3])
                r2 = float(tab.sigma_rad[s]) ** 2 / float(scn.n_obs_per_slot)
                S = H @ P @ H.T + np.eye(2) * r2
                S = 0.5 * (S + S.T)
                L = np.linalg.cholesky(S)
                b = burned_at.get(j)
                post_burn = b is not None and k > b.node and not b.detected
                acquired = noise.u[j, k, i] < a.p_acq
                z = noise.z[j, k, i]
                if post_burn:
                    b.n_obs_after += 1
                    dtheta = H @ delta[j]
                    in_fov = float(np.linalg.norm(dtheta)) <= float(tab.fov_half_rad[s])
                    if acquired and in_fov:
                        nu = dtheta + L @ z
                        nis = float(nu @ np.linalg.solve(S, nu))
                        if nis > thr:
                            b.detected, b.detect_how = True, "nis"
                    elif acquired and not in_fov and a.p_acq >= model.miss_p_acq_min:
                        b.detected, b.detect_how = True, "miss"
                    if b.detected:
                        b.latency_h = float(tab.t_nodes[k] - b.t_burn_s) / 3600.0
                        b.detect_sensor = a.sensor_id
                elif acquired and (b is None or k <= b.node):
                    # quiet observation (no burn, or before the burn).  nu ~ N(0, S) so the exceedance
                    # rate is alpha by construction (bookkeeping, not calibration -- see module docstring);
                    # observations after a declared detection are not tested (the tasker does not re-plan)
                    n_quiet += 1
                    nu = L @ z
                    if float(nu @ np.linalg.solve(S, nu)) > thr:
                        n_fa += 1
                # expected (scheduled) update, sequential within the slot as the engine does
                _g, _p, P_plus = expected_gain(tab, s, j, k, P, tr.x_nodes[k], kind=scn.gain_kind,
                                               acquisition=scn.acquisition, n_obs=scn.n_obs_per_slot, n_tiles=scn.search_tiles)
                if P_plus is not None:
                    P_local[j] = P_plus
            # self-check: the replayed post-update sigma must equal the engine's recorded sigma at this node
            for j in np.nonzero(n_in_slot)[0]:
                eng = float(res.sigma_km[j, k])
                rep = float(np.sqrt(max(np.trace(P_local[j][:3, :3]), 0.0)))
                max_rel = max(max_rel, abs(rep - eng) / max(eng, 1e-12))
        # advance the burn displacements to node k+1 (started at the burn node, not at t0)
        for j, b in burned_at.items():
            if k == b.node:
                delta[j] = scn.tracks[j].phi[k] @ np.concatenate([np.zeros(3), b.dv_kms])
            elif k > b.node:
                delta[j] = scn.tracks[j].phi[k] @ delta[j]
    return n_quiet, n_fa, max_rel


# ---------------------------------------------------------------------------
# one (draw, architecture)
# ---------------------------------------------------------------------------
def _score_architecture(arch: Architecture, tracks, t_nodes: np.ndarray, burns_template: list[Optional[BurnRecord]],
                        cfg: EvalConfig, model: ManeuverModel, noise: _Noise, draw: int,
                        custody_null: Optional[float]) -> tuple[DrawScores, float]:
    t_a = time.perf_counter()
    sensors = arch.all_sensors()
    table = build_visibility_table(sensors, tracks, sigma_arcsec=cfg.sigma_arcsec)
    scn = Scenario(tracks=tracks, table=table, slot_s=cfg.slot_min * 60.0, gain_kind=cfg.gain_kind,  # type: ignore[arg-type]
                   acquisition=cfg.acquisition)  # type: ignore[arg-type]
    if custody_null is None:
        custody_null = null_custody_pct(scn, cfg.custody_threshold_km)
    policy = _RecordingGreedy()
    res = run_schedule(scn, policy, method="greedy", meta={"gain_kind": cfg.gain_kind})
    m = custody_metrics(res, cfg.custody_threshold_km)
    horizon_h = float(t_nodes[-1] - t_nodes[0]) / 3600.0
    cov, cov_obj = _coverage(table)
    rev_mean, rev_max, censored, gaps = _revisit(res, horizon_h)
    burns = [None if b is None else BurnRecord(b.object_id, b.node, b.t_burn_s, b.dv_kms.copy(), censored_h=b.censored_h)
             for b in burns_template]
    n_quiet, n_fa, replay_err = _detect(res, policy, burns, model, noise)
    per_object = {}
    for j, tr in enumerate(scn.tracks):
        po = m["per_object"][j]
        b = burns[j]
        per_object[tr.object_id] = {
            "coverage_pct": 100.0 * float(cov_obj[j]), "custody_pct": po["custody_pct"], "n_obs": po["n_obs"],
            "mean_tslo_h": po["mean_tslo_h"], "revisit_mean_h": float(rev_mean[j]), "revisit_censored": bool(censored[j]),
            "burn": None if b is None else b.as_dict(),
        }
    blocked: dict[str, int] = {}
    for counts in res.idle_reason_counts.values():
        for name, n in counts.items():
            blocked[name] = blocked.get(name, 0) + int(n)
    ds = DrawScores(
        draw=draw, t0_s=float(t_nodes[0]), coverage_pct=100.0 * cov, custody_pct=m["overall"]["custody_pct"],
        custody_pct_null=float(custody_null), mean_tslo_h=m["overall"]["mean_tslo_h"],
        revisit_mean_h=float(rev_mean.mean()), revisit_max_h=float(rev_max.max()), n_objects_censored=int(censored.sum()),
        n_observations=len(res.assignments), n_quiet_obs_sampled=n_quiet, n_false_alarms=n_fa,
        burns=[b for b in burns if b is not None], per_object=per_object, blocked_reasons=blocked,
        runtime_s=time.perf_counter() - t_a, revisit_gaps_h=gaps, replay_sigma_max_rel_err=replay_err,
    )
    return ds, float(custody_null)


def _run_draw(d: int, seed_d: int, archs: Sequence[Architecture], object_ids: Sequence[str], cfg: EvalConfig,
              model: ManeuverModel) -> tuple[list[DrawScores], dict]:
    t_a = time.perf_counter()
    rng = np.random.default_rng(seed_d)
    offset = rng.uniform(0.0, cfg.phasing_span())
    t0 = cfg.t0_s + offset
    t_nodes = slot_grid(t0, t0 + cfg.horizon_days * DAY_S, cfg.slot_min)
    tracks = build_tracks(object_ids, t_nodes, q_psd=cfg.q_psd, sigma0_pos_km=cfg.initial_sigma_km,
                          sigma0_vel_kms=cfg.initial_sigma_vel_kms)
    if cfg.target_radius_m is not None or cfg.target_albedo is not None:
        # reference object population (studio "target_radius_m / target_albedo"): overrides catalog physical parameters
        tracks = [replace(tr, radius_m=float(cfg.target_radius_m) if cfg.target_radius_m is not None else tr.radius_m,
                          albedo=float(cfg.target_albedo) if cfg.target_albedo is not None else tr.albedo) for tr in tracks]
    burns = _sample_burns(rng, object_ids, t_nodes, model, cfg.horizon_days)
    t_b = time.perf_counter()
    # common random numbers: one noise array set per draw, indexed by (object, slot, i-th observation), shared by every architecture
    depth = max(len(a.all_sensors()) for a in archs)
    noise = _Noise.draw(rng, len(object_ids), int(t_nodes.size - 1), depth)
    out = []
    custody_null = None
    for a in archs:
        ds, custody_null = _score_architecture(a, tracks, t_nodes, burns, cfg, model, noise, d, custody_null)
        out.append(ds)
    meta = {"draw": d, "seed": seed_d, "t0_offset_days": offset / DAY_S, "t0_tdb_s": float(t0), "n_slots": int(t_nodes.size - 1),
            "n_burns": int(sum(b is not None for b in burns)),
            "burns": [b.as_dict() for b in burns if b is not None],
            "tracks_s": t_b - t_a, "total_s": time.perf_counter() - t_a}
    return out, meta


# ---------------------------------------------------------------------------
# aggregation
# ---------------------------------------------------------------------------
def _stat(values: Sequence[float]) -> dict:
    v = np.asarray([x for x in values if x is not None and np.isfinite(x)], dtype=np.float64)
    if v.size == 0:
        return {"mean": None, "p05": None, "p95": None, "n": 0}
    return {"mean": float(v.mean()), "p05": float(np.percentile(v, 5)), "p95": float(np.percentile(v, 95)), "n": int(v.size)}


def summarize(name: str, arch: Architecture, draws: list[DrawScores]) -> ArchitectureScore:
    per_draw = {
        "coverage_pct": [d.coverage_pct for d in draws],
        "custody_pct": [d.custody_pct for d in draws],
        "custody_pct_null": [d.custody_pct_null for d in draws],
        "mean_tslo_h": [d.mean_tslo_h for d in draws],
        "revisit_mean_h": [d.revisit_mean_h for d in draws],
        "revisit_max_h": [d.revisit_max_h for d in draws],
        "n_observations": [d.n_observations for d in draws],
    }
    stats = {}
    for key, vals in per_draw.items():
        st = _stat(vals)
        stats[key] = st["mean"]
        stats[f"{key}_p05"] = st["p05"]
        stats[f"{key}_p95"] = st["p95"]
    # revisit_p95_h: 95th percentile of the OBSERVED intervals pooled over objects and draws (discriminates architectures);
    # revisit_p95_censored_h: over (object, draw) cells of the per-object mean, censored cells at the horizon (saturates)
    pooled = [g for d in draws for g in d.revisit_gaps_h]
    cell_rev = [po["revisit_mean_h"] for d in draws for po in d.per_object.values()]
    stats["revisit_p95_h"] = _stat(pooled)["p95"]
    stats["revisit_p95_censored_h"] = _stat(cell_rev)["p95"]
    stats["n_revisit_intervals"] = len(pooled)
    stats["revisit_censored_objects_pct"] = (100.0 * sum(d.n_objects_censored for d in draws)
                                             / max(1, sum(len(d.per_object) for d in draws)))
    stats["replay_sigma_max_rel_err"] = float(max((d.replay_sigma_max_rel_err for d in draws), default=0.0))
    burns = [b for d in draws for b in d.burns]
    detected = [b for b in burns if b.detected]
    lat = _stat([b.latency_h for b in detected])
    lat_c = _stat([b.latency_h if b.detected else b.censored_h for b in burns])
    stats.update({
        "n_burns": len(burns), "n_detected": len(detected),
        "detections_pct": 100.0 * len(detected) / len(burns) if burns else None,
        "detections_by_nis": sum(1 for b in detected if b.detect_how == "nis"),
        "detections_by_miss": sum(1 for b in detected if b.detect_how == "miss"),
        "detection_latency_mean_h": lat["mean"], "detection_latency_p05_h": lat["p05"], "detection_latency_p95_h": lat["p95"],
        "detection_latency_censored_mean_h": lat_c["mean"], "detection_latency_censored_p95_h": lat_c["p95"],
        "n_quiet_obs_sampled": int(sum(d.n_quiet_obs_sampled for d in draws)),
        "n_false_alarms": int(sum(d.n_false_alarms for d in draws)),
    })
    nq = stats["n_quiet_obs_sampled"]
    stats["false_alarm_rate_per_obs"] = stats["n_false_alarms"] / nq if nq else None
    stats["n_draws"] = len(draws)
    stats["runtime_s"] = float(sum(d.runtime_s for d in draws))
    stats.update(arch.cost_proxies())
    n_sp = max(1, arch.n_space_sensors)
    stats["custody_pct_per_space_sensor"] = (stats["custody_pct"] / n_sp) if stats["custody_pct"] is not None and arch.n_space_sensors else None
    stats["coverage_pct_per_space_sensor"] = (stats["coverage_pct"] / n_sp) if stats["coverage_pct"] is not None and arch.n_space_sensors else None
    # per-object pooled over draws
    per_object: dict[str, dict] = {}
    oids = list(draws[0].per_object) if draws else []
    for oid in oids:
        rows = [d.per_object[oid] for d in draws]
        ob = [b for d in draws for b in d.burns if b.object_id == oid]
        od = [b for b in ob if b.detected]
        per_object[oid] = {
            "coverage_pct": float(np.mean([r["coverage_pct"] for r in rows])),
            "custody_pct": float(np.mean([r["custody_pct"] for r in rows])),
            "mean_tslo_h": float(np.mean([r["mean_tslo_h"] for r in rows])),
            "revisit_mean_h": float(np.mean([r["revisit_mean_h"] for r in rows])),
            "revisit_censored_pct": 100.0 * float(np.mean([r["revisit_censored"] for r in rows])),
            "n_obs_mean": float(np.mean([r["n_obs"] for r in rows])),
            "n_burns": len(ob), "n_detected": len(od),
            "detection_latency_mean_h": _stat([b.latency_h for b in od])["mean"],
        }
    blocked: dict[str, int] = {}
    for d in draws:
        for k, v in d.blocked_reasons.items():
            blocked[k] = blocked.get(k, 0) + v
    top = sorted(blocked.items(), key=lambda kv: -kv[1])[:3]
    stats["top_blocking_reasons"] = [{"reason": k, "count": v} for k, v in top]
    notes = arch.notes
    if top:
        notes = (notes + " " if notes else "") + "Top visibility blockers: " + ", ".join(f"{k} ({v})" for k, v in top) + "."
    if stats["n_burns"] and not stats["n_detected"]:
        notes += " No injected maneuver was detected within the horizon."
    if stats["revisit_censored_objects_pct"] > 0:
        notes += (f" {stats['revisit_censored_objects_pct']:.0f} % of (object, draw) cells were observed fewer than twice "
                  f"(revisit censored at the horizon).")
    return ArchitectureScore(name=name, architecture=arch.as_dict(), draws=draws, stats=stats, per_object=per_object, notes=notes)


# ---------------------------------------------------------------------------
# persistent spawn process pool (draws are independent; each worker builds the catalog truth once)
# ---------------------------------------------------------------------------
_POOL: Optional[ProcessPoolExecutor] = None
_POOL_WORKERS = 0
_POOL_LOCK = threading.Lock()
MAX_PROCESS_WORKERS = 4


def _run_draw_task(args) -> tuple[int, tuple[list[DrawScores], dict]]:
    """Picklable per-draw task for the process pool (module-level so ``spawn`` can import it)."""
    d, seed_d, archs, ids, cfg, model = args
    return d, _run_draw(d, seed_d, archs, ids, cfg, model)


def _process_pool(workers: int) -> ProcessPoolExecutor:
    """Lazily created, reused ``spawn`` pool (grown if a later call asks for more workers)."""
    global _POOL, _POOL_WORKERS
    workers = max(1, min(int(workers), MAX_PROCESS_WORKERS, os.cpu_count() or 1))
    with _POOL_LOCK:
        if _POOL is None or _POOL_WORKERS < workers:
            if _POOL is not None:
                _POOL.shutdown(wait=False, cancel_futures=True)
            _POOL = ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context("spawn"))
            _POOL_WORKERS = workers
        return _POOL


def _reset_process_pool() -> None:
    global _POOL, _POOL_WORKERS
    with _POOL_LOCK:
        if _POOL is not None:
            try:
                _POOL.shutdown(wait=False, cancel_futures=True)
            except Exception:  # pragma: no cover
                pass
        _POOL, _POOL_WORKERS = None, 0


atexit.register(_reset_process_pool)


# ---------------------------------------------------------------------------
# public entry point
# ---------------------------------------------------------------------------
def evaluate(
    architectures: Sequence[Architecture],
    n_mc: int = 8,
    horizon_days: Optional[float] = None,
    seed: int = 0,
    object_set: str | Sequence[str] = "all_simulated",
    maneuver_model: ManeuverModel | dict | None = None,
    config: EvalConfig | None = None,
    workers: int = 1,
    executor: str = "process",
) -> EvaluationResult:
    """Score ``architectures`` over ``n_mc`` common-random-number draws (see module docstring).

    ``horizon_days`` overrides ``config.horizon_days`` when given (default 2 days without a
    config).  ``object_set`` is ``'all_simulated'`` or a list of SIMULATED object ids (real
    Horizons objects are refused: no burn is ever simulated on a real spacecraft, and their
    cached spans do not cover the randomised windows).

    ``workers > 1`` runs the draws in parallel; every draw owns its own generator so the result
    is bit-identical to the sequential run whatever the executor.  ``executor='process'`` (the
    default) uses a module-level, lazily created, persistent ``spawn`` process pool (``fork`` is
    not used: unsafe inside a threaded server); ``'thread'`` uses a thread pool.  Measured on the
    4 presets × 8 draws × 2 days (15-core laptop): sequential 9.8 s, 4 threads 8.5 s (GIL-bound),
    4 spawn processes 4.7 s cold (interpreter + catalog truth build per worker) and ~2.5 s warm
    (pool reused).  A broken pool falls back to the sequential path and says so in ``timing``.
    """
    t_start = time.perf_counter()
    if not architectures:
        raise ValueError("at least one architecture is required")
    names = [a.name for a in architectures]
    if len(set(names)) != len(names):
        raise ValueError(f"architecture names must be unique: {names}")
    n_mc = int(n_mc)
    if n_mc < 1:
        raise ValueError("n_mc must be >= 1")
    if isinstance(maneuver_model, dict):
        mm = dict(maneuver_model)
        if "dv_mps_range" in mm:
            mm["dv_mps_range"] = tuple(float(v) for v in mm["dv_mps_range"])
        maneuver_model = ManeuverModel(**mm)
    model = maneuver_model or ManeuverModel()
    cfg = config if config is not None else EvalConfig()
    if horizon_days is not None:
        cfg = replace(cfg, horizon_days=float(horizon_days))
    from selene.objects.catalog import get_catalog
    cat = get_catalog()
    ids = resolve_objects(object_set, cat)
    real = [i for i in ids if cat.get(i).notional is None]
    if real:
        raise ValueError(f"architecture studies run on SIMULATED objects only; real (Horizons) objects refused: {real}")
    for a in architectures:
        a.all_sensors()  # validates: raises on an empty architecture / unknown platform
    seeds = [int(s.generate_state(1)[0]) for s in np.random.SeedSequence(int(seed)).spawn(n_mc)]
    per_arch: list[list[DrawScores]] = [[] for _ in architectures]
    metas: list[dict] = [{} for _ in range(n_mc)]

    def work(d: int):
        return d, _run_draw(d, seeds[d], architectures, ids, cfg, model)

    t_setup = time.perf_counter() - t_start
    used = "sequential"
    note = ""
    if workers and workers > 1 and n_mc > 1:
        if executor == "process":
            try:
                pool = _process_pool(int(workers))
                results = list(pool.map(_run_draw_task, [(d, seeds[d], list(architectures), list(ids), cfg, model) for d in range(n_mc)]))
                used = f"process x{min(int(workers), n_mc)}"
            except (BrokenProcessPool, OSError, RuntimeError, PicklingError) as e:   # pragma: no cover - environment dependent
                _reset_process_pool()
                note = f"process pool unavailable ({type(e).__name__}: {e}); ran sequentially"
                results = [work(d) for d in range(n_mc)]
        elif executor == "thread":
            with ThreadPoolExecutor(max_workers=int(workers)) as ex:
                results = list(ex.map(work, range(n_mc)))
            used = f"thread x{min(int(workers), n_mc)}"
        else:
            raise ValueError("executor must be 'process' or 'thread'")
    else:
        results = [work(d) for d in range(n_mc)]
    for d, (scores, meta) in sorted(results, key=lambda r: r[0]):
        metas[d] = meta
        for i, ds in enumerate(scores):
            per_arch[i].append(ds)
    scores = [summarize(a.name, a, per_arch[i]) for i, a in enumerate(architectures)]
    timing = {"setup_s": t_setup, "total_s": time.perf_counter() - t_start, "workers": int(workers), "executor": used,
              "per_draw_s": [m["total_s"] for m in metas], "mean_draw_s": float(np.mean([m["total_s"] for m in metas]))}
    if note:
        timing["note"] = note
    return EvaluationResult(scores=scores, object_ids=ids, n_mc=n_mc, seed=int(seed), config=cfg, maneuver_model=model,
                            timing=timing, draws_meta=metas)
