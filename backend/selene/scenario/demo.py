"""Scripted 2-minute demo scenario (Milestone 9): a notional-actor spacecraft in a DRO performs an
unannounced burn, the ground network loses it in lunar glare, its uncertainty balloons, reachability
says where it could go, the SELENE tasker redirects a space-based observer, custody is regained and
the maneuver is characterised; an analyst brief is generated.  **Everything is SIMULATED** and
attributed to a notional actor.

Nothing in the story is scripted *numerically*: every state, visibility verdict, residual,
covariance, reachable set, schedule and estimate below is produced by the project's engines
(:mod:`selene.dynamics`, :mod:`selene.sensors`, :mod:`selene.od`, :mod:`selene.maneuver`,
:mod:`selene.reachability`, :mod:`selene.tasking`).  What *is* chosen by hand, and why
(``meta.epoch_choice_rationale`` repeats this):

* **Epoch** ``2026-02-23T00:00 → 2026-03-01T00:00 UTC`` (6 days, hourly frames).  Scanning the cached
  window (2026-02-15 .. 03-30) with the real visibility model shows the DRO object stays within 12°
  of the Moon on the sky; under the phase-dependent lunar-glare model (3° at new Moon → 15° at full
  Moon, ``sensors/constraints.py``) the nine ground sites see it only on the nights of Feb 23–25
  (illuminated fraction 0.33 → 0.6, object 10–11° from the Moon) and are blind from Feb 25 ~11 UTC
  through full Moon (Mar 3) and beyond.  The story therefore starts with routine nightly ground
  tracklets and loses the object exactly when the Moon brightens — a computed outcome, not a script.
* **Burn** ``2026-02-25T08:30 UTC``, 30 m/s (notional actor; the magnitude is an assumption inside
  the 20–40 m/s class).  The *direction* was taken from the reachability engine: among 256 sampled
  30 m/s burn directions it is the one whose coasting arc passes closest to the Earth-Moon L1 point
  (≈ 260 km at +41.8 h, refined on the dense solution — ``metrics.truth_geometry`` carries the
  computed value — against ≈ 1 400–4 000 km for the unperturbed DRO over the cached span).  Note
  that the burned arc *grazes* L1 and stays in the lunar realm: it does **not** transit the L1 neck
  (``reachability/regions.py`` gateway definition), and the bundle says so (``closest_approach``
  truth event, ``truth_geometry.l1_transit``).  It is applied 30 min after the last pre-burn ground
  tracklet so that exactly one post-burn ground observation (Siding Spring, 10:00 UTC) catches the
  anomaly before the glare closes the ground window.
* **Reachability budget** 100 m/s / 168 h for the alert: an *assumed* analyst planning budget
  bounding the plausible single-maneuver class of a medium bus (the 30 m/s truth lies inside it).
  Reported fractions are what the sampler returns, including the honest results that the NRHO relay
  corridor is reachable only at the budget limit and that an L1 **neck transit** (realm change through
  the L1 neighbourhood, ``reachability/regions.py``) is reached by only a small share of the sampled
  burns, at the budget limit; the alert and the brief quote the transit count, the closest approach of
  the reachable set to L1 and the number of samples that merely dip past the x = x_L1 plane and return
  (``metrics.reachability.gateway_diagnostics``).  An earlier version of the classifier counted those
  dips as transits (14 % of rays at >= 10 m/s); that headline was a classification artefact.
* **Custody thresholds** on the particle-cloud RSS position sigma: CUSTODY < 100 km (the tasking
  module's default), DEGRADED 100–1000 km, LOST > 1000 km.
* **Covariance re-opening** after a detection (``_Reopen``): the filter prior is widened by the
  covariance of an impulsive Δv of unknown size (σ = 25 m/s, an assumption that must bound the
  burn class) applied at an unknown epoch uniformly distributed in the last observation gap,
  ``P += σ²·[[Δt²/3 I, Δt/2 I], [Δt/2 I, I]]``.  This is what makes the cloud balloon honestly: a
  single post-burn angles-only tracklet cannot resolve the Δv.

Outputs: the bundle dict documented in :func:`build_scenario` (frames, events, metrics, brief), written
to ``data/demo/scenario.json`` + ``scenario_meta.json`` by ``python -m selene.scenario.demo --rebuild``.
"""
from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

from selene.constants import DATA_DIR, L_STAR, T_STAR
from selene.dynamics import frames
from selene.dynamics.ephemeris import EphemParams, get_ephemeris
from selene.maneuver.config import DetectorConfig, EstimatorConfig
from selene.maneuver.detection import detect
from selene.maneuver.estimation import estimate_impulsive_dv
from selene.maneuver.synthetic import Burn, generate_measurements, initial_covariance, sample_initial_state, truth_with_burn
from selene.objects.catalog import get_catalog
from selene.od.particles import linear_covariance, propagate_cloud
from selene.od.ukf import UKFConfig, ukf_run
from selene.reachability.regions import RegionInputs, default_regions, neck_passage_diagnostics
from selene.reachability.sampling import ReachabilityConfig, compute_reachability, refine_min_dv
from selene.reachability.tasking_hint import sensor_hints
from selene.scenario.brief import gateway_sentence, generate_brief
from selene.sensors.constraints import angular_separation, illuminated_fraction
from selene.sensors.observers import DEFAULT_OBSERVERS, observer_state
from selene.sensors.reasons import AVAILABILITY_REASONS, REASON_NAMES, primary_reason, reason_names
from selene.sensors.sites import DEFAULT_SITES, site_frame
from selene.sensors.visibility import evaluate, observe, sensor_geometry
from selene.tasking.greedy import run_greedy
from selene.tasking.scenario import make_scenario
from selene.time import J2000_JD, seconds_since_j2000_tdb

__all__ = ["build_scenario", "write_bundle", "DEMO_DIR", "SCENARIO_PATH", "META_PATH", "DemoConfig"]

DEMO_DIR = DATA_DIR / "demo"
SCENARIO_PATH = DEMO_DIR / "scenario.json"
META_PATH = DEMO_DIR / "scenario_meta.json"

HOUR = 3600.0
ARCSEC = np.pi / (180.0 * 3600.0)
PROTAGONIST = "SIM-DRO-01"
RELAY = "SIM-NRHO-RELAY-01"
TITLE = "Unannounced DRO departure — custody loss and recovery (SIMULATED)"
DISCLAIMER = ("SIMULATED scenario. Every object, event and burn is notional and attributed to a 'notional actor' "
              "or 'notional allied operator'; no real country, operator or spacecraft is implicated. Sensor "
              "specifications are assumed representative values. This is a defensive awareness and traffic-safety "
              "product: reachability and tasking describe where an object could be and where to look, never intent "
              "or engagement.")

#: Per-frame ground-blind label when no site has the object above its elevation limit at night (not a glare effect).
NO_SITE_AVAILABLE = "no_site_available"
#: Event kinds emitted beyond the spec list: the SIMULATED-truth burn marker (frontend headline.ts maps it to 'burn')
#: and the truth closest-approach marker (the burned arc grazes L1 without transiting the neck).
EXTRA_EVENT_KINDS = {"maneuver": "SIMULATED-truth marker of the burn itself (never visible to the operator at that instant)",
                     "closest_approach": "SIMULATED-truth marker of the closest approach to L1 (a graze, not a neck transit)"}
#: Time conventions of the bundle (also published as ``meta.time_fields``): ``t`` / ``t_rel_s`` are seconds since
#: ``meta.t0_utc``; ``t_utc`` is ISO UTC; metrics ``*_rel_s`` are seconds since t0; ``tasking.attempts[].t_s`` and the
#: filter re-open events carry absolute TDB seconds past J2000 like every other API route.
TIME_FIELDS = {"t": "seconds since meta.t0_utc (frontend contract)", "t_rel_s": "same as t",
               "t_utc": "ISO-8601 UTC with Z", "*_rel_s": "metrics: seconds since meta.t0_utc",
               "t_s": "absolute TDB seconds past J2000 (same meaning as in every other API route)"}

#: 30 m/s burn direction (GCRF unit vector) from the reachability sampler: closest approach to L1 (see module docstring).
BURN_DIR_GCRF = np.array([-0.88729302, 0.39949375, 0.23046875])
BURN_DIR_GCRF /= np.linalg.norm(BURN_DIR_GCRF)


@dataclass
class DemoConfig:
    t0_utc: str = "2026-02-23T00:00:00"
    t1_utc: str = "2026-03-01T00:00:00"
    burn_utc: str = "2026-02-25T08:30:00"
    burn_mps: float = 30.0
    burn_dir_gcrf: np.ndarray = field(default_factory=lambda: BURN_DIR_GCRF.copy())
    frame_dt_s: float = HOUR
    obs_cadence_s: float = 2 * HOUR          # routine ground tracklet cadence
    followup_cadence_s: float = 2 * HOUR     # space-observer cadence after re-acquisition
    followup_span_s: float = 36 * HOUR
    sigma_arcsec: float = 1.0
    prior_pos_km: float = 5.0                # converged-track catalogue covariance at t0 (assumption, see ASSUMPTIONS)
    prior_vel_mps: float = 0.2
    q_psd: float = 1e-18
    reopen_sigma_dv_mps: float = 25.0
    reopen_nis_trigger: float = 50.0         # ~ p < 1e-11 for chi2_2: a gross, unambiguous exceedance
    custody_km: float = 100.0
    lost_km: float = 1000.0
    dv_budget_mps: float = 100.0
    horizon_h: float = 168.0
    n_dirs: int = 96
    n_particles: int = 1500
    cloud_points_max: int = 800
    cloud_points_tight: int = 250            # exported points when sigma < 50 km (a dot anyway)
    reach_points_max: int = 800
    slot_min: float = 60.0
    search_fields_per_slot: int = 25        # mosaic budget per tasked slot (assumption; 60 min / 25 fields ~ 2.4 min per field)
    escalation_delay_s: float = HOUR           # re-planning + slew latency between the loss declaration and the first tasked look (assumption)
    escalation_gain: str = "logdet"           # tasker objective for the escalation segment (see build_scenario)
    playback_s: float = 120.0
    seed: int = 0
    fast: bool = False

    @classmethod
    def make(cls, seed: int = 0, fast: bool = False) -> "DemoConfig":
        c = cls(seed=int(seed), fast=bool(fast))
        if fast:
            c.n_dirs, c.n_particles = 32, 400
            c.cloud_points_max, c.cloud_points_tight, c.reach_points_max = 300, 120, 300
        return c


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _utc(t_s: float) -> str:
    from astropy.time import Time
    return str(Time(J2000_JD, float(t_s) / 86400.0, format="jd", scale="tdb").utc.isot)[:19] + "Z"


def _utc_many(t_s) -> list[str]:
    from astropy.time import Time
    t = np.atleast_1d(np.asarray(t_s, dtype=np.float64))
    iso = Time(J2000_JD, t / 86400.0, format="jd", scale="tdb").utc.isot
    return [str(x)[:19] + "Z" for x in np.atleast_1d(iso)]


def _r(x, nd: int):
    """Round a float / array to ``nd`` decimals and return plain python (NaN -> None)."""
    if isinstance(x, np.ndarray):
        return np.round(x.astype(np.float64), nd).tolist()
    if isinstance(x, (list, tuple)):
        return [_r(v, nd) for v in x]
    if x is None:
        return None
    xf = float(x)
    return round(xf, nd) if math.isfinite(xf) else None


def _flat(points: np.ndarray, nd: int = 6) -> list:
    return np.round(np.asarray(points, dtype=np.float64).reshape(-1), nd).tolist()


def _subsample(n: int, k: int) -> np.ndarray:
    return np.arange(n) if n <= k else np.linspace(0, n - 1, k).astype(int)


def _custody(sigma_km: float, cfg: DemoConfig) -> str:
    if sigma_km < cfg.custody_km:
        return "CUSTODY"
    return "DEGRADED" if sigma_km < cfg.lost_km else "LOST"


_UI = {"CUSTODY": "held", "DEGRADED": "degraded", "LOST": "lost"}


class _Reopen:
    """Maneuver-robust covariance re-opening hook for :func:`selene.od.ukf.ukf_run` (see module docstring)."""

    def __init__(self, sigma_dv_mps: float, nis_trigger: float):
        self.s2 = (sigma_dv_mps * 1e-3) ** 2
        self.trigger = float(nis_trigger)
        self.events: list[dict] = []

    def __call__(self, ctx: dict) -> Optional[np.ndarray]:
        if not (ctx["nis"] > self.trigger):
            return None
        dt = max(float(ctx["dt_s"]), 60.0)
        I3 = np.eye(3)
        Q = np.zeros((6, 6))
        Q[:3, :3] = self.s2 * dt**2 / 3.0 * I3
        Q[:3, 3:] = Q[3:, :3] = self.s2 * dt / 2.0 * I3
        Q[3:, 3:] = self.s2 * I3
        self.events.append({"t_s": float(ctx["t_s"]), "nis": float(ctx["nis"]), "gap_s": dt,
                            "sensor_id": getattr(ctx["meas"], "sensor_id", None)})
        return np.asarray(ctx["P_pred"]) + Q


def _cloud_rng(seed: int, anchor_t_s: float) -> np.random.Generator:
    """Deterministic particle generator per (seed, anchor epoch): the custody-decay pass and the frame pass
    therefore draw identical particles for the same filter posterior."""
    return np.random.default_rng([int(seed), int(round(anchor_t_s))])


def _los_rot(R_t: np.ndarray, u_gcrf: np.ndarray) -> np.ndarray:
    """GCRF direction -> rotating-frame axes (rotation only)."""
    return R_t.T @ u_gcrf


def _refine_closest_approach(state_at, grid: np.ndarray, k_min: int, point_rot: np.ndarray) -> tuple[float, float]:
    """Closest approach of a trajectory to a rotating-frame landmark, refined by bounded 1-D minimisation of the
    distance on the dense solution inside the grid cell around the grid minimum (same approach as the orbit
    library's perilune refinement).  Returns (t_s, distance_km)."""
    from scipy.optimize import minimize_scalar

    p = np.asarray(point_rot, dtype=np.float64)

    def dist_km(t: float) -> float:
        st = np.asarray(state_at(float(t)), dtype=np.float64).reshape(1, 6)
        r = frames.gcrf_to_rot(st, np.array([float(t)]))[0, :3]
        return float(np.linalg.norm(r - p) * L_STAR)

    lo = float(grid[max(k_min - 1, 0)])
    hi = float(grid[min(k_min + 1, len(grid) - 1)])
    if hi <= lo:
        return float(grid[k_min]), dist_km(float(grid[k_min]))
    res = minimize_scalar(dist_km, bounds=(lo, hi), method="bounded", options={"xatol": 1.0})
    t_best = float(res.x)
    d_best = float(res.fun)
    d_grid = dist_km(float(grid[k_min]))
    return (t_best, d_best) if d_best <= d_grid else (float(grid[k_min]), d_grid)


# ---------------------------------------------------------------------------
# builder
# ---------------------------------------------------------------------------
def build_scenario(seed: int = 0, fast: bool = False, config: Optional[DemoConfig] = None, _debug: Optional[dict] = None) -> dict:
    """Run the full engine chain and return the demo bundle.

    Bundle keys: ``meta`` (title, t0_utc, t1_utc, duration_s, playback_speed, epoch_choice_rationale,
    disclaimer, time_fields, ...), ``frames`` (hourly; each with ``t`` = ``t_rel_s`` seconds since t0, ``t_utc``,
    ``objects``, ``clouds``, ``sensors``, optional ``reachable``, ``observations``), ``events`` (chronological),
    ``metrics`` (custody_timeline, sigma_timeline, detection_latency_h, dv_true_mps, dv_est_mps, dv_est_err_pct,
    dir_err_deg, regained_after_h, ...), ``brief`` (markdown).  Field aliases for the frontend:
    ``t`` (= ``t_rel_s``), ``custody_ui`` ('held'/'degraded'/'lost'), ``sigma_pos_km`` (= ``sigma_km``),
    ``target_id``, ``boresight_rot`` and cloud ``points`` (= ``points_rot``, flat xyz).  Object positions are
    ``pos_rot`` (nd) and ``pos_gcrf_km`` only.  See :data:`TIME_FIELDS` for the time conventions.
    """
    cfg = config or DemoConfig.make(seed, fast)
    tic = time.perf_counter()
    timing: dict = {}
    log: list[str] = []
    rng = np.random.default_rng(cfg.seed)

    # ---- epochs ---------------------------------------------------------------------------
    t0 = float(seconds_since_j2000_tdb(cfg.t0_utc))
    t1 = float(seconds_since_j2000_tdb(cfg.t1_utc))
    tb = float(seconds_since_j2000_tdb(cfg.burn_utc))
    n_frames = int(round((t1 - t0) / cfg.frame_dt_s)) + 1
    grid = t0 + cfg.frame_dt_s * np.arange(n_frames)
    grid_utc = _utc_many(grid)
    cat = get_catalog()
    e = cat.get(PROTAGONIST)
    phys = e.physical or {}
    radius_m, albedo = float(phys.get("radius_m", 1.0)), float(phys.get("albedo", 0.2))
    params = EphemParams(srp=True, cr_area_mass=float(phys.get("cr_area_mass_m2_kg", 0.0)))
    ground = list(DEFAULT_SITES)
    space = list(DEFAULT_OBSERVERS)
    sensors_all = ground + space
    sim_ids = [x.id for x in cat.objects("simulated")]
    others = [i for i in sim_ids if i != PROTAGONIST]

    # ---- truth with the injected burn -------------------------------------------------------
    t_a = time.perf_counter()
    x_true0 = np.asarray(cat.state_at(PROTAGONIST, t0), dtype=np.float64)
    burn = Burn(tb, cfg.burn_mps * 1e-3 * cfg.burn_dir_gcrf)
    truth = truth_with_burn(x_true0, t0, t1 + 2 * 86400.0, burn, params)
    truth_grid = truth.at(grid)
    x_b_pre = truth.arcs[0].at(tb)
    from selene.maneuver.synthetic import frame_center, rtn_basis, vnb_basis
    center = frame_center(x_b_pre, tb, "auto")
    dv_rtn_true = rtn_basis(x_b_pre, tb, center) @ burn.dv_kms * 1e3
    dv_vnb_true = vnb_basis(x_b_pre, tb, center) @ burn.dv_kms * 1e3
    timing["truth_s"] = time.perf_counter() - t_a

    # ---- routine ground custody: tracklets at obs_cadence from the first visible site ---------
    t_a = time.perf_counter()
    obs_epochs = t0 + cfg.obs_cadence_s * np.arange(1, int((t1 - t0) / cfg.obs_cadence_s) + 1)
    ground_meas = generate_measurements(truth.at, obs_epochs, ground, cfg.sigma_arcsec, rng, radius_m, albedo, 1)
    # why the ground network is blind at each hourly epoch.  A site that is in daylight or has the object below
    # its minimum elevation (sensors/reasons.py AVAILABILITY_REASONS) could not have observed *anything*, so the
    # Moon bit it may also carry says nothing about glare.  The per-frame reason is therefore taken only over the
    # AVAILABLE sites (night, object above the elevation limit): the majority of their highest-priority
    # target-specific reason, or NO_SITE_AVAILABLE when no site could point at the object at all.
    eph = get_ephemeris()
    sun_g = eph.position("sun", grid)
    moon_g = eph.position("moon", grid)
    illum = illuminated_fraction(sun_g, moon_g)
    sep_moon_deg = np.degrees(angular_separation(truth_grid[:, :3], moon_g))
    ground_reason_grid: list[np.ndarray] = []
    ground_visible_grid = np.zeros(n_frames, dtype=bool)
    for s in ground:
        pos, _, zen = sensor_geometry(s, grid)
        mask, _, _, _ = evaluate(s, pos, zen, truth_grid[:, :3], sun_g, moon_g, radius_m, albedo)
        ground_visible_grid |= mask == 0
        ground_reason_grid.append(mask)
    reason_stack = np.stack(ground_reason_grid)            # (9, T)
    site_ids = [s.id for s in ground]
    dominant_reason: list[Optional[str]] = []
    avail_sites: list[list[str]] = []                       # sites that were available but blocked, per frame
    for k in range(n_frames):
        if ground_visible_grid[k]:
            dominant_reason.append(None)
            avail_sites.append([])
            continue
        col = reason_stack[:, k]
        av = np.where((col & AVAILABILITY_REASONS) == 0)[0]
        avail_sites.append([site_ids[i] for i in av])
        if av.size == 0:
            dominant_reason.append(NO_SITE_AVAILABLE)
            continue
        pr = primary_reason(col[av])
        vals, counts = np.unique(pr[pr != 0], return_counts=True)
        dominant_reason.append(REASON_NAMES.get(int(vals[np.argmax(counts)]), None) if vals.size else None)
    n_avail = np.array([len(a) for a in avail_sites])
    timing["ground_obs_s"] = time.perf_counter() - t_a
    log.append(f"ground tracklets: {len(ground_meas)} ({sum(1 for m in ground_meas if m.t_s < tb)} pre-burn)")

    # ---- UKF pass 1 (ground only) + detection ------------------------------------------------
    t_a = time.perf_counter()
    P0 = initial_covariance(cfg.prior_pos_km, cfg.prior_vel_mps)
    x0 = sample_initial_state(truth.at(t0), P0, rng)
    ukf_cfg = UKFConfig(q_psd=cfg.q_psd)
    reopen = _Reopen(cfg.reopen_sigma_dv_mps, cfg.reopen_nis_trigger)
    run1 = ukf_run(ground_meas, x0, P0, t0, params=params, config=ukf_cfg, object_id=PROTAGONIST, inflate=reopen)
    det_cfg = DetectorConfig(alpha=0.01)
    report1 = detect(run1, det_cfg, truth=truth.at, params=params)
    timing["ukf1_detect_s"] = time.perf_counter() - t_a
    if not report1.declared or report1.first_declared_t_s is None:
        raise RuntimeError(f"demo scenario: maneuver not declared on the ground track (status {report1.status})")
    t_det = float(report1.first_declared_t_s)
    i_det = int(np.searchsorted(run1.t_s, t_det))
    j_pre = i_det - 1
    x_pre, P_pre, t_pre = run1.x[j_pre].copy(), run1.P[j_pre].copy(), float(run1.t_s[j_pre])
    det_meas = run1.meas[i_det]
    log.append(f"declared at {_utc(t_det)} (NIS {run1.nis[i_det]:.0f}, sensor {det_meas.sensor_id}), latency {(t_det - tb) / HOUR:.2f} h")

    # ---- reachability from the last good estimate over the burn window -----------------------
    t_a = time.perf_counter()
    window_h = (t_det - t_pre) / HOUR
    burn_epochs = tuple(sorted({0.0, round(window_h / 2, 3), round(window_h, 3)}))
    rcfg = ReachabilityConfig(dv_budget_mps=cfg.dv_budget_mps, horizon_h=cfg.horizon_h, n_dirs=cfg.n_dirs,
                              burn_epochs_h=burn_epochs, cr_area_mass=params.cr_area_mass)
    regions = default_regions()
    rs = compute_reachability(x_pre, t_pre, rcfg, object_id=PROTAGONIST, regions=regions)
    refined: dict[str, dict] = {}
    if not cfg.fast:
        for st in rs.region_stats:
            if st.n_hit and st.min_dv_mps and st.min_dv_mps > 0:
                r = refine_min_dv(rs, st.key)
                if r:
                    refined[st.key] = r
    region_rows = []
    gateway_diag: dict[str, dict] = {}
    for st in rs.region_stats:
        reg = next(r for r in regions if r.key == st.key)
        min_dv = st.min_dv_mps
        if st.key in refined:
            min_dv = refined[st.key]["dv_mps"]
        row = {"key": st.key, "name": st.name, "kind": st.kind, "fraction": _r(st.fraction, 4),
               "sample_fraction": _r(st.sample_fraction, 4), "n_hit": int(st.n_hit),
               "earliest_h": _r(st.earliest_h, 1), "earliest_nominal_h": _r(st.earliest_nominal_h, 1), "min_dv_mps": _r(min_dv, 1),
               "nominal_hits": bool(st.nominal_hits), "newly_reachable": bool(st.newly_reachable),
               "why_it_matters": reg.why_it_matters}
        if reg.kind == "gateway":
            # what the reachable set does at the neck even when nothing transits: closest approach, plane crossings that
            # return, realm changes far from the libration point (the brief and the alert quote these numbers verbatim)
            row["diagnostics"] = gateway_diag[st.key] = _gateway_diagnostics(rs, reg, st)
        region_rows.append(row)
    timing["reachability_s"] = time.perf_counter() - t_a

    # ---- custody decay from the detection posterior -> t_lost ------------------------------
    t_a = time.perf_counter()
    x_det, P_det = run1.x[i_det].copy(), run1.P[i_det].copy()
    decay_grid = grid[grid > t_det + 1e-6]
    decay_grid = decay_grid[decay_grid <= t_det + 4 * 86400.0]
    decay = propagate_cloud(x_det, P_det, t_det, decay_grid, n=cfg.n_particles, rng=_cloud_rng(cfg.seed, t_det),
                            params=params, with_linear=False, object_id=PROTAGONIST)
    sig_decay = decay.metrics()["sigma_pos_km"]
    lost_idx = np.where(sig_decay > cfg.lost_km)[0]
    t_lost = float(decay_grid[lost_idx[0]]) if lost_idx.size else float(decay_grid[-1])
    custody_lost = bool(lost_idx.size)
    deg_idx = np.where(sig_decay > cfg.custody_km)[0]
    sigma_det = float(np.sqrt(np.trace(P_det[:3, :3])))
    t_degraded = t_det if sigma_det > cfg.custody_km else (float(decay_grid[deg_idx[0]]) if deg_idx.size else None)
    timing["decay_s"] = time.perf_counter() - t_a
    log.append(f"custody lost (sigma > {cfg.lost_km:.0f} km) at {_utc(t_lost)}; sigma at detection {sigma_det:.0f} km")

    # ---- tasking: routine segment A (space observers on the other objects), then escalation B --
    # Escalation starts one slot after the loss is declared (re-planning / slew latency, an assumption);
    # if custody never formally drops below the LOST threshold the escalation still starts at the end of
    # the decay horizon so the story reports what the network can do.
    t_a = time.perf_counter()
    t_esc = (t_lost if custody_lost else float(decay_grid[-1])) + cfg.escalation_delay_s
    t_esc = min(t_esc, t1 - cfg.slot_min * 60.0)
    seg_a = make_scenario(others, sensors=space, t0_s=t0, t1_s=t_esc, slot_min=cfg.slot_min, q_psd=cfg.q_psd)
    res_a = run_greedy(seg_a)
    # reachable-set pointing hints for every space sensor at the escalation epoch (mosaic size = p90 spread)
    k_esc = int(np.argmin(np.abs(rs.t_grid_s - t_esc)))
    hint0 = sensor_hints(rs, space, radius_m=radius_m, albedo=albedo, step_indices=np.array([k_esc]))
    # the scheduler and the truth verification share ONE search budget: a mosaic of at most ``n_tiles`` fields per slot
    n_tiles = int(max(1, cfg.search_fields_per_slot))
    seg_b_end = min(t1, t_esc + 48 * HOUR)
    seg_b = make_scenario(sim_ids, sensors=space, t0_s=t_esc, t1_s=seg_b_end, slot_min=cfg.slot_min, q_psd=cfg.q_psd,
                          priorities={PROTAGONIST: 3.0}, search_tiles=n_tiles, gain_kind=cfg.escalation_gain)
    # covariance continuity: others continue from segment A, the protagonist from its re-opened filter covariance
    pa = {tr.object_id: res_a.P_final[j] for j, tr in enumerate(seg_a.tracks)}
    _, P_lin = linear_covariance(x_det, P_det, t_det, np.array([t_esc]), params)
    for tr in seg_b.tracks:
        tr.P0 = P_lin[0].copy() if tr.object_id == PROTAGONIST else pa[tr.object_id].copy()
    res_b = run_greedy(seg_b)
    sensor_order = {s_.id: i for i, s_ in enumerate(space)}   # the greedy's own sequential sensor order
    prot_assign = sorted([a for a in res_b.assignments if a.object_id == PROTAGONIST], key=lambda a: (a.slot, sensor_order[a.sensor_id]))
    timing["tasking_s"] = time.perf_counter() - t_a
    log.append(f"tasker: {len(prot_assign)} protagonist assignments in segment B (search_tiles={n_tiles}, gain={cfg.escalation_gain})")

    # ---- re-acquisition: verify every tasked look against the SIMULATED truth ----------------
    t_a = time.perf_counter()
    space_by_id = {s.id: s for s in space}

    def _verify(sid: str, t_task: float) -> dict:
        """Does sensor ``sid``, pointed at the reachable-set centroid with a mosaic covering the p90 spread,
        actually see the true object at ``t_task``?  (visibility model + pointing geometry; never assumed)."""
        sen = space_by_id[sid]
        k = int(np.argmin(np.abs(rs.t_grid_s - t_task)))
        h = sensor_hints(rs, [sen], radius_m=radius_m, albedo=albedo, step_indices=np.array([k]))["steps"][0].sensors[0]
        xt = truth.at(t_task)
        pos, _, _zen = sensor_geometry(sen, t_task)
        mask, mag, rng_km, _ = evaluate(sen, pos[0], None, xt[:3], eph.position("sun", t_task), eph.position("moon", t_task), radius_m, albedo)
        vis = bool(mask == 0)
        rec = {"t_s": float(t_task), "t_utc": _utc(t_task), "sensor_id": sid, "truth_visible": vis,
               "truth_block_reason": reason_names(int(mask)) if not vis else [],
               "pointing_gcrf": h.pointing_gcrf, "ra_deg": h.ra_deg, "dec_deg": h.dec_deg,
               "spread_p90_deg": h.spread_p90_deg, "n_fields_p90": h.n_fields_p90, "fov_deg": h.fov_deg,
               "visible_fraction": h.visible_fraction, "fov_capture_fraction": h.fov_capture_fraction,
               "range_km": float(rng_km), "magnitude": float(mag)}
        if vis and h.pointing_gcrf is not None:
            los = xt[:3] - pos[0]
            los /= np.linalg.norm(los)
            off = float(np.degrees(np.arccos(np.clip(los @ np.asarray(h.pointing_gcrf), -1, 1))))
            # n_fields_p90 = ceil((2 p90 / fov)^2)  =>  a budget of n_tiles fields covers a radius 0.5 fov sqrt(n_tiles)
            fov = float(sen.fov_deg)
            mosaic_radius = max(0.5 * fov, min(float(h.spread_p90_deg or 0.0), 0.5 * fov * math.sqrt(n_tiles)))
            rec.update(offset_deg=off, mosaic_radius_deg=mosaic_radius, n_fields_used=int(min(n_tiles, h.n_fields_p90 or 1)),
                       acquired=bool(off <= mosaic_radius))
        else:
            rec.update(offset_deg=None, mosaic_radius_deg=None, n_fields_used=0, acquired=False)
        return rec

    attempts: list[dict] = []
    reacq: Optional[dict] = None          # the escalation slot: {"t_s", "sensors": [acquiring records], "primary": record}
    slots = sorted({a.slot for a in prot_assign})
    for sl in slots:
        rows = [a for a in prot_assign if a.slot == sl]
        recs = [_verify(a.sensor_id, a.t_s) for a in rows]
        attempts.extend(recs)
        got = [r for r in recs if r["acquired"]]
        if got:
            reacq = {"t_s": float(rows[0].t_s), "t_utc": _utc(rows[0].t_s), "sensors": got, "primary": got[0],
                     "not_acquired": [r for r in recs if not r["acquired"]],
                     "blocked": [{"sensor_id": h.sensor_id, "reason": h.dominant_block_reason, "visible_fraction": h.visible_fraction}
                                 for h in hint0["steps"][0].sensors if h.sensor_id not in {a.sensor_id for a in rows}]}
            break
    if reacq is None and not prot_assign:  # the scheduler never picked it: try the hint ordering directly
        for h in hint0["steps"][0].sensors:
            if h.visible_fraction > 0:
                rec = _verify(h.sensor_id, t_esc)
                attempts.append(rec)
                if rec["acquired"]:
                    reacq = {"t_s": t_esc, "t_utc": _utc(t_esc), "sensors": [rec], "primary": rec, "not_acquired": [], "blocked": []}
                    break
    # follow-ups: the scheduler's own plan for the protagonist at the follow-up cadence (each look passes the
    # visibility model through ``observe``; pointing at the filter prediction is assumed once re-acquired)
    follow_meas = []
    if reacq is not None:
        t_re = reacq["t_s"]
        stride = max(1, int(round(cfg.followup_cadence_s / (cfg.slot_min * 60.0))))
        acq_ids = {r["sensor_id"] for r in reacq["sensors"]}
        for a in prot_assign:
            if a.t_s < t_re - 1e-6 or a.t_s > t_re + cfg.followup_span_s + 1e-6:
                continue
            if (a.t_s - t_re) / (cfg.slot_min * 60.0) % stride > 1e-6:
                continue
            if abs(a.t_s - t_re) < 1e-6 and a.sensor_id not in acq_ids:
                continue  # a look that did not acquire in the escalation slot yields no tracklet
            m = observe(space_by_id[a.sensor_id], truth.at(a.t_s), a.t_s, sigma_arcsec=cfg.sigma_arcsec, rng=rng,
                        target_radius_m=radius_m, albedo=albedo)
            if m is not None:
                follow_meas.append(m)
    timing["reacq_s"] = time.perf_counter() - t_a
    log.append(f"re-acquisition: {'OK by ' + ','.join(r['sensor_id'] for r in reacq['sensors']) if reacq else 'FAILED'} "
               f"({len(attempts)} looks verified), {len(follow_meas)} space tracklets")

    # ---- UKF pass 2 over the full observation set + detection + dv estimation -----------------
    t_a = time.perf_counter()
    all_meas = sorted(ground_meas + follow_meas, key=lambda m: (m.t_s, m.sensor_id))
    reopen2 = _Reopen(cfg.reopen_sigma_dv_mps, cfg.reopen_nis_trigger)
    run = ukf_run(all_meas, x0, P0, t0, params=params, config=ukf_cfg, object_id=PROTAGONIST, inflate=reopen2)
    report = detect(run, det_cfg, truth=truth.at, params=params)
    timing["ukf2_detect_s"] = time.perf_counter() - t_a
    t_a = time.perf_counter()
    est = None
    est_err: Optional[str] = None
    n_post_used = 0
    post_meas = [m for m in run.meas if m.t_s >= t_det - 1e-6]
    if len(post_meas) >= 2:
        ecfg = EstimatorConfig(n_post_obs=10, n_grid=5 if cfg.fast else 9)
        try:
            est = estimate_impulsive_dv(x_pre, t_pre, post_meas, (t_pre, t_det), P_pre=P_pre, params=params, cfg=ecfg)
            n_post_used = est.n_obs
        except Exception as ex:  # noqa: BLE001 - reported honestly
            est_err = f"{type(ex).__name__}: {ex}"
    else:
        est_err = "fewer than 2 post-burn observations: the maneuver cannot be characterised"
    timing["dv_estimate_s"] = time.perf_counter() - t_a
    if _debug is not None:
        _debug.update(run=run, run1=run1, truth=truth, burn=burn, params=params, x_pre=x_pre, P_pre=P_pre, t_pre=t_pre, t_det=t_det,
                      post_meas=post_meas, est=est, rs=rs, reacq=reacq, report=report, x0=x0, P0=P0, t0=t0,
                      res_a=res_a, res_b=res_b, seg_b=seg_b, hint0=hint0, t_lost=t_lost, attempts=attempts)
    t_char = float(post_meas[n_post_used - 1].t_s) if est is not None else None

    # ---- particle clouds on the frame grid from the final filter posteriors --------------------
    t_a = time.perf_counter()
    anchors = [(t0, x0, P0)] + [(float(run.t_s[k]), run.x[k], run.P[k]) for k in range(len(run))]
    anchor_t = np.array([a[0] for a in anchors])
    owner = np.searchsorted(anchor_t, grid, side="right") - 1    # anchor index for each frame
    cloud_pos_rot = [None] * n_frames
    cloud_sigma = np.zeros(n_frames)
    cloud_mean_rot = np.zeros((n_frames, 3))
    cloud_mean_gcrf = np.zeros((n_frames, 3))
    fr_grid = frames.rotating_frame(grid)
    for a_idx in np.unique(owner):
        fidx = np.where(owner == a_idx)[0]
        ta, xa, Pa = anchors[a_idx]
        tq = grid[fidx]
        pc = propagate_cloud(xa, Pa, ta, tq, n=cfg.n_particles, rng=_cloud_rng(cfg.seed, ta), params=params, with_linear=False, object_id=PROTAGONIST)
        sig = pc.metrics()["sigma_pos_km"]
        for q, fi in enumerate(fidx):
            st = pc.states[q]
            rot = frames.gcrf_to_rot(st, np.full(st.shape[0], grid[fi]))[:, :3]
            cloud_pos_rot[fi] = rot
            cloud_sigma[fi] = float(sig[q])
            cloud_mean_rot[fi] = rot.mean(axis=0)
            cloud_mean_gcrf[fi] = st[:, :3].mean(axis=0)
    timing["clouds_s"] = time.perf_counter() - t_a

    # ---- custody timeline, regain -------------------------------------------------------------
    status = [_custody(s, cfg) for s in cloud_sigma]
    t_regained = None
    if reacq is not None:
        after = np.where((grid >= reacq["t_s"] - 1e-6) & (cloud_sigma < cfg.custody_km))[0]
        if after.size:
            t_regained = float(grid[after[0]])
    # honest bookkeeping: the first frame at/after t_lost that is LOST (particle sigma), if any
    lost_frames = np.where((grid >= t_det) & (cloud_sigma > cfg.lost_km))[0]
    t_lost_frame = float(grid[lost_frames[0]]) if lost_frames.size else None

    # ---- truth region entries -----------------------------------------------------------------
    w_tstar = np.asarray(fr_grid.omega) * T_STAR     # instantaneous frame rate in units of 1/T* (regions.py Jacobi rescale)
    rot_truth = frames.gcrf_to_rot(truth_grid, grid)
    inp = RegionInputs(rot_truth[:, :3], truth_grid[:, :3], truth_grid[:, :3] - moon_g, rot_truth[:, 3:6], truth_grid[:, 3:6], omega_t_star=w_tstar)
    entries = []
    for reg in regions:
        m = reg.contains(inp)
        if m.any():
            k = int(np.argmax(m))
            entries.append({"key": reg.key, "name": reg.name, "t_s": float(grid[k]), "frame": k, "kind": reg.kind})
    l1_rot = np.asarray(regions[0].params["center_rot_nd"], dtype=np.float64)
    d_l1_km = np.linalg.norm(rot_truth[:, :3] - l1_rot, axis=1) * L_STAR
    # closest approach refined on the dense truth solution (the hourly-grid minimum overstated it by ~250 km)
    t_l1_min, d_l1_min_km = _refine_closest_approach(truth.at, grid, int(np.argmin(d_l1_km)), l1_rot)
    l1_transit = any(en["key"] == "l1_gateway" for en in entries)
    relay_grid = cat.state_at(RELAY, grid)
    d_relay_km = np.linalg.norm(truth_grid[:, :3] - relay_grid[:, :3], axis=1)
    # the same quantities for the UNPERTURBED orbit over the window (what the burn changed)
    quiet_grid = cat.state_at(PROTAGONIST, grid)
    d_l1_quiet_km = np.linalg.norm(frames.gcrf_to_rot(quiet_grid, grid)[:, :3] - l1_rot, axis=1) * L_STAR
    _, d_l1_quiet_min_km = _refine_closest_approach(lambda t: cat.state_at(PROTAGONIST, t), grid, int(np.argmin(d_l1_quiet_km)), l1_rot)

    # ---- frames ------------------------------------------------------------------------------
    t_a = time.perf_counter()
    other_truth = {oid: cat.state_at(oid, grid) for oid in others}
    other_rot = {oid: frames.gcrf_to_rot(other_truth[oid], grid)[:, :3] for oid in others}
    # covariance series of the other objects: segment A nodes, then B, then a propagated tail (segment C)
    sig_other = {oid: np.full(n_frames, np.nan) for oid in others}
    seg_c = None
    res_c = None
    if seg_b_end < t1 - 1e-6:
        seg_c = make_scenario(sim_ids, sensors=space, t0_s=seg_b_end, t1_s=t1, slot_min=cfg.slot_min, q_psd=cfg.q_psd,
                              priorities={PROTAGONIST: 3.0}, search_tiles=n_tiles)
        pb = {tr.object_id: res_b.P_final[j] for j, tr in enumerate(seg_b.tracks)}
        for tr in seg_c.tracks:
            tr.P0 = pb[tr.object_id].copy()
        res_c = run_greedy(seg_c)
    for res, scn in ((res_a, seg_a), (res_b, seg_b), (res_c, seg_c)):
        if res is None:
            continue
        nodes = scn.table.t_nodes
        fi = np.round((nodes - t0) / cfg.frame_dt_s).astype(int)
        ok = (fi >= 0) & (fi < n_frames) & (np.abs(nodes - (t0 + fi * cfg.frame_dt_s)) < 1.0)
        for j, tr in enumerate(scn.tracks):
            if tr.object_id in sig_other:
                sig_other[tr.object_id][fi[ok]] = res.sigma_km[j][ok]
    assign_by_frame: dict[int, dict[str, str]] = {}
    for res in (res_a, res_b, res_c):
        if res is None:
            continue
        for a in res.assignments:
            fi = int(round((a.t_s - t0) / cfg.frame_dt_s))
            assign_by_frame.setdefault(fi, {})[a.sensor_id] = a.object_id
    # the UKF track of the protagonist for the sensor that re-acquired it, by frame
    prot_obs_by_frame: dict[int, list] = {}
    for k, m in enumerate(run.meas):
        fi = int(round((m.t_s - t0) / cfg.frame_dt_s))
        prot_obs_by_frame.setdefault(fi, []).append((m, float(np.linalg.norm(run.innov[k]) / ARCSEC)))
    site_pos_rot = {}
    for s in ground:
        frs = site_frame(s, grid)
        site_pos_rot[s.id] = frames.gcrf_pos_to_rot(frs.pos_km, grid)
        site_pos_rot[s.id + "_gcrf"] = frs.pos_km
    obs_pos_rot = {}
    for s in space:
        st = observer_state(s, grid)
        obs_pos_rot[s.id] = frames.gcrf_to_rot(st, grid)[:, :3]
        obs_pos_rot[s.id + "_gcrf"] = st[:, :3]
    all_pos_gcrf = {oid: other_truth[oid] for oid in others}
    all_pos_gcrf[PROTAGONIST] = truth_grid
    reach_t_idx = {}
    for k_frame in range(n_frames):
        kk = int(np.argmin(np.abs(rs.t_grid_s - grid[k_frame])))
        reach_t_idx[k_frame] = kk if abs(rs.t_grid_s[kk] - grid[k_frame]) < 1.0 else None
    t_reach_show_end = t_regained if t_regained is not None else t1
    # boresights for looks at the protagonist: the tasked reachable-set centroid in the re-acquisition slot, the filter
    # estimate (particle-cloud mean) afterwards -- never the truth, which the operator does not know
    k_reacq = int(round((reacq["t_s"] - t0) / cfg.frame_dt_s)) if reacq is not None else None
    reacq_pointing = {r["sensor_id"]: np.asarray(r["pointing_gcrf"], dtype=np.float64) for r in (reacq["sensors"] if reacq else [])
                      if r.get("pointing_gcrf") is not None}
    frames_out = []
    for k in range(n_frames):
        t = float(grid[k])
        objs = []
        for oid in sim_ids:
            if oid == PROTAGONIST:
                sg = float(cloud_sigma[k])
                stt = status[k]
                pr = rot_truth[k, :3]
                pg = truth_grid[k, :3]
            else:
                sgv = sig_other[oid][k]
                sg = float(sgv) if np.isfinite(sgv) else float(np.nanmax(sig_other[oid])) if np.isfinite(sig_other[oid]).any() else 10.0
                stt = _custody(sg, cfg)
                pr = other_rot[oid][k]
                pg = other_truth[oid][k, :3]
            o = {"id": oid, "pos_rot": _r(pr, 6), "pos_gcrf_km": _r(pg, 1),
                 "custody": stt, "custody_ui": _UI[stt], "sigma_km": _r(sg, 2), "sigma_pos_km": _r(sg, 2)}
            if oid == PROTAGONIST:
                o["pos_est_rot"] = _r(cloud_mean_rot[k], 6)
            objs.append(o)
        pts = cloud_pos_rot[k]
        n_keep = cfg.cloud_points_tight if cloud_sigma[k] < 50.0 else cfg.cloud_points_max
        sel = _subsample(pts.shape[0], n_keep)
        flat = _flat(pts[sel], 6)
        clouds = [{"object_id": PROTAGONIST, "points_rot": flat, "points": flat, "sigma_km": _r(cloud_sigma[k], 2),
                   "n_particles": int(cfg.n_particles), "n_exported": int(sel.size)}]
        sens = []
        R_t = fr_grid.R[k]
        for s in sensors_all:
            is_ground = s.kind == "ground"
            d = {"id": s.id, "kind": s.kind, "fov_deg": float(s.fov_deg), "active": False, "tasked_object": None, "target_id": None}
            d["pos_rot"] = _r(site_pos_rot[s.id][k] if is_ground else obs_pos_rot[s.id][k], 6)
            spos = site_pos_rot[s.id + "_gcrf"][k] if is_ground else obs_pos_rot[s.id + "_gcrf"][k]
            target = None
            if is_ground:
                for m, _res in prot_obs_by_frame.get(k, []):
                    if m.sensor_id == s.id:
                        target = PROTAGONIST
            else:
                for m, _res in prot_obs_by_frame.get(k, []):
                    if m.sensor_id == s.id:
                        target = PROTAGONIST
                if target is None:
                    target = assign_by_frame.get(k, {}).get(s.id)
            if target is not None:
                if target == PROTAGONIST and k == k_reacq and s.id in reacq_pointing:
                    u, basis = reacq_pointing[s.id].copy(), "reachable_set_centroid"
                elif target == PROTAGONIST:
                    u, basis = cloud_mean_gcrf[k] - spos, "filter_estimate"
                else:
                    u, basis = all_pos_gcrf[target][k, :3] - spos, "reference_trajectory"   # linear-covariance tasking, no estimate
                u = u / np.linalg.norm(u)
                ur = _r(_los_rot(R_t, u), 5)
                d.update(active=True, tasked_object=target, target_id=target, pointing_rot=ur, boresight_rot=ur, pointing_basis=basis)
            sens.append(d)
        obs_rows = [{"sensor_id": m.sensor_id, "object_id": PROTAGONIST, "residual_arcsec": _r(res_as, 3),
                     "magnitude": _r(m.magnitude, 2)} for m, res_as in prot_obs_by_frame.get(k, [])]
        for sid, oid in assign_by_frame.get(k, {}).items():
            if oid != PROTAGONIST:
                obs_rows.append({"sensor_id": sid, "object_id": oid, "residual_arcsec": None,
                                 "note": "linear-covariance tasking look (no measurement realisation)"})
        fr = {"t": float(t - t0), "t_rel_s": float(t - t0), "t_utc": grid_utc[k], "objects": objs, "clouds": clouds,
              "sensors": sens, "observations": obs_rows,
              "ground_blind_reason": dominant_reason[k], "ground_sites_available": int(n_avail[k]),
              "moon_illum": _r(illum[k], 3), "moon_sep_deg": _r(sep_moon_deg[k], 2)}
        if t_det - 1e-6 <= t <= t_reach_show_end + 1e-6 and reach_t_idx[k] is not None:
            kk = reach_t_idx[k]
            act = rs.active[:, kk]
            prs = rs.states_rot[act, kk, :3]
            sel = _subsample(prs.shape[0], cfg.reach_points_max)
            flat = _flat(prs[sel], 5)
            fr["reachable"] = {"points_rot": flat, "points": flat, "regions": region_rows, "dv_budget_mps": cfg.dv_budget_mps,
                               "horizon_h": cfg.horizon_h, "t_h": _r(rs.t_h[kk], 2), "n_active": int(act.sum())}
        frames_out.append(fr)
    timing["frames_s"] = time.perf_counter() - t_a

    # ---- metrics -------------------------------------------------------------------------------
    dv_true = burn.magnitude_mps
    dv_est = dv_est_err_pct = dir_err_deg = dv_sigma = dir_sigma = None
    t_burn_est = t_burn_sigma = None
    est_block = None
    if est is not None:
        dv_est = float(est.magnitude_mps)
        dv_est_err_pct = float(100.0 * (dv_est - dv_true) / dv_true)
        cosang = float(est.direction_gcrf @ (burn.dv_kms / np.linalg.norm(burn.dv_kms)))
        dir_err_deg = float(np.degrees(np.arccos(np.clip(cosang, -1, 1))))
        dv_sigma = float(est.magnitude_sigma_mps)
        dir_sigma = float(est.direction_sigma_deg)
        t_burn_est, t_burn_sigma = float(est.t_burn_s), float(est.t_burn_sigma_s)
        est_block = {"dv_mps": _r(dv_est, 2), "dv_sigma_mps": _r(dv_sigma, 2), "direction_gcrf": _r(est.direction_gcrf, 4),
                     "direction_rot": _r(_los_rot(frames.rotating_frame(est.t_burn_s).R, est.direction_gcrf), 4),
                     "direction_sigma_deg": _r(dir_sigma, 2), "dv_rtn_mps": _r(est.dv_rtn_mps, 2), "dv_vnb_mps": _r(est.dv_vnb_mps, 2),
                     "frame_center": est.frame_center, "t_burn_utc": _utc(est.t_burn_s), "t_burn_sigma_s": _r(t_burn_sigma, 1),
                     "t_burn_err_s": _r(est.t_burn_s - tb, 1), "n_obs": int(est.n_obs), "converged": bool(est.converged),
                     "residual_rms_arcsec": _r(est.residual_rms_arcsec, 3), "reduced_chi2": _r(est.reduced_chi2, 3),
                     "classification": est.classification, "grid": est.grid}
    nees = report.summary.get("nees") or {}
    health = report.summary.get("filter_health", {})
    custody_frac = float(np.mean([s == "CUSTODY" for s in status]))
    # ONE definition of "sigma at detection" everywhere (events, frames, metrics): the particle-cloud RSS position
    # sigma at the detection frame, which is what drives the custody thresholds; the UKF covariance after the
    # re-open is reported separately under its own name.
    k_det_frame = int(np.argmin(np.abs(grid - t_det)))
    sigma_det_cloud = float(cloud_sigma[k_det_frame])
    metrics = {
        "protagonist": PROTAGONIST, "relay": RELAY,
        "time_fields": TIME_FIELDS,
        "custody_timeline": [{"t_rel_s": float(grid[k] - t0), "t_utc": grid_utc[k], "status": status[k]} for k in range(n_frames)],
        "sigma_timeline": [{"t_rel_s": float(grid[k] - t0), "sigma_km": _r(cloud_sigma[k], 2)} for k in range(n_frames)],
        "custody_pct": _r(100.0 * custody_frac, 1),
        "custody_status_counts": {s: int(sum(1 for x in status if x == s)) for s in ("CUSTODY", "DEGRADED", "LOST")},
        "t_burn_utc": _utc(tb), "t_burn_rel_s": tb - t0,
        "t_detect_utc": _utc(t_det), "t_detect_rel_s": t_det - t0, "detect_sensor": det_meas.sensor_id, "detect_nis": _r(run.nis[i_det], 1),
        "detect_residual_arcsec": _r(float(np.linalg.norm(run.innov[i_det]) / ARCSEC), 2),
        "detection_latency_h": _r((t_det - tb) / HOUR, 3),
        "t_degraded_utc": _utc(t_degraded) if t_degraded is not None else None,
        "t_degraded_rel_s": (t_degraded - t0) if t_degraded is not None else None,
        "t_lost_utc": _utc(t_lost) if custody_lost else None, "t_lost_rel_s": (t_lost - t0) if custody_lost else None,
        "loss_reason": _loss_reason(dominant_reason, avail_sites, grid, t_det, t_lost, t1),
        "ground_blind_reason_counts": _reason_counts(dominant_reason, grid, t_det, t1),
        "moon_illum_at_loss": _r(illum[int(np.argmin(np.abs(grid - t_lost)))], 3),
        "moon_sep_deg_at_loss": _r(sep_moon_deg[int(np.argmin(np.abs(grid - t_lost)))], 2),
        "sigma_at_detection_km": _r(sigma_det_cloud, 1),
        "sigma_at_detection_definition": "particle-cloud RSS position sigma at the detection frame (the custody metric); "
                                         "ukf_sigma_at_detection_km is sqrt(trace P_pos) of the re-opened filter covariance",
        "ukf_sigma_at_detection_km": _r(sigma_det, 1),
        "max_sigma_km": _r(float(cloud_sigma.max()), 1),
        "t_max_sigma_utc": grid_utc[int(np.argmax(cloud_sigma))],
        "tasking": _tasking_metrics(reacq, attempts, t0, n_tiles, len(follow_meas), cfg),
        "t_regained_utc": _utc(t_regained) if t_regained is not None else None,
        "t_regained_rel_s": (t_regained - t0) if t_regained is not None else None,
        "regained_after_h": _r((t_regained - t_lost) / HOUR, 2) if (t_regained is not None and custody_lost) else None,
        "regained_after_burn_h": _r((t_regained - tb) / HOUR, 2) if t_regained is not None else None,
        "sigma_after_regain_km": _r(float(cloud_sigma[int(np.argmin(np.abs(grid - t_regained)))]), 2) if t_regained is not None else None,
        "dv_true_mps": _r(dv_true, 3), "dv_true_dir_gcrf": _r(burn.dv_kms / np.linalg.norm(burn.dv_kms), 5),
        "dv_true_rtn_mps": _r(dv_rtn_true, 2), "dv_true_vnb_mps": _r(dv_vnb_true, 2), "dv_true_frame_center": center,
        "dv_est_mps": _r(dv_est, 3), "dv_est_sigma_mps": _r(dv_sigma, 3), "dv_est_err_pct": _r(dv_est_err_pct, 2),
        "dir_err_deg": _r(dir_err_deg, 3), "dir_sigma_deg": _r(dir_sigma, 3),
        "t_burn_est_utc": _utc(t_burn_est) if t_burn_est is not None else None, "t_burn_est_sigma_s": _r(t_burn_sigma, 1),
        "t_burn_est_err_s": _r(t_burn_est - tb, 1) if t_burn_est is not None else None,
        "dv_estimate": est_block, "dv_estimate_error": est_err, "t_characterised_utc": _utc(t_char) if t_char else None,
        "detection": {"status": report.status, "declared": bool(report.declared), "alpha": det_cfg.alpha,
                      "alpha_familywise_per_test": report.summary.get("alpha_familywise_per_test"),
                      "threshold_nis": _r(report.summary.get("threshold_nis"), 2),
                      "threshold_nis_familywise": _r(report.summary.get("threshold_nis_familywise"), 2),
                      "n_updates": int(report.summary.get("n_updates", 0)), "n_tested": int(report.summary.get("n_tested", 0)),
                      "counts": report.summary.get("counts"), "mean_nis_tested": _r(report.summary.get("mean_nis"), 2),
                      "baseline_established": bool(health.get("baseline_established")), "warnings": health.get("warnings", []),
                      "nees_mean": _r(nees.get("mean_nees"), 2), "nees_bounds": _r(nees.get("mean_bounds"), 2),
                      "nees_consistent": nees.get("consistent"), "nees_n_epochs": nees.get("n_epochs"),
                      "nees_note": "single-run statistic: the band assumes N independent epochs, but successive errors of one "
                                   "sequential run are strongly correlated (the effective N is much smaller), so a below-band mean "
                                   "here is indicative of a conservative covariance, not a failed realism test; the Monte-Carlo "
                                   "realism study (od/realism.py, tests/test_od_realism.py) is the calibrated check",
                      "nees_baseline_mean": _r(health.get("nees_baseline_mean"), 2), "nees_baseline_bound": _r(health.get("nees_baseline_bound"), 2),
                      "reopen_events": reopen2.events},
        "reachability": {"dv_budget_mps": cfg.dv_budget_mps, "horizon_h": cfg.horizon_h, "n_samples": int(rs.n_samples),
                         "n_dirs": int(rcfg.n_dirs), "burn_epochs_h": list(burn_epochs), "t_ref_utc": _utc(t_pre),
                         "regions": region_rows, "refined_min_dv": refined, "n_terminated": int(rs.meta["n_terminated"]),
                         "gateway_diagnostics": gateway_diag,
                         "gateway_definition": "neck transit (reachability/regions.py): enter the ball of 0.05 nd around L1/L2 from one "
                                               "realm proper, cross the plane x = x_L inside it and reach the other realm proper "
                                               "(|x - x_L| > 0.05 nd) with C < C_L; grazes, plane dips that return and passages "
                                               "unresolved at the horizon are not transits"},
        "truth_geometry": {"min_dist_to_L1_km": _r(d_l1_min_km, 0), "t_min_dist_to_L1_utc": _utc(t_l1_min),
                           "min_dist_to_L1_grid_km": _r(float(d_l1_km.min()), 0),
                           "min_dist_to_L1_note": "closest approach refined by bounded 1-D minimisation on the dense truth solution "
                                                  "(min_dist_to_L1_grid_km is the hourly-frame minimum)",
                           "min_dist_to_L1_unperturbed_km": _r(d_l1_quiet_min_km, 0),
                           "l1_transit": bool(l1_transit),
                           "l1_transit_note": "True only if the SIMULATED truth passes through the L1 neck into the Earth realm "
                                              "(reachability/regions.py); a graze that stays in the lunar realm is not a transit",
                           "min_dist_to_relay_km": _r(float(d_relay_km.min()), 0), "t_min_dist_to_relay_utc": grid_utc[int(np.argmin(d_relay_km))],
                           "region_entries": [{**{k: v for k, v in en.items() if k != "t_s"}, "t_rel_s": en["t_s"] - t0,
                                               "t_utc": _utc(en["t_s"])} for en in entries]},
        "observations": {"n_total": len(run.meas), "n_ground": len(ground_meas), "n_ground_pre_burn": int(sum(1 for m in ground_meas if m.t_s < tb)),
                         "n_ground_post_burn": int(sum(1 for m in ground_meas if m.t_s >= tb)), "n_space_followup": len(follow_meas),
                         "by_sensor": _count_by_sensor(run.meas), "sigma_arcsec": cfg.sigma_arcsec, "cadence_h": cfg.obs_cadence_s / HOUR},
        "filter": {"kind": run.meta.get("filter"), "q_psd_km2_s3": cfg.q_psd, "prior_pos_km": cfg.prior_pos_km, "prior_vel_mps": cfg.prior_vel_mps,
                   "reopen_sigma_dv_mps": cfg.reopen_sigma_dv_mps, "reopen_nis_trigger": cfg.reopen_nis_trigger,
                   "n_particles": cfg.n_particles, "custody_km": cfg.custody_km, "lost_km": cfg.lost_km},
        "other_objects_custody": {oid: {"custody_pct": _r(100.0 * float(np.nanmean(sig_other[oid] < cfg.custody_km)), 1),
                                        "max_sigma_km": _r(float(np.nanmax(sig_other[oid])), 1)} for oid in others},
        "tasking_segments": {"A": _seg_summary(res_a), "B": _seg_summary(res_b), "C": _seg_summary(res_c) if res_c else None},
    }

    # ---- events ---------------------------------------------------------------------------------
    events = _build_events(cfg, t0, t1, tb, t_det, t_degraded, t_lost if custody_lost else None, t_regained, t_char, run, i_det,
                           reacq, attempts, entries, region_rows, metrics, est_block, status, grid, grid_utc, cloud_sigma, follow_meas,
                           dominant_reason, avail_sites, illum, sep_moon_deg, d_l1_km,
                           closest_l1=(t_l1_min, d_l1_min_km, l1_transit, d_l1_quiet_min_km))

    # ---- meta + brief ---------------------------------------------------------------------------
    duration = float(t1 - t0)
    meta = {
        "title": TITLE, "t0_utc": _utc(t0), "t1_utc": _utc(t1), "duration_s": duration,
        "playback_s": cfg.playback_s, "playback_speed": duration / cfg.playback_s, "frame_dt_s": cfg.frame_dt_s, "n_frames": n_frames,
        "protagonist_id": PROTAGONIST, "relay_id": RELAY, "fast": cfg.fast, "seed": cfg.seed,
        "epoch_choice_rationale": EPOCH_RATIONALE,
        "burn_rationale": BURN_RATIONALE.format(closest_km=d_l1_min_km, closest_h=(t_l1_min - tb) / HOUR, quiet_km=d_l1_quiet_min_km,
                                                transit="transits" if l1_transit else "does not transit"),
        "assumptions": ASSUMPTIONS,
        "disclaimer": DISCLAIMER, "engines": ENGINES, "build_log": log, "time_fields": TIME_FIELDS,
        "event_kinds": sorted({e["kind"] for e in events}), "extra_event_kinds": EXTRA_EVENT_KINDS,
        "search_fields_per_slot": n_tiles,
    }
    brief = generate_brief(meta, metrics, events)
    meta["brief_generated_utc"] = _now_utc()
    timing["total_s"] = time.perf_counter() - tic
    meta["timing_s"] = {k: round(v, 2) for k, v in timing.items()}
    return {"meta": meta, "frames": frames_out, "events": events, "metrics": metrics, "brief": brief}


# ---------------------------------------------------------------------------
def _gateway_diagnostics(rs, reg, st) -> dict:
    """Neck diagnostics of a reachable set for one gateway region (plain numbers for the alert / brief)."""
    L_rot = np.asarray(reg.params["center_rot_nd"], dtype=np.float64)
    radius = float(reg.params["radius_nd"])
    margin = float(reg.params.get("realm_margin_nd", radius))
    dg = neck_passage_diagnostics(rs.states_rot[..., :3], L_rot, radius, margin, rs.active, rs.hits[reg.key])
    k = dg.get("closest_epoch_index")
    nominal_d = np.linalg.norm(rs.nominal_rot[:, :3] - L_rot, axis=1) * L_STAR
    return {"libration_point": reg.name.split()[0], "far_realm": "Earth realm" if reg.key == "l1_gateway" else "exterior realm",
            "n_samples": int(dg["n"]), "n_transit": int(st.n_hit),
            "n_enter_ball": int(dg["n_enter_ball"]), "n_cross_plane_in_ball": int(dg["n_cross_plane_in_ball"]),
            "n_cross_and_return": int(dg["n_cross_and_return"]), "n_reach_far_realm": int(dg["n_reach_far_realm"]),
            "n_far_realm_other_route": int(dg["n_far_realm_other_route"]),
            "other_route_min_crossing_km": _r(dg["other_route_min_crossing_km"], 0),
            "max_depth_past_plane_km": _r(dg["max_depth_past_plane_km"], 0),
            "closest_km": _r(dg["closest_km"], 0), "t_closest_h": _r(rs.t_h[k], 1) if k is not None else None,
            "closest_dv_mps": _r(rs.dv_mps[dg["closest_sample"]], 1) if dg.get("closest_sample") is not None else None,
            "nominal_closest_km": _r(float(nominal_d.min()), 0), "nominal_t_closest_h": _r(rs.t_h[int(np.argmin(nominal_d))], 1),
            "radius_km": _r(radius * L_STAR, 0), "realm_margin_km": _r(margin * L_STAR, 0),
            "definition": "transit = enters the neck ball from one realm proper, crosses the plane inside it and reaches the other "
                          "realm proper (|x - x_L| > margin) with C < C_L; counts are samples (not rays)"}


def _tasking_metrics(reacq, attempts, t0, n_tiles, n_follow, cfg) -> dict:
    base = {"search_tiles_used_by_scheduler": n_tiles, "search_budget_note": "mosaic budget per tasked slot shared by the scheduler's "
            "acquisition model and the truth verification (a look acquires iff the truth is within the radius min(p90 spread, "
            "0.5 FOV sqrt(budget)) of the tasked boresight)", "attempts": attempts, "n_followup_obs": n_follow,
            "scheduler": f"greedy ({cfg.escalation_gain} gain, FOV acquisition model), priority 3 on the object of interest",
            "escalation_delay_h": cfg.escalation_delay_s / HOUR}
    if reacq is None:
        base.update(sensor_id=None, sensor_ids=[], t_task_utc=None, t_task_rel_s=None, pointing_ra_deg=None, pointing_dec_deg=None,
                    n_fields=None, spread_p90_deg=None, offset_deg=None, range_km=None, magnitude=None, sensors=[], not_acquired=[], blocked=[])
        return base
    pr = reacq["primary"]
    base.update(sensor_id=pr["sensor_id"], sensor_ids=[r["sensor_id"] for r in reacq["sensors"]], t_task_utc=reacq["t_utc"],
                t_task_rel_s=reacq["t_s"] - t0, pointing_ra_deg=_r(pr["ra_deg"], 3), pointing_dec_deg=_r(pr["dec_deg"], 3),
                n_fields=pr["n_fields_p90"], n_fields_used=pr.get("n_fields_used"), mosaic_radius_deg=_r(pr.get("mosaic_radius_deg"), 3),
                spread_p90_deg=_r(pr["spread_p90_deg"], 2), offset_deg=_r(pr["offset_deg"], 3),
                range_km=_r(pr["range_km"], 0), magnitude=_r(pr["magnitude"], 2),
                sensors=[{k: (_r(v, 3) if isinstance(v, float) else v) for k, v in r.items() if k != "pointing_gcrf"} for r in reacq["sensors"]],
                not_acquired=reacq["not_acquired"], blocked=reacq["blocked"])
    return base


def _loss_reason(dominant: list, avail_sites: list, grid: np.ndarray, t_det: float, t_lost: float, t1: float) -> dict:
    """Honest attribution of the ground-network blindness.

    Per frame, ``dominant`` is the majority target-specific reason over the sites that were AVAILABLE (night, object
    above the elevation limit), or ``no_site_available`` when no site could point at the object at all (daylight /
    below elevation everywhere -- not a glare effect).  ``dominant`` here is the most common target-specific reason over
    the post-detection blind frames that had at least one available site; the no-site frames are counted separately.
    """
    from collections import Counter
    between = Counter(d for d, t in zip(dominant, grid) if t_det <= t <= t_lost and d)
    after = Counter(d for d, t in zip(dominant, grid) if t_det <= t <= t1 and d)
    k_lost = int(np.argmin(np.abs(grid - t_lost)))
    specific = Counter({k: v for k, v in after.items() if k != NO_SITE_AVAILABLE})
    dominant_specific = specific.most_common(1)[0][0] if specific else None
    n_blind = int(sum(after.values()))
    return {"dominant": dominant_specific or (NO_SITE_AVAILABLE if after else None),
            "at_loss": dominant[k_lost], "available_sites_at_loss": list(avail_sites[k_lost]),
            "counts_between_detection_and_loss": dict(between), "counts_after_detection": dict(after),
            "n_blind_frames_after_detection": n_blind,
            "n_frames_no_site_available_after_detection": int(after.get(NO_SITE_AVAILABLE, 0)),
            "n_frames_available_site_blocked_after_detection": int(sum(specific.values())),
            "note": "per hourly frame: majority highest-priority target-specific reason over the AVAILABLE ground sites (night, object "
                    "above the elevation limit); 'no_site_available' when every site was in daylight or had the object below its "
                    "elevation limit. 'dominant' is the most common target-specific reason over the post-detection blind frames with "
                    "an available site; the no-site frames are reported separately and are not attributed to glare."}


def _reason_counts(dominant: list, grid: np.ndarray, t_a: float, t_b: float) -> dict:
    from collections import Counter
    return dict(Counter(d for d, t in zip(dominant, grid) if t_a <= t <= t_b and d))


def _count_by_sensor(meas) -> dict:
    from collections import Counter
    return dict(Counter(m.sensor_id for m in meas))


def _seg_summary(res) -> dict:
    from selene.tasking.metrics import custody_metrics
    m = custody_metrics(res)["overall"]
    return {k: (round(v, 3) if isinstance(v, float) else v) for k, v in m.items()
            if k in ("custody_pct", "mean_tslo_h", "n_observations", "n_slots", "horizon_h", "n_objects", "n_sensors")}


def _now_utc() -> str:
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _blind_phrase(reason: Optional[str], avail: list[str]) -> str:
    """Plain statement of why the ground network is blind at one instant (per-site truth, no priority collapse)."""
    if reason == NO_SITE_AVAILABLE:
        return "no ground site has the object above its elevation limit at night (daylight or below the horizon everywhere)"
    if reason == "moon_exclusion":
        return (f"every site that has the object up at night ({', '.join(avail) or 'none'}) is blocked by lunar glare "
                f"(object inside the phase-dependent Moon exclusion cone)")
    return f"every available site ({', '.join(avail) or 'none'}) is blocked: {reason}"


def _build_events(cfg, t0, t1, tb, t_det, t_degraded, t_lost, t_regained, t_char, run, i_det, reacq, attempts, entries, region_rows,
                  metrics, est_block, status, grid, grid_utc, cloud_sigma, follow_meas, dominant_reason, avail_sites, illum, sep_moon,
                  d_l1_km, closest_l1=None) -> list:
    """Chronological feed.  Same-epoch ordering is fixed by ``rank`` (lower first): truth marker -1, status/detection 1,
    tasking 2, re-acquisition tracklet 3, custody change 4, follow-up tracklets 5, characterisation 6, brief 9.
    Event times: ``t`` = ``t_rel_s`` seconds since t0 and ``t_utc`` (see :data:`TIME_FIELDS`)."""
    ev: list[dict] = []
    R_TRUTH, R_OBS, R_DETECT, R_STATUS, R_ALERT, R_TASK, R_REACQ, R_CUSTODY, R_FOLLOW, R_CHAR, R_BRIEF = -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9

    def add(t_s, kind, severity, text, object_id=None, data=None, rank=0):
        ev.append({"t": float(t_s - t0), "t_rel_s": float(t_s - t0), "t_utc": _utc(t_s), "kind": kind, "severity": severity,
                   "text": text, "object_id": object_id, "data": data or {}, "_rank": rank})

    # custody nominal
    add(t0, "custody_nominal", "info",
        f"{PROTAGONIST} under routine custody: ground-network tracklets every {cfg.obs_cadence_s / HOUR:.0f} h when a site can see it "
        f"(Moon {illum[0] * 100:.0f} % illuminated, object {sep_moon[0]:.1f} deg from the Moon). Prior 1-sigma {cfg.prior_pos_km:.0f} km / "
        f"{cfg.prior_vel_mps:.1f} m/s; the space observers are allocated to the other catalogued SIMULATED objects.", PROTAGONIST,
        {"show_layers": ["objects", "sensors"]}, rank=R_STATUS)
    # observations (ground pre-burn, then thinned space follow-ups)
    t_re = reacq["t_s"] if reacq is not None else None
    last_follow_event = t_re if t_re is not None else -np.inf   # the re-acquisition slot has its own event below
    for k, m in enumerate(run.meas):
        res_as = float(np.linalg.norm(run.innov[k]) / ARCSEC)
        if k == i_det:
            continue
        rank = R_OBS
        if t_re is not None and m.t_s >= t_re - 1e-6:
            # space follow-ups: one feed entry per 6 h (every tracklet is still in the frames' observation lists)
            if m.t_s - last_follow_event < 6 * HOUR - 1e-6:
                continue
            last_follow_event = m.t_s
            rank = R_FOLLOW
        txt = f"{m.sensor_id} tracklet on {PROTAGONIST}: residual {res_as:.2f}\" (NIS {run.nis[k]:.1f}), sigma_pos {np.sqrt(np.trace(run.P[k][:3, :3])):.1f} km"
        add(m.t_s, "observation", "info", txt, PROTAGONIST, {"sensor_id": m.sensor_id, "residual_arcsec": round(res_as, 3),
                                                              "nis": round(float(run.nis[k]), 2), "magnitude": round(float(m.magnitude), 2)}, rank=rank)
    # the burn itself (truth, SIMULATED) -- shown as an info marker so the audience knows what happened
    add(tb, "maneuver", "info",
        f"[SIMULATED truth] notional actor performs an unannounced {cfg.burn_mps:.0f} m/s burn on {PROTAGONIST} "
        f"(RTN {metrics['dv_true_rtn_mps'][0]:+.1f}/{metrics['dv_true_rtn_mps'][1]:+.1f}/{metrics['dv_true_rtn_mps'][2]:+.1f} m/s, "
        f"{metrics['dv_true_frame_center']}-centred). Not visible to the operator at this instant.", PROTAGONIST,
        {"dv_mps": cfg.burn_mps, "truth": True}, rank=R_TRUTH)
    # detection
    m = run.meas[i_det]
    res_as = float(np.linalg.norm(run.innov[i_det]) / ARCSEC)
    add(t_det, "maneuver_detected", "alert",
        f"MANEUVER DETECTED on {PROTAGONIST}: {m.sensor_id} tracklet is {res_as:.0f}\" off the predicted position "
        f"(NIS {run.nis[i_det]:.0f} vs family-wise threshold {metrics['detection']['threshold_nis_familywise']:.1f} at alpha = {metrics['detection']['alpha']:g}). "
        f"Filter covariance re-opened for an unknown impulsive dv (sigma {cfg.reopen_sigma_dv_mps:.0f} m/s over the "
        f"{(t_det - run.t_s[i_det - 1]) / HOUR:.1f} h gap).", PROTAGONIST,
        {"sensor_id": m.sensor_id, "residual_arcsec": round(res_as, 2), "nis": round(float(run.nis[i_det]), 1),
         "confidence": 1.0 - 1e-6, "show_layers": ["clouds"]}, rank=R_DETECT)
    # custody DEGRADED at the status change itself; the text also says when and why the ground network goes blind
    k_det = int(np.argmin(np.abs(grid - t_det)))
    k_blind = next((k for k in range(k_det + 1, len(grid)) if dominant_reason[k]), None)
    k_glare = next((k for k in range(k_det + 1, len(grid)) if dominant_reason[k] == "moon_exclusion"), None)
    blind_txt = ""
    if k_blind is not None:
        blind_txt = (f" Ground network blind from {grid_utc[k_blind]}: {_blind_phrase(dominant_reason[k_blind], avail_sites[k_blind])}"
                     f" (Moon {illum[k_blind] * 100:.0f} % illuminated, object {sep_moon[k_blind]:.1f} deg from it, glare exclusion "
                     f"{3 + 12 * illum[k_blind]:.1f} deg).")
        if k_glare is not None and k_glare != k_blind:
            blind_txt += (f" From {grid_utc[k_glare]} the sites that do have it up at night ({', '.join(avail_sites[k_glare])}) are blocked by "
                          f"lunar glare alone.")
    t_dg = t_degraded if t_degraded is not None else (float(grid[k_blind]) if k_blind is not None else None)
    if t_dg is not None:
        k_dg = int(np.argmin(np.abs(grid - t_dg)))
        add(t_dg, "custody_degraded", "warn",
            f"CUSTODY DEGRADED on {PROTAGONIST}: particle-cloud sigma_pos {cloud_sigma[k_dg]:.0f} km > {cfg.custody_km:.0f} km "
            f"(a single post-burn angles-only tracklet cannot resolve the dv)." + blind_txt +
            f" Uncertainty growing at ~{cfg.reopen_sigma_dv_mps * 86.4:.0f} km/day.", PROTAGONIST,
            {"sigma_km": round(float(cloud_sigma[k_dg]), 1), "t_blind_utc": grid_utc[k_blind] if k_blind is not None else None,
             "reason": dominant_reason[k_blind] if k_blind is not None else None,
             "glare_from_utc": grid_utc[k_glare] if k_glare is not None else None}, rank=R_STATUS)
    if t_lost is not None:
        k_l = int(np.argmin(np.abs(grid - t_lost)))
        add(t_lost, "custody_lost", "alert",
            f"CUSTODY LOST on {PROTAGONIST}: particle-cloud sigma_pos {cloud_sigma[k_l]:.0f} km > {cfg.lost_km:.0f} km with no observation since "
            f"{_utc(t_det)}. Ground: {_blind_phrase(dominant_reason[k_l], avail_sites[k_l])} (dominant reason '{dominant_reason[k_l] or 'n/a'}'). "
            f"Escalating to the space network.", PROTAGONIST,
            {"sigma_km": round(float(cloud_sigma[k_l]), 1), "reason": dominant_reason[k_l], "available_sites": avail_sites[k_l],
             "show_layers": ["clouds", "reachable"]}, rank=R_STATUS)
    t_alert = t_det
    # reachability alert
    n_rays = int(metrics["reachability"]["n_samples"])
    # headline = regions that the burn OPENS (not on the unperturbed path); regions the quiet orbit visits anyway carry no
    # information about the maneuver and are listed separately so the alert never flags routine geometry as new
    new = [r for r in region_rows if r["n_hit"] and not r["nominal_hits"]]
    routine = [r for r in region_rows if r["nominal_hits"]]
    parts = [f"{r['name']}: {100 * r['fraction']:.0f} % of sampled rays, earliest +{r['earliest_h']:.0f} h, >= {r['min_dv_mps']:.0f} m/s"
             for r in new]
    if not parts:
        parts.append("no high-value region that the unperturbed orbit does not already visit")
    routine_txt = ""
    if routine:
        routine_txt = (" Already on the unperturbed path (not attributable to the burn): "
                       + "; ".join(f"{r['name']} (first entry +{r['earliest_nominal_h']:.0f} h with no burn)" for r in routine) + ".")
    nrho = next((r for r in region_rows if r["key"] == "nrho_corridor"), None)
    relay_txt = ""
    if nrho is not None:
        # a 0-vs-1 ray difference flips with the noise seed at this sampling resolution, so both cases are worded as 'marginal'
        if nrho["n_hit"]:
            relay_txt = (f" The corridor of the notional allied relay ({RELAY}, 9:2 NRHO) is marginal at this sampling resolution: "
                         f"{nrho['n_hit']} of {n_rays} rays, only at the budget limit (>= {nrho['min_dv_mps']:.0f} m/s, earliest +{nrho['earliest_h']:.0f} h).")
        else:
            relay_txt = (f" The corridor of the notional allied relay ({RELAY}, 9:2 NRHO) is not reached by any of the {n_rays} sampled rays within "
                         f"this budget and horizon (at or below the sampling resolution: a marginal hit at the budget limit cannot be excluded).")
        relay_txt += " Recommendation: increased custody on the object and conjunction screening for the relay; no closer approach is implied."
    gateway_txt = " " + " ".join(gateway_sentence(r, cfg.horizon_h) for r in region_rows if r["kind"] == "gateway")
    add(t_alert, "reachability_alert", "warn",
        f"REACHABILITY (awareness only): from the last good state with an assumed {cfg.dv_budget_mps:.0f} m/s budget over {cfg.horizon_h:.0f} h, "
        f"{PROTAGONIST} could newly enter: " + "; ".join(parts) + "." + routine_txt + gateway_txt + relay_txt, PROTAGONIST,
        {"dv_budget_mps": cfg.dv_budget_mps, "horizon_h": cfg.horizon_h, "regions": region_rows,
         "newly_reachable": [r["key"] for r in new], "on_unperturbed_path": [r["key"] for r in routine],
         "gateway_diagnostics": {r["key"]: r.get("diagnostics") for r in region_rows if r["kind"] == "gateway"},
         "show_layers": ["reachable"]}, rank=R_ALERT)
    # tasking
    tk = metrics["tasking"]
    if reacq is not None:
        pr = reacq["primary"]
        got_txt = "; ".join(f"{r['sensor_id']} ({r['n_fields_used']} of {r['n_fields_p90']} field(s) of {r['fov_deg']:.0f} deg needed for the p90 spread "
                            f"{r['spread_p90_deg']:.2f} deg, range {r['range_km']:,.0f} km, mag {r['magnitude']:.1f})" for r in reacq["sensors"])
        miss_txt = "; ".join(f"{r['sensor_id']} (truth visible: {r['truth_visible']}, offset {r['offset_deg']}, {r['truth_block_reason']})"
                             for r in reacq["not_acquired"])
        blk_txt = "; ".join(f"{b['sensor_id']} ({b['reason'] or 'not selected'})" for b in reacq["blocked"])
        add(reacq["t_s"], "tasking_update", "info",
            f"SELENE tasker redirects {len(reacq['sensors'])} space sensor(s) to {PROTAGONIST}, boresights at the reachable-set centroid "
            f"(primary {pr['sensor_id']}: RA {pr['ra_deg']:.2f} deg, Dec {pr['dec_deg']:+.2f} deg): {got_txt}."
            + (f" Looks that did not acquire: {miss_txt}." if miss_txt else "")
            + (f" Not tasked: {blk_txt}." if blk_txt else "")
            + f" Scheduler: greedy {cfg.escalation_gain} gain, FOV acquisition model with a search budget of {tk['search_tiles_used_by_scheduler']} "
              f"fields per slot, priority 3 on {PROTAGONIST}.", PROTAGONIST,
            {"sensor_id": pr["sensor_id"], "sensor_ids": [r["sensor_id"] for r in reacq["sensors"]], "target_id": PROTAGONIST,
             "ra_deg": pr["ra_deg"], "dec_deg": pr["dec_deg"], "n_fields": pr["n_fields_used"], "show_layers": ["sensors", "reachable"]}, rank=R_TASK)
        add(reacq["t_s"], "observation", "info",
            f"RE-ACQUIRED {PROTAGONIST}: " + "; ".join(f"{r['sensor_id']} truth {r['offset_deg']:.2f} deg from boresight (mosaic radius {r['mosaic_radius_deg']:.2f} deg)"
                                                        for r in reacq["sensors"])
            + f". Follow-up tracklets per the tasker's plan every {cfg.followup_cadence_s / HOUR:.0f} h.", PROTAGONIST,
            {"sensor_id": pr["sensor_id"], "sensor_ids": [r["sensor_id"] for r in reacq["sensors"]], "reacquired": True}, rank=R_REACQ)
        for a in attempts:
            if a["t_s"] < reacq["t_s"] - 1e-6:
                add(a["t_s"], "tasking_update", "warn",
                    f"Tasked look by {a['sensor_id']} did not acquire {PROTAGONIST} (truth visible: {a['truth_visible']}, "
                    f"offset {a['offset_deg']}, reasons {a['truth_block_reason']}).", PROTAGONIST, {"sensor_id": a["sensor_id"], "acquired": False},
                    rank=R_TASK)
    else:
        add(t_alert + HOUR, "tasking_update", "alert",
            f"No space observer could re-acquire {PROTAGONIST} within the window ({len(attempts)} looks verified). "
            "Custody remains lost; this is reported as-is.", PROTAGONIST, {"acquired": False}, rank=R_TASK)
    if t_regained is not None:
        k_r = int(np.argmin(np.abs(grid - t_regained)))
        n_sp = sum(1 for m_ in follow_meas if m_.t_s <= t_regained + 1e-6)
        add(t_regained, "custody_regained", "info",
            f"CUSTODY REGAINED on {PROTAGONIST}: cloud sigma_pos {cloud_sigma[k_r]:.1f} km < {cfg.custody_km:.0f} km after "
            f"{n_sp} space-observer tracklet(s) ({metrics['regained_after_h'] or 0:.1f} h after loss, "
            f"{metrics['regained_after_burn_h']:.1f} h after the burn).", PROTAGONIST,
            {"sigma_km": round(float(cloud_sigma[k_r]), 2), "show_layers": ["clouds"]}, rank=R_CUSTODY)
    # region entries (truth): with the neck-transit gateway definition a region entry means a real realm change
    for en in entries:
        if en["t_s"] <= t0 + 1e-6:
            continue
        k = en["frame"]
        add(en["t_s"], "entered_region", "warn" if en["kind"] in ("gateway", "corridor") else "info",
            f"[SIMULATED truth] {PROTAGONIST} enters {en['name']} ({d_l1_km[k]:,.0f} km from L1); custody status at the time: {status[k]}.",
            PROTAGONIST, {"region": en["key"], "custody": status[k]}, rank=R_OBS)
    # closest approach to L1 (truth): the burned arc grazes the neck; say so instead of calling a graze an entry
    if closest_l1 is not None:
        t_ca, d_ca, transit, d_quiet = closest_l1
        if t0 < t_ca < t1:
            k = int(np.argmin(np.abs(grid - t_ca)))
            add(t_ca, "closest_approach", "info",
                f"[SIMULATED truth] {PROTAGONIST} passes {d_ca:,.0f} km from the Earth-Moon L1 point (+{(t_ca - tb) / HOUR:.1f} h after the burn; "
                f"the unperturbed orbit would have come no closer than {d_quiet:,.0f} km in this window) and "
                f"{'transits the L1 neck into the Earth realm' if transit else 'does NOT transit the L1 neck: it stays in the lunar realm'}; "
                f"custody status at the time: {status[k]}.", PROTAGONIST,
                {"region": "l1_gateway", "distance_km": round(float(d_ca), 0), "transit": bool(transit), "custody": status[k]}, rank=R_OBS)
    if t_char is not None and est_block is not None:
        add(t_char, "maneuver_characterised", "info",
            f"MANEUVER CHARACTERISED: dv {est_block['dv_mps']:.2f} +/- {est_block['dv_sigma_mps']:.2f} m/s "
            f"(truth {cfg.burn_mps:.0f} m/s, error {metrics['dv_est_err_pct']:+.2f} %), direction +/- {est_block['direction_sigma_deg']:.2f} deg "
            f"({metrics['dir_err_deg']:.2f} deg from truth), RTN {est_block['dv_rtn_mps'][0]:+.1f}/{est_block['dv_rtn_mps'][1]:+.1f}/{est_block['dv_rtn_mps'][2]:+.1f} m/s, "
            f"epoch {est_block['t_burn_utc']} +/- {est_block['t_burn_sigma_s']:.0f} s ({est_block['t_burn_err_s']:+.0f} s from truth); "
            f"{est_block['classification']['primary']}.", PROTAGONIST,
            # ``direction`` is GCRF (same frame as metrics.dv_true_dir_gcrf and the maneuver route's direction_gcrf);
            # the rotating-frame version is published under its own, frame-tagged name
            {"dv_mps": est_block["dv_mps"], "dv_sigma_mps": est_block["dv_sigma_mps"],
             "direction": est_block["direction_gcrf"], "direction_frame": "gcrf", "direction_gcrf": est_block["direction_gcrf"],
             "direction_rot": est_block["direction_rot"], "direction_rot_frame": "earth_moon_rotating_nd",
             "confidence": 1.0 - metrics["detection"]["alpha"], "dv_rtn_mps": est_block["dv_rtn_mps"]}, rank=R_CHAR)
        t_brief = max(t_char, t_regained or t_char) + 1.0
    elif est_block is None:
        t_brief = (t_regained or (t_lost or t_det)) + HOUR
        add(t_brief - 1.0, "maneuver_characterised", "warn",
            f"Maneuver NOT characterised: {metrics['dv_estimate_error']}", PROTAGONIST, {"error": metrics["dv_estimate_error"]}, rank=R_CHAR)
    else:
        t_brief = t_char + 1.0
    t_brief = min(t_brief, t1)
    add(t_brief, "brief_ready", "info", "Analyst brief generated (SIMULATED — notional actor).", PROTAGONIST, {"show_layers": ["brief"]}, rank=R_BRIEF)
    ev.sort(key=lambda d: (d["t_rel_s"], d["_rank"], d["kind"]))
    for d in ev:
        d.pop("_rank", None)
    return ev


EPOCH_RATIONALE = (
    "Window 2026-02-23 .. 2026-03-01 UTC (6 days, hourly frames) inside the cached DE440s/Horizons span. Scanning the span with "
    "the real visibility model: SIM-DRO-01 stays within ~12 deg of the Moon on the sky, so under the phase-dependent lunar-glare "
    "exclusion (3 deg at new Moon to 15 deg at full Moon) the nine ground sites can only track it on the nights of Feb 23-25 "
    "(Moon 33-60 % illuminated, exclusion 7-10 deg, object 10-11 deg from the Moon) and are blind from Feb 25 ~11 UTC through "
    "full Moon (Mar 3). The burn is placed 30 min after the last pre-burn ground tracklet so that exactly one post-burn "
    "ground observation catches the anomaly before the glare closes the window; the loss is therefore computed, not scripted. "
    "Known fragility: that tracklet clears the assumed phase-dependent glare cone by only ~0.1 deg, so a different glare "
    "parameterisation or a 30-min epoch shift leaves no post-burn ground observation and the builder raises instead of "
    "inventing a detection."
)
#: formatted at build time with the refined closest-approach numbers (``{closest_km}``, ``{closest_h}``, ``{quiet_km}``, ``{transit}``)
BURN_RATIONALE = (
    "30 m/s (assumed, inside the 20-40 m/s class) at 2026-02-25T08:30Z. Direction: of 256 Fibonacci-sampled 30 m/s burn "
    "directions propagated with the reachability engine, the one whose coasting arc passes closest to the Earth-Moon L1 point "
    "({closest_km:,.0f} km at +{closest_h:.1f} h after the burn, refined on the dense solution, vs {quiet_km:,.0f} km for the "
    "unperturbed DRO in this window). The burned arc {transit} the L1 neck (it grazes L1 and stays in the lunar realm), which "
    "the bundle states explicitly. The notional actor's intent is never inferred."
)
ASSUMPTIONS = [
    "All sensors are notional (assumed representative specs); astrometric noise 1 arcsec (1-sigma, isotropic on-sky).",
    "Routine custody: one ground tracklet every 2 h when any site can see the object; space observers are allocated to the "
    "libration-point objects until the tasker escalates.",
    "Filter: UKF with DE440s Earth+Moon+Sun+SRP dynamics, process-noise PSD 1e-18 km^2/s^3; the t0 prior (5 km / 0.2 m/s, 1-sigma) "
    "stands for the catalogue covariance of an object that has been under custody for weeks (the 20 km / 2 m/s acquisition prior "
    "would drift past 100 km in the 8-h gaps between ground sites); after a gross NIS exceedance the prior is re-opened for an "
    "impulsive dv of sigma 25 m/s at an unknown epoch in the last gap (must bound the burn class).",
    "Custody thresholds on the particle-cloud RSS position sigma: < 100 km CUSTODY, 100-1000 km DEGRADED, > 1000 km LOST.",
    "Reachability: assumed 100 m/s planning budget, 168 h horizon, impulsive burns on a Fibonacci sphere x absolute magnitude ladder "
    "x burn epochs spanning the detection gap; fractions are fractions of sampled (direction, epoch) rays.",
    "Tasking: greedy scheduler (trace gain for routine custody, log-det information gain for the escalation because re-acquisition "
    "must resolve range and velocity, not just plane-of-sky position) with the FOV acquisition model; every tasked look is verified "
    "against the SIMULATED truth (visibility model + mosaic of fields around the reachable-set centroid), never assumed; the scheduler's "
    "acquisition model and the verification share one search budget of 25 fields per 60-min slot (no explicit slew/exposure timeline); "
    "one slot of re-planning latency between the loss declaration and the first tasked look.",
    "Ground-blind attribution: per hourly frame only the sites that are available (astronomical night, object above the minimum elevation) "
    "are asked why they are blocked; frames where no site is available are reported as 'no_site_available', not as glare.",
    "Force model has no lunar gravity field or planetary perturbations beyond the Sun; light-time and aberration are not modelled.",
]
ENGINES = ["selene.dynamics (DE440s N-body + SRP)", "selene.sensors (photometry, exclusions, glare, eclipse)", "selene.od.ukf", "selene.od.particles",
           "selene.maneuver.detection / estimation", "selene.reachability.sampling / tasking_hint", "selene.tasking.greedy"]


# ---------------------------------------------------------------------------
def write_bundle(bundle: dict, path: Path = SCENARIO_PATH, meta_path: Path = META_PATH) -> tuple[Path, Path]:
    path.parent.mkdir(parents=True, exist_ok=True)
    txt = json.dumps(bundle, separators=(",", ":"), allow_nan=False)
    path.write_text(txt)
    meta = {"meta": bundle["meta"], "metrics": bundle["metrics"], "n_events": len(bundle["events"]),
            "n_frames": len(bundle["frames"]), "bundle_bytes": len(txt.encode("utf-8"))}
    meta_path.write_text(json.dumps(meta, indent=1, allow_nan=False))
    return path, meta_path


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Build the SELENE demo scenario bundle (SIMULATED).")
    ap.add_argument("--rebuild", action="store_true", help="run the engines and write data/demo/scenario.json")
    ap.add_argument("--fast", action="store_true", help="reduced sample counts (< 60 s)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=str, default=None, help="output path (default data/demo/scenario.json)")
    args = ap.parse_args(argv)
    if not args.rebuild:
        ap.print_help()
        return 0
    tic = time.perf_counter()
    bundle = build_scenario(seed=args.seed, fast=args.fast)
    out = Path(args.out) if args.out else SCENARIO_PATH
    meta_out = out.with_name(out.stem + "_meta.json")
    p, mp = write_bundle(bundle, out, meta_out)
    m = bundle["metrics"]
    print(f"wrote {p} ({p.stat().st_size / 1e6:.2f} MB) and {mp} in {time.perf_counter() - tic:.1f} s")
    print(f"frames={len(bundle['frames'])} events={len(bundle['events'])} latency_h={m['detection_latency_h']} "
          f"lost={m['t_lost_utc']} regained={m['t_regained_utc']} dv_true={m['dv_true_mps']} dv_est={m['dv_est_mps']} "
          f"err%={m['dv_est_err_pct']} dir_err={m['dir_err_deg']} loss_reason={m['loss_reason']['dominant']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
