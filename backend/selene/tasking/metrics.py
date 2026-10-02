"""Custody metrics and baseline schedulers (random, round-robin) for honest comparisons.

Custody definition
------------------
An object is *in custody* at a slot node when its RSS position sigma ``sqrt(tr P_pos)`` is below
``custody_threshold_km`` (default 100 km, configurable).  ``custody_pct`` is the percentage of
slot nodes (post-update covariance at ``t_k``, i.e. the knowledge state right after the slot's
observations) satisfying this -- a time fraction on the slot grid, not a count of observations.
Objects that start outside the threshold therefore show poor custody until somebody actually
observes them; that is the point.  Conversely an object whose prior is already well inside the
threshold can stay "in custody" for a long time with *no* observations at all (on the API default
-- 10 km prior, q = 1e-18 km²/s³, 48 h -- every object does), so ``custody_pct`` only
discriminates policies when the prior is stale or sensors are scarce.  :func:`null_custody_pct`
gives the no-observation reference and ``custody_pct_observed`` the custody fraction restricted
to nodes at/after each object's first observation.

Time since last observation (TSLO) at node ``k`` is ``t_k − t_last_obs``; for an object never
observed it is measured from ``t0`` and the object is flagged ``never_observed``.

Baselines
---------
* ``random``: each sensor picks uniformly among its visible + slew-feasible objects (seeded).
* ``round_robin``: each sensor cycles through the object list in a fixed order and takes the
  next visible + feasible one after its previous pick.

Both use the same engine, visibility table and update equations as the greedy and MILP policies.
"""
from __future__ import annotations



import numpy as np

from selene.tasking.greedy import Scenario, ScheduleResult, SlotContext, run_schedule

__all__ = [
    "DEFAULT_CUSTODY_THRESHOLD_KM",
    "custody_metrics",
    "null_custody_pct",
    "run_null",
    "RandomPolicy",
    "RoundRobinPolicy",
    "run_random",
    "run_round_robin",
    "comparison_row",
]

DEFAULT_CUSTODY_THRESHOLD_KM = 100.0


# ---------------------------------------------------------------------------
# metrics
# ---------------------------------------------------------------------------
def custody_metrics(res: ScheduleResult, custody_threshold_km: float = DEFAULT_CUSTODY_THRESHOLD_KM) -> dict:
    """Per-object and overall custody statistics for a :class:`ScheduleResult`."""
    scn = res.scenario
    t = res.t_nodes
    t0 = float(t[0])
    K1 = t.size
    thr = float(custody_threshold_km)
    per_object = []
    custody_mask = res.sigma_km < thr                     # (J, K+1)
    tslo_h = np.where(np.isfinite(res.last_obs_t), (t[None, :] - res.last_obs_t) / 3600.0, (t[None, :] - t0) / 3600.0)
    for j, tr in enumerate(scn.tracks):
        n_obs = int(res.n_obs_object[j])
        per_object.append({
            "id": tr.object_id,
            "name": tr.meta.get("name", tr.object_id),
            "orbit_type": tr.meta.get("orbit_type", ""),
            "label": tr.meta.get("label", ""),
            "priority": tr.priority,
            "custody_pct": 100.0 * float(custody_mask[j].mean()),
            "n_obs": n_obs,
            "never_observed": n_obs == 0,
            "mean_tslo_h": float(tslo_h[j].mean()),
            "max_tslo_h": float(tslo_h[j].max()),
            "initial_sigma_km": float(res.sigma_km[j, 0]),
            "final_sigma_km": float(res.sigma_km[j, -1]),
            "min_sigma_km": float(res.sigma_km[j].min()),
            "max_sigma_km": float(res.sigma_km[j].max()),
            "sigma_series_km": res.sigma_km[j].tolist(),
            "tslo_series_h": tslo_h[j].tolist(),
            "in_custody_series": custody_mask[j].astype(bool).tolist(),
            "observed_by": sorted({a.sensor_id for a in res.assignments if a.object_id == tr.object_id}),
        })
    S = scn.n_sensors
    K = scn.n_slots
    n_assign = len(res.assignments)
    sensors = []
    for s, sid in enumerate(res.sensor_ids):
        n = int(res.n_obs_sensor[s])
        avail = K - int(res.idle_no_candidate[s])
        sensors.append({
            "id": sid,
            "n_obs": n,
            "utilisation_pct": 100.0 * n / K if K else 0.0,
            "slots_with_candidates": avail,
            "utilisation_when_available_pct": 100.0 * n / avail if avail else 0.0,
            "slots_without_candidates": int(res.idle_no_candidate[s]),
            "blocked_reason_counts": res.idle_reason_counts.get(sid, {}),
            "objects_observed": sorted({a.object_id for a in res.assignments if a.sensor_id == sid}),
        })
    summed_trace = res.trace_pos.sum(axis=0)             # (K+1,)
    observed_mask = np.isfinite(res.last_obs_t)          # (J, K+1) node at/after the object's first observation
    overall = {
        "method": res.method,
        "custody_threshold_km": thr,
        "custody_pct": 100.0 * float(custody_mask.mean()) if custody_mask.size else 0.0,
        "custody_pct_min_object": float(min(o["custody_pct"] for o in per_object)) if per_object else 0.0,
        "objects_in_custody_at_end": int(custody_mask[:, -1].sum()) if custody_mask.size else 0,
        "n_objects": scn.n_objects,
        "n_sensors": S,
        "n_slots": K,
        "slot_min": scn.slot_s / 60.0,
        "horizon_h": (float(t[-1]) - t0) / 3600.0,
        "n_observations": n_assign,
        "n_objects_never_observed": int(sum(1 for o in per_object if o["never_observed"])),
        "mean_tslo_h": float(tslo_h.mean()) if tslo_h.size else 0.0,
        "max_tslo_h": float(tslo_h.max()) if tslo_h.size else 0.0,
        "summed_trace_initial_km2": float(summed_trace[0]) if summed_trace.size else 0.0,
        "summed_trace_final_km2": float(summed_trace[-1]) if summed_trace.size else 0.0,
        "summed_trace_mean_km2": float(summed_trace.mean()) if summed_trace.size else 0.0,
        "summed_trace_series_km2": summed_trace.tolist(),
        "rss_sigma_final_km": float(np.sqrt(summed_trace[-1])) if summed_trace.size else 0.0,
        "mean_final_sigma_km": float(res.sigma_km[:, -1].mean()) if res.sigma_km.size else 0.0,
        "total_gain": res.total_gain(),
        "sensor_utilisation_pct": 100.0 * n_assign / (S * K) if S * K else 0.0,
        "runtime_s": res.runtime_s,
        "custody_note": "custody is a covariance criterion: an object whose prior is already inside the threshold "
                        "stays 'in custody' without observations for as long as the propagated sigma holds "
                        "(check n_objects_never_observed, mean_tslo_h and custody_pct_null); custody_pct_observed is "
                        "the custody fraction over (object, node) cells at/after the object's first observation only",
        "custody_pct_observed": (100.0 * float(custody_mask[observed_mask].mean()) if observed_mask.any() else 0.0),
        "n_object_nodes_observed": int(observed_mask.sum()),
    }
    overall.update({k: v for k, v in res.meta.items() if (k.startswith("objective") and k != "objective_windows")
                    or k in ("solver", "n_windows", "fallbacks", "n_filled_idle", "n_greedy_kept", "budget_exhausted",
                             "n_budget_exhausted", "surrogate_note")})
    return {"per_object": per_object, "sensors": sensors, "overall": overall}


def comparison_row(res: ScheduleResult, custody_threshold_km: float = DEFAULT_CUSTODY_THRESHOLD_KM) -> dict:
    """Compact one-line summary used by the ``compare`` mode."""
    m = custody_metrics(res, custody_threshold_km)["overall"]
    keys = ("method", "custody_pct", "custody_pct_min_object", "mean_tslo_h", "max_tslo_h", "n_observations",
            "n_objects_never_observed", "summed_trace_final_km2", "summed_trace_mean_km2", "rss_sigma_final_km",
            "mean_final_sigma_km", "total_gain", "sensor_utilisation_pct", "runtime_s")
    row = {k: m[k] for k in keys}
    for k in ("objective_milp_total", "objective_plan_total", "objective_greedy_surrogate_total", "solver", "n_windows",
              "fallbacks", "n_filled_idle", "n_greedy_kept", "budget_exhausted"):
        if k in m:
            row[k] = m[k]
    return row


# ---------------------------------------------------------------------------
# baseline policies
# ---------------------------------------------------------------------------
class RandomPolicy:
    def __init__(self, seed: int = 0):
        self.rng = np.random.default_rng(seed)

    def __call__(self, k: int, ctx: SlotContext) -> dict:
        plan = {}
        for s in range(ctx.scn.n_sensors):
            js = [j for j in range(ctx.scn.n_objects) if ctx.feasible(s, j)]
            if js:
                plan[s] = int(self.rng.choice(js))
        return plan


class RoundRobinPolicy:
    def __init__(self):
        self.pointer: dict[int, int] = {}

    def __call__(self, k: int, ctx: SlotContext) -> dict:
        plan = {}
        J = ctx.scn.n_objects
        for s in range(ctx.scn.n_sensors):
            start = self.pointer.get(s, -1) + 1
            for step in range(J):
                j = (start + step) % J
                if ctx.feasible(s, j):
                    plan[s] = j
                    self.pointer[s] = j
                    break
        return plan


def run_null(scn: Scenario) -> ScheduleResult:
    """No observations at all: the covariance-only reference every policy should beat."""
    return run_schedule(scn, lambda k, ctx: {}, method="null")


def null_custody_pct(scn: Scenario, custody_threshold_km: float = DEFAULT_CUSTODY_THRESHOLD_KM) -> float:
    """Custody % obtained with zero observations (propagated prior only)."""
    return custody_metrics(run_null(scn), custody_threshold_km)["overall"]["custody_pct"]


def run_random(scn: Scenario, seed: int = 0) -> ScheduleResult:
    return run_schedule(scn, RandomPolicy(seed), method="random", meta={"seed": seed})


def run_round_robin(scn: Scenario) -> ScheduleResult:
    return run_schedule(scn, RoundRobinPolicy(), method="round_robin")
