"""Rolling-horizon assignment optimisation with ``scipy.optimize.milp`` (HiGHS), local-search fallback.

At node ``k`` the optimiser plans the next ``H`` slots (default 4 = 80 min at 20-min slots), commits
only slot ``k`` (receding horizon), lets the engine apply the real sequential covariance updates,
and re-plans at ``k+1``.

Surrogate objective (stated honestly)
-------------------------------------
Within a window the covariances are **frozen at the window start**: object ``j``'s covariance is
propagated (no updates) to each slot κ and the myopic gain ``g[s,κ,j]`` of every feasible
(sensor, slot, object) triple is evaluated there.  Summing myopic gains over-counts repeated looks
at the same object, so diminishing returns are modelled with **rank weights**: if object ``j``
receives ``n`` observations in the window with (weighted) stand-alone gains sorted
``g_(1) ≥ g_(2) ≥ … ≥ g_(n)``, its value is::

    V_j = Σ_{m=1..n} f_j[m] · g_(m) + b_j · [n ≥ 1],        0 ≤ f_j[m] ≤ 1, f_j[1] = 1

where ``f_j[m] = G_j^{(m)} / g_j^{(m)}`` is the ratio of the m-th *sequential marginal* gain to the
m-th best stand-alone gain when the m best candidates are applied one after another to the frozen
covariance (exact when the chosen set is precisely those candidates, an approximation otherwise),
and ``b_j = revisit_weight · ḡ · TSLO_j/24 h`` is a revisit bonus for stale objects (ḡ = mean
positive gain in the window).  Every candidate therefore has a **non-negative** marginal value --
the previous formulation charged the m-th observation an *absolute* penalty taken from the m-th
best candidate regardless of which candidate was chosen, which made low-gain looks net-negative
and left ~30 % of sensor-slots idle (found in review; see ``n_filled_idle``, now ≈ 0).

MILP (``solve_window_milp``)::

    variables  x[i] ∈ {0,1}        triple i = (s, κ, j) is observed
               z[i,m] ∈ {0,1}      triple i occupies rank m of object j
    maximise   Σ_i Σ_m f_j[m]·g_i·z[i,m]  +  Σ_j b_j·Σ_{i∈j} z[i,0]
    s.t.       Σ_{j} x[s,κ,j] ≤ 1                        (one object per sensor per slot)
               Σ_m z[i,m] = x[i]                         (an observed triple has exactly one rank)
               Σ_{i∈j} z[i,m] ≤ 1                        (one triple per rank)
               x[s,κ,j] + x[s,κ+1,j'] ≤ 1  if the slew j→j' exceeds slew_rate·slot

Because ``f_j`` is non-increasing in ``m``, an optimal solution assigns the chosen triples to
ranks in decreasing order of gain (rearrangement inequality), so the MILP value equals ``Σ_j V_j``.
The reported ``objective_milp`` is the **raw solver value** of this surrogate on each window (not
clamped), next to ``objective_greedy_surrogate`` for a greedy plan built on the same window; the
committed plan is the better of the two (``n_greedy_kept`` counts how often the greedy plan had to
be kept, which only happens on a solver time-out).  None of these numbers is the realised
information gain; the end-to-end custody metrics of the committed schedule are reported by
:mod:`selene.tasking.metrics` exactly as for every other policy.

Runtime guard: ``time_budget_s`` is a wall-clock budget for the whole run.  Each window's solver
time limit is the smaller of ``time_limit_s`` and the remaining budget per remaining window; once
the budget is exhausted the policy stops building windows and falls back to the myopic greedy
rule for the remaining slots (counted in ``n_budget_exhausted``, flagged ``budget_exhausted``).

If ``scipy.optimize.milp`` is unavailable (SciPy < 1.9) or the solver fails on a window, a
first-improvement local search (single-slot re-assignment moves) over the same surrogate is used
and counted in ``fallbacks``.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from selene.tasking.greedy import GreedyPolicy, Scenario, ScheduleResult, SlotContext, run_schedule
from selene.tasking.information import gain as _gain_fn
from selene.tasking.information import angles_jacobian, acquisition_probability, predict_covariance, update_covariance

try:  # SciPy >= 1.9
    from scipy.optimize import Bounds, LinearConstraint, milp
    from scipy.sparse import coo_matrix

    HAVE_MILP = True
except Exception:  # pragma: no cover - old SciPy
    HAVE_MILP = False

__all__ = ["HAVE_MILP", "WindowProblem", "build_window", "solve_window_milp", "solve_window_local_search",
           "surrogate_objective", "greedy_surrogate_plan", "HorizonPolicy", "run_milp"]


# ---------------------------------------------------------------------------
# window problem
# ---------------------------------------------------------------------------
@dataclass
class WindowProblem:
    k0: int
    slots: list[int]                        # absolute node indices κ in the window
    triples: list[tuple[int, int, int]]     # (s, κ, j) feasible
    gains: np.ndarray                       # priority-weighted gain per triple
    raw_gains: np.ndarray                   # unweighted gain per triple
    F: dict[int, np.ndarray]                # j -> rank weights f_j[m] ∈ [0, 1], m = 0..M_j-1 (f_j[0] = 1)
    bonus: np.ndarray                       # (J,) revisit bonus for the first observation
    slew_conflicts: list[tuple[int, int]]   # pairs of triple indices that cannot both be chosen
    meta: dict = field(default_factory=dict)

    @property
    def n_x(self) -> int:
        return len(self.triples)

    def rank_weight(self, j: int, m: int) -> float:
        Fj = self.F.get(j)
        if Fj is None or Fj.size == 0:
            return 1.0 if m == 0 else 0.0
        return float(Fj[min(m, Fj.size - 1)])


def _rank_weights(scn: Scenario, ctx: SlotContext, j: int, cands: list[tuple[int, int, float, np.ndarray]]) -> np.ndarray:
    """f_j[m] = (marginal gain of the m-th best candidate after the m−1 better ones have been
    applied) / (its stand-alone gain) on the frozen covariance; in [0, 1] by submodularity (clamped)."""
    order = sorted(range(len(cands)), key=lambda i: -cands[i][2])
    tab = scn.table
    # start from the predicted covariance at the earliest candidate slot; apply all updates there
    # (ignoring intra-window propagation: the surrogate freezes the covariance)
    k_ref = min(c[1] for c in cands)
    P = ctx.predicted_P(j, k_ref)
    f = np.ones(len(cands))
    for m, i in enumerate(order):
        s, kappa, g_alone, _P_pred = cands[i]
        H = angles_jacobian(tab.obs_pos[s, kappa], scn.tracks[j].x_nodes[kappa][:3])
        p = 1.0
        if scn.acquisition != "none":
            p = acquisition_probability(P, tab.los[s, j, kappa], float(tab.range_km[s, j, kappa]),
                                        float(tab.fov_half_rad[s]), scn.search_tiles)
        P_new = update_covariance(P, H, float(tab.sigma_rad[s]), p, scn.n_obs_per_slot, scn.acquisition)
        marginal = _gain_fn(P, P_new, scn.gain_kind)
        f[m] = 1.0 if m == 0 else (min(1.0, max(0.0, marginal / g_alone)) if g_alone > 0 else 0.0)
        P = P_new
    return np.minimum.accumulate(f)   # enforce non-increasing rank weights


def build_window(ctx: SlotContext, horizon: int, revisit_weight: float = 0.05, max_obs_per_object: Optional[int] = None) -> WindowProblem:
    """Enumerate feasible (sensor, slot, object) triples and their frozen-covariance gains.

    ``max_obs_per_object`` caps the number of ranks per object (default ``n_sensors · n_slots``,
    the most observations an object can receive in the window); the weight of the last enumerated
    rank is reused beyond it.  Slew conflicts are only generated between consecutive slots within
    the window (an idle slot in between is *not* credited with extra slew budget here, unlike the
    engine's own feasibility check, so the window plan is conservative for slow sensors).
    """
    scn = ctx.scn
    tab = scn.table
    k0 = ctx.k
    K = scn.n_slots
    slots = list(range(k0, min(K, k0 + horizon)))
    S, J = scn.n_sensors, scn.n_objects
    triples: list[tuple[int, int, int]] = []
    raw: list[float] = []
    per_obj: dict[int, list[tuple[int, int, float, np.ndarray]]] = {j: [] for j in range(J)}
    P_pred: dict[tuple[int, int], np.ndarray] = {}
    for j in range(J):
        P = ctx.P[j].copy()
        tr = scn.tracks[j]
        for kappa in slots:
            if kappa > k0:
                P = predict_covariance(P, tr.phi[kappa - 1], tr.Q[kappa - 1])
            P_pred[(j, kappa)] = P
    for s in range(S):
        for kappa in slots:
            for j in range(J):
                if not tab.visible[s, j, kappa]:
                    continue
                if kappa == k0 and not ctx.feasible(s, j):
                    continue
                g, _p, _Pp = ctx.evaluate(s, j, P_pred[(j, kappa)], kappa)
                if g <= 0.0:
                    continue
                triples.append((s, kappa, j))
                raw.append(g)
                per_obj[j].append((s, kappa, g, P_pred[(j, kappa)]))
    raw_g = np.asarray(raw, dtype=np.float64)
    w = np.array([scn.tracks[j].priority for (_s, _k, j) in triples]) if triples else np.zeros(0)
    gains = raw_g * w
    cap = int(max_obs_per_object) if max_obs_per_object else S * len(slots)
    F = {}
    for j, c in per_obj.items():
        if not c:
            continue
        c_sorted = sorted(c, key=lambda t: -t[2])[:cap]
        F[j] = _rank_weights(scn, ctx, j, c_sorted)
    gbar = float(raw_g.mean()) if raw_g.size else 0.0
    bonus = np.array([revisit_weight * gbar * ctx.tslo_h(j) / 24.0 for j in range(J)])
    # slew conflicts between consecutive slots of the same sensor (only where the limit binds)
    conflicts: list[tuple[int, int]] = []
    idx = {t: i for i, t in enumerate(triples)}
    for s in range(S):
        limit = float(tab.slew_rate_rad_s[s]) * scn.slot_s * scn.slew_fraction
        if limit >= np.pi:
            continue
        for kappa in slots[:-1]:
            A = tab.los[s, :, kappa]          # (J,3)
            B = tab.los[s, :, kappa + 1]
            cosang = np.clip(A @ B.T, -1.0, 1.0)
            sep = np.arccos(cosang)           # (J,J)
            bad = np.argwhere(sep > limit)
            for j1, j2 in bad:
                a, b = idx.get((s, kappa, int(j1))), idx.get((s, kappa + 1, int(j2)))
                if a is not None and b is not None:
                    conflicts.append((a, b))
    n_z = int(sum(len(c) ** 2 for c in per_obj.values()))      # rank variables the MILP will use
    return WindowProblem(k0, slots, triples, gains, raw_g, F, bonus, conflicts,
                         meta={"n_x": len(triples), "n_z": n_z, "n_slots": len(slots), "n_conflicts": len(conflicts)})


# ---------------------------------------------------------------------------
# objective evaluation (shared by MILP, greedy-surrogate and local search)
# ---------------------------------------------------------------------------
def surrogate_objective(wp: WindowProblem, chosen: set[int]) -> float:
    """Value of a set of chosen triple indices under the window surrogate (rank-weighted sum)."""
    by_obj: dict[int, list[float]] = {}
    for i in chosen:
        by_obj.setdefault(wp.triples[i][2], []).append(float(wp.gains[i]))
    val = 0.0
    for j, gs in by_obj.items():
        gs.sort(reverse=True)
        val += sum(wp.rank_weight(j, m) * g for m, g in enumerate(gs)) + float(wp.bonus[j])
    return val


def _valid(wp: WindowProblem, chosen: set[int], i: int) -> bool:
    s, kappa, _j = wp.triples[i]
    for c in chosen:
        if c == i:
            continue
        s2, k2, _ = wp.triples[c]
        if s2 == s and k2 == kappa:
            return False
    for a, b in wp.slew_conflicts:
        if (a == i and b in chosen) or (b == i and a in chosen):
            return False
    return True


def greedy_surrogate_plan(wp: WindowProblem) -> set[int]:
    """Greedy on the window surrogate: slot by slot, sensor by sensor, best marginal value."""
    chosen: set[int] = set()
    by_slot_sensor: dict[tuple[int, int], list[int]] = {}
    for i, (s, kappa, _j) in enumerate(wp.triples):
        by_slot_sensor.setdefault((kappa, s), []).append(i)
    base = surrogate_objective(wp, chosen)
    for kappa in wp.slots:
        for s in range(max((t[0] for t in wp.triples), default=-1) + 1):
            best_i, best_val = None, base
            for i in by_slot_sensor.get((kappa, s), []):
                if not _valid(wp, chosen, i):
                    continue
                v = surrogate_objective(wp, chosen | {i})
                if v > best_val + 1e-12:
                    best_i, best_val = i, v
            if best_i is not None:
                chosen.add(best_i)
                base = best_val
    return chosen


def solve_window_local_search(wp: WindowProblem, start: Optional[set[int]] = None, max_passes: int = 20) -> tuple[set[int], float]:
    """First-improvement local search over single (sensor, slot) re-assignments."""
    chosen = set(start) if start is not None else greedy_surrogate_plan(wp)
    best = surrogate_objective(wp, chosen)
    slots_sensors = sorted({(kappa, s) for (s, kappa, _j) in wp.triples})
    by_ss: dict[tuple[int, int], list[int]] = {}
    for i, (s, kappa, _j) in enumerate(wp.triples):
        by_ss.setdefault((kappa, s), []).append(i)
    for _ in range(max_passes):
        improved = False
        for key in slots_sensors:
            current = [i for i in chosen if (wp.triples[i][1], wp.triples[i][0]) == key]
            cur = current[0] if current else None
            options = [None] + [i for i in by_ss[key] if i != cur]
            for opt in options:
                trial = set(chosen)
                if cur is not None:
                    trial.discard(cur)
                if opt is not None:
                    if not _valid(wp, trial, opt):
                        continue
                    trial.add(opt)
                v = surrogate_objective(wp, trial)
                if v > best + 1e-12:
                    chosen, best, improved = trial, v, True
                    break
        if not improved:
            break
    return chosen, best


def solve_window_milp(wp: WindowProblem, time_limit_s: float = 5.0, mip_rel_gap: float = 1e-4) -> tuple[Optional[set[int]], float, str]:
    """Solve the window MILP; returns (chosen triple indices, surrogate objective, status)."""
    if not HAVE_MILP:
        return None, float("nan"), "milp_unavailable"
    n_x = wp.n_x
    if n_x == 0:
        return set(), 0.0, "empty"
    # ranks per object: at most the number of its candidate triples
    members: dict[int, list[int]] = {}
    for i, (_s, _k, j) in enumerate(wp.triples):
        members.setdefault(j, []).append(i)
    z_index: dict[tuple[int, int], int] = {}
    col = n_x
    for j, mem in members.items():
        for i in mem:
            for m in range(len(mem)):
                z_index[(i, m)] = col
                col += 1
    n_var = col
    c = np.zeros(n_var)                                    # maximise -> minimise negative
    for (i, m), ci in z_index.items():
        j = wp.triples[i][2]
        c[ci] = -(wp.rank_weight(j, m) * float(wp.gains[i]) + (float(wp.bonus[j]) if m == 0 else 0.0))
    rows, cols, vals, lo, hi = [], [], [], [], []
    r = 0
    # one object per sensor-slot
    groups: dict[tuple[int, int], list[int]] = {}
    for i, (s, kappa, _j) in enumerate(wp.triples):
        groups.setdefault((s, kappa), []).append(i)
    for grp in groups.values():
        for i in grp:
            rows.append(r); cols.append(i); vals.append(1.0)
        lo.append(-np.inf); hi.append(1.0); r += 1
    for j, mem in members.items():
        M = len(mem)
        # Σ_m z[i,m] − x[i] = 0
        for i in mem:
            for m in range(M):
                rows.append(r); cols.append(z_index[(i, m)]); vals.append(1.0)
            rows.append(r); cols.append(i); vals.append(-1.0)
            lo.append(0.0); hi.append(0.0); r += 1
        # one triple per rank
        for m in range(M):
            for i in mem:
                rows.append(r); cols.append(z_index[(i, m)]); vals.append(1.0)
            lo.append(-np.inf); hi.append(1.0); r += 1
    # slew conflicts
    for a, b in wp.slew_conflicts:
        rows.append(r); cols.append(a); vals.append(1.0)
        rows.append(r); cols.append(b); vals.append(1.0)
        lo.append(-np.inf); hi.append(1.0); r += 1
    A = coo_matrix((vals, (rows, cols)), shape=(r, n_var)).tocsr()
    res = milp(c, constraints=LinearConstraint(A, np.array(lo), np.array(hi)), integrality=np.ones(n_var),
               bounds=Bounds(0.0, 1.0), options={"time_limit": float(time_limit_s), "disp": False, "mip_rel_gap": float(mip_rel_gap)})
    if res.x is None or res.status not in (0, 1):   # 0 optimal, 1 time limit with feasible solution
        return None, float("nan"), f"milp_failed:{res.message}"
    x = np.asarray(res.x[:n_x])
    chosen = {int(i) for i in np.flatnonzero(x > 0.5)}
    status = "optimal" if res.status == 0 else "time_limit"
    return chosen, surrogate_objective(wp, chosen), status


# ---------------------------------------------------------------------------
# policy
# ---------------------------------------------------------------------------
class HorizonPolicy:
    """Receding-horizon policy: plan ``horizon`` slots, commit the first.

    ``fill_idle``: a sensor the window plan leaves idle in the committed slot although it has a
    feasible positive-gain object is given its best one by the myopic greedy rule and counted in
    ``n_filled`` (with the rank-weighted surrogate every candidate has non-negative value, so this
    is expected to be ≈ 0 and is reported so the reader can check).

    ``time_budget_s``: wall-clock budget for the whole run (see module docstring).
    """

    def __init__(self, horizon: int = 4, revisit_weight: float = 0.05, time_limit_s: float = 5.0, use_milp: bool = True,
                 fill_idle: bool = True, mip_rel_gap: float = 1e-4, time_budget_s: Optional[float] = None,
                 n_windows_expected: Optional[int] = None):
        self.horizon = int(max(1, horizon))
        self.revisit_weight = float(revisit_weight)
        self.time_limit_s = float(time_limit_s)
        self.use_milp = bool(use_milp and HAVE_MILP)
        self.fill_idle = bool(fill_idle)
        self.mip_rel_gap = float(mip_rel_gap)
        self.time_budget_s = None if time_budget_s is None else float(time_budget_s)
        self.n_windows_expected = n_windows_expected
        self.log: list[dict] = []
        self.n_filled = 0
        self.n_greedy_kept = 0
        self.n_budget_exhausted = 0
        self._t_first: Optional[float] = None
        self._greedy = GreedyPolicy()

    def _remaining_budget(self, now: float) -> Optional[float]:
        if self.time_budget_s is None:
            return None
        if self._t_first is None:
            self._t_first = now
        return self.time_budget_s - (now - self._t_first)

    def __call__(self, k: int, ctx: SlotContext) -> dict:
        t0 = time.perf_counter()
        remaining = self._remaining_budget(t0)
        if remaining is not None and remaining <= 0.0:
            # budget exhausted: myopic greedy for the rest of the run, no window built
            self.n_budget_exhausted += 1
            plan = self._greedy(k, ctx)
            self.log.append({"slot": k, "n_x": 0, "n_z": 0, "n_conflicts": 0, "objective_milp": float("nan"),
                             "objective_plan": float("nan"), "objective_greedy_surrogate": float("nan"),
                             "status": "budget_exhausted:greedy", "n_filled": 0, "build_s": 0.0,
                             "total_s": time.perf_counter() - t0})
            return plan
        wp = build_window(ctx, self.horizon, self.revisit_weight)
        t_build = time.perf_counter() - t0
        greedy_set = greedy_surrogate_plan(wp)
        obj_greedy = surrogate_objective(wp, greedy_set)
        chosen, obj_raw, status = (None, float("nan"), "local_search")
        if self.use_milp:
            limit = self.time_limit_s
            if remaining is not None:
                n_left = max(1, (self.n_windows_expected or ctx.scn.n_slots) - k)
                limit = max(0.05, min(self.time_limit_s, (remaining - t_build) / n_left))
            chosen, obj_raw, status = solve_window_milp(wp, limit, self.mip_rel_gap)
        if chosen is None:
            chosen, obj_raw = solve_window_local_search(wp, start=greedy_set)
            status = "local_search" if status == "local_search" else status + "->local_search"
        obj_plan = obj_raw
        if obj_raw < obj_greedy - 1e-9:   # never commit something worse than the greedy surrogate plan
            chosen, obj_plan, status = greedy_set, obj_greedy, status + "(greedy_kept)"
            self.n_greedy_kept += 1
        plan = {}
        for i in chosen:
            s, kappa, j = wp.triples[i]
            if kappa == k:
                plan[s] = j
        n_filled = 0
        if self.fill_idle:
            P_local = [P.copy() for P in ctx.P]
            for s in range(ctx.scn.n_sensors):
                if s in plan:
                    g, p, Pp = ctx.evaluate(s, plan[s], P_local[plan[s]])
                    if Pp is not None:
                        P_local[plan[s]] = Pp
            for s in range(ctx.scn.n_sensors):
                if s in plan:
                    continue
                best = None
                for c in ctx.candidates(s, P_local):
                    if c.gain > 0 and (best is None or c.score > best.score):
                        best = c
                if best is not None:
                    plan[s] = best.j
                    P_local[best.j] = best.P_plus
                    n_filled += 1
        self.n_filled += n_filled
        self.log.append({"slot": k, "n_x": wp.n_x, "n_z": wp.meta["n_z"],
                         "n_conflicts": len(wp.slew_conflicts), "objective_milp": float(obj_raw),
                         "objective_plan": float(obj_plan), "objective_greedy_surrogate": float(obj_greedy),
                         "status": status, "n_filled": n_filled, "build_s": t_build, "total_s": time.perf_counter() - t0})
        return plan


def run_milp(scn: Scenario, horizon: int = 4, revisit_weight: float = 0.05, time_limit_s: float = 5.0,
             use_milp: bool = True, fill_idle: bool = True, time_budget_s: Optional[float] = None) -> ScheduleResult:
    pol = HorizonPolicy(horizon, revisit_weight, time_limit_s, use_milp, fill_idle, time_budget_s=time_budget_s,
                        n_windows_expected=scn.n_slots)
    res = run_schedule(scn, pol, method="milp" if pol.use_milp else "local_search")
    log = pol.log
    solved = [e for e in log if not e["status"].startswith("budget_exhausted")]
    n_fallback = sum(1 for e in solved if "local_search" in e["status"])
    res.meta.update({
        "solver": "scipy.optimize.milp (HiGHS)" if pol.use_milp else "local_search",
        "horizon_slots": horizon,
        "revisit_weight": revisit_weight,
        "fill_idle": fill_idle,
        "n_windows": len(log),
        "n_windows_solved": len(solved),
        "fallbacks": n_fallback,
        "n_filled_idle": pol.n_filled,
        "n_greedy_kept": pol.n_greedy_kept,
        "n_budget_exhausted": pol.n_budget_exhausted,
        "budget_exhausted": pol.n_budget_exhausted > 0,
        "time_budget_s": time_budget_s,
        "objective_milp_total": float(sum(e["objective_milp"] for e in solved)),
        "objective_plan_total": float(sum(e["objective_plan"] for e in solved)),
        "objective_greedy_surrogate_total": float(sum(e["objective_greedy_surrogate"] for e in solved)),
        "objective_windows": [{k: v for k, v in e.items() if k in ("slot", "objective_milp", "objective_plan",
                                                                     "objective_greedy_surrogate", "status", "n_filled")}
                              for e in log],
        "mean_window_solve_s": float(np.mean([e["total_s"] for e in solved])) if solved else 0.0,
        "surrogate_note": "objective_milp is the raw solver value of the window surrogate (rank-weighted gains with "
                          "covariances frozen at each window start, see tasking/optimize.py), not a realised quantity; "
                          "compare against objective_greedy_surrogate on the same windows and judge the policy by the "
                          "realised custody metrics reported alongside",
    })
    return res
