"""Unified object catalog: SIMULATED notional spacecraft + real JPL Horizons spacecraft.

* **Notional objects** (:mod:`selene.objects.notional`, ``kind='simulated'``): truth is the
  DE440s ephemeris model (Earth + Moon + Sun point masses + cannonball SRP with the object's
  C_R·A/m) propagated from the demo epoch.  The truth arcs are built lazily on first access and
  cached per object over ``[epoch - 1 d, epoch + 14 d]`` (dense integrator output for exact
  interpolation, plus a 10-minute grid :class:`~selene.dynamics.propagate.Trajectory`).  Times
  outside the window are served by deterministic on-demand extension of the arc.
* **Horizons objects** (:mod:`selene.objects.horizons`, ``kind='horizons'``, ``is_real=True``):
  JPL-published navigation ephemerides interpolated with a cubic Hermite spline; only objects
  whose cached span contains the demo epoch are listed.

Everything is deterministic (fixed epoch, fixed tolerances, no random numbers).  Units: km,
km/s, TDB seconds past J2000 (``t_s``); nondimensional rotating-frame output on request.

Ephemeris insertion correction for unstable libration-point orbits
--------------------------------------------------------------------
A CR3BP halo/Lyapunov state mapped into the real force model departs its orbit within one or two
revolutions (stability index 50-1000: the unstable mode e-folds in 1-3 days, and the CR3BP /
ephemeris mismatch of ~1 % L* seeds it); several of the raw mappings impact the Moon inside two
weeks.  Real libration-point spacecraft fly ephemeris-converged quasi-halos with ~m/s-per-month
station-keeping.  For objects flagged ``ephem_fit`` we therefore solve a small boundary-value
problem once: Gauss-Newton on the epoch *velocity* (3 unknowns, STM-based Jacobian, backtracking)
minimising the GCRF position residual to the CR3BP reference orbit (same phase advance) at 8
checkpoints spread over the 14-day window.  The epoch *position* is untouched, the corrections
are 5-30 m/s (reported per object as ``ephem_fit_dv_mps``), and the resulting uncontrolled arc
stays within ~10 000 km of the reference orbit instead of leaving it.  This is a modelling device
for building a believable SIMULATED catalog, not an estimate of anything real.
"""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Literal

import numpy as np

from selene.constants import DAY_S, L_STAR
from selene.dynamics import frames
from selene.dynamics.ephemeris import BodyCache, EphemParams, get_ephemeris, propagate_ephemeris
from selene.dynamics.propagate import Frame, Trajectory
from selene.objects import horizons as hz
from selene.objects.notional import (
    NOTIONAL_OBJECTS,
    SIMULATED_LABEL,
    NotionalObject,
    initial_state_gcrf,
    reference_rot_state,
)
from selene.time import tdb_jd_to_utc_iso
from selene.dynamics.ephemeris import tdb_s_to_jd

__all__ = [
    "CatalogEntry",
    "Catalog",
    "TruthArcs",
    "get_catalog",
    "fit_epoch_velocity",
    "TRUTH_WINDOW_S",
    "TRUTH_GRID_S",
]

#: cached truth window relative to the demo epoch [s] and grid step [s]
TRUTH_WINDOW_S = (-1.0 * DAY_S, 14.0 * DAY_S)
TRUTH_GRID_S = 600.0
#: on-demand extension limit (either side of the window) [s]
MAX_EXTENSION_S = 120.0 * DAY_S
_RTOL = 1e-11
_ATOL = 1e-11


# ---------------------------------------------------------------------------
# ephemeris insertion correction
# ---------------------------------------------------------------------------
def fit_epoch_velocity(
    obj: NotionalObject,
    s0_gcrf: np.ndarray,
    t0_s: float,
    params: EphemParams,
    span_s: float = 14.0 * DAY_S,
    n_checkpoints: int = 8,
    stages: tuple[float, ...] = (0.125, 0.25, 0.5, 0.75, 1.0),
    max_iter: int = 15,
    max_step_kms: float = 0.05,
) -> tuple[np.ndarray, dict]:
    """Gauss-Newton fit of the epoch velocity so the ephemeris arc tracks the CR3BP reference.

    Residual: GCRF position error at ``n_checkpoints`` epochs ``t0 + k·span/n`` between the
    ephemeris propagation of ``[r0, v]`` and the CR3BP reference state of ``obj`` (phase advanced
    by Δt / period, mapped with :func:`frames.rot_pos_to_gcrf`).  Jacobian ∂r(t_k)/∂v0 is the
    STM block Φ_rv(t_k, t0).  Because the raw state may leave the orbit (or hit the Moon) before
    the end of the window, the span is grown in ``stages`` (fractions of ``span_s``), each stage
    warm-started from the previous solution; a stage whose initial propagation fails is skipped.
    Steps are capped at ``max_step_kms`` and backtracked (halved, up to 7 times) until the RMS
    residual decreases.  Returns ``(state, info)``: Δv [m/s], iterations and the RMS / max
    residual [km] at the start of the full-span stage (if it could be evaluated) and after.
    """
    s0 = np.asarray(s0_gcrf, dtype=np.float64).copy()
    s = s0.copy()
    total_iters = 0
    rms_before = max_before = None
    for frac in stages:
        span = span_s * frac
        n_ck = max(2, int(round(n_checkpoints * frac)))
        tk = t0_s + np.linspace(span / n_ck, span, n_ck)
        ref_g = frames.rot_pos_to_gcrf(reference_rot_state(obj, tk - t0_s)[:, :3], tk)
        t_eval = np.concatenate([[t0_s], tk])

        def rj(state, stm):
            sol = propagate_ephemeris(state, t0_s, t_eval_s=t_eval, params=params, stm=stm, rtol=_RTOL, atol=_ATOL)
            if not sol.success or sol.y.shape[1] != len(t_eval):
                return None, None
            Y = sol.y[:, 1:]
            res = (Y[:3].T - ref_g).ravel()
            J = np.concatenate([Y[6:, k].reshape(6, 6)[:3, 3:6] for k in range(n_ck)], axis=0) if stm else None
            return res, J

        res, J = rj(s, True)
        if res is None:
            continue  # raw state does not survive this span yet; a shorter stage already ran or the next one will
        rms = float(np.sqrt(np.mean(res**2)))
        if frac == 1.0 and rms_before is None:
            rms_before, max_before = rms, float(np.max(np.linalg.norm(res.reshape(-1, 3), axis=1)))
        for _ in range(max_iter):
            dv, *_ = np.linalg.lstsq(J, -res, rcond=None)
            n = np.linalg.norm(dv)
            if n > max_step_kms:
                dv *= max_step_kms / n
            accepted = False
            for _ in range(8):
                trial = s.copy()
                trial[3:] += dv
                res_t, _ = rj(trial, False)
                if res_t is not None and float(np.sqrt(np.mean(res_t**2))) < rms:
                    s, accepted = trial, True
                    break
                dv *= 0.5
            total_iters += 1
            if not accepted:
                break
            res, J = rj(s, True)
            if res is None:
                break
            new_rms = float(np.sqrt(np.mean(res**2)))
            converged = rms - new_rms < 1e-3 * rms
            rms = new_rms
            if converged:
                break
    # final statistics over the full span
    n_ck = n_checkpoints
    tk = t0_s + np.linspace(span_s / n_ck, span_s, n_ck)
    ref_g = frames.rot_pos_to_gcrf(reference_rot_state(obj, tk - t0_s)[:, :3], tk)
    sol = propagate_ephemeris(s, t0_s, t_eval_s=np.concatenate([[t0_s], tk]), params=params, rtol=_RTOL, atol=_ATOL)
    if not sol.success:
        raise RuntimeError(f"fit_epoch_velocity({obj.id}): fitted state does not survive the window: {sol.message}")
    res_f = sol.y[:3, 1:].T - ref_g
    info = {
        "dv_mps": float(np.linalg.norm(s[3:] - s0[3:]) * 1e3),
        "dv_kms_vec": (s[3:] - s0[3:]).tolist(),
        "iterations": int(total_iters),
        "rms_full_span_start_km": rms_before,
        "max_full_span_start_km": max_before,
        "rms_after_km": float(np.sqrt(np.mean(res_f**2))),
        "max_after_km": float(np.max(np.linalg.norm(res_f, axis=1))),
        "n_checkpoints": int(n_checkpoints),
        "span_days": span_s / DAY_S,
        "note": "one-time insertion correction so the uncontrolled ephemeris arc shadows the CR3BP reference "
                "(*_full_span_start_km: residual when the full-span stage began, i.e. after the shorter "
                "warm-up stages; None = the state did not yet survive the full window)",
    }
    return s, info


# ---------------------------------------------------------------------------
# truth arcs
# ---------------------------------------------------------------------------
@dataclass
class TruthArcs:
    """Dense ephemeris-model truth for one notional object: a list of contiguous arcs, each a
    :class:`Trajectory` in GCRF with the integrator's dense output in ``meta['sol']``."""

    object_id: str
    epoch_s: float
    state_epoch: np.ndarray
    params: EphemParams
    arcs: list[Trajectory] = field(default_factory=list)
    build_time_s: float = 0.0
    fit_info: dict | None = None
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    @property
    def t_min(self) -> float:
        return min(float(a.t_s.min()) for a in self.arcs)

    @property
    def t_max(self) -> float:
        return max(float(a.t_s.max()) for a in self.arcs)

    def _arc_for(self, t: float) -> Trajectory | None:
        for a in self.arcs:
            if a.t_s.min() - 1e-6 <= t <= a.t_s.max() + 1e-6:
                return a
        return None

    def _extend(self, t_target: float) -> None:
        """Deterministically extend the outermost arc to cover ``t_target``."""
        with self._lock:
            if self._arc_for(t_target) is not None:
                return
            if t_target > self.t_max:
                if t_target - self.epoch_s > MAX_EXTENSION_S:
                    raise ValueError(f"{self.object_id}: t beyond the {MAX_EXTENSION_S / DAY_S:.0f}-day extension limit")
                last = max(self.arcs, key=lambda a: a.t_s.max())
                t_start = float(last.t_s.max())
                s_start = last.at(t_start)
                t_end = t_target + 1.0 * DAY_S
            else:
                if self.epoch_s - t_target > MAX_EXTENSION_S:
                    raise ValueError(f"{self.object_id}: t beyond the {MAX_EXTENSION_S / DAY_S:.0f}-day extension limit")
                last = min(self.arcs, key=lambda a: a.t_s.min())
                t_start = float(last.t_s.min())
                s_start = last.at(t_start)
                t_end = t_target - 1.0 * DAY_S
            self.arcs.append(_dense_arc(self.object_id, s_start, t_start, t_end, self.params))

    def at(self, t_s) -> np.ndarray:
        """GCRF state(s) [km, km/s]; (6,) for scalar input, (N,6) for arrays."""
        t = np.asarray(t_s, dtype=np.float64)
        scalar = t.ndim == 0
        tt = np.atleast_1d(t)
        out = np.empty((tt.size, 6))
        for t_ext in (float(tt.min()), float(tt.max())):
            if self._arc_for(t_ext) is None:
                self._extend(t_ext)
        for a in self.arcs:
            m = (tt >= a.t_s.min() - 1e-6) & (tt <= a.t_s.max() + 1e-6)
            if m.any():
                out[m] = a.at(tt[m])
        return out[0] if scalar else out

    def grid(self) -> Trajectory:
        """The cached window on a :data:`TRUTH_GRID_S` grid (GCRF)."""
        t0 = self.epoch_s + TRUTH_WINDOW_S[0]
        t1 = self.epoch_s + TRUTH_WINDOW_S[1]
        ts = np.arange(t0, t1 + 0.5 * TRUTH_GRID_S, TRUTH_GRID_S)
        meta = {"model": "ephemeris", "object_id": self.object_id, "srp": self.params.srp,
                "cr_area_mass": self.params.cr_area_mass, "grid_s": TRUTH_GRID_S}
        return Trajectory(ts, self.at(ts), "gcrf_km", meta)


def _dense_arc(object_id: str, s_start: np.ndarray, t_start: float, t_end: float, params: EphemParams) -> Trajectory:
    direction = 1.0 if t_end >= t_start else -1.0
    ts = np.arange(t_start, t_end + direction * 0.5 * TRUTH_GRID_S, direction * TRUTH_GRID_S)
    if ts[-1] * direction < t_end * direction:
        ts = np.append(ts, t_end)
    sol = propagate_ephemeris(s_start, t_start, t_eval_s=ts, params=params, dense_output=True, rtol=_RTOL, atol=_ATOL)
    if not sol.success:
        raise RuntimeError(f"truth propagation failed for {object_id}: {sol.message}")
    order = np.argsort(ts)
    tr = Trajectory(ts[order], sol.y[:6].T[order], "gcrf_km",
                    {"model": "ephemeris", "object_id": object_id, "sol": sol.sol, "sol_frame": "gcrf_km",
                     "nfev": int(sol.nfev)})
    return tr


# ---------------------------------------------------------------------------
# catalog entries
# ---------------------------------------------------------------------------
Kind = Literal["simulated", "horizons"]


@dataclass
class CatalogEntry:
    id: str
    name: str
    kind: Kind
    is_real: bool
    actor: str
    orbit_type: str
    orbit_ref: str | None
    source: str
    description: str
    role: str
    epoch_s: float
    physical: dict | None = None
    notional: NotionalObject | None = field(default=None, repr=False)
    horizons: hz.HorizonsObject | None = field(default=None, repr=False)
    notes: list[str] = field(default_factory=list)

    @property
    def label(self) -> str:
        return SIMULATED_LABEL if self.kind == "simulated" else "REAL (JPL Horizons)"


def _horizons_entry(o: hz.HorizonsObject, epoch_s: float) -> CatalogEntry:
    regime = o.regime
    orbit_type = {
        "nrho": "NRHO (Horizons)",
        "lunar_orbit": "lunar orbit (Horizons)",
        "lunar_halo": "lunar halo (Horizons)",
        "heo": "high Earth orbit (Horizons)",
        "cislunar": "cislunar (Horizons)",
    }.get(regime, f"{regime} (Horizons)")
    md = o.metadata
    notes = [
        "Real spacecraft ephemeris as published by JPL Horizons (navigation-team reconstruction/prediction); "
        "interpolated with a cubic Hermite spline between cached samples.",
        f"Cached span (UTC): {o.span[0]} .. {o.span[1]}; step {md.get('step')}; "
        f"leave-one-out interpolation error {md.get('interp_leave_one_out_max_km', {}).get('hermite', 'n/a')} km.",
        "No events, maneuvers or intent are ever simulated for real objects.",
    ]
    return CatalogEntry(
        id=f"HZ{o.id}",
        name=o.name.replace(" (spacecraft)", ""),
        kind="horizons",
        is_real=True,
        actor="real spacecraft (operator per public record)",
        orbit_type=orbit_type,
        orbit_ref=None,
        source=f"JPL Horizons id {o.id} ({md.get('trajectory_source') or 'Horizons'}), fetched {md.get('fetched_utc')}",
        description=f"{o.name}: real cislunar object, regime '{regime}'.",
        role="real spacecraft (reference / custody object)",
        epoch_s=epoch_s,
        physical=None,
        horizons=o,
        notes=notes,
    )


def _notional_entry(o: NotionalObject, epoch_s: float) -> CatalogEntry:
    src = (f"SIMULATED: CR3BP library record {o.orbit_ref} at phase {o.phase:.2f}"
           if o.is_library_orbit else "SIMULATED: Moon-centred Keplerian elements (Ely 2005 ELFO)")
    return CatalogEntry(
        id=o.id, name=o.name, kind="simulated", is_real=False, actor=o.actor, orbit_type=o.orbit_type,
        orbit_ref=o.orbit_ref, source=src, description=o.description, role=o.role, epoch_s=epoch_s,
        physical=o.physical.as_dict(), notional=o,
        notes=["SIMULATED object; all states and events are synthetic and attributed to a notional actor.",
               "Truth: DE440s Earth+Moon+Sun point masses + cannonball SRP, propagated from the demo epoch."],
    )


# ---------------------------------------------------------------------------
class Catalog:
    """Unified catalog (see module docstring)."""

    def __init__(self, epoch_s: float | None = None, include_horizons: bool = True):
        self.epoch_s = frames.DEMO_EPOCH_TDB_S if epoch_s is None else float(epoch_s)
        self.epoch_utc = tdb_jd_to_utc_iso(tdb_s_to_jd(self.epoch_s))
        self._entries: dict[str, CatalogEntry] = {}
        for o in NOTIONAL_OBJECTS:
            self._entries[o.id] = _notional_entry(o, self.epoch_s)
        if include_horizons:
            for ho in hz.load_horizons_objects(contemporaneous_only=True, at_utc=self.epoch_utc):
                e = _horizons_entry(ho, self.epoch_s)
                self._entries[e.id] = e
        self._truth: dict[str, TruthArcs] = {}
        self._lock = threading.Lock()
        self._body_cache = BodyCache(self.epoch_s + TRUTH_WINDOW_S[0] - DAY_S, self.epoch_s + TRUTH_WINDOW_S[1] + DAY_S)

    # -- listing -----------------------------------------------------------
    def objects(self, kind: Kind | None = None) -> list[CatalogEntry]:
        es = list(self._entries.values())
        return es if kind is None else [e for e in es if e.kind == kind]

    def ids(self) -> list[str]:
        return list(self._entries)

    def __contains__(self, obj_id: str) -> bool:
        return obj_id in self._entries

    def __len__(self) -> int:
        return len(self._entries)

    def get(self, obj_id: str) -> CatalogEntry:
        try:
            return self._entries[obj_id]
        except KeyError:
            raise KeyError(f"unknown object {obj_id!r}; known ids: {self.ids()}") from None

    # -- truth -------------------------------------------------------------
    def truth(self, obj_id: str) -> TruthArcs:
        """Cached ephemeris truth of a notional object (built on first access)."""
        e = self.get(obj_id)
        if e.notional is None:
            raise ValueError(f"{obj_id} is a real (Horizons) object; it has no simulated truth")
        with self._lock:
            tr = self._truth.get(obj_id)
            if tr is None:
                tr = self._build_truth(e.notional)
                self._truth[obj_id] = tr
            return tr

    def _build_truth(self, o: NotionalObject) -> TruthArcs:
        t_start = time.perf_counter()
        params = EphemParams(srp=True, cr_area_mass=o.physical.cr_area_mass, cache=self._body_cache)
        s0 = initial_state_gcrf(o, self.epoch_s)
        fit_info = None
        if o.ephem_fit and o.is_library_orbit:
            s0, fit_info = fit_epoch_velocity(o, s0, self.epoch_s, params, span_s=TRUTH_WINDOW_S[1])
        fwd = _dense_arc(o.id, s0, self.epoch_s, self.epoch_s + TRUTH_WINDOW_S[1], params)
        bwd = _dense_arc(o.id, s0, self.epoch_s, self.epoch_s + TRUTH_WINDOW_S[0], params)
        tr = TruthArcs(o.id, self.epoch_s, s0, params, [bwd, fwd], fit_info=fit_info)
        tr.build_time_s = time.perf_counter() - t_start
        return tr

    def truth_window_s(self, obj_id: str) -> tuple[float, float]:
        """``(t_min, t_max)`` TDB seconds for which a state of ``obj_id`` can be served.

        Notional objects: the catalog epoch ± :data:`MAX_EXTENSION_S` (the cached window plus deterministic
        on-demand extension); Horizons objects: the cached ephemeris span.  Routes check requested windows
        against this and answer 400 instead of letting the extension's ``ValueError`` surface as a 500."""
        e = self.get(obj_id)
        if e.notional is not None:
            return self.epoch_s - MAX_EXTENSION_S, self.epoch_s + MAX_EXTENSION_S
        assert e.horizons is not None
        return float(e.horizons.t0), float(e.horizons.t1)

    def truth_window_utc(self, obj_id: str) -> list[str]:
        """:meth:`truth_window_s` as two ISO UTC strings (trailing Z)."""
        return [tdb_jd_to_utc_iso(tdb_s_to_jd(t))[:19] + "Z" for t in self.truth_window_s(obj_id)]

    def truth_grid(self, obj_id: str) -> Trajectory:
        """10-minute-grid GCRF :class:`Trajectory` of a notional object over the cached window."""
        return self.truth(obj_id).grid()

    def epoch_state(self, obj_id: str) -> np.ndarray:
        """GCRF state at the catalog epoch (notional: the (possibly fitted) insertion state)."""
        return self.state_at(obj_id, self.epoch_s)

    # -- states ------------------------------------------------------------
    def state_at(self, obj_id: str, t_s) -> np.ndarray:
        """Earth-centred GCRF state [km, km/s] at TDB ``t_s``; (6,) for scalar, (N,6) for arrays.

        Horizons objects raise ``ValueError`` for a scalar time outside the cached span and
        return NaN rows for array times outside it."""
        e = self.get(obj_id)
        if e.notional is not None:
            return self.truth(obj_id).at(t_s)
        assert e.horizons is not None
        t = np.asarray(t_s, dtype=np.float64)
        if t.ndim == 0:
            s = e.horizons.state_at(float(t))
            if s is None:
                raise ValueError(f"{obj_id}: t={float(t):.1f} s outside the cached Horizons span "
                                 f"[{e.horizons.t0:.1f}, {e.horizons.t1:.1f}]")
            return s
        return e.horizons.states_at(t)

    def trajectory(self, obj_id: str, t0_s: float, t1_s: float, n: int = 200, frame: Frame = "gcrf_km") -> Trajectory:
        """Sampled trajectory on ``n`` evenly spaced epochs in ``frame`` ∈ {'gcrf_km','rot_nd','moon_km'}.
        For Horizons objects epochs outside the cached span are dropped (ValueError if none remain)."""
        n = int(n)
        if n < 2:
            raise ValueError("n must be >= 2")
        if frame not in ("gcrf_km", "rot_nd", "moon_km"):
            raise ValueError(f"unknown frame {frame!r}")
        ts = np.linspace(float(t0_s), float(t1_s), n)
        e = self.get(obj_id)
        states = self.state_at(obj_id, ts)
        if e.horizons is not None:
            ok = np.isfinite(states).all(axis=1)
            if not ok.any():
                raise ValueError(f"{obj_id}: requested span lies entirely outside the cached Horizons span")
            ts, states = ts[ok], states[ok]
        meta = {"object_id": obj_id, "kind": e.kind, "n_requested": n}
        tr = Trajectory(ts, states, "gcrf_km", meta)
        return tr.to_frame(frame)

    # -- summaries ---------------------------------------------------------
    def summary(self, obj_id: str, with_epoch_state: bool = True) -> dict:
        """JSON-friendly description of one object (what the API serves)."""
        e = self.get(obj_id)
        d = {
            "id": e.id, "name": e.name, "kind": e.kind, "is_real": e.is_real, "label": e.label,
            "actor": e.actor, "orbit_type": e.orbit_type, "orbit_ref": e.orbit_ref, "role": e.role,
            "description": e.description, "source": e.source, "notes": list(e.notes),
            "epoch_utc": self.epoch_utc, "epoch_tdb_s": self.epoch_s, "physical": e.physical,
        }
        if e.notional is not None:
            o = e.notional
            d.update(phase=o.phase, period_s=o.period_s(), tags=list(o.tags), ephem_fit=o.ephem_fit,
                     radius_m=o.physical.radius_m, albedo=o.physical.albedo)
            if o.is_library_orbit:
                rec = o.record()
                d["orbit_record"] = {"id": rec.id, "family": rec.family, "branch": rec.branch,
                                     "period_days": rec.period_days, "jacobi": rec.jacobi,
                                     "stability_index": rec.stability_index, "tags": list(rec.tags),
                                     "params": {k: v for k, v in rec.params.items() if k.endswith("_km")}}
            else:
                d["kepler_moon"] = o.kepler.as_dict() if o.kepler else None
        else:
            ho = e.horizons
            d.update(horizons_id=ho.id, span_utc=list(ho.span), supported_window_utc=self.truth_window_utc(obj_id),
                     regime=ho.regime, n_samples=len(ho),
                     interp_leave_one_out_km=ho.metadata.get("interp_leave_one_out_max_km"))
        if with_epoch_state:
            s = self.epoch_state(obj_id)
            d["state_gcrf_km"] = [float(v) for v in s]
            d["state_rot_nd"] = [float(v) for v in frames.gcrf_to_rot(s, self.epoch_s)]
            d["geocentric_range_km"] = float(np.linalg.norm(s[:3]))
            d["selenocentric_range_km"] = float(np.linalg.norm(frames.gcrf_to_moon_centered(s, self.epoch_s)[:3]))
            if e.notional is not None:
                tr = self.truth(obj_id)
                d["truth"] = {
                    "model": "DE440s Earth+Moon+Sun point masses + cannonball SRP",
                    "window_tdb_s": [tr.t_min, tr.t_max],
                    "supported_window_tdb_s": list(self.truth_window_s(obj_id)),
                    "supported_window_utc": self.truth_window_utc(obj_id),
                    "supported_window_note": "states (and synthetic observations / OD runs) are served inside the catalog "
                                             f"epoch +/- {MAX_EXTENSION_S / DAY_S:.0f} days by deterministic extension of the cached arc",
                    "grid_s": TRUTH_GRID_S,
                    "build_time_s": round(tr.build_time_s, 3),
                    "ephem_fit": tr.fit_info,
                }
        return d

    def summaries(self, kind: Kind | None = None) -> list[dict]:
        return [self.summary(e.id) for e in self.objects(kind)]


# ---------------------------------------------------------------------------
_CATALOG: Catalog | None = None
_CAT_LOCK = threading.Lock()


def get_catalog(reset: bool = False) -> Catalog:
    """Process-wide cached :class:`Catalog` at the demo epoch."""
    global _CATALOG
    with _CAT_LOCK:
        if _CATALOG is None or reset:
            _CATALOG = Catalog()
        return _CATALOG


def nd_to_km(x: float) -> float:  # convenience for callers that only import this module
    return x * L_STAR
