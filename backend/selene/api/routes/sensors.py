"""GET /api/sensors and GET /api/sensors/{id}/visibility.

Epochs without an ephemeris (a real Horizons object outside its cached span) are reported as ``no_ephemeris``:
``visible`` / ``reasons`` / ``magnitude`` / ``range_km`` / ``phase_deg`` are ``null`` there, ``reason_names`` is
``["no_ephemeris"]``, the ``no_ephemeris`` flag array marks them and ``fraction_visible`` is taken over the
evaluated epochs only (``n_evaluated``).  A window with no evaluable epoch at all is a 400 naming the cached span.
Nothing is fabricated: the old behaviour evaluated NaN geometry and answered 'too_faint' / 'daylight' for epochs
the catalog knows nothing about.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel

from selene.api.routes.catalog import parse_utc, tdb_s_to_utc_iso
from selene.dynamics.frames import DEMO_EPOCH_UTC
from selene.sensors.observers import DEFAULT_OBSERVERS, orbit_summary
from selene.sensors.reasons import ALL_REASONS, REASON_NAMES
from selene.sensors.sites import DEFAULT_SITES, SPEC_DISCLAIMER
from selene.sensors.visibility import visibility

router = APIRouter(prefix="/sensors", tags=["sensors"])

_SENSORS = {s.id: s for s in DEFAULT_SITES}
_SENSORS.update({o.id: o for o in DEFAULT_OBSERVERS})


class SensorListResponse(BaseModel):
    ground: list[dict]
    space: list[dict]
    note: str
    reason_bits: dict[str, int]


NO_EPHEMERIS = "no_ephemeris"


class VisibilityResponse(BaseModel):
    sensor_id: str
    object_id: str
    t0: str                     # normalised ISO UTC (same as t0_utc; kept for existing clients)
    t1: str
    t0_utc: str
    t1_utc: str
    n: int
    n_evaluated: int            # epochs with an ephemeris (fraction_visible is over these)
    n_no_ephemeris: int
    fraction_visible: float
    t_s: list[float]            # TDB seconds past J2000 (alias tdb_s)
    tdb_s: list[float]
    epochs_utc: list[str]
    visible: list[Optional[bool]]          # null = no ephemeris at that epoch
    magnitude: list[Optional[float]]
    reasons: list[Optional[int]]
    reason_names: list[list[str]]          # ["no_ephemeris"] where the object has no ephemeris
    range_km: list[Optional[float]]
    phase_deg: list[Optional[float]]
    no_ephemeris: list[bool]
    ephemeris_span_utc: Optional[list[str]] = None   # the object's served window (Horizons: cached span)
    note: Optional[str] = None


@router.get("", response_model=SensorListResponse)
def list_sensors():
    space = []
    for o in DEFAULT_OBSERVERS:
        d = o.as_dict()
        d["orbit"] = orbit_summary(o)
        space.append(d)
    return SensorListResponse(
        ground=[s.as_dict() for s in DEFAULT_SITES],
        space=space,
        note=SPEC_DISCLAIMER + " Space observers are notional platforms on CR3BP periodic orbits.",
        reason_bits={REASON_NAMES[b]: int(b) for b in ALL_REASONS},
    )


def _object_track(object_id: str, t_s: np.ndarray) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """``(positions (N,3) GCRF km, has_ephemeris (N,) bool, served window [utc0, utc1])`` of a catalog object.

    404 for an unknown object; 400 when no requested epoch lies inside the object's served window (a Horizons
    object outside its cached span, or a notional object beyond the extension limit), naming that window.
    Epochs outside the window are NaN rows flagged False in ``has_ephemeris`` (never evaluated)."""
    from selene.objects.catalog import get_catalog

    cat = get_catalog()
    if object_id not in cat:
        raise HTTPException(404, detail=f"unknown object_id {object_id!r}; known ids: {cat.ids()}")
    lo, hi = cat.truth_window_s(object_id)
    span_utc = cat.truth_window_utc(object_id)
    has = (t_s >= lo - 1e-6) & (t_s <= hi + 1e-6)
    if not has.any():
        raise HTTPException(400, detail=f"requested window lies entirely outside the ephemeris span of {object_id} "
                                        f"({span_utc[0]} .. {span_utc[1]}); no visibility can be evaluated")
    st = np.full((t_s.size, 6), np.nan)
    try:
        st[has] = np.asarray(cat.state_at(object_id, t_s[has]), dtype=float).reshape(-1, 6)
    except ValueError as e:
        raise HTTPException(400, detail=f"could not evaluate object {object_id!r}: {e}")
    has &= np.isfinite(st).all(axis=1)          # a Horizons gap inside the span is also 'no ephemeris'
    if not has.any():
        raise HTTPException(400, detail=f"no epoch of the requested window has an ephemeris for {object_id} "
                                        f"(span {span_utc[0]} .. {span_utc[1]})")
    return st[:, :3], has, span_utc


@router.get("/{sensor_id}/visibility", response_model=VisibilityResponse)
def sensor_visibility(
    sensor_id: str,
    object_id: str = Query(..., description="catalog object id"),
    t0: str = Query(DEMO_EPOCH_UTC, description="UTC ISO start"),
    t1: Optional[str] = Query(None, description="UTC ISO end (default t0 + 7 days)"),
    n: int = Query(169, ge=2, le=5000),
    radius_m: float = Query(1.0, gt=0),
    albedo: float = Query(0.2, gt=0, le=1),
):
    sensor = _SENSORS.get(sensor_id)
    if sensor is None:
        raise HTTPException(404, detail=f"unknown sensor {sensor_id!r}; known: {sorted(_SENSORS)}")
    t0_s = parse_utc(t0, "t0")
    t1_s = parse_utc(t1, "t1") if t1 else t0_s + 7 * 86400.0
    if t1_s <= t0_s:
        raise HTTPException(400, detail="t1 must be after t0")
    t_s = np.linspace(t0_s, t1_s, int(n))
    track, has, span_utc = _object_track(object_id, t_s)
    ev = visibility(sensor, track[has], t_s[has], radius_m, albedo).as_dict()
    n_eval = int(has.sum())
    # scatter the evaluated epochs back onto the requested grid; the rest is null / 'no_ephemeris'
    def _scatter(vals, fill=None):
        out = [fill] * t_s.size
        for j, k in enumerate(np.flatnonzero(has)):
            out[k] = vals[j]
        return out
    t0_iso, t1_iso = tdb_s_to_utc_iso(t0_s)[0], tdb_s_to_utc_iso(t1_s)[0]
    n_missing = int(t_s.size - n_eval)
    note = None
    if n_missing:
        note = (f"{n_missing} of {t_s.size} epochs have no ephemeris for {object_id} (served window {span_utc[0]} .. "
                f"{span_utc[1]}); they are reported as 'no_ephemeris' with null visibility and excluded from fraction_visible")
    return VisibilityResponse(sensor_id=sensor_id, object_id=object_id, t0=t0_iso, t1=t1_iso, t0_utc=t0_iso, t1_utc=t1_iso,
                              n=int(n), n_evaluated=n_eval, n_no_ephemeris=n_missing,
                              fraction_visible=float(ev["fraction_visible"]),
                              t_s=t_s.tolist(), tdb_s=t_s.tolist(), epochs_utc=tdb_s_to_utc_iso(t_s),
                              visible=_scatter(ev["visible"]), magnitude=_scatter(ev["magnitude"]),
                              reasons=_scatter(ev["reasons"]), reason_names=_scatter(ev["reason_names"], [NO_EPHEMERIS]),
                              range_km=_scatter(ev["range_km"]), phase_deg=_scatter(ev["phase_deg"]),
                              no_ephemeris=(~has).tolist(), ephemeris_span_utc=span_utc, note=note)
