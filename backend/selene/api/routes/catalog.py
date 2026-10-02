"""GET /api/catalog/objects, /api/catalog/objects/{id}, /api/catalog/objects/{id}/trajectory.

Also hosts two small helpers shared by the core routes: :func:`parse_utc` (ISO UTC -> TDB
seconds past J2000) and :func:`round_floats` (recursive payload rounding).
"""
from __future__ import annotations

import math
from typing import Literal, Optional

import numpy as np
from astropy.time import Time
from fastapi import APIRouter, HTTPException, Query

from selene.api.schemas import CatalogOut, ObjectSummary, TrajectoryOut
from selene.dynamics.frames import DEMO_EPOCH_UTC
from selene.objects.catalog import get_catalog
from selene.time import J2000_JD, seconds_since_j2000_tdb

router = APIRouter(prefix="/catalog", tags=["catalog"])

DISCLAIMER = (
    "Objects labelled SIMULATED are synthetic spacecraft placed on published cislunar orbit types; "
    "their states, physical parameters and events are notional and attributed to a 'notional actor', "
    "never to a real country, operator or spacecraft. Objects of kind 'horizons' are real spacecraft "
    "ephemerides republished by JPL Horizons; no events are ever simulated for them."
)


# ---------------------------------------------------------------------------
# helpers shared by the core routes
def kernel_span_tdb_s() -> tuple[float, float]:
    """(start, end) TDB seconds of the DE440s Earth-Moon coverage (cached; 1849-12-26 .. 2150-01-22)."""
    global _KERNEL_SPAN
    if _KERNEL_SPAN is None:
        from selene.dynamics.ephemeris import get_ephemeris

        _KERNEL_SPAN = get_ephemeris().span_tdb_s()
    return _KERNEL_SPAN


_KERNEL_SPAN: Optional[tuple[float, float]] = None


def parse_utc(s: str, what: str = "epoch", *, check_span: bool = True) -> float:
    """ISO-8601 UTC (optionally with a trailing 'Z') -> TDB seconds past J2000; 400 on failure.

    With ``check_span`` (default) an epoch outside the DE440s kernel coverage is also a 400 that
    names the valid span, instead of a ``jplephem.OutOfRangeError`` 500 deep inside a route.
    """
    try:
        txt = str(s).strip()
        if txt.endswith("Z") or txt.endswith("z"):
            txt = txt[:-1]
        t = float(seconds_since_j2000_tdb(txt))
    except Exception as e:  # noqa: BLE001
        raise HTTPException(400, detail=f"could not parse {what} {s!r} as an ISO UTC time: {e}")
    if check_span:
        try:
            lo, hi = kernel_span_tdb_s()
        except FileNotFoundError:
            raise   # app-level handler -> 503 with the download hint
        if not (lo <= t <= hi):
            raise HTTPException(400, detail=f"{what} {s!r} is outside the DE440s ephemeris coverage "
                                            f"{tdb_s_to_utc_iso(lo)[0][:10]} .. {tdb_s_to_utc_iso(hi)[0][:10]}")
    return t


def tdb_s_to_utc_iso(t_s) -> list[str]:
    """Vectorised TDB seconds past J2000 -> ISO UTC strings, millisecond precision, trailing ``Z``.

    The ``Z`` is deliberate: ``new Date("2026-03-01T00:00:00.000")`` in a browser is parsed as
    *local* time, so every epoch the API echoes carries the UTC designator.
    """
    t = np.atleast_1d(np.asarray(t_s, dtype=np.float64))
    iso = Time(J2000_JD + t / 86400.0, format="jd", scale="tdb").utc.isot
    return [str(x)[:23] + "Z" for x in np.atleast_1d(iso)]


def ensure_z(iso: str) -> str:
    """Append the UTC designator to an ISO string that lacks an offset."""
    s = str(iso)
    return s if (s.endswith("Z") or s.endswith("z") or "+" in s[10:]) else s + "Z"


def err_detail(e: BaseException) -> str:
    """Error message for a 4xx ``detail``: ``str(KeyError('x'))`` is ``"'x'"`` (extra quotes), use its argument."""
    if isinstance(e, KeyError) and e.args:
        return str(e.args[0])
    return str(e)


def round_floats(x, ndigits: int = 6):
    """Recursively round floats (NaN/inf -> None) in nested lists/dicts/arrays."""
    if isinstance(x, dict):
        return {k: round_floats(v, ndigits) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [round_floats(v, ndigits) for v in x]
    if isinstance(x, np.ndarray):
        return round_floats(x.tolist(), ndigits)
    if isinstance(x, (float, np.floating)):
        xf = float(x)
        return round(xf, ndigits) if math.isfinite(xf) else None
    if isinstance(x, (np.integer,)):
        return int(x)
    return x


def _summary_or_404(obj_id: str) -> dict:
    cat = get_catalog()
    if obj_id not in cat:
        raise HTTPException(404, detail=f"unknown object {obj_id!r}; known ids: {cat.ids()}")
    return _summary(cat, obj_id)


def _summary(cat, obj_id: str) -> dict:
    o = round_floats(cat.summary(obj_id), 6)
    o["epoch_utc"] = ensure_z(o["epoch_utc"])
    return o


# ---------------------------------------------------------------------------
@router.get("/objects", response_model=CatalogOut)
def list_objects(kind: Optional[Literal["simulated", "horizons"]] = Query(None)):
    cat = get_catalog()
    objs = [_summary(cat, e.id) for e in cat.objects(kind)]
    return CatalogOut(
        epoch_utc=ensure_z(cat.epoch_utc),
        n_simulated=sum(1 for o in objs if o["kind"] == "simulated"),
        n_real=sum(1 for o in objs if o["kind"] == "horizons"),
        disclaimer=DISCLAIMER,
        objects=[ObjectSummary(**o) for o in objs],
    )


@router.get("/objects/{obj_id}", response_model=ObjectSummary)
def get_object(obj_id: str):
    return ObjectSummary(**_summary_or_404(obj_id))


@router.get("/objects/{obj_id}/trajectory", response_model=TrajectoryOut)
def object_trajectory(
    obj_id: str,
    t0: str = Query(DEMO_EPOCH_UTC, description="UTC ISO start"),
    t1: Optional[str] = Query(None, description="UTC ISO end (default t0 + 7 days)"),
    n: int = Query(200, ge=2, le=5000),
    frame: Literal["gcrf_km", "rot_nd", "moon_km"] = Query("gcrf_km"),
):
    cat = get_catalog()
    if obj_id not in cat:
        raise HTTPException(404, detail=f"unknown object {obj_id!r}; known ids: {cat.ids()}")
    t0_s = parse_utc(t0, "t0")
    t1_s = parse_utc(t1, "t1") if t1 else t0_s + 7 * 86400.0
    if t1_s <= t0_s:
        raise HTTPException(400, detail="t1 must be after t0")
    try:
        tr = cat.trajectory(obj_id, t0_s, t1_s, n, frame)
    except ValueError as e:
        raise HTTPException(400, detail=str(e))
    except RuntimeError as e:
        raise HTTPException(500, detail=f"propagation failed: {e}")
    nd = 8 if frame == "rot_nd" else 5
    units = {"gcrf_km": "Earth-centred GCRF, km and km/s",
             "moon_km": "Moon-centred, GCRF axes, km and km/s",
             "rot_nd": "Earth-Moon rotating frame, nondimensional (length = instantaneous Earth-Moon distance, time T*)"}[frame]
    e = cat.get(obj_id)
    return TrajectoryOut(
        object_id=obj_id,
        kind=e.kind,
        frame=frame,
        units=units,
        t0_utc=tdb_s_to_utc_iso(t0_s)[0],
        t1_utc=tdb_s_to_utc_iso(t1_s)[0],
        n=len(tr),
        epochs_utc=tdb_s_to_utc_iso(tr.t_s),
        tdb_s=round_floats(tr.t_s, 3),
        states=round_floats(tr.states, nd),
        meta=round_floats({k: v for k, v in tr.meta.items() if k != "sol"}, 6),
    )
