"""GET /api/orbits/families and /api/orbits/records/{id} (CR3BP periodic-orbit library)."""
from __future__ import annotations

from functools import lru_cache
from typing import Optional

import numpy as np
from fastapi import APIRouter, HTTPException, Query

from selene.api.routes.catalog import parse_utc, round_floats
from selene.api.schemas import FamiliesOut, FamilyOut, MemberOut, RecordOut
from selene.constants import L_STAR, MU, T_STAR
from selene.dynamics import frames
from selene.orbits.library import OrbitRecord, get_library

router = APIRouter(prefix="/orbits", tags=["orbits"])

_NOTE = (
    "CR3BP Earth-Moon periodic orbits in the rotating frame, nondimensional (barycentric origin, "
    "Moon at (1-mu, 0, 0), length unit L*, time unit T*). Members are evenly thinned per family; "
    "root-found synodic-resonant NRHO records (tags NRHO_9:2, NRHO_4:1, NRHO_3:1) are always included. "
    "In the DE440s force model these orbits are quasi-periodic reference orbits, not exact solutions."
)


def _thin(records: list[OrbitRecord], max_members: int) -> list[OrbitRecord]:
    """Evenly spaced subset of the regular members (first and last kept) + all located NRHO records."""
    regular = [r for r in records if "_nrho_" not in r.id]
    located = [r for r in records if "_nrho_" in r.id]
    if max_members <= 0 or len(regular) <= max_members:
        keep = regular
    else:
        idx = np.unique(np.round(np.linspace(0, len(regular) - 1, max_members)).astype(int))
        keep = [regular[i] for i in idx]
    return keep + located


def _member(lib, r: OrbitRecord, n_samples: int) -> dict:
    return {
        "id": r.id,
        "ic": r.ic,
        "period": r.period_nd,
        "period_days": r.period_days,
        "jacobi": r.jacobi,
        "stability": r.stability_index,
        "closure_error": r.closure_error,
        "tags": list(r.tags),
        "params": {k: v for k, v in r.params.items() if isinstance(v, (int, float))},
        "samples_rot": round_floats(lib.sample(r, n_samples), 6),
    }


@lru_cache(maxsize=16)
def _families_payload(n_samples: int, max_members: int, family: Optional[str]) -> dict:
    lib = get_library()
    fams = []
    for key in lib.family_keys():
        if family and key != family and not key.startswith(family):
            continue
        recs = lib.members(key)
        if not recs:
            continue
        meta = lib.families.get(key, {})
        chosen = _thin(recs, max_members)
        fams.append({
            "name": key,
            "family": recs[0].family,
            "branch": recs[0].branch,
            "n_total": len(recs),
            "n_returned": len(chosen),
            "period_days_range": meta.get("period_days_range", [min(r.period_days for r in recs), max(r.period_days for r in recs)]),
            "jacobi_range": meta.get("jacobi_range", [min(r.jacobi for r in recs), max(r.jacobi for r in recs)]),
            "references": list(meta.get("references", [])),
            "members": [_member(lib, r, n_samples) for r in chosen],
        })
    return round_floats({
        "mu": MU, "L_star_km": L_STAR, "T_star_s": T_STAR, "n_samples": n_samples, "note": _NOTE, "families": fams,
    }, 9)


@router.get("/families", response_model=FamiliesOut)
def families(
    n_samples: int = Query(200, ge=2, le=2000, description="positions per member over one period"),
    max_members: int = Query(12, ge=0, le=100, description="evenly thinned members per family (0 = all)"),
    family: Optional[str] = Query(None, description="restrict to one family key, e.g. 'L2_halo_S' or prefix 'L2_halo'"),
):
    payload = _families_payload(int(n_samples), int(max_members), family)
    if not payload["families"]:
        raise HTTPException(404, detail=f"no family matches {family!r}; keys: {get_library().family_keys()}")
    return FamiliesOut(**payload)


@router.get("/records/{record_id}", response_model=RecordOut)
def record(
    record_id: str,
    n_samples: int = Query(400, ge=2, le=5000),
    gcrf_epoch: Optional[str] = Query(None, description="UTC ISO; if given, also return the orbit mapped to GCRF km "
                                                        "over one period starting at this epoch (instantaneous-frame "
                                                        "mapping, display only)"),
):
    lib = get_library()
    if record_id not in lib.by_id:
        raise HTTPException(404, detail=f"unknown orbit record {record_id!r}")
    r = lib[record_id]
    d = r.as_dict()
    d["samples_rot"] = round_floats(lib.sample(r, int(n_samples)), 8)
    if gcrf_epoch:
        t0 = parse_utc(gcrf_epoch, "gcrf_epoch")
        ts = t0 + np.linspace(0.0, r.period_nd, int(n_samples)) * T_STAR
        pos = frames.rot_pos_to_gcrf(lib.sample(r, int(n_samples)), ts)
        d["samples_gcrf_km"] = round_floats(pos, 3)
    return RecordOut(**round_floats(d, 12))
