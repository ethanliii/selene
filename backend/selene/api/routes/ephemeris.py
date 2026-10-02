"""GET /api/ephemeris/bodies — DE440s Earth/Moon/Sun positions and the exact rotating-frame basis.

For every epoch the response carries the instantaneous Earth-Moon rotating frame of
:mod:`selene.dynamics.frames` (columns x̂ ŷ ẑ of R in GCRF, the Earth-Moon distance d and the
barycentre offset r_bary = μ r_moon) so the frontend can switch frames exactly:

    r_rot_nd = Rᵀ (r_gcrf − r_bary) / d_km        r_gcrf = r_bary + d_km · R r_rot_nd
"""
from __future__ import annotations

from typing import Optional

import numpy as np
from fastapi import APIRouter, HTTPException, Query

from selene.api.routes.catalog import parse_utc, round_floats, tdb_s_to_utc_iso
from selene.api.schemas import BasisOut, BodiesOut, RotFrameOut
from selene.constants import L_STAR, MU, T_STAR
from selene.dynamics import frames
from selene.dynamics.cr3bp import lagrange_points
from selene.dynamics.ephemeris import get_ephemeris
from selene.dynamics.frames import DEMO_EPOCH_UTC

router = APIRouter(prefix="/ephemeris", tags=["ephemeris"])

_NOTE = (
    "Positions are Earth-centred GCRF/ICRF km from JPL DE440s (Earth is the origin). rot_frame gives the Moon "
    "and the CR3BP Lagrange points of the instantaneous Earth-Moon rotating frame in GCRF km (display aid: the "
    "libration regions are not fixed points in the ephemeris model). basis.R is row-major 3x3 with columns "
    "x_hat, y_hat, z_hat; r_rot_nd = R^T (r_gcrf - r_bary) / d_km."
)


@router.get("/bodies", response_model=BodiesOut)
def bodies(
    t0: str = Query(DEMO_EPOCH_UTC, description="UTC ISO start"),
    t1: Optional[str] = Query(None, description="UTC ISO end (default t0 + 14 days)"),
    n: int = Query(337, ge=1, le=5000, description="number of evenly spaced epochs (default hourly over 14 d)"),
):
    t0_s = parse_utc(t0, "t0")
    t1_s = parse_utc(t1, "t1") if t1 else t0_s + 14 * 86400.0
    if n > 1 and t1_s < t0_s:
        raise HTTPException(400, detail="t1 must not be before t0")
    ts = np.linspace(t0_s, t1_s, int(n)) if n > 1 else np.array([t0_s])
    eph = get_ephemeris()
    moon = eph.moon_state(ts)[:, :3]
    sun = eph.position("sun", ts, "earth")
    sun = np.atleast_2d(sun)
    f = frames.rotating_frame(ts)
    R = f.R                      # (N,3,3) columns x̂ ŷ ẑ
    d = f.d_km                   # (N,)
    rb = f.r_bary_gcrf           # (N,3)
    L = lagrange_points(MU)      # (5,3) nd
    # r_gcrf = r_bary + d * R @ r_nd  for each Lagrange point and for the Moon (1-μ,0,0)
    lag = rb[:, None, :] + d[:, None, None] * np.einsum("nij,kj->nki", R, L)   # (N,5,3)
    moon_rot = rb + d[:, None] * R[:, :, 0] * (1.0 - MU)
    return BodiesOut(
        source="de440s",
        epochs=tdb_s_to_utc_iso(ts),
        tdb_s=round_floats(ts, 3),
        earth=[[0.0, 0.0, 0.0] for _ in range(len(ts))],
        moon=round_floats(moon, 3),
        sun=round_floats(sun, 1),
        rot_frame=RotFrameOut(
            moon=round_floats(moon_rot, 3),
            l1=round_floats(lag[:, 0], 3), l2=round_floats(lag[:, 1], 3), l3=round_floats(lag[:, 2], 3),
            l4=round_floats(lag[:, 3], 3), l5=round_floats(lag[:, 4], 3),
        ),
        basis=BasisOut(
            R=round_floats(R.reshape(len(ts), 9), 12),
            d_km=round_floats(d, 6),
            r_bary_km=round_floats(rb, 6),
            omega_rad_s=round_floats(f.omega, 15),
        ),
        lagrange_rot_nd={f"l{i + 1}": round_floats(L[i], 9) for i in range(5)},
        mu=MU,
        L_star_km=L_STAR,
        T_star_s=T_STAR,
        note=_NOTE,
    )
