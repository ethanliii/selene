"""Reference data from JPL's Three-Body Periodic Orbit Catalog for validating the SELENE library.

API: https://ssd-api.jpl.nasa.gov/periodic_orbits.api  (doc: https://ssd-api.jpl.nasa.gov/doc/periodic_orbits.html,
version 1.0, 2020 June).  Queried families (Earth-Moon system)::

    family=halo&libr=1&branch=N / S,  family=halo&libr=2&branch=N / S,
    family=lyapunov&libr=1 / 2,  family=dro,  family=resonant&branch=31 / 21

A compact subset (≤ 60 members per family plus a (Jacobi, period) poly-line of ≤ 400 points
along the family) is cached in ``data/orbits/jpl_reference.json`` with the query URL and
access date so the validation tests run offline.

Normalisation differences (documented, see ``jpl_mu_difference``)
------------------------------------------------------------------
* JPL mass ratio μ = 1.215058560962404e-2; SELENE μ = GM_Moon/(GM_Earth+GM_Moon) from DE440
  = 1.2150584395829193e-2 (difference 1.2e-9, i.e. 1e-7 relative).  Nondimensional CR3BP
  initial conditions, periods and Jacobi constants are therefore directly comparable to
  ~1e-7; the validation tests re-converge JPL members with *JPL's* μ for an exact comparison
  and also report the shift when SELENE's μ is used.
* JPL's length unit is 389 703.26 km and its time unit 382 981.29 s (a different convention
  for the Earth-Moon distance / mean motion); SELENE uses L* = 384 400 km and
  T* = sqrt(L*³/(GM_E+GM_M)) ≈ 375 190 s.  Dimensionless quantities agree; km and day
  conversions differ by ~1.4 % (length) and ~2.1 % (time).  In particular the "9:2 synodic
  resonant" NRHO is defined through the *physical* synodic month and SELENE's T*
  (Zimovan-Spreen et al. 2020), not through JPL's time unit.
* JPL's frame and sign conventions match SELENE's rotating frame (barycentric, x toward the
  Moon, z along the angular velocity).  JPL lists each orbit at one perpendicular xz-plane
  crossing; which of the two crossings is used varies between families, so comparisons are
  made in the crossing-independent (Jacobi, period) space, or by re-converging from JPL's IC.
"""
from __future__ import annotations

import argparse
import json
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from selene.constants import DATA_DIR, MU

__all__ = [
    "JPL_API",
    "JPL_QUERIES",
    "JPL_MU",
    "REFERENCE_PATH",
    "fetch_family",
    "compact_family",
    "build_reference",
    "load_jpl_reference",
    "distance_to_curve",
    "jpl_mu_difference",
]

JPL_API = "https://ssd-api.jpl.nasa.gov/periodic_orbits.api"
JPL_MU = 1.215058560962404e-2
REFERENCE_PATH = DATA_DIR / "orbits" / "jpl_reference.json"

# our family key -> JPL query (without the API base)
JPL_QUERIES = {
    "L1_lyapunov": "sys=earth-moon&family=lyapunov&libr=1",
    "L2_lyapunov": "sys=earth-moon&family=lyapunov&libr=2",
    "L1_halo_N": "sys=earth-moon&family=halo&libr=1&branch=N",
    "L1_halo_S": "sys=earth-moon&family=halo&libr=1&branch=S",
    "L2_halo_N": "sys=earth-moon&family=halo&libr=2&branch=N",
    "L2_halo_S": "sys=earth-moon&family=halo&libr=2&branch=S",
    "DRO": "sys=earth-moon&family=dro",
    "resonant_3:1": "sys=earth-moon&family=resonant&branch=31",
    "resonant_2:1": "sys=earth-moon&family=resonant&branch=21",
}
# scratch file names used by --raw-dir (curl downloads)
_RAW_NAMES = {
    "L1_lyapunov": "lyap_L1.json", "L2_lyapunov": "lyap_L2.json",
    "L1_halo_N": "halo_L1_N.json", "L1_halo_S": "halo_L1_S.json",
    "L2_halo_N": "halo_L2_N.json", "L2_halo_S": "halo_L2_S.json",
    "DRO": "dro.json", "resonant_3:1": "res_31.json", "resonant_2:1": "res_21.json",
}


def jpl_mu_difference() -> dict:
    return {"jpl_mu": JPL_MU, "selene_mu": MU, "abs_diff": JPL_MU - MU, "rel_diff": (JPL_MU - MU) / MU}


def fetch_family(query: str, timeout: float = 90.0) -> dict:
    """GET one family from the JPL API (network).  Returns the parsed JSON payload."""
    url = f"{JPL_API}?{query}"
    req = urllib.request.Request(url, headers={"User-Agent": "SELENE-orbit-library/0.1"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        payload = json.loads(resp.read().decode("utf-8"))
    if "data" not in payload:
        raise RuntimeError(f"JPL API returned no data for {url}: {payload}")
    payload["_url"] = url
    return payload


def _order_along_family(key: str, a: np.ndarray) -> np.ndarray:
    """Index order following the family curve.  Planar families are monotone in x0; halo
    families are chained by nearest neighbour in normalised (x, z, vy) from the near-planar end."""
    if "halo" not in key:
        return np.argsort(a[:, 0])
    P = a[:, [0, 2, 4]].copy()
    P = (P - P.mean(0)) / (P.std(0) + 1e-12)
    n = len(P)
    used = np.zeros(n, bool)
    i = int(np.argmin(np.abs(a[:, 2])))
    order = [i]
    used[i] = True
    for _ in range(n - 1):
        d = np.linalg.norm(P - P[i], axis=1)
        d[used] = np.inf
        i = int(np.argmin(d))
        order.append(i)
        used[i] = True
    return np.array(order)


def _curve_indices(C: np.ndarray, T: np.ndarray, n_curve: int) -> np.ndarray:
    """Indices (into the family-ordered arrays) of the (Jacobi, period) poly-line vertices: the
    union of ``n_curve`` vertices uniform in catalogue index (dense where the catalogue is
    dense), ``n_curve`` vertices uniform in *arc length* of the range-normalised (C, T) curve
    (resolves bends such as the period fold of the L1 halo family and the steep small-DRO end),
    and the four extremal members (min/max C and T) so the poly-line reaches the catalogue
    limits.  Index sampling alone left the L1 halo poly-line 3.5e-4 short of the catalogued
    minimum period (a spurious 2.6e-4 distance for our member at the fold); arc length alone
    starves the C ≈ 3 region of the L1 halo family, whose catalogue extends to C ≈ 0.2."""
    n = len(C)
    if n <= 2 * n_curve:
        return np.arange(n)
    idx_u = np.linspace(0, n - 1, n_curve).round().astype(int)
    c = (C - C.min()) / (np.ptp(C) + 1e-300)
    t = (T - T.min()) / (np.ptp(T) + 1e-300)
    s = np.concatenate([[0.0], np.cumsum(np.hypot(np.diff(c), np.diff(t)))])
    idx_s = np.searchsorted(s, np.linspace(0.0, s[-1], n_curve)).clip(0, n - 1)
    ext = [np.argmin(C), np.argmax(C), np.argmin(T), np.argmax(T)]
    return np.unique(np.concatenate([idx_u, idx_s, ext]))


def compact_family(key: str, payload: dict, max_members: int = 60, n_curve: int = 400) -> dict:
    fields = payload["fields"]
    a = np.array(payload["data"], dtype=np.float64)
    order = _order_along_family(key, a)
    a = a[order]
    n = len(a)
    idx = np.unique(np.linspace(0, n - 1, min(max_members, n)).round().astype(int))
    jc, pc = fields.index("jacobi"), fields.index("period")
    cidx = _curve_indices(a[:, jc], a[:, pc], n_curve)
    return {
        "query_url": payload.get("_url", f"{JPL_API}?{JPL_QUERIES[key]}"),
        "family": payload.get("family"),
        "libration_point": payload.get("libration_point"),
        "branch": payload.get("branch"),
        "count_total": int(payload.get("count", n)),
        "limits": payload.get("limits"),
        "fields": fields,
        "ordering": "x0 ascending" if "halo" not in key else "nearest-neighbour chain in (x,z,vy) from the near-planar end",
        "members": [[float(v) for v in row] for row in a[idx]],
        "curve_ct": [[float(a[i, jc]), float(a[i, pc])] for i in cidx],
    }


def build_reference(out_path: Path | str = REFERENCE_PATH, raw_dir: Path | str | None = None, max_members: int = 60,
                    keys=None, verbose: bool = True) -> dict:
    """Fetch (or read from ``raw_dir``) every family in :data:`JPL_QUERIES` and write the compact
    reference file.  Returns the written dict."""
    keys = list(keys or JPL_QUERIES.keys())
    out = {
        "source": "NASA/JPL Three-Body Periodic Orbits API v1.0 (ssd-api.jpl.nasa.gov/periodic_orbits.api)",
        "doc": "https://ssd-api.jpl.nasa.gov/doc/periodic_orbits.html",
        "access_date_utc": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
        "mu_note": jpl_mu_difference(),
        "system": None,
        "families": {},
    }
    for k in keys:
        raw = Path(raw_dir) / _RAW_NAMES[k] if raw_dir else None
        if raw is not None and raw.exists():
            payload = json.loads(raw.read_text())
            payload["_url"] = f"{JPL_API}?{JPL_QUERIES[k]}"
        else:
            payload = fetch_family(JPL_QUERIES[k])
        if out["system"] is None:
            out["system"] = payload.get("system")
        out["families"][k] = compact_family(k, payload, max_members=max_members)
        if verbose:
            print(f"  {k:14s} total={out['families'][k]['count_total']:6d} kept={len(out['families'][k]['members'])}")
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, separators=(",", ":")))
    return out


def load_jpl_reference(path: Path | str = REFERENCE_PATH) -> dict:
    return json.loads(Path(path).read_text())


def distance_to_curve(points: np.ndarray, curve: np.ndarray) -> np.ndarray:
    """Euclidean distance from each point (N,2) to the poly-line ``curve`` (M,2)."""
    P = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    C = np.asarray(curve, dtype=np.float64).reshape(-1, 2)
    A, B = C[:-1], C[1:]
    AB = B - A
    L2 = np.sum(AB**2, axis=1) + 1e-300
    out = np.empty(len(P))
    for i, p in enumerate(P):
        t = np.clip(np.sum((p - A) * AB, axis=1) / L2, 0.0, 1.0)
        proj = A + t[:, None] * AB
        out[i] = np.min(np.linalg.norm(proj - p, axis=1))
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description="Fetch JPL periodic-orbit reference data")
    ap.add_argument("--fetch", action="store_true")
    ap.add_argument("--raw-dir", default=None, help="directory of previously downloaded raw API responses")
    ap.add_argument("--out", default=str(REFERENCE_PATH))
    ap.add_argument("--max-members", type=int, default=60)
    args = ap.parse_args(argv)
    if args.fetch or args.raw_dir:
        build_reference(args.out, raw_dir=args.raw_dir, max_members=args.max_members)
        print(f"wrote {args.out}")
    else:
        ref = load_jpl_reference(args.out)
        print(f"access {ref['access_date_utc']}: " + ", ".join(f"{k}:{len(v['members'])}" for k, v in ref["families"].items()))


if __name__ == "__main__":
    main()
