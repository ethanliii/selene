"""Cached catalogue of CR3BP periodic orbits (the SELENE orbit library).

Build once (``python -m selene.orbits.library --rebuild``), persisted under ``data/orbits``:

* ``orbits.parquet``  -- one row per :class:`OrbitRecord` (initial conditions, period, Jacobi
  constant, stability, closure error, geometry parameters, tags; list fields JSON-encoded)
* ``families.json``   -- family metadata: generation parameters, counts, literature references

Units: initial conditions are nondimensional Earth-Moon rotating-frame states
(L* = 384 400 km, T* ≈ 3.7519e5 s, μ from DE440 GMs); ``period_days`` uses T*; geometry
parameters are in km (× L*).  Everything here is CR3BP: when flown in the DE440s ephemeris
model these orbits are *quasi*-periodic and need station-keeping; the library is used for
seeding, visualisation, candidate sensor orbits and notional object placement.
"""
from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from selene.constants import DATA_DIR, L_STAR, MU, T_STAR
from selene.dynamics.cr3bp import jacobi, propagate_cr3bp
from selene.orbits.families import FAMILY_SPECS, FamilyMember, FamilyResult, generate_family, locate_synodic_nrhos
from selene.orbits.stability import stability_index

__all__ = ["OrbitRecord", "OrbitLibrary", "get_library", "ORBITS_DIR", "PARQUET_PATH", "FAMILIES_PATH"]

ORBITS_DIR = DATA_DIR / "orbits"
PARQUET_PATH = ORBITS_DIR / "orbits.parquet"
FAMILIES_PATH = ORBITS_DIR / "families.json"

_ID_PREFIX = {
    "L1_lyapunov": "L1_lyap",
    "L2_lyapunov": "L2_lyap",
    "L1_halo_N": "L1_halo_N",
    "L1_halo_S": "L1_halo_S",
    "L2_halo_N": "L2_halo_N",
    "L2_halo_S": "L2_halo_S",
    "DRO": "DRO",
    "resonant_3:1": "RES_31",
    "resonant_2:1": "RES_21",
}


@dataclass
class OrbitRecord:
    id: str
    family: str                 # 'L1_lyapunov', 'L2_halo', 'DRO', 'resonant_3:1', ...
    branch: str                 # 'N' / 'S' for halos, '' otherwise
    ic: list                    # 6 floats, rotating frame, nondimensional
    period_nd: float
    period_days: float
    jacobi: float
    stability_index: float      # nu = ½(|λmax| + 1/|λmax|)
    eigenvalues: list           # |λ_i|, 6 floats, decreasing
    closure_error: float        # |X(T) - X0|_inf, independent propagation (rtol 2.3e-14)
    params: dict                # Az_km, x0, perilune_km, apolune_km, perigee_km, amplitude_km, ...
    tags: list = field(default_factory=list)
    method: str = "single_shooting"
    index: int = 0

    @property
    def ic_array(self) -> np.ndarray:
        return np.asarray(self.ic, dtype=np.float64)

    @property
    def family_key(self) -> str:
        return f"{self.family}_{self.branch}" if self.branch else self.family

    @property
    def stable(self) -> bool:
        return self.stability_index <= 1.0 + 1e-6

    def to_row(self) -> dict:
        x = self.ic
        return {
            "id": self.id,
            "family": self.family,
            "branch": self.branch,
            "index": int(self.index),
            "x0": x[0], "y0": x[1], "z0": x[2], "vx0": x[3], "vy0": x[4], "vz0": x[5],
            "period_nd": self.period_nd,
            "period_days": self.period_days,
            "jacobi": self.jacobi,
            "stability_index": self.stability_index,
            "closure_error": self.closure_error,
            "method": self.method,
            "eigenvalues": json.dumps(self.eigenvalues),
            "params": json.dumps(self.params),
            "tags": json.dumps(self.tags),
        }

    @classmethod
    def from_row(cls, r: dict) -> "OrbitRecord":
        return cls(
            id=r["id"], family=r["family"], branch=r["branch"] or "",
            ic=[float(r[k]) for k in ("x0", "y0", "z0", "vx0", "vy0", "vz0")],
            period_nd=float(r["period_nd"]), period_days=float(r["period_days"]), jacobi=float(r["jacobi"]),
            stability_index=float(r["stability_index"]), eigenvalues=json.loads(r["eigenvalues"]),
            closure_error=float(r["closure_error"]), params=json.loads(r["params"]), tags=json.loads(r["tags"]),
            method=r["method"], index=int(r["index"]),
        )

    def as_dict(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------------------
# record construction
# ---------------------------------------------------------------------------
def _record_from_member(key: str, i: int, m: FamilyMember, fr: FamilyResult, rid: str | None = None, mu: float = MU) -> OrbitRecord:
    st = stability_index(m.ic, m.period, mu)
    tags = list(dict.fromkeys(m.tags + (["stable"] if st.stable else [])))
    params = dict(m.geometry)
    params.update({"x0": float(m.ic[0]), "z0": float(m.ic[2]), "vy0": float(m.ic[4]),
                   "efold_periods": st.efold_time, "monodromy_det": st.det, "unit_eig_error": st.unit_pair_error})
    params.update({k: v for k, v in m.seed.items() if isinstance(v, (int, float))})
    return OrbitRecord(
        id=rid or f"{_ID_PREFIX[key]}_{i:03d}",
        family=fr.family,
        branch=fr.branch,
        ic=[float(v) for v in m.ic],
        period_nd=float(m.period),
        period_days=float(m.period * T_STAR / 86400.0),
        jacobi=float(jacobi(m.ic, mu)),
        stability_index=float(st.nu),
        eigenvalues=[float(v) for v in np.abs(st.eigenvalues)],
        closure_error=float(m.closure_error),
        params=params,
        tags=tags,
        method=m.method,
        index=i,
    )


def _build_family(key: str, quick: bool = False) -> tuple[list, dict]:
    """Worker: generate one family and turn it into records (+ metadata)."""
    t0 = time.time()
    fr = generate_family(key, quick=quick)
    records = [_record_from_member(key, i, m, fr) for i, m in enumerate(fr.members)]
    located = []
    if key == "L2_halo_S":
        for m in locate_synodic_nrhos(fr):
            tag = [t for t in m.tags if t.startswith("NRHO_")][0]
            rid = f"{_ID_PREFIX[key]}_nrho_{tag.split('_')[1].replace(':', '_')}"
            located.append(_record_from_member(key, len(records) + len(located), m, fr, rid=rid))
    records += located
    periods = [r.period_days for r in records]
    meta = {
        "family": fr.family,
        "branch": fr.branch,
        "count": len(records),
        "generation": fr.params,
        "references": fr.references,
        "log": fr.log,
        "elapsed_s": round(time.time() - t0, 2),
        "period_days_range": [min(periods), max(periods)],
        "jacobi_range": [min(r.jacobi for r in records), max(r.jacobi for r in records)],
        "stability_index_range": [min(r.stability_index for r in records), max(r.stability_index for r in records)],
        "max_closure_error": max(r.closure_error for r in records),
        "located": [r.id for r in located],
    }
    return records, meta


# ---------------------------------------------------------------------------
class OrbitLibrary:
    def __init__(self, records: list[OrbitRecord], families: dict, meta: dict | None = None):
        self.records = list(records)
        self.families = families
        self.meta = meta or {}
        self.by_id = {r.id: r for r in self.records}

    # -- build / persist -------------------------------------------------
    @classmethod
    def build(cls, quick: bool = False, parallel: bool = True, verbose: bool = True, keys=None) -> "OrbitLibrary":
        t0 = time.time()
        keys = list(keys or FAMILY_SPECS.keys())
        results: dict[str, tuple[list, dict]] = {}
        if parallel and len(keys) > 1:
            try:
                from concurrent.futures import ProcessPoolExecutor

                with ProcessPoolExecutor(max_workers=min(len(keys), 8)) as ex:
                    futs = {k: ex.submit(_build_family, k, quick) for k in keys}
                    for k, f in futs.items():
                        results[k] = f.result()
                        if verbose:
                            print(f"  built {k:14s} {results[k][1]['count']:4d} members  ({results[k][1]['elapsed_s']:.1f} s)")
            except Exception as e:  # pragma: no cover - fallback path
                if verbose:
                    print(f"  parallel build failed ({e!r}); falling back to serial")
                results = {}
        if not results:
            for k in keys:
                results[k] = _build_family(k, quick)
                if verbose:
                    print(f"  built {k:14s} {results[k][1]['count']:4d} members  ({results[k][1]['elapsed_s']:.1f} s)")
        records = []
        families = {}
        for k in keys:
            recs, meta = results[k]
            records += recs
            families[k] = meta
        meta = {
            "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "quick": quick,
            "build_time_s": round(time.time() - t0, 1),
            "mu": MU,
            "L_star_km": L_STAR,
            "T_star_s": T_STAR,
            "n_records": len(records),
            "ic_convention": "rotating frame, nondimensional; perpendicular xz-plane crossing farther from the Moon "
                             "(apolune for NRHOs); halo branch N/S = sign of z there",
            "closure_check": "independent full-period DOP853 propagation at rtol=2.3e-14, atol=1e-14",
            "normalisation_note": "JPL periodic-orbit catalogue uses the same μ to ~1e-9 but lunit=389703 km and "
                                  "tunit=382981 s; nondimensional ICs/periods are directly comparable, km/day values are not",
        }
        return cls(records, families, meta)

    def save(self, directory: Path | str = ORBITS_DIR) -> None:
        import pyarrow as pa
        import pyarrow.parquet as pq

        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        rows = [r.to_row() for r in self.records]
        table = pa.Table.from_pylist(rows)
        pq.write_table(table, directory / "orbits.parquet", compression="zstd")
        payload = {"meta": self.meta, "families": self.families}
        (directory / "families.json").write_text(json.dumps(payload, indent=1, default=_json_default))

    @classmethod
    def load(cls, directory: Path | str = ORBITS_DIR) -> "OrbitLibrary":
        import pyarrow.parquet as pq

        directory = Path(directory)
        table = pq.read_table(directory / "orbits.parquet")
        records = [OrbitRecord.from_row(r) for r in table.to_pylist()]
        payload = json.loads((directory / "families.json").read_text())
        return cls(records, payload.get("families", {}), payload.get("meta", {}))

    @staticmethod
    def exists(directory: Path | str = ORBITS_DIR) -> bool:
        directory = Path(directory)
        return (directory / "orbits.parquet").exists() and (directory / "families.json").exists()

    # -- queries ---------------------------------------------------------
    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, rid: str) -> OrbitRecord:
        return self.by_id[rid]

    def family_keys(self) -> list[str]:
        return list(self.families.keys())

    def members(self, family_key: str) -> list[OrbitRecord]:
        """Records of a family key such as 'L2_halo_S', 'DRO', 'resonant_3:1'."""
        return [r for r in self.records if r.family_key == family_key]

    def find(
        self,
        family: str | None = None,
        tag: str | None = None,
        branch: str | None = None,
        period_days: float | None = None,
        jacobi: float | None = None,
        perilune_km: float | None = None,
        Az_km: float | None = None,
        nearest: bool = True,
        exclude_located: bool = False,
    ):
        """Filter by family (``'L2_halo'`` or family key ``'L2_halo_S'``), branch and tag, then
        rank by closeness in the given numeric criteria (each normalised by the candidate
        spread).  ``nearest=True`` returns the single best record (or None); ``nearest=False``
        returns the ranked list."""
        c = self.records
        if family is not None:
            c = [r for r in c if r.family == family or r.family_key == family]
        if branch is not None:
            c = [r for r in c if r.branch == branch.upper()]
        if tag is not None:
            c = [r for r in c if tag in r.tags]
        if exclude_located:  # drop the root-found synodic-resonant NRHO records
            c = [r for r in c if "_nrho_" not in r.id]
        crit = []
        if period_days is not None:
            crit.append((np.array([r.period_days for r in c]), period_days))
        if jacobi is not None:
            crit.append((np.array([r.jacobi for r in c]), jacobi))
        if perilune_km is not None:
            crit.append((np.array([r.params.get("perilune_km", np.nan) for r in c]), perilune_km))
        if Az_km is not None:
            crit.append((np.array([r.params.get("Az_km", np.nan) for r in c]), Az_km))
        if crit and c:
            score = np.zeros(len(c))
            for vals, target in crit:
                scale = np.nanmax(vals) - np.nanmin(vals) if len(vals) > 1 else 1.0
                scale = scale if scale > 0 else max(abs(target), 1.0)
                score += ((vals - target) / scale) ** 2
            order = np.argsort(np.nan_to_num(score, nan=np.inf))
            c = [c[i] for i in order]
        if nearest:
            return c[0] if c else None
        return c

    # -- sampling --------------------------------------------------------
    def _rec(self, record) -> OrbitRecord:
        return self.by_id[record] if isinstance(record, str) else record

    def sample_states(self, record, n: int = 400, rtol: float = 1e-11, atol: float = 1e-11, mu: float = MU) -> np.ndarray:
        """(n, 6) rotating-frame nondimensional states over one period, t = linspace(0, T, n)."""
        r = self._rec(record)
        n = int(n)
        if n < 2:
            raise ValueError(f"sample_states needs n >= 2 (got {n}): t = linspace(0, T, n)")
        ts = np.linspace(0.0, r.period_nd, n)
        sol = propagate_cr3bp(r.ic_array, t_eval=ts, mu=mu, rtol=rtol, atol=atol)
        if not sol.success:
            raise RuntimeError(f"sampling {r.id} failed: {sol.message}")
        return sol.y[:6].T.copy()

    def sample(self, record, n: int = 400, **kw) -> np.ndarray:
        """(n, 3) rotating-frame nondimensional positions over one period."""
        return self.sample_states(record, n, **kw)[:, :3]

    def sample_km(self, record, n: int = 400, **kw) -> np.ndarray:
        """(n, 3) rotating-frame positions in km (× L*), barycentric origin."""
        return self.sample(record, n, **kw) * L_STAR

    # -- reporting -------------------------------------------------------
    def summary_table(self) -> str:
        lines = [f"{'family':14s} {'n':>4s} {'period [d]':>16s} {'Jacobi C':>16s} {'nu (stab idx)':>18s} {'max closure':>12s} {'perilune [km]':>18s}"]
        for k, meta in self.families.items():
            recs = self.members(k)
            if not recs:
                continue
            per = [r.params.get("perilune_km", np.nan) for r in recs]
            lines.append(
                f"{k:14s} {len(recs):4d} {min(r.period_days for r in recs):7.2f}..{max(r.period_days for r in recs):<7.2f} "
                f"{min(r.jacobi for r in recs):7.4f}..{max(r.jacobi for r in recs):<7.4f} "
                f"{min(r.stability_index for r in recs):8.2f}..{max(r.stability_index for r in recs):<8.2f} "
                f"{max(r.closure_error for r in recs):12.1e} {np.nanmin(per):8.0f}..{np.nanmax(per):<8.0f}"
            )
        tagged = [r for r in self.records if any(t.startswith("NRHO_") for t in r.tags)]
        for r in tagged:
            lines.append(
                f"  {r.id}: tags={r.tags} T={r.period_days:.4f} d  ic={np.round(r.ic, 6).tolist()}  "
                f"perilune={r.params['perilune_km']:.0f} km apolune={r.params['apolune_km']:.0f} km nu={r.stability_index:.3f} closure={r.closure_error:.1e}"
            )
        lines.append(f"total records: {len(self.records)}  build_time_s={self.meta.get('build_time_s')}  generated={self.meta.get('generated_utc')}")
        return "\n".join(lines)


def _json_default(o):
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, float) and not np.isfinite(o):
        return None
    return str(o)


# ---------------------------------------------------------------------------
_LIB: OrbitLibrary | None = None


def get_library(rebuild: bool = False, directory: Path | str = ORBITS_DIR, quick: bool = False) -> OrbitLibrary:
    """Cached loader (module-level singleton).  Loads ``data/orbits``; builds (and saves) it if
    missing or ``rebuild=True``."""
    global _LIB
    if _LIB is not None and not rebuild:
        return _LIB
    if rebuild or not OrbitLibrary.exists(directory):
        lib = OrbitLibrary.build(quick=quick, verbose=False)
        lib.save(directory)
    else:
        lib = OrbitLibrary.load(directory)
    _LIB = lib
    return lib


def main(argv=None):
    ap = argparse.ArgumentParser(description="SELENE periodic-orbit library")
    ap.add_argument("--rebuild", action="store_true", help="regenerate all families and overwrite data/orbits")
    ap.add_argument("--quick", action="store_true", help="coarser continuation steps (development)")
    ap.add_argument("--serial", action="store_true", help="disable the process pool")
    ap.add_argument("--out", default=str(ORBITS_DIR))
    args = ap.parse_args(argv)
    if args.rebuild:
        print(f"building orbit library (quick={args.quick}) ...")
        t0 = time.time()
        lib = OrbitLibrary.build(quick=args.quick, parallel=not args.serial)
        lib.save(args.out)
        print(f"saved {len(lib)} records to {args.out} in {time.time() - t0:.1f} s")
    else:
        t0 = time.time()
        lib = OrbitLibrary.load(args.out)
        print(f"loaded {len(lib)} records in {time.time() - t0:.3f} s")
    print(lib.summary_table())


if __name__ == "__main__":
    main()
