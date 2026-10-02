"""Tests of the committed orbit library (data/orbits): closure, Jacobi conservation, literature
values, family coverage, NRHO properties, queries, sampling and load time."""
import json
import time

import numpy as np
import pytest

from selene.constants import L_STAR, MU, R_MOON, T_STAR
from selene.dynamics.cr3bp import jacobi, propagate_cr3bp
from selene.orbits.families import NRHO_PERILUNE_BOUNDS_KM, SYNODIC_MONTH_DAYS
from selene.orbits.library import FAMILIES_PATH, PARQUET_PATH, OrbitLibrary, get_library

MIN_COUNTS = {
    "L1_lyapunov": 25, "L2_lyapunov": 25,
    "L1_halo_N": 30, "L1_halo_S": 30, "L2_halo_N": 30, "L2_halo_S": 30,
    "DRO": 30, "resonant_3:1": 10, "resonant_2:1": 10,
}
# Closure |X(T) - X0|_inf is measured with an independent full-period DOP853 propagation at
# rtol = 2.3e-14 / atol = 1e-14 (the tightest scipy accepts).  At the corrector's own 1e-12
# the same ICs show up to ~9e-10 for the most unstable Lyapunov members (|lambda_max| ~ 1e3
# amplifies the integration error), which measures the integrator, not the IC.
CLOSURE_RTOL, CLOSURE_ATOL = 2.3e-14, 1e-14
CLOSURE_TOL = 1e-10
# Only near-rectilinear members (multiple shooting through a ~2 000-17 000 km perilune) get
# the relaxed 1e-9 bound allowed by the spec; resonant families must meet 1e-10 like the rest
# (their low-perigee tails were trimmed at the roundoff floor of the closure check, see
# families.gen_resonant).
CLOSURE_TOL_SENSITIVE = 1e-9


@pytest.fixture(scope="module")
def lib() -> OrbitLibrary:
    if not OrbitLibrary.exists():
        pytest.skip("orbit library not built (python -m selene.orbits.library --rebuild)")
    return get_library()


@pytest.fixture(scope="module")
def propagated(lib):
    """One tight full-period propagation per record: closure and Jacobi drift."""
    out = {}
    for r in lib.records:
        ts = np.linspace(0.0, r.period_nd, 101)
        sol = propagate_cr3bp(r.ic_array, t_eval=ts, rtol=CLOSURE_RTOL, atol=CLOSURE_ATOL)
        assert sol.success, r.id
        C = jacobi(sol.y[:6].T)
        out[r.id] = {
            "closure": float(np.max(np.abs(sol.y[:6, -1] - r.ic_array))),
            "jacobi_drift": float(np.max(np.abs(C - C[0]))),
            "jacobi": float(C[0]),
        }
    return out


def test_files_exist_and_are_compact():
    assert PARQUET_PATH.exists() and FAMILIES_PATH.exists()
    total = sum(p.stat().st_size for p in PARQUET_PATH.parent.iterdir() if p.is_file())
    assert total < 3_000_000, f"data/orbits is {total/1e6:.2f} MB"


def test_library_loads_fast():
    import pyarrow.parquet  # noqa: F401  (import cost is paid once at module import, not per load)

    t0 = time.perf_counter()
    lib2 = OrbitLibrary.load()
    dt = time.perf_counter() - t0
    assert len(lib2) > 200
    assert dt < 0.5, f"load took {dt:.3f} s"


def test_family_member_counts(lib):
    for key, n_min in MIN_COUNTS.items():
        assert key in lib.families, key
        assert len(lib.members(key)) >= n_min, (key, len(lib.members(key)))
    assert len({r.id for r in lib.records}) == len(lib.records)  # unique ids


def test_every_record_closes(lib, propagated):
    worst = {}
    n_sensitive = 0
    for r in lib.records:
        c = propagated[r.id]["closure"]
        sensitive = r.method == "multiple_shooting" or "NRHO" in r.tags
        n_sensitive += sensitive
        tol = CLOSURE_TOL_SENSITIVE if sensitive else CLOSURE_TOL
        assert c < tol, f"{r.id} closure {c:.2e} >= {tol:g} (method={r.method})"
        # the stored value was measured with the same tolerance -> consistent to the noise level
        assert abs(c - r.closure_error) < 1e-9, r.id
        worst[r.family_key] = max(worst.get(r.family_key, 0.0), c)
    print("max closure per family:", {k: f"{v:.1e}" for k, v in worst.items()})
    # explicit statement of the two regimes (spec: 1e-10, with 1e-9 allowed only for
    # NRHO / multiple-shooting members)
    assert n_sensitive > 0
    assert max(propagated[r.id]["closure"] for r in lib.records
               if not (r.method == "multiple_shooting" or "NRHO" in r.tags)) < CLOSURE_TOL
    assert max(v for k, v in worst.items() if "lyap" in k or "DRO" in k or "resonant" in k) < CLOSURE_TOL
    assert max(worst.values()) < CLOSURE_TOL_SENSITIVE


def test_jacobi_conserved_along_each_orbit(lib, propagated):
    drift = {r.id: propagated[r.id]["jacobi_drift"] for r in lib.records}
    worst_id = max(drift, key=drift.get)
    assert drift[worst_id] < 1e-9, (worst_id, drift[worst_id])
    for r in lib.records:
        assert abs(propagated[r.id]["jacobi"] - r.jacobi) < 1e-10, r.id


def test_jacobi_conserved_on_sampled_states(lib):
    for key in ("L2_halo_S", "DRO", "resonant_3:1", "L1_lyapunov"):
        r = lib.members(key)[len(lib.members(key)) // 2]
        S = lib.sample_states(r, n=300)
        C = jacobi(S)
        assert S.shape == (300, 6)
        assert np.max(np.abs(C - r.jacobi)) < 1e-8, key
        assert np.linalg.norm(S[-1, :3] - S[0, :3]) < 1e-7  # one full period


def test_klmr_l1_lyapunov_member(lib):
    # Koon, Lo, Marsden & Ross (2011) §4.2: x0 = 0.8234, ẏ0 = 0.1263, T = 2.7430
    recs = lib.members("L1_lyapunov")
    r = min(recs, key=lambda r: abs(r.ic[0] - 0.8234))
    assert abs(r.ic[0] - 0.8234) < 1e-6
    assert abs(r.period_nd - 2.7430) < 1e-3
    assert abs(r.ic[4] - 0.1263) < 1e-3
    assert r.stability_index > 1000  # strongly unstable (JPL: 1180.75)


def test_nrho_9_2_record(lib):
    r = lib.find(tag="NRHO_9:2")
    assert r is not None and r.family == "L2_halo" and r.branch == "S"
    # period: 2/9 of the mean synodic month (Zimovan-Spreen et al. 2020; Lee 2019)
    assert abs(r.period_days - 2.0 / 9.0 * SYNODIC_MONTH_DAYS) < 1e-6
    assert abs(r.period_days - 6.56) < 0.1
    assert 3000.0 < r.params["perilune_km"] < 3600.0
    assert 65_000.0 < r.params["apolune_km"] < 72_000.0
    # nearly linearly stable (published stability index ~1.3)
    assert abs(r.stability_index) < 2.0
    # apolune IC on the southern side, consistent with Zimovan-Spreen et al. (1.0221, -0.1821, -0.1030)
    assert abs(r.ic[0] - 1.0221) < 2e-3 and abs(r.ic[2] + 0.1821) < 2e-3 and abs(r.ic[4] + 0.1030) < 2e-3
    assert r.closure_error < 1e-9


def test_nrho_regime_tags_and_other_synodic_resonances(lib):
    for libr in (1, 2):
        lo, hi = NRHO_PERILUNE_BOUNDS_KM[libr]
        for br in ("N", "S"):
            recs = lib.members(f"L{libr}_halo_{br}")
            nrho = [r for r in recs if "NRHO" in r.tags]
            assert len(nrho) >= 5, (libr, br)
            for r in nrho:
                assert lo <= r.params["perilune_km"] <= hi
            for r in recs:
                assert r.params["perilune_km"] > R_MOON + 50.0  # no lunar impactors in the library
            # outside the regime the halos are strongly unstable
            assert max(r.stability_index for r in recs if "NRHO" not in r.tags) > 100
    for p, q in ((4, 1), (3, 1)):
        r = lib.find(tag=f"NRHO_{p}:{q}")
        assert r is not None
        assert abs(r.period_days - q / p * SYNODIC_MONTH_DAYS) < 1e-6
        assert r.closure_error < 1e-9


def _true_perilune_km(r) -> float:
    """Independent perilune: dense-output propagation (rtol 1e-12) + bounded minimisation of the
    Moon distance around the minimum of a 2 000-point grid."""
    from scipy.optimize import minimize_scalar

    sol = propagate_cr3bp(r.ic_array, tf=r.period_nd, rtol=1e-12, atol=1e-12, dense_output=True)
    moon = np.array([1.0 - MU, 0.0, 0.0])

    def f(t):
        return float(np.linalg.norm(sol.sol(t)[:3] - moon))

    ts = np.linspace(0.0, r.period_nd, 2000)
    d = np.array([f(t) for t in ts])
    i = int(np.argmin(d))
    res = minimize_scalar(f, bounds=(ts[max(i - 1, 0)], ts[min(i + 1, len(ts) - 1)]), method="bounded",
                          options={"xatol": 1e-12})
    return min(res.fun, d[i]) * L_STAR


def test_perilune_is_refined_not_grid_sampled(lib):
    """The stored perilune must be the true minimum Moon distance, not a coarse grid minimum
    (which is biased high by up to ~400 km for near-rectilinear members and hid lunar
    impactors in an earlier build).  Checked on the lowest-perilune member of every halo
    branch, the three located NRHOs and the lowest L2 Lyapunov member."""
    picks = []
    for key in ("L1_halo_N", "L1_halo_S", "L2_halo_N", "L2_halo_S", "L2_lyapunov"):
        picks.append(min(lib.members(key), key=lambda r: r.params["perilune_km"]))
    picks += [lib.find(tag=t) for t in ("NRHO_9:2", "NRHO_4:1", "NRHO_3:1")]
    for r in picks:
        true_km = _true_perilune_km(r)
        assert abs(true_km - r.params["perilune_km"]) < 1.0, (r.id, true_km, r.params["perilune_km"])
        assert true_km > R_MOON + 50.0, (r.id, true_km)   # family terminated before lunar impact
    # the 9:2 NRHO perilune agrees with the published ~3 250 km (Zimovan-Spreen et al. 2020)
    assert abs(lib.find(tag="NRHO_9:2").params["perilune_km"] - 3250.0) < 60.0


def test_sample_rejects_degenerate_n(lib):
    r = lib.members("DRO")[0]
    for n in (0, 1):
        with pytest.raises(ValueError):
            lib.sample(r, n=n)
    assert lib.sample(r, n=2).shape == (2, 3)


def test_stability_contrast_nrho_vs_mid_halo(lib):
    nrho = lib.find(tag="NRHO_9:2")
    mid = lib.find(family="L1_halo_S", Az_km=25_000.0)
    assert 10_000 < mid.params["Az_km"] < 45_000
    assert nrho.stability_index < 2.0 < 10.0 < mid.stability_index
    # stability flag consistent with nu
    for r in lib.records:
        assert ("stable" in r.tags) == (r.stability_index <= 1 + 1e-6)
        assert len(r.eigenvalues) == 6 and r.eigenvalues[0] >= r.eigenvalues[-1]


def test_dro_family_span(lib):
    recs = lib.members("DRO")
    xs = np.array([r.ic[0] for r in recs])
    T = np.array([r.period_days for r in recs])
    assert len(recs) >= 30
    assert np.all(np.diff(xs) < 0)
    assert T.min() <= 2.0 and T.max() >= 20.0
    assert all(r.stability_index <= 1 + 1e-6 for r in recs)  # DROs are linearly stable
    assert all(r.ic[2] == 0 and r.ic[5] == 0 for r in recs)
    assert all(r.ic[4] > 0 for r in recs)                     # retrograde (clockwise) about the Moon


def test_resonant_families(lib):
    for p, q in ((3, 1), (2, 1)):
        recs = lib.members(f"resonant_{p}:{q}")
        assert len(recs) >= 10
        for r in recs:
            assert f"RESONANT_{p}:{q}" in r.tags
            assert abs(r.period_nd - 2 * np.pi * q) / (2 * np.pi * q) < 0.07   # ≈ q synodic revolutions
            assert r.params["perigee_km"] > 40_000.0                              # above GEO
        xs = [r.ic[0] for r in recs]
        assert np.all(np.diff(xs) < 0)


def test_halo_branch_convention(lib):
    for libr in (1, 2):
        N = lib.members(f"L{libr}_halo_N")
        S = lib.members(f"L{libr}_halo_S")
        assert all(r.ic[2] > 0 for r in N) and all(r.ic[2] < 0 for r in S)
        # mirror symmetry: the (Jacobi, period) curves coincide
        cN = np.array(sorted((r.jacobi, r.period_nd) for r in N if "_nrho_" not in r.id))
        cS = np.array(sorted((r.jacobi, r.period_nd) for r in S if "_nrho_" not in r.id))
        n = min(len(cN), len(cS))
        assert np.allclose(cN[:n], cS[:n], atol=5e-3)
        # record IC is the crossing farther from the Moon: apolune for NRHOs
        for r in N + S:
            r2 = np.hypot(r.ic[0] - (1 - MU), r.ic[2]) * L_STAR
            assert abs(r2 - r.params["apolune_km"]) < 0.02 * r.params["apolune_km"]


def test_find_and_sample(lib):
    r = lib.find(family="DRO", period_days=14.0)
    assert r.family == "DRO"
    others = lib.find(family="DRO", period_days=14.0, nearest=False)
    assert others[0].id == r.id and len(others) == len(lib.members("DRO"))
    assert abs(r.period_days - 14.0) == min(abs(o.period_days - 14.0) for o in others)
    r2 = lib.find(family="L2_halo", branch="S", perilune_km=3300.0)
    assert 2500 < r2.params["perilune_km"] < 4500
    assert lib.find(family="nope") is None
    P = lib.sample(r2, n=400)
    assert P.shape == (400, 3)
    assert np.linalg.norm(P[-1] - P[0]) < 1e-7
    assert np.min(np.linalg.norm(P - np.array([1 - MU, 0, 0]), axis=1)) * L_STAR < 1.3 * r2.params["perilune_km"]
    assert lib.sample_km("DRO_000", n=10).shape == (10, 3)
    assert lib["DRO_000"].family == "DRO"


def test_families_json_metadata(lib):
    meta = json.loads(FAMILIES_PATH.read_text())
    assert meta["meta"]["mu"] == pytest.approx(MU)
    assert meta["meta"]["L_star_km"] == L_STAR and meta["meta"]["T_star_s"] == pytest.approx(T_STAR)
    for key, fam in meta["families"].items():
        assert fam["count"] == len(lib.members(key))
        assert fam["references"] and fam["generation"]
        assert fam["max_closure_error"] < 1e-9
