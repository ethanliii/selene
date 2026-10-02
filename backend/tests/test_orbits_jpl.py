"""Validation of the SELENE orbit library against JPL's Three-Body Periodic Orbit Catalog
(cached subset in data/orbits/jpl_reference.json; see selene/orbits/jpl_reference.py for the
normalisation differences)."""
import numpy as np
import pytest

from selene.constants import MU
from selene.dynamics.cr3bp import jacobi
from selene.orbits.jpl_reference import (
    JPL_MU,
    JPL_QUERIES,
    REFERENCE_PATH,
    distance_to_curve,
    jpl_mu_difference,
    load_jpl_reference,
)
from selene.orbits.library import OrbitLibrary, get_library
from selene.orbits.shooting import closure_error, shoot_symmetric

N_RECONVERGE = 4
# Distance in (Jacobi, period) nondimensional space from the cached ~700-vertex poly-line.
# Against JPL's *full* catalogue poly-line (3 000-12 000 members per family) every record of
# the committed library lies within 6.1e-7 (mean <= 1e-7); the cached poly-line's chords add
# up to 7.4e-5 (L1 halo) / 4.1e-5 (DRO) where the family curve bends, hence 2e-4 here.
CT_CURVE_TOL = 2e-4


@pytest.fixture(scope="module")
def ref():
    if not REFERENCE_PATH.exists():
        pytest.skip("JPL reference file missing (python -m selene.orbits.jpl_reference --fetch)")
    return load_jpl_reference()


@pytest.fixture(scope="module")
def lib():
    if not OrbitLibrary.exists():
        pytest.skip("orbit library not built")
    return get_library()


def _pick_members(ref_fam: dict, recs: list, n: int) -> list:
    """JPL members closest in (C, T) to n evenly spread records of our family (so the
    comparison is made inside the range the library covers)."""
    mem = np.array(ref_fam["members"], dtype=np.float64)
    ours = np.array([[r.jacobi, r.period_nd] for r in recs])
    picks = []
    for i in np.unique(np.linspace(0, len(ours) - 1, n).round().astype(int)):
        d = np.hypot(mem[:, 6] - ours[i, 0], mem[:, 7] - ours[i, 1])
        j = int(np.argmin(d))
        if j not in picks:
            picks.append(j)
    return [mem[j] for j in picks]


def test_reference_file_provenance(ref):
    assert "ssd-api.jpl.nasa.gov" in ref["source"]
    assert ref["access_date_utc"] >= "2026-01-01"
    assert set(JPL_QUERIES) <= set(ref["families"])
    for key, fam in ref["families"].items():
        assert fam["query_url"].endswith(JPL_QUERIES[key])
        assert 3 <= len(fam["members"]) <= 60
        assert fam["fields"] == ["x", "y", "z", "vx", "vy", "vz", "jacobi", "period", "stability"]
        assert len(fam["curve_ct"]) >= 50
    # JPL's own Jacobi value agrees with ours evaluated at their IC (same μ to 1e-9)
    for key, fam in ref["families"].items():
        for row in fam["members"][:: max(1, len(fam["members"]) // 6)]:
            assert abs(jacobi(row[:6], JPL_MU) - row[6]) < 1e-9, key
            # with SELENE's μ the Jacobi constant shifts by δμ·|∂C/∂μ| where, with the primaries
            # at x = -μ and 1-μ,  |∂C/∂μ| ≤ 2/r1 + 2/r2 + 2(1-μ)/r1² + 2μ/r2²  (large only for
            # members hugging a primary, e.g. JPL's smallest DROs / L2 Lyapunov at 800 km altitude)
            r1 = np.hypot(row[0] + MU, row[2])
            r2 = np.hypot(row[0] - 1 + MU, row[2])
            dmu = abs(JPL_MU - MU)
            bound = 2.0 * dmu * (2 / r1 + 2 / r2 + 2 * (1 - MU) / r1**2 + 2 * MU / r2**2) + 1e-9
            assert abs(jacobi(row[:6], MU) - row[6]) < bound, key


def test_mu_and_unit_differences_are_documented(ref):
    d = jpl_mu_difference()
    assert abs(float(ref["system"]["mass_ratio"]) - JPL_MU) < 1e-15
    assert abs(d["abs_diff"]) < 1e-8 and abs(d["rel_diff"]) < 1e-6
    # JPL normalises with its own length/time units (not 384 400 km / T*)
    assert abs(float(ref["system"]["lunit"]) - 389703.26) < 1.0
    assert abs(float(ref["system"]["tunit"]) - 382981.29) < 1.0


@pytest.mark.parametrize("key", list(JPL_QUERIES))
def test_shooting_reconverges_from_jpl_ic(ref, lib, key):
    recs = [r for r in lib.members(key) if "_nrho_" not in r.id]
    members = _pick_members(ref["families"][key], recs, N_RECONVERGE)
    assert len(members) >= 3
    mode = "fix_z0" if "halo" in key else "planar"
    worst = {"dT_jpl_mu": 0.0, "closure": 0.0, "dIC": 0.0, "dT_selene_mu": 0.0}
    for row in members:
        X, T_jpl = row[:6], row[7]
        # (a) exact comparison: same μ as JPL.  The corrector is started from a *perturbed*
        # JPL IC (vy0 by 1e-4 relative, x0 by 1e-4 for halos) so that Newton steps are
        # actually taken: starting exactly at the JPL IC the first residual is already below
        # tolerance and the assertions would be trivially satisfied.
        Xp = X.copy()
        Xp[4] *= 1.0 + 1e-4
        if mode == "fix_z0":
            Xp[0] += 1e-4
        r = shoot_symmetric(Xp, mode=mode, mu=JPL_MU, t_half=0.5 * T_jpl)
        assert r.converged, (key, X, T_jpl, r.message)
        assert r.iterations >= 2, (key, r.iterations)  # the perturbation was corrected, not accepted
        c = closure_error(r.ic, r.period, mu=JPL_MU)
        assert c < 1e-9, (key, X, c)
        assert abs(r.period - T_jpl) < 1e-6, (key, r.period, T_jpl)
        assert np.max(np.abs(r.ic - X)) < 1e-7, (key, r.ic, X)
        # and the unperturbed JPL IC is already periodic under our EOM (validates EOM + μ)
        assert closure_error(X, T_jpl, mu=JPL_MU) < 1e-8, key
        # (b) with SELENE's μ (δμ = 1.2e-9) the re-converged orbit shifts slightly along the
        # family; with z0 held fixed the period shift is ~1e-7 in general but reaches ~1e-5
        # near the z0-fold of the L2 halo family (dT/dμ|z0 is large there), so only a loose
        # bound is meaningful here -- the exact comparison is (a).
        r2 = shoot_symmetric(X, mode=mode, mu=MU, t_half=0.5 * T_jpl)
        assert r2.converged and abs(r2.period - T_jpl) < 1e-4
        # JPL's stability index is the same quantity as ours (within the μ shift)
        worst["dT_jpl_mu"] = max(worst["dT_jpl_mu"], abs(r.period - T_jpl))
        worst["closure"] = max(worst["closure"], c)
        worst["dIC"] = max(worst["dIC"], float(np.max(np.abs(r.ic - X))))
        worst["dT_selene_mu"] = max(worst["dT_selene_mu"], abs(r2.period - T_jpl))
    print(key, {k: f"{v:.1e}" for k, v in worst.items()})


@pytest.mark.parametrize("key", list(JPL_QUERIES))
def test_family_lies_on_jpl_jacobi_period_curve(ref, lib, key):
    curve = np.array(ref["families"][key]["curve_ct"], dtype=np.float64)
    recs = lib.members(key)
    pts = np.array([[r.jacobi, r.period_nd] for r in recs])
    d = distance_to_curve(pts, curve)
    print(key, f"max {d.max():.2e} mean {d.mean():.2e}")
    assert d.max() < CT_CURVE_TOL, (key, d.max(), recs[int(np.argmax(d))].id)
    # and the library spans a non-trivial part of the catalogued family
    lim = ref["families"][key]["limits"]
    pmin, pmax = float(lim["period"][0]), float(lim["period"][1])
    assert pts[:, 1].min() >= pmin - 1e-3 and pts[:, 1].max() <= pmax + 1e-3
    assert (pts[:, 1].max() - pts[:, 1].min()) > 0.05 * (pmax - pmin)


def test_jpl_stability_index_matches_ours(ref, lib):
    """Stability index definition agrees with JPL's catalogue for members we re-converge."""
    from selene.orbits.stability import stability_index

    for key in ("L1_lyapunov", "L2_halo_S", "DRO"):
        recs = [r for r in lib.members(key) if "_nrho_" not in r.id]
        for row in _pick_members(ref["families"][key], recs, 3):
            r = shoot_symmetric(row[:6], mode="fix_z0" if "halo" in key else "planar", mu=JPL_MU, t_half=0.5 * row[7])
            st = stability_index(r.ic, r.period, mu=JPL_MU)
            assert abs(st.nu - row[8]) <= 0.01 * max(row[8], 1.0) + 1e-3, (key, st.nu, row[8])
