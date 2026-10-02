"""M1 acceptance tests for the DE440s ephemeris access and the n-body force model (PLAN §2.2)."""
import time

import numpy as np
import pytest

from selene.constants import AU_KM, GM_EARTH, GM_MOON, MU, P_SRP_1AU
from selene.dynamics import ephemeris as e
from selene.dynamics.frames import DEMO_EPOCH_TDB_S
from selene.time import seconds_since_j2000_tdb

DAY = 86400.0
T2026 = seconds_since_j2000_tdb("2026-01-01T00:00:00")


@pytest.fixture(scope="module")
def eph():
    return e.get_ephemeris()


def test_singleton(eph):
    assert e.get_ephemeris() is eph


def test_moon_distance_and_speed_across_2026(eph):
    ts = T2026 + np.arange(0, 365 * DAY, 3600.0)
    sm = eph.moon_state(ts)
    assert sm.shape == (ts.size, 6)
    d = np.linalg.norm(sm[:, :3], axis=1)
    v = np.linalg.norm(sm[:, 3:], axis=1)
    # perigee/apogee extremes of the lunar orbit (Meeus, Astronomical Algorithms, ch. 50)
    assert 356_000 < d.min() < 365_000
    assert 400_000 < d.max() < 407_000
    assert 0.95 < v.min() and v.max() < 1.12
    assert abs(v.mean() - 1.02) < 0.03  # mean orbital speed ≈ 1.02 km/s


def test_sun_distance_across_2026(eph):
    ts = T2026 + np.arange(0, 365 * DAY, 6 * 3600.0)
    r = np.linalg.norm(eph.position("sun", ts, "earth"), axis=1) / AU_KM
    assert 0.983 <= r.min() <= 0.9835  # perihelion
    assert 1.0165 <= r.max() <= 1.017  # aphelion


def test_scalar_and_array_shapes(eph):
    t = DEMO_EPOCH_TDB_S
    assert eph.position("moon", t).shape == (3,)
    assert eph.state("moon", t).shape == (6,)
    assert eph.state("sun", np.array([t, t + 1.0])).shape == (2, 6)
    assert np.allclose(eph.state("moon", np.array([t]))[0], eph.state("moon", t))
    # center bookkeeping: moon wrt earth == -(earth wrt moon); ssb chain consistent
    assert np.allclose(eph.state("moon", t, "earth"), -eph.state("earth", t, "moon"))
    assert np.allclose(
        eph.position("sun", t, "earth"),
        eph.position("sun", t, "ssb") - eph.position("earth", t, "ssb"),
        atol=1e-6,
    )
    with pytest.raises(ValueError):
        eph.position("mars", t)


def test_velocity_is_time_derivative_of_position(eph):
    # jplephem evaluates the Chebyshev offset in seconds since the segment start (~3e9 s), so
    # its internal time resolution is ~0.5 µs -> ~0.5 mm position noise -> ~1e-8 km/s in a
    # 20-s central difference.  Tolerance set accordingly (analytic derivative is far better).
    t = DEMO_EPOCH_TDB_S
    h = 10.0
    for body in ("moon", "sun"):
        v = eph.state(body, t)[3:]
        v_fd = (eph.position(body, t + h) - eph.position(body, t - h)) / (2 * h)
        assert np.linalg.norm(v - v_fd) < 1e-7 * max(1.0, np.linalg.norm(v))


def test_barycenter_consistent_with_mass_ratio(eph):
    ts = DEMO_EPOCH_TDB_S + np.arange(5) * DAY
    emb = eph.earth_moon_barycenter_state(ts)
    moon = eph.moon_state(ts)
    # DE440 EMRAT and the GM ratio agree to ~1e-10 -> metre-level over 4700 km
    assert np.max(np.abs(emb[:, :3] - MU * moon[:, :3])) < 1e-3


def test_body_cache_accuracy(eph):
    t0 = DEMO_EPOCH_TDB_S
    cache = e.BodyCache(t0, t0 + 30 * DAY, dt_s=600.0)
    tq = t0 + np.random.default_rng(0).uniform(0, 30 * DAY, 300)
    err_m = max(np.linalg.norm(cache.position("moon", t) - eph.position("moon", t)) for t in tq)
    err_s = max(np.linalg.norm(cache.position("sun", t) - eph.position("sun", t)) for t in tq)
    assert err_m < 1.0 and err_s < 1.0, f"cache err moon {err_m:.2e} km, sun {err_s:.2e} km"
    # documented sub-metre accuracy for dt = 600 s
    assert err_m < 1e-3 and err_s < 1e-3, f"cache err moon {err_m:.2e} km, sun {err_s:.2e} km"
    # vectorised path agrees with the numba scalar path
    assert np.allclose(cache.positions("moon", tq[:5]), np.array([cache.position("moon", t) for t in tq[:5]]))
    assert cache.covers(t0) and cache.covers(t0 + 30 * DAY) and not cache.covers(t0 + 40 * DAY)


def test_tidal_acceleration_magnitude():
    """Along the Earth-Moon line the net lunar perturbation (direct - indirect) on a GEO-range
    point is exactly GM_M [1/(d-r)² - 1/d²] toward the Moon (≈ 2 GM_M r/d³ to leading order)."""
    t = DEMO_EPOCH_TDB_S
    params = e.EphemParams(bodies=("moon",))
    rm = e.get_ephemeris().position("moon", t)
    d = np.linalg.norm(rm)
    xhat = rm / d
    rr = 42_164.0
    r = rr * xhat
    a = e.nbody_accel(t, r, params) + GM_EARTH * r / np.linalg.norm(r) ** 3  # remove central
    expected = GM_MOON * (1.0 / (d - rr) ** 2 - 1.0 / d**2)
    leading = 2 * GM_MOON * rr / d**3
    assert abs(np.dot(a, xhat) - expected) / expected < 1e-9
    assert abs(expected - leading) / leading < 0.25  # leading-order sanity (r/d ≈ 0.11)
    assert np.linalg.norm(a - np.dot(a, xhat) * xhat) < 1e-9 * expected


def test_indirect_term_present():
    """Without the indirect term, the Sun's acceleration at r ≈ 0 would be GM_S/AU² ≈ 5.9e-6 km/s²;
    with it, the net perturbation at the Earth's center vanishes."""
    t = DEMO_EPOCH_TDB_S
    params = e.EphemParams(bodies=("sun",))
    r = np.array([1.0, 0.0, 0.0])  # 1 km from the Earth centre
    a = e.nbody_accel(t, r, params) + GM_EARTH * r / np.linalg.norm(r) ** 3
    assert np.linalg.norm(a) < 1e-12  # tidal ≈ 2 GM_S r / AU³ ≈ 8e-14


def test_srp_magnitude_direction_and_shadow():
    t = DEMO_EPOCH_TDB_S
    eph = e.get_ephemeris()
    rs = eph.position("sun", t)
    shat = rs / np.linalg.norm(rs)
    params = e.EphemParams(bodies=(), srp=True, cr_area_mass=1.0)
    r = 100_000.0 * shat  # sunward of Earth, lit
    a = e.srp_accel(t, r, params)
    expected = P_SRP_1AU * 1e-3 * (AU_KM / np.linalg.norm(r - rs)) ** 2  # km/s²
    assert abs(np.linalg.norm(a) - expected) / expected < 1e-9
    # pushes away from the Sun: along (r - r_sun), i.e. toward Earth for a sunward object
    assert np.dot(a, shat) < 0 and np.dot(a, (r - rs) / np.linalg.norm(r - rs)) > 0
    # cylindrical Earth shadow: directly behind the Earth at 10 000 km, 1000 km off-axis
    perp = np.cross(shat, [0, 0, 1.0])
    perp /= np.linalg.norm(perp)
    assert np.linalg.norm(e.srp_accel(t, -10_000.0 * shat + 1000.0 * perp, params)) == 0.0
    assert np.linalg.norm(e.srp_accel(t, -10_000.0 * shat + 7000.0 * perp, params)) > 0.0
    # srp flag off -> no SRP in nbody_accel
    assert np.allclose(e.nbody_accel(t, r, e.EphemParams(bodies=())), -GM_EARTH * r / np.linalg.norm(r) ** 3)


def test_nbody_jacobian_vs_finite_differences():
    t = DEMO_EPOCH_TDB_S
    params = e.EphemParams(cache=e.BodyCache(t, t + DAY))
    r = np.array([2.0e5, 1.0e5, 3.0e4])
    G = e.nbody_jacobian(t, r, params)
    h = 1e-2
    G_fd = np.zeros((3, 3))
    for j in range(3):
        dp, dm = r.copy(), r.copy()
        dp[j] += h
        dm[j] -= h
        G_fd[:, j] = (e.nbody_accel(t, dp, params) - e.nbody_accel(t, dm, params)) / (2 * h)
    rel = np.max(np.abs(G - G_fd)) / np.max(np.abs(G))
    assert rel < 1e-6, f"nbody Jacobian rel err {rel:.2e}"
    assert np.allclose(G, G.T)
    # full 42-state derivative: A-matrix columns vs FD of nbody_eom
    s = np.concatenate([r, [0.3, 0.5, 0.1]])
    A = e.nbody_eom_stm(t, np.concatenate([s, np.eye(6).ravel()]), params)[6:].reshape(6, 6)
    A_fd = np.zeros((6, 6))
    for j in range(6):
        dp, dm = s.copy(), s.copy()
        dp[j] += h
        dm[j] -= h
        A_fd[:, j] = (e.nbody_eom(t, dp, params) - e.nbody_eom(t, dm, params)) / (2 * h)
    rel = np.max(np.abs(A - A_fd)) / np.max(np.abs(A))
    assert rel < 1e-6, f"nbody_eom_stm rel err {rel:.2e}"


def test_two_body_energy_conservation_geo_10_days():
    t0 = DEMO_EPOCH_TDB_S
    r0 = 42_164.0
    s0 = np.array([r0, 0, 0, 0, np.sqrt(GM_EARTH / r0), 0])
    params = e.EphemParams(bodies=())
    sol = e.propagate_ephemeris(s0, t0, t0 + 10 * DAY, t_eval_s=t0 + np.linspace(0, 10 * DAY, 241), params=params)
    assert sol.success
    E = 0.5 * np.sum(sol.y[3:] ** 2, axis=0) - GM_EARTH / np.linalg.norm(sol.y[:3], axis=0)
    rel = float(np.max(np.abs(E - E[0])) / abs(E[0]))
    assert rel < 1e-10, f"specific energy relative drift {rel:.2e}"
    # period check: after exactly 10 sidereal GEO periods the object returns to its start
    T = 2 * np.pi * np.sqrt(r0**3 / GM_EARTH)
    sol2 = e.propagate_ephemeris(s0, t0, t0 + 10 * T, params=params)
    assert np.linalg.norm(sol2.y[:3, -1] - s0[:3]) < 1e-3


def test_propagated_stm_matches_sensitivity():
    t0 = DEMO_EPOCH_TDB_S
    s0 = np.array([3.0e5, 5.0e4, 2.0e4, -0.2, 0.9, 0.1])
    tf = t0 + 2 * DAY
    cache = e.BodyCache(t0, tf)
    params = e.EphemParams(cache=cache)
    sol = e.propagate_ephemeris(s0, t0, tf, params=params, stm=True)
    phi = sol.y[6:, -1].reshape(6, 6)
    phi_fd = np.zeros((6, 6))
    steps = np.array([1.0, 1.0, 1.0, 1e-5, 1e-5, 1e-5])
    for j in range(6):
        dp, dm = s0.copy(), s0.copy()
        dp[j] += steps[j]
        dm[j] -= steps[j]
        yp = e.propagate_ephemeris(dp, t0, tf, params=params).y[:6, -1]
        ym = e.propagate_ephemeris(dm, t0, tf, params=params).y[:6, -1]
        phi_fd[:, j] = (yp - ym) / (2 * steps[j])
    # scale columns to compare dimensionless entries
    scale = np.outer(1.0 / np.array([1, 1, 1, 1e-3, 1e-3, 1e-3]), np.array([1, 1, 1, 1e-3, 1e-3, 1e-3]))
    rel = np.max(np.abs((phi - phi_fd) * scale)) / np.max(np.abs(phi * scale))
    assert rel < 1e-5, f"propagated STM vs FD rel err {rel:.2e}"


def test_propagation_speed_7_days():
    t0 = DEMO_EPOCH_TDB_S
    s0 = np.array([3.2e5, 0.0, 0.0, 0.0, 0.25, 0.05])  # L1-ish range, slow dynamics
    e.propagate_ephemeris(s0, t0, t0 + DAY, rtol=1e-10, atol=1e-10)  # warm numba/jit caches
    tic = time.perf_counter()
    sol = e.propagate_ephemeris(s0, t0, t0 + 7 * DAY, rtol=1e-10, atol=1e-10)
    dt = time.perf_counter() - tic
    assert sol.success
    assert dt < 2.0, f"7-day propagation took {dt:.3f} s (nfev={sol.nfev})"
    # also a fast-dynamics case (GEO, ~7 revolutions)
    sg = np.array([42_164.0, 0, 0, 0, np.sqrt(GM_EARTH / 42_164.0), 0])
    tic = time.perf_counter()
    sol = e.propagate_ephemeris(sg, t0, t0 + 7 * DAY, rtol=1e-10, atol=1e-10)
    dt = time.perf_counter() - tic
    assert dt < 2.0, f"7-day GEO propagation took {dt:.3f} s (nfev={sol.nfev})"
