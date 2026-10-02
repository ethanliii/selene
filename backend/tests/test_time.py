import math

from selene import time as st


def test_j2000_epoch():
    # J2000.0 = 2000-01-01T12:00:00 TT = 11:58:55.816 UTC
    jd = st.utc_to_tdb_jd("2000-01-01T11:58:55.816")
    assert abs(jd - st.J2000_JD) < 1e-7  # < 10 ms


def test_roundtrip():
    iso = "2026-10-02T00:00:00.000"
    jd = st.utc_to_tdb_jd(iso)
    assert st.tdb_jd_to_utc_iso(jd).startswith("2026-10-02T00:00:00")


def test_grid():
    g = st.utc_grid("2026-10-02T00:00:00", "2026-10-03T00:00:00", 25)
    assert len(g) == 25
    assert math.isclose(g[-1] - g[0], 1.0, abs_tol=1e-9)
