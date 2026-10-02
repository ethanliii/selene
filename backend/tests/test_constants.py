import math

from selene import constants as c


def test_gm_values_match_de440():
    assert math.isclose(c.GM_EARTH, 398600.435507, rel_tol=1e-9)
    assert math.isclose(c.GM_MOON, 4902.800118, rel_tol=1e-9)
    assert math.isclose(c.GM_SUN, 1.32712440041e11, rel_tol=1e-9)


def test_cr3bp_normalisation():
    assert math.isclose(c.MU, 0.0121505856, rel_tol=1e-7)
    assert 375_000 < c.T_STAR < 375_400
    assert math.isclose(c.V_STAR, 1.0245, rel_tol=1e-3)
    assert math.isclose(c.SYNODIC_PERIOD_S / 86400, 27.28, rel_tol=1e-3)


def test_unit_roundtrip():
    assert math.isclose(c.km_to_nd(c.nd_to_km(0.3)), 0.3)
    assert math.isclose(c.kms_to_nd(c.nd_to_kms(1.1)), 1.1)
