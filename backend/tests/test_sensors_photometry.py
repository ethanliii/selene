import numpy as np
import pytest

from selene.sensors.photometry import (
    M_SUN,
    detectable,
    lambertian_phase_function,
    phase_angle,
    visual_magnitude,
)


def _lambert_sphere_flux_ratio(albedo: float, rho: float, R: float, phi: float, n: int = 400) -> float:
    """Derivation-independent reference: direct surface integration of a Lambertian sphere.

    BRDF = albedo/pi (sr^-1).  Sun irradiance E (set to 1) along direction s, observer along o at
    distance R >> rho; the reflected flux at the observer is  sum_i (a/pi) E (n_i.s)+ (n_i.o)+ dA_i / R^2
    and the Sun's own flux at the observer is E, so the ratio is independent of E.  This uses
    nothing from the module under test (no phase function, no cross-section convention).
    """
    o = np.array([0.0, 0.0, 1.0])
    s = np.array([np.sin(phi), 0.0, np.cos(phi)])
    th = (np.arange(n) + 0.5) / n * np.pi
    ph = (np.arange(2 * n) + 0.5) / (2 * n) * 2 * np.pi
    TH, PH = np.meshgrid(th, ph, indexing="ij")
    normals = np.stack([np.sin(TH) * np.cos(PH), np.sin(TH) * np.sin(PH), np.cos(TH)], axis=-1)
    cos_s = np.clip(normals @ s, 0.0, None)
    cos_o = np.clip(normals @ o, 0.0, None)
    dA = rho**2 * np.sin(TH) * (np.pi / n) * (2 * np.pi / (2 * n))
    return float(np.sum((albedo / np.pi) * cos_s * cos_o * dA) / R**2)


def test_phase_function_endpoints():
    assert lambertian_phase_function(0.0) == pytest.approx(2.0 / (3.0 * np.pi), rel=1e-12)
    assert lambertian_phase_function(np.pi) == pytest.approx(0.0, abs=1e-15)
    # monotonically decreasing in phase angle
    p = lambertian_phase_function(np.linspace(0, np.pi, 50))
    assert np.all(np.diff(p) < 0)


def test_phase_function_normalisation_per_projected_area():
    # Energy conservation: a Lambertian sphere reflects albedo * (pi rho^2) * E in total, so with
    # p(phi) defined per unit *projected* area (A = pi rho^2) the integral of p over 4pi sr is 1.
    n = 20000
    phi = (np.arange(n) + 0.5) / n * np.pi          # midpoint rule, dOmega = 2 pi sin(phi) dphi
    integral = float(np.sum(lambertian_phase_function(phi) * 2 * np.pi * np.sin(phi)) * (np.pi / n))
    assert integral == pytest.approx(1.0, rel=1e-6)


def test_visual_magnitude_hand_calculation():
    # 1 m radius, Lambert albedo 0.2, zero phase, 400 000 km -- by hand (see module docstring):
    #   flux ratio = a * pi * rho^2 * p(0) / R^2 = (2a/3) rho^2 / R^2  (geometric albedo 2a/3)
    #              = 0.133333 * 1e-6 km^2 / 1.6e11 km^2 = 8.3333e-19
    #   m = -26.74 - 2.5 log10(8.3333e-19) = -26.74 + 45.198 = 18.458
    rho_km, R_km, a = 1.0e-3, 4.0e5, 0.2
    expected = M_SUN - 2.5 * np.log10((2.0 * a / 3.0) * rho_km**2 / R_km**2)
    got = float(visual_magnitude(1.0, a, 0.0, R_km))
    assert got == pytest.approx(expected, abs=1e-9)
    assert got == pytest.approx(18.46, abs=0.01)   # explicit docstring value
    # Published xGEO detectability statements put ~1-2 m objects at lunar distance near 18-20 mag.
    assert 17.0 <= got <= 22.0


@pytest.mark.parametrize("phi_deg", [0.0, 30.0, 60.0, 90.0, 135.0])
def test_visual_magnitude_matches_surface_integration(phi_deg):
    # Derivation-independent check of the whole formula (normalisation + phase dependence)
    # against a brute-force surface integral of a Lambertian sphere.
    a, rho_km, R_km = 0.3, 1.5e-3, 3.0e5
    phi = np.deg2rad(phi_deg)
    ref = M_SUN - 2.5 * np.log10(_lambert_sphere_flux_ratio(a, rho_km, R_km, phi))
    got = float(visual_magnitude(1.5, a, phi, R_km))
    assert got == pytest.approx(ref, abs=2e-3), f"phi={phi_deg}: {got} vs integral {ref}"


def test_specular_sphere_cross_check():
    # Independent anchor from the same family of models: a perfectly specular (mirror) sphere of
    # radius rho scatters uniformly into 4pi sr, giving flux ratio a rho^2 / (4 R^2); its diffuse
    # counterpart at zero phase is brighter by 4 * (2/3) = 8/3 (p_g = 2a/3 vs 1/4), i.e. 1.065 mag.
    a, rho_km, R_km = 0.2, 1e-3, 4e5
    m_spec = M_SUN - 2.5 * np.log10(a * rho_km**2 / (4 * R_km**2))
    m_diff = float(visual_magnitude(1.0, a, 0.0, R_km))
    assert m_spec - m_diff == pytest.approx(2.5 * np.log10(8.0 / 3.0), abs=1e-9)


def test_magnitude_scaling_laws():
    m1 = visual_magnitude(1.0, 0.2, 0.0, 4e5)
    # doubling range -> +1.505 mag ; doubling radius -> -1.505 mag ; halving albedo -> +0.753
    assert visual_magnitude(1.0, 0.2, 0.0, 8e5) - m1 == pytest.approx(5 * np.log10(2), abs=1e-9)
    assert visual_magnitude(2.0, 0.2, 0.0, 4e5) - m1 == pytest.approx(-5 * np.log10(2), abs=1e-9)
    assert visual_magnitude(1.0, 0.1, 0.0, 4e5) - m1 == pytest.approx(2.5 * np.log10(2), abs=1e-9)
    # dark side is infinitely faint
    assert np.isinf(visual_magnitude(1.0, 0.2, np.pi, 4e5))


def test_phase_angle_geometry():
    sun = np.array([1.5e8, 0.0, 0.0])
    tgt = np.array([4e5, 0.0, 0.0])
    # observer on the Sun side of the target -> phase ~0 ; behind the target -> phase ~pi
    assert phase_angle(sun, np.array([5e5, 0, 0]), tgt) == pytest.approx(0.0, abs=1e-12)
    assert phase_angle(sun, np.array([0.0, 0, 0]), tgt) == pytest.approx(np.pi, abs=1e-12)
    # observer at right angle -> pi/2
    assert phase_angle(sun, np.array([4e5, 1e5, 0]), tgt) == pytest.approx(np.pi / 2, rel=1e-6)
    # broadcasting
    out = phase_angle(sun, np.zeros((7, 1, 3)), np.tile(tgt, (1, 5, 1)))
    assert out.shape == (7, 5)


def test_detectable_with_margin():
    m = np.array([18.0, 19.0, 20.0])
    assert detectable(m, 19.5).tolist() == [True, True, False]
    assert detectable(m, 19.5, margin_mag=1.0).tolist() == [True, False, False]
