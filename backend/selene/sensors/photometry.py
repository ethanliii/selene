"""Photometric detectability of a diffusely reflecting sphere.

Model (PLAN §2.5)
-----------------
Diffuse (Lambertian) sphere.  With the Hejduk / Krag / McCue convention the reflected flux is
written as *albedo × cross-sectional area × phase function*, where the phase function is
normalised per unit **projected area** A = π ρ²::

    p(φ) = (2 / (3 π²)) · [ (π − φ) cos φ + sin φ ],       p(0) = 2/(3π) ≈ 0.2122,  p(π) = 0

Apparent visual magnitude of a sphere of radius ρ and Lambertian (Bond/normal) albedo ``a`` at
range R from the observer, illuminated by the Sun (m☉ = −26.74 at 1 AU; the object–Sun distance is
taken as 1 AU, which for cislunar objects is correct to < 0.3 % in distance, < 0.006 mag)::

    m = m☉ − 2.5 log10( a · π ρ² · p(φ) / R² )

The factor π is the cross-section: ``a · π ρ² · p(φ)`` is the effective reflecting area.  At zero
phase ``a π p(0) = 2a/3``, i.e. the geometric albedo of a Lambertian sphere is p_g = 2a/3
(classical result, e.g. Russell 1916; Hapke 2012 §12), so the formula is identical to the
planetary form ``m = m☉ − 2.5 log10( p_g Φ(φ) ρ² / R² )`` with Φ(0) = 1.  Note the distinction:
``albedo`` here is the surface (Lambert) albedo, **not** the geometric albedo; a 0.2 Lambert
albedo corresponds to p_g ≈ 0.133.  Energy check: ∫ p(φ) dΩ over the sphere equals 1 only with
A = π ρ², which is what the unit tests verify by direct surface integration.

ρ and R **must be in the same length unit**; the public function takes ρ in metres and R in
kilometres and converts internally.  References: Hejduk, "Specular and diffuse components in
spherical satellite photometric modeling", AMOS 2011 (eq. 1–3: F_diff with A = π r²); Krag,
"Visible magnitude of typical satellites in synchronous orbits", 1974; McCue, Williams & Morford,
Planet. Space Sci. 19 (1971) 851.

Hand-checked example (``radius_m=1``, ``albedo=0.2``, ``phi=0``, ``range_km=400_000``)::

    a π ρ² p(0) / R² = 0.2 · π · 1 m² · 0.21221 / (4.0e8 m)² = 0.133333 / 1.6e17 = 8.3333e-19
    m = −26.74 − 2.5 log10(8.3333e-19) = −26.74 − 2.5 · (−18.0792) = −26.74 + 45.198 = 18.46

i.e. a 2 m diameter, 20 %-albedo object at lunar distance is a ~18.5 mag target at zero phase
(~19.7 mag at 90° phase), consistent with public statements that xGEO custody needs ~1 m-class
apertures reaching ~19–20 mag (see e.g. AFRL Oracle and the Space Force Cislunar Highway Patrol
System descriptions).

Not modelled: specular glints, shape/attitude-dependent lightcurves, atmospheric extinction and
sky brightness beyond the per-site limiting magnitude, and the opposition surge.
"""
from __future__ import annotations

import numpy as np

__all__ = [
    "M_SUN",
    "phase_angle",
    "lambertian_phase_function",
    "visual_magnitude",
    "detectable",
]

#: Apparent V magnitude of the Sun at 1 AU: −26.74 (Cox 2000, Allen's Astrophysical Quantities,
#: 4th ed., Table 14.2; Willmer 2018, ApJS 236, 47 gives −26.76).  Not an IAU-defined quantity —
#: IAU 2015 Resolution B2 fixes only the *bolometric* zero points (m_bol,☉ = −26.832).
M_SUN = -26.74


def _unit(v):
    v = np.asarray(v, dtype=np.float64)
    return v / np.linalg.norm(v, axis=-1, keepdims=True)


def phase_angle(r_sun, r_obs, r_tgt) -> np.ndarray:
    """Sun–target–observer phase angle φ [rad].

    All inputs are positions in a common frame (km), broadcastable to a common ``(..., 3)``
    shape.  φ = 0 means the target is fully illuminated as seen by the observer (opposition);
    φ = π means the observer looks at the dark side (target between observer and Sun).
    """
    r_sun = np.asarray(r_sun, dtype=np.float64)
    r_obs = np.asarray(r_obs, dtype=np.float64)
    r_tgt = np.asarray(r_tgt, dtype=np.float64)
    to_sun = r_sun - r_tgt
    to_obs = r_obs - r_tgt
    cross = np.linalg.norm(np.cross(to_sun, to_obs), axis=-1)
    dot = np.sum(to_sun * to_obs, axis=-1)
    return np.arctan2(cross, dot)


def lambertian_phase_function(phi) -> np.ndarray:
    """Diffuse-sphere phase function p(φ) = (2/(3π²))[(π−φ)cosφ + sinφ], φ in rad, φ ∈ [0, π]."""
    phi = np.clip(np.asarray(phi, dtype=np.float64), 0.0, np.pi)
    return (2.0 / (3.0 * np.pi**2)) * ((np.pi - phi) * np.cos(phi) + np.sin(phi))


def visual_magnitude(radius_m, albedo, phi, range_km) -> np.ndarray:
    """Apparent visual magnitude of a Lambertian sphere, m = m☉ − 2.5 log10(a π ρ² p(φ) / R²).

    Parameters
    ----------
    radius_m : object radius ρ [m]
    albedo   : Lambert (Bond/normal) surface albedo ``a`` (dimensionless, 0–1); the geometric
               albedo of the sphere is 2a/3
    phi      : phase angle [rad]
    range_km : observer–target range [km]

    Returns ``+inf`` where the phase function is zero (φ = π) or the inputs are degenerate.

    >>> float(np.round(visual_magnitude(1.0, 0.2, 0.0, 4.0e5), 2))
    18.46
    """
    radius_km = np.asarray(radius_m, dtype=np.float64) * 1e-3
    albedo = np.asarray(albedo, dtype=np.float64)
    rng = np.asarray(range_km, dtype=np.float64)
    p = lambertian_phase_function(phi)
    p = np.where(p < 1e-12, 0.0, p)  # p(π) is ~1e-17 in floating point; treat as dark (+inf mag)
    with np.errstate(divide="ignore", invalid="ignore"):
        flux_ratio = albedo * np.pi * radius_km**2 * p / rng**2   # A = π ρ² (cross-section)
        m = M_SUN - 2.5 * np.log10(flux_ratio)
    return np.where(np.isfinite(m), m, np.inf)


def detectable(magnitude, limiting_mag, margin_mag: float = 0.0) -> np.ndarray:
    """``True`` where ``magnitude <= limiting_mag - margin_mag`` (brighter is numerically smaller).

    ``margin_mag`` is an optional SNR safety margin (e.g. 0.5 mag) for conservative scheduling.
    """
    return np.asarray(magnitude) <= (np.asarray(limiting_mag, dtype=np.float64) - float(margin_mag))
