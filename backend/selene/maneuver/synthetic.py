"""Synthetic truth-with-burn and measurement generation for maneuver tests and the API route.

Everything here is SIMULATED: burns are injected into a notional object's ephemeris-model
truth and attributed to a "notional actor".  Units: km, km/s, TDB seconds past J2000.

Local frames
------------
For a centre body c (Earth or Moon) with relative state (r, v):

* RTN: R̂ = r/|r|, N̂ = (r×v)/|r×v|, T̂ = N̂×R̂   (radial, transverse ≈ along-track, normal)
* VNB: V̂ = v/|v|, N̂ as above, B̂ = V̂×N̂            (velocity, normal, bi-normal)

Named injection directions (unit vectors in GCRF at the burn epoch): ``prograde``/``retrograde``
(±V̂), ``radial_out``/``radial_in`` (±R̂), ``normal``/``anti_normal`` (±N̂), ``toward_moon``,
``toward_earth``, ``toward_l1``, ``toward_l2`` (unit vector from the object to that point).
The frame centre follows :class:`EstimatorConfig.center` ('auto' = Moon when within
``moon_primary_radius_km`` of the Moon).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional, Sequence

import numpy as np

from selene.constants import GM_EARTH, GM_MOON
from selene.dynamics import frames
from selene.dynamics.cr3bp import lagrange_points
from selene.dynamics.ephemeris import EphemParams, get_ephemeris
from selene.dynamics.propagate import Trajectory, propagate
from selene.sensors.visibility import ARCSEC, Measurement, Sensor, observe, measurement_model

__all__ = [
    "Burn", "PiecewiseTruth", "frame_center", "rtn_basis", "vnb_basis", "direction_unit",
    "truth_with_burn", "generate_measurements", "renoise", "sample_initial_state", "initial_covariance",
]

DIRECTION_NAMES = ("prograde", "retrograde", "radial_out", "radial_in", "normal", "anti_normal",
                   "toward_moon", "toward_earth", "toward_l1", "toward_l2")


@dataclass
class Burn:
    """Impulsive Δv [km/s, GCRF components] applied at TDB ``t_s``."""

    t_s: float
    dv_kms: np.ndarray

    def __post_init__(self):
        self.dv_kms = np.asarray(self.dv_kms, dtype=np.float64).ravel()[:3]

    @property
    def magnitude_mps(self) -> float:
        return float(np.linalg.norm(self.dv_kms) * 1e3)

    def as_dict(self) -> dict:
        return {"t_s": float(self.t_s), "dv_kms_gcrf": self.dv_kms.tolist(), "magnitude_mps": self.magnitude_mps}


# ---------------------------------------------------------------------------
# local frames
# ---------------------------------------------------------------------------
def frame_center(x_gcrf: np.ndarray, t_s: float, center: str = "auto", moon_radius_km: float = 100_000.0) -> str:
    if center in ("earth", "moon"):
        return center
    r_moon = get_ephemeris().position("moon", float(t_s))
    return "moon" if np.linalg.norm(np.asarray(x_gcrf)[:3] - r_moon) < moon_radius_km else "earth"


def _relative_state(x_gcrf: np.ndarray, t_s: float, center: str) -> np.ndarray:
    x = np.asarray(x_gcrf, dtype=np.float64)
    if center == "moon":
        return x - get_ephemeris().moon_state(float(t_s))
    return x.copy()


def rtn_basis(x_gcrf: np.ndarray, t_s: float, center: str = "earth") -> np.ndarray:
    """Rows R̂, T̂, N̂ (3,3) in GCRF for the state relative to ``center``."""
    s = _relative_state(x_gcrf, t_s, center)
    r, v = s[:3], s[3:]
    R = r / np.linalg.norm(r)
    h = np.cross(r, v)
    N = h / np.linalg.norm(h)
    T = np.cross(N, R)
    return np.vstack([R, T, N])


def vnb_basis(x_gcrf: np.ndarray, t_s: float, center: str = "earth") -> np.ndarray:
    """Rows V̂, N̂, B̂ (3,3) in GCRF for the state relative to ``center``."""
    s = _relative_state(x_gcrf, t_s, center)
    r, v = s[:3], s[3:]
    V = v / np.linalg.norm(v)
    h = np.cross(r, v)
    N = h / np.linalg.norm(h)
    B = np.cross(V, N)
    return np.vstack([V, N, B])


def direction_unit(name: str, x_gcrf: np.ndarray, t_s: float, center: str = "auto",
                   moon_radius_km: float = 100_000.0) -> np.ndarray:
    """GCRF unit vector for a named burn direction (see module docstring)."""
    name = name.lower()
    c = frame_center(x_gcrf, t_s, center, moon_radius_km)
    x = np.asarray(x_gcrf, dtype=np.float64)
    if name in ("prograde", "retrograde"):
        u = vnb_basis(x, t_s, c)[0]
        return u if name == "prograde" else -u
    if name in ("radial_out", "radial_in"):
        u = rtn_basis(x, t_s, c)[0]
        return u if name == "radial_out" else -u
    if name in ("normal", "anti_normal"):
        u = rtn_basis(x, t_s, c)[2]
        return u if name == "normal" else -u
    if name == "toward_earth":
        d = -x[:3]
    elif name == "toward_moon":
        d = get_ephemeris().position("moon", float(t_s)) - x[:3]
    elif name in ("toward_l1", "toward_l2"):
        L = lagrange_points()
        idx = 0 if name == "toward_l1" else 1
        d = frames.rot_pos_to_gcrf(L[idx], float(t_s)) - x[:3]
    else:
        raise ValueError(f"unknown direction {name!r}; choose from {DIRECTION_NAMES}")
    return d / np.linalg.norm(d)


# ---------------------------------------------------------------------------
# truth
# ---------------------------------------------------------------------------
@dataclass
class PiecewiseTruth:
    """Dense truth made of contiguous ephemeris arcs; ``at(t)`` evaluates the right arc."""

    arcs: list[Trajectory]
    burn: Optional[Burn] = None
    params: EphemParams = field(default_factory=EphemParams)

    @property
    def t0(self) -> float:
        return float(self.arcs[0].t_s[0])

    @property
    def t1(self) -> float:
        return float(self.arcs[-1].t_s[-1])

    def at(self, t_s) -> np.ndarray:
        t = np.asarray(t_s, dtype=np.float64)
        scalar = t.ndim == 0
        tt = np.atleast_1d(t)
        out = np.empty((tt.size, 6))
        for i, a in enumerate(self.arcs):
            lo, hi = float(a.t_s[0]), float(a.t_s[-1])
            if i == len(self.arcs) - 1:
                m = tt >= lo - 1e-6
            else:
                m = (tt >= lo - 1e-6) & (tt < hi)
            if m.any():
                out[m] = a.at(tt[m])
        return out[0] if scalar else out

    __call__ = at


def truth_with_burn(x0, t0_s: float, t1_s: float, burn: Burn | None = None,
                    params: EphemParams | None = None, rtol: float = 1e-11, atol: float = 1e-11) -> PiecewiseTruth:
    """Ephemeris-model truth from ``x0`` at ``t0_s`` to ``t1_s`` with an optional impulsive burn.

    The pre-burn arc is integrated to the burn epoch, Δv is added to the velocity, and the
    post-burn arc continues to ``t1_s``; both arcs keep the integrator's dense output.
    """
    params = params or EphemParams()
    x0 = np.asarray(x0, dtype=np.float64).ravel()[:6]
    arcs: list[Trajectory] = []
    if burn is None or not (t0_s < burn.t_s < t1_s) or np.linalg.norm(burn.dv_kms) == 0.0:
        arcs.append(propagate(x0, t0_s, np.array([t0_s, t1_s]), params=params, rtol=rtol, atol=atol, keep_dense=True))
        return PiecewiseTruth(arcs, None if burn is None else burn, params)
    pre = propagate(x0, t0_s, np.array([t0_s, burn.t_s]), params=params, rtol=rtol, atol=atol, keep_dense=True)
    xb = pre.at(burn.t_s).copy()
    xb[3:] += burn.dv_kms
    post = propagate(xb, burn.t_s, np.array([burn.t_s, t1_s]), params=params, rtol=rtol, atol=atol, keep_dense=True)
    return PiecewiseTruth([pre, post], burn, params)


# ---------------------------------------------------------------------------
# measurements
# ---------------------------------------------------------------------------
def generate_measurements(truth: Callable[[float], np.ndarray], epochs: Sequence[float], sensors: Sequence[Sensor],
                          sigma_arcsec: float = 1.0, rng=None, radius_m: float = 1.0, albedo: float = 0.2,
                          max_per_epoch: int = 1) -> list[Measurement]:
    """Visible-only noisy RA/Dec measurements at ``epochs`` from the first ``max_per_epoch``
    sensors (in the given order) that can see the target.  Uses the catalog photometry
    (``radius_m``, ``albedo``) so faint/glare/eclipse epochs are *dropped*, which is how gaps
    arise naturally in the synthetic runs."""
    rng = np.random.default_rng(0) if rng is None else rng
    out: list[Measurement] = []
    for t in epochs:
        x = np.asarray(truth(float(t)), dtype=np.float64).ravel()
        n_here = 0
        for s in sensors:
            m = observe(s, x, float(t), sigma_arcsec=sigma_arcsec, rng=rng, target_radius_m=radius_m,
                        albedo=albedo, require_visible=True)
            if m is not None:
                out.append(m)
                n_here += 1
                if n_here >= max_per_epoch:
                    break
    return out


def renoise(meas: Sequence[Measurement], rng, sigma_arcsec: float | None = None) -> list[Measurement]:
    """Fresh noise realisation of a measurement set (noiseless truth kept in ``m.truth``),
    reproducing :func:`observe`'s isotropic on-sky noise.  Cheap: no geometry is recomputed."""
    out = []
    for m in meas:
        sig = float(sigma_arcsec) * ARCSEC if sigma_arcsec is not None else float(m.sigma_rad)
        ra0, dec0 = m.truth.get("ra_rad"), m.truth.get("dec_rad")
        if ra0 is None:
            ra0, dec0 = measurement_model(m.observer_pos_gcrf, m.observer_pos_gcrf)  # pragma: no cover
        n1, n2 = rng.standard_normal(2)
        dec_m = float(dec0) + sig * n1
        ra_m = float(np.mod(ra0 + sig * n2 / np.cos(dec_m), 2.0 * np.pi))
        out.append(Measurement(t_s=m.t_s, sensor_id=m.sensor_id, ra_rad=ra_m, dec_rad=dec_m, sigma_rad=sig,
                               observer_pos_gcrf=m.observer_pos_gcrf, observer_vel_gcrf=m.observer_vel_gcrf,
                               visible=m.visible, magnitude=m.magnitude, reasons=m.reasons,
                               light_time_s=m.light_time_s, truth=dict(m.truth)))
    return out


def initial_covariance(sigma_pos_km: float = 20.0, sigma_vel_mps: float = 2.0) -> np.ndarray:
    P0 = np.zeros((6, 6))
    P0[:3, :3] = np.eye(3) * sigma_pos_km**2
    P0[3:, 3:] = np.eye(3) * (sigma_vel_mps * 1e-3) ** 2
    return P0


def sample_initial_state(x_true, P0, rng) -> np.ndarray:
    """Draw x0 ~ N(x_true, P0) so the filter starts statistically consistent."""
    L = np.linalg.cholesky(np.asarray(P0, dtype=np.float64))
    return np.asarray(x_true, dtype=np.float64).ravel()[:6] + L @ rng.standard_normal(6)
