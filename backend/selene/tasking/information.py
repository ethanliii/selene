"""Linearised covariance bookkeeping and expected information gain for sensor tasking.

This module is a *linear covariance analysis* engine for many objects at once: it never
simulates measurement noise realisations.  Each object carries a reference trajectory (the
catalog truth / current best estimate) and a 6x6 covariance ``P`` [km², km²/s, km²/s²] that is

* **propagated** between slot nodes with the ephemeris-model state transition matrix (STM)
  Φ(t_{k+1}, t_k) from :func:`selene.dynamics.ephemeris.propagate_ephemeris` (``stm=True``) plus
  continuous-white-noise-acceleration process noise, and
* **updated** by the linearised angles-only (RA/Dec) measurement of a chosen sensor.

Equations
---------
Prediction (Kalman / batch covariance analysis, e.g. Tapley, Schutz & Born 2004 §4.9)::

    P⁻_{k+1} = Φ P⁺_k Φᵀ + Q(Δt),
    Q(Δt) = q · [[Δt³/3 I, Δt²/2 I], [Δt²/2 I, Δt I]]          (Bar-Shalom, Li & Kirubarajan 2001, §6.2.2)

with ``q`` the (isotropic) acceleration power spectral density [km²/s³] standing in for
un-modelled accelerations (SRP mis-modelling, out-gassing, small un-announced thrusting).
Default ``q = 1e-18 km²/s³``: for a 1 h correlation time that is σ_a ≈ 1.7e-8 m/s², about 30 %
of the SRP acceleration on a 1500 kg / 12 m² bus (4.7e-8 m/s²) -- an *assumption*, not a
measured value.

Measurement: topocentric GCRF (ra, dec) of the object from the sensor.  The Jacobian rows are
scaled so the two components are on-sky angles with isotropic noise, matching the
``[d_ra·cos(dec), d_dec]`` innovation convention used by the OD track::

    H = [[cos(dec)·∂ra/∂r, 0₃], [∂dec/∂r, 0₃]]   (2x6, rad/km),   R = σ² I₂   (σ = sensor sigma, rad)

Update in Joseph form (numerically symmetric)::

    S = H P⁻ Hᵀ + R,  K = P⁻ Hᵀ S⁻¹,  P⁺_det = (I − K H) P⁻ (I − K H)ᵀ + K R Kᵀ

Acquisition (``acquisition='fov'``, the default): the object is only measured if it actually
falls inside the sensor's field of view when the sensor points at the predicted position.  With
an isotropic on-sky prediction error σ_θ = σ_⊥/ρ (σ_⊥ the RSS position sigma perpendicular to the
line of sight over √2, ρ the range) and a circular field of half-angle θ_f, the detection
probability is the Rayleigh CDF ``p = 1 − exp(−n_tiles·θ_f²/(2 σ_θ²))`` (``n_tiles`` fields
mosaicked around the prediction, default 1).  The *expected* posterior covariance is then the
Bernoulli mixture of the two outcomes, which share the same mean in a linear covariance analysis::

    P⁺ = p · P⁺_det + (1 − p) · P⁻          (Bar-Shalom, Li & Kirubarajan 2001 §6.6, PDAF with
                                             P_D = p, P_G = 1 and no clutter)

so the expected trace / max-eigenvalue reduction is exactly ``p`` times that of a certain
detection, and custody that is lost (σ_θ ≫ θ_f, p ≈ 0) **stays lost** under pointed observations
until a search (``n_tiles``) or another sensor with a wider field is used.  The older
"information reduction factor" model ``acquisition='irf'`` (R → R/p; expected Fisher
information) is kept as an option and is documented as **optimistic for small p**: it recovers a
1e5 km prior in two or three looks at p ≈ 1e-4, because the expected *information* is linear in
``p`` while the expected *covariance* is not.  ``acquisition='none'`` assumes p = 1 whenever the
object is visible.

Expected gain of one observation, selectable (all ≥ 0 by construction; the low-level default is
the information-theoretic ``'logdet'``, the schedulers default to ``'trace'`` because custody is a
position criterion -- see :mod:`selene.tasking.greedy`)::

    'logdet' : I = ½ [ln det P⁻ − ln det P⁺]            (mutual information, nats)
    'trace'  : ΔT = tr P⁻_pos − tr P⁺_pos               (km², position block)
    'maxeig' : Δλ = λ_max(P⁻_pos) − λ_max(P⁺_pos)       (km²)

Design note: because measurement noise is never sampled, the estimate *mean* stays on the
reference trajectory; all visibility and geometry is therefore evaluated on the truth track.
This is the standard assumption of covariance-based tasking studies and is stated in every
API response (``meta.assumptions``).  Units: km, s, km/s, rad.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Literal, Optional, Sequence

import numpy as np

from selene.dynamics.ephemeris import BodyCache, EphemParams, get_ephemeris, propagate_ephemeris
from selene.sensors.constraints import angular_separation
from selene.sensors.reasons import reason_names
from selene.sensors.visibility import ARCSEC, Sensor, evaluate, measurement_jacobian, measurement_model, sensor_geometry

__all__ = [
    "GainKind",
    "DEFAULT_Q_PSD",
    "ObjectTrack",
    "VisibilityTable",
    "build_tracks",
    "build_visibility_table",
    "symplectic_inverse",
    "process_noise",
    "predict_covariance",
    "angles_jacobian",
    "acquisition_probability",
    "update_covariance",
    "gain",
    "expected_gain",
    "sigma_pos_km",
    "slew_feasible",
]

GainKind = Literal["logdet", "trace", "maxeig"]
AcquisitionModel = Literal["fov", "irf", "none"]

#: default acceleration PSD [km²/s³] (see module docstring; an assumption)
DEFAULT_Q_PSD = 1e-18
#: default sensor astrometric noise [arcsec] (1 m-class telescope vs. Gaia frame; see visibility.py)
DEFAULT_SIGMA_ARCSEC = 1.0


# ---------------------------------------------------------------------------
# per-object bookkeeping
# ---------------------------------------------------------------------------
@dataclass
class ObjectTrack:
    """Reference trajectory + STM chain of one object on the slot grid.

    ``t_nodes`` (K+1,), ``x_nodes`` (K+1, 6) GCRF km/km/s, ``phi`` (K, 6, 6) with
    ``phi[k] = Φ(t_{k+1}, t_k)``, ``Q`` (K, 6, 6) process noise per step, ``P0`` prior covariance.
    """

    object_id: str
    t_nodes: np.ndarray
    x_nodes: np.ndarray
    phi: np.ndarray
    Q: np.ndarray
    P0: np.ndarray
    radius_m: float = 1.0
    albedo: float = 0.2
    priority: float = 1.0
    kind: str = "simulated"
    meta: dict = field(default_factory=dict)

    @property
    def n_slots(self) -> int:
        return len(self.t_nodes) - 1


def process_noise(dt_s: float, q_psd: float) -> np.ndarray:
    """Continuous white-noise-acceleration process noise (6x6) for a step of ``dt_s`` seconds."""
    dt = float(dt_s)
    I3 = np.eye(3)
    Q = np.zeros((6, 6))
    Q[:3, :3] = q_psd * dt**3 / 3.0 * I3
    Q[:3, 3:] = q_psd * dt**2 / 2.0 * I3
    Q[3:, :3] = Q[:3, 3:]
    Q[3:, 3:] = q_psd * dt * I3
    return Q


def predict_covariance(P: np.ndarray, phi: np.ndarray, Q: np.ndarray) -> np.ndarray:
    """P⁻ = Φ P Φᵀ + Q (symmetrised)."""
    Pn = phi @ P @ phi.T + Q
    return 0.5 * (Pn + Pn.T)


_J_SYMPLECTIC = np.block([[np.zeros((3, 3)), np.eye(3)], [-np.eye(3), np.zeros((3, 3))]])


def symplectic_inverse(Phi: np.ndarray) -> np.ndarray:
    """Φ⁻¹ = −J Φᵀ J for the STM of a Hamiltonian flow in canonical (r, v) coordinates.

    The point-mass N-body model (time-dependent potential, position-independent indirect and
    SRP terms) is Hamiltonian with H = ½|v|² + U(r, t), so its STM is symplectic and this inverse
    is exact up to the integrator's own symplecticity error (|det Φ − 1| ≈ 1e-10 here) -- no
    matrix inversion, hence no sensitivity to the km / km·s⁻¹ unit disparity.
    """
    return -_J_SYMPLECTIC @ Phi.T @ _J_SYMPLECTIC


def _stm_chain(x0: np.ndarray, t_nodes: np.ndarray, params: EphemParams, rtol: float, atol: float) -> tuple[np.ndarray, dict]:
    """Φ(t_{k+1}, t_k) for all k from a single STM propagation: Φ(k+1,k) = Φ(k+1,0) Φ(k,0)⁻¹ using
    the symplectic inverse.  ``det Φ = 1`` is used as a numerical health check; if it fails the
    chain is rebuilt segment by segment (identity reset at each node).  Verified against the
    segment-wise chain to ~1e-8 relative (1e-6 for a 2-h-period lunar orbiter) in the tests.
    """
    sol = propagate_ephemeris(x0, float(t_nodes[0]), t_eval_s=t_nodes, params=params, stm=True, rtol=rtol, atol=atol)
    if not sol.success or sol.y.shape[1] != t_nodes.size:
        raise RuntimeError(f"STM propagation failed: {sol.message}")
    Phi0 = sol.y[6:].T.reshape(-1, 6, 6)          # Φ(t_k, t_0)
    K = t_nodes.size - 1
    phi = np.empty((K, 6, 6))
    for k in range(K):
        phi[k] = Phi0[k + 1] @ symplectic_inverse(Phi0[k])
    dets = np.linalg.det(phi)
    info = {"method": "single_propagation_symplectic_inverse",
            "max_abs_det_minus_1": float(np.max(np.abs(dets - 1.0))), "nfev": int(sol.nfev)}
    if info["max_abs_det_minus_1"] > 1e-6:
        # segment-wise rebuild (identity reset each step) -- slower but unconditionally stable
        states = sol.y[:6].T
        for k in range(K):
            s = propagate_ephemeris(states[k], float(t_nodes[k]), tf_s=float(t_nodes[k + 1]), params=params,
                                    stm=True, rtol=rtol, atol=atol)
            if not s.success:
                raise RuntimeError(f"segment STM propagation failed: {s.message}")
            phi[k] = s.y[6:, -1].reshape(6, 6)
        info["method"] = "segmentwise"
        info["max_abs_det_minus_1_after"] = float(np.max(np.abs(np.linalg.det(phi) - 1.0)))
    return phi, info


def build_tracks(
    object_ids: Sequence[str],
    t_nodes: np.ndarray,
    *,
    q_psd: float = DEFAULT_Q_PSD,
    sigma0_pos_km: float = 10.0,
    sigma0_vel_kms: float = 1e-4,
    priorities: Optional[dict[str, float]] = None,
    catalog=None,
    rtol: float = 1e-10,
    atol: float = 1e-10,
) -> list[ObjectTrack]:
    """Reference states, STM chains and process noise for ``object_ids`` on ``t_nodes``.

    Prior covariance ``P0 = diag(σ_pos² I₃, σ_vel² I₃)`` (isotropic; a documented simplification --
    real post-fit covariances are strongly along-track-elongated).  Simulated objects use the
    catalog's own truth force model (Earth + Moon + Sun + SRP with the object's C_R·A/m) so the STM
    matches the truth dynamics; Horizons objects use the point-mass model without SRP.
    """
    from selene.objects.catalog import get_catalog  # local import: catalog is heavy and optional for unit tests

    cat = catalog or get_catalog()
    t_nodes = np.asarray(t_nodes, dtype=np.float64)
    cache = BodyCache(float(t_nodes[0]) - 86400.0, float(t_nodes[-1]) + 86400.0)
    priorities = priorities or {}
    P0 = np.diag([sigma0_pos_km**2] * 3 + [sigma0_vel_kms**2] * 3)
    dts = np.diff(t_nodes)
    Q = np.stack([process_noise(dt, q_psd) for dt in dts]) if dts.size else np.zeros((0, 6, 6))
    tracks = []
    for oid in object_ids:
        e = cat.get(oid)
        x_nodes = np.asarray(cat.state_at(oid, t_nodes), dtype=np.float64)
        if not np.all(np.isfinite(x_nodes)):
            raise ValueError(f"{oid}: reference trajectory is not available over the whole window")
        if e.notional is not None:
            tp = cat.truth(oid).params
            params = EphemParams(srp=tp.srp, cr_area_mass=tp.cr_area_mass, bodies=tp.bodies, cache=cache)
            radius_m, albedo = float(e.notional.physical.radius_m), float(e.notional.physical.albedo)
        else:
            params = EphemParams(cache=cache)
            radius_m, albedo = 1.0, 0.2  # unknown real object: generic 2 m-class assumption
        t_start = time.perf_counter()
        phi, info = _stm_chain(x_nodes[0], t_nodes, params, rtol, atol)
        info["stm_time_s"] = time.perf_counter() - t_start
        tracks.append(ObjectTrack(
            object_id=oid, t_nodes=t_nodes, x_nodes=x_nodes, phi=phi, Q=Q, P0=P0.copy(),
            radius_m=radius_m, albedo=albedo, priority=float(priorities.get(oid, 1.0)), kind=e.kind,
            meta={"orbit_type": e.orbit_type, "name": e.name, "label": e.label, "stm": info},
        ))
    return tracks


# ---------------------------------------------------------------------------
# visibility table on the slot grid
# ---------------------------------------------------------------------------
@dataclass
class VisibilityTable:
    """Visibility of every object from every sensor at every slot node (arrays ``(S, J, K+1)``).

    ``obs_pos`` (S, K+1, 3) sensor GCRF positions; ``los`` (S, J, K+1, 3) unit lines of sight
    (for slew feasibility); ``sigma_rad`` (S,) per-sensor astrometric noise; ``fov_half_rad`` (S,).
    """

    sensor_ids: list[str]
    object_ids: list[str]
    t_nodes: np.ndarray
    visible: np.ndarray
    reasons: np.ndarray
    magnitude: np.ndarray
    range_km: np.ndarray
    obs_pos: np.ndarray
    los: np.ndarray
    sigma_rad: np.ndarray
    fov_half_rad: np.ndarray
    slew_rate_rad_s: np.ndarray
    build_time_s: float = 0.0

    @property
    def n_sensors(self) -> int:
        return len(self.sensor_ids)

    @property
    def n_objects(self) -> int:
        return len(self.object_ids)

    def reasons_at(self, s: int, j: int, k: int) -> list[str]:
        return reason_names(int(self.reasons[s, j, k]))

    def fraction_visible(self) -> np.ndarray:
        """(S, J) fraction of slot nodes at which object j is visible from sensor s."""
        return self.visible[:, :, :-1].mean(axis=2) if self.visible.shape[2] > 1 else self.visible.mean(axis=2)


def build_visibility_table(
    sensors: Sequence[Sensor],
    tracks: Sequence[ObjectTrack],
    *,
    sigma_arcsec: float | dict[str, float] = DEFAULT_SIGMA_ARCSEC,
    margin_mag: float = 0.0,
) -> VisibilityTable:
    """Evaluate :func:`selene.sensors.visibility.evaluate` for all (sensor, object) pairs on the grid.

    Geometry is computed once per sensor (ground sites: astropy ITRS→GCRS, cached per grid) and
    the Sun/Moon once for the whole grid; cost is O(S·J·K) vectorised numpy, ≈ 0.3 s for
    9 × 8 × 145.
    """
    t_start = time.perf_counter()
    t_nodes = np.asarray(tracks[0].t_nodes, dtype=np.float64) if tracks else np.zeros(0)
    S, J, K1 = len(sensors), len(tracks), t_nodes.size
    eph = get_ephemeris()
    sun = eph.position("sun", t_nodes).reshape(-1, 3)
    moon = eph.position("moon", t_nodes).reshape(-1, 3)
    visible = np.zeros((S, J, K1), dtype=bool)
    reasons = np.zeros((S, J, K1), dtype=np.int64)
    mag = np.full((S, J, K1), np.inf)
    rng_km = np.zeros((S, J, K1))
    obs_pos = np.zeros((S, K1, 3))
    los = np.zeros((S, J, K1, 3))
    sig = np.zeros(S)
    fov = np.zeros(S)
    slew = np.zeros(S)
    for s, sensor in enumerate(sensors):
        pos, _vel, zen = sensor_geometry(sensor, t_nodes)
        obs_pos[s] = pos
        sig[s] = (sigma_arcsec.get(sensor.id, DEFAULT_SIGMA_ARCSEC) if isinstance(sigma_arcsec, dict) else float(sigma_arcsec)) * ARCSEC
        fov[s] = 0.5 * np.deg2rad(float(sensor.fov_deg))
        slew[s] = np.deg2rad(float(getattr(sensor, "slew_rate_deg_s", 1.0)))
        for j, tr in enumerate(tracks):
            tgt = tr.x_nodes[:, :3]
            mask, m, r, _phi = evaluate(sensor, pos, zen, tgt, sun, moon, tr.radius_m, tr.albedo, margin_mag)
            visible[s, j] = mask == 0
            reasons[s, j] = mask
            mag[s, j] = m
            rng_km[s, j] = r
            d = tgt - pos
            los[s, j] = d / np.linalg.norm(d, axis=1, keepdims=True)
    return VisibilityTable(
        sensor_ids=[x.id for x in sensors], object_ids=[tr.object_id for tr in tracks], t_nodes=t_nodes,
        visible=visible, reasons=reasons, magnitude=mag, range_km=rng_km, obs_pos=obs_pos, los=los,
        sigma_rad=sig, fov_half_rad=fov, slew_rate_rad_s=slew, build_time_s=time.perf_counter() - t_start,
    )


# ---------------------------------------------------------------------------
# measurement linearisation, acquisition probability, update, gain
# ---------------------------------------------------------------------------
def angles_jacobian(observer_pos: np.ndarray, target_pos: np.ndarray) -> np.ndarray:
    """2x6 on-sky Jacobian ``[[cos(dec)·∂ra/∂r, 0], [∂dec/∂r, 0]]`` (rad/km)."""
    H3 = measurement_jacobian(observer_pos, target_pos)   # (2,3)
    _ra, dec = measurement_model(observer_pos, target_pos)
    H = np.zeros((2, 6))
    H[0, :3] = np.cos(dec) * H3[0]
    H[1, :3] = H3[1]
    return H


def acquisition_probability(P: np.ndarray, los_unit: np.ndarray, range_km: float, fov_half_rad: float,
                            n_tiles: int = 1) -> float:
    """P(object inside a circular field of half-angle ``fov_half_rad`` when pointed at the mean).

    σ_θ² = ½ tr(P_⊥)/ρ² with P_⊥ the position covariance projected perpendicular to the line of
    sight (isotropic approximation), p = 1 − exp(−n_tiles·θ_f²/(2σ_θ²)).  ``n_tiles > 1`` models a
    mosaic of adjacent fields around the prediction (area scaling of a single field; a
    simplification of a real search pattern).
    """
    u = np.asarray(los_unit, dtype=np.float64)
    Ppos = P[:3, :3]
    proj = np.eye(3) - np.outer(u, u)
    var_perp = 0.5 * float(np.trace(proj @ Ppos @ proj))
    if var_perp <= 0.0:
        return 1.0
    sig_theta2 = var_perp / float(range_km) ** 2
    return float(1.0 - np.exp(-0.5 * max(1, int(n_tiles)) * fov_half_rad**2 / sig_theta2))


def update_covariance(P: np.ndarray, H: np.ndarray, sigma_rad: float, p_acq: float = 1.0, n_obs: int = 1,
                      model: AcquisitionModel = "fov") -> np.ndarray:
    """Expected posterior covariance of an angles-only observation.

    ``n_obs`` independent frames in the slot (R → R/n_obs).  ``p_acq`` is the detection
    probability; how it enters depends on ``model`` (see module docstring):

    * ``'fov'``  -- Bernoulli mixture ``p·P⁺_det + (1−p)·P⁻`` (expected covariance; default),
    * ``'irf'``  -- information reduction factor ``R → R/p`` (expected information; optimistic),
    * ``'none'`` -- certain detection (``p_acq`` ignored).
    """
    p = 1.0 if model == "none" else float(min(1.0, max(0.0, p_acq)))
    if p <= 1e-12:
        return P.copy()
    r2 = (sigma_rad**2) / float(n_obs)
    if model == "irf":
        r2 = r2 / p
    R = np.eye(2) * r2
    S = H @ P @ H.T + R
    K = np.linalg.solve(S.T, (P @ H.T).T).T        # P Hᵀ S⁻¹
    IKH = np.eye(6) - K @ H
    Pn = IKH @ P @ IKH.T + K @ R @ K.T
    Pn = 0.5 * (Pn + Pn.T)
    if model == "fov" and p < 1.0:
        Pn = p * Pn + (1.0 - p) * P
    return Pn


def _logdet(P: np.ndarray) -> float:
    sign, ld = np.linalg.slogdet(P)
    if sign <= 0:
        # fall back to eigenvalues clipped at a tiny floor (P should be SPD)
        w = np.clip(np.linalg.eigvalsh(P), 1e-300, None)
        return float(np.sum(np.log(w)))
    return float(ld)


def gain(P_minus: np.ndarray, P_plus: np.ndarray, kind: GainKind = "logdet") -> float:
    """Utility of going from ``P_minus`` to ``P_plus`` (≥ 0; tiny negative round-off clamped)."""
    if kind == "logdet":
        g = 0.5 * (_logdet(P_minus) - _logdet(P_plus))
    elif kind == "trace":
        g = float(np.trace(P_minus[:3, :3]) - np.trace(P_plus[:3, :3]))
    elif kind == "maxeig":
        g = float(np.linalg.eigvalsh(P_minus[:3, :3])[-1] - np.linalg.eigvalsh(P_plus[:3, :3])[-1])
    else:
        raise ValueError(f"unknown gain kind {kind!r}")
    return max(0.0, g)


def expected_gain(
    table: VisibilityTable, s: int, j: int, k: int, P: np.ndarray, x_target: np.ndarray, *,
    kind: GainKind = "logdet", acquisition: AcquisitionModel = "fov", n_obs: int = 1, n_tiles: int = 1,
) -> tuple[float, float, Optional[np.ndarray]]:
    """Expected gain of sensor ``s`` observing object ``j`` at node ``k`` given covariance ``P``.

    Returns ``(gain, p_acq, P_plus)``; ``(0.0, 0.0, None)`` when the object is not visible.
    """
    if not table.visible[s, j, k]:
        return 0.0, 0.0, None
    obs = table.obs_pos[s, k]
    H = angles_jacobian(obs, x_target[:3])
    p = 1.0
    if acquisition != "none":
        p = acquisition_probability(P, table.los[s, j, k], float(table.range_km[s, j, k]), float(table.fov_half_rad[s]), n_tiles)
    P_plus = update_covariance(P, H, float(table.sigma_rad[s]), p, n_obs, acquisition)
    return gain(P, P_plus, kind), p, P_plus


def sigma_pos_km(P: np.ndarray) -> float:
    """RSS position sigma sqrt(tr P_pos) [km]."""
    return float(np.sqrt(max(0.0, np.trace(P[:3, :3]))))


def slew_feasible(boresight: Optional[np.ndarray], los_new: np.ndarray, slew_rate_rad_s: float, elapsed_s: float,
                  fraction: float = 1.0) -> bool:
    """True if the sensor can turn from ``boresight`` to ``los_new`` within ``fraction`` of the time
    ``elapsed_s`` available for the slew.  The engine passes the time since the sensor's previous
    pointing (one slot for back-to-back observations, more after idle slots, so an idle sensor
    accumulates slew budget); ``boresight=None`` (no previous pointing) is always feasible."""
    if boresight is None:
        return True
    sep = float(angular_separation(boresight, los_new))
    return sep <= slew_rate_rad_s * elapsed_s * fraction + 1e-12
