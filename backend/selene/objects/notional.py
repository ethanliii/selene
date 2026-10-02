"""SIMULATED spacecraft catalogue (notional objects on published cislunar orbit types).

Every object in this module is **SIMULATED**.  Events, states and physical parameters are
attributed to a "notional actor" or a "notional allied operator" and never to a real country,
operator or spacecraft.  Physical parameters (bus radius, albedo, area, mass, C_R) are *design
assumptions* chosen to be representative of small/medium spacecraft buses and are flagged as
notional in every API response.

Placement
---------
Each object references either

* a record of the CR3BP periodic-orbit library (``orbit_ref`` = library record id), placed at
  ``phase`` (fraction of the period measured from the library initial condition, which is the
  perpendicular xz-plane crossing farther from the Moon), or
* a Moon-centred Keplerian element set (``orbit_ref = 'keplerian_moon'``), used for the
  elliptical lunar frozen orbit (ELFO).

The state at the demo epoch (:data:`selene.dynamics.frames.DEMO_EPOCH_TDB_S`) is obtained by
propagating the CR3BP initial condition by ``phase * period`` and mapping the rotating-frame
state to GCRF with :func:`selene.dynamics.frames.rot_to_gcrf` (see
:func:`initial_state_gcrf`).  Units: km, km/s, s, rad (degrees only in element specs).

References
----------
* Ely, T. A. (2005), "Stable Constellations of Frozen Elliptical Inclined Lunar Orbits",
  *J. Astronautical Sciences* 53(3):301-316 -- frozen ELFO a ≈ 6541 km, e = 0.6, i = 56.2°,
  ω = 90° (the Lidov-Kozai frozen condition for the Earth third-body perturbation).
* Ely & Lieb (2006), "Constellations of elliptical inclined lunar orbits providing polar and
  global coverage", *J. Astronautical Sciences* 54(1) -- same element set for a south-pole
  communications constellation.
* Zimovan-Spreen, Howell & Davis (2020), CMDA 132:28 -- 9:2 synodic-resonant L2 southern NRHO.
* Whitley & Martinez (2016), "Options for staging orbits in cislunar space", IEEE Aerospace --
  DRO / NRHO / halo staging orbit comparison used to choose the orbit classes represented here.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np

from selene.constants import GM_MOON, R_MOON
from selene.dynamics import frames
from selene.dynamics.cr3bp import propagate_cr3bp
from selene.orbits.library import OrbitRecord, get_library

__all__ = [
    "Physical",
    "KeplerianMoon",
    "NotionalObject",
    "NOTIONAL_OBJECTS",
    "get_notional",
    "notional_ids",
    "cr3bp_phase_state",
    "keplerian_moon_state_gcrf",
    "initial_state_gcrf",
    "reference_rot_state",
    "SIMULATED_LABEL",
    "NOTIONAL_ACTOR",
    "NOTIONAL_ALLY",
]

SIMULATED_LABEL = "SIMULATED"
NOTIONAL_ACTOR = "notional actor"
NOTIONAL_ALLY = "notional allied operator"


# ---------------------------------------------------------------------------
# physical / orbit specs
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Physical:
    """Notional photometric and SRP parameters.

    radius_m   characteristic (equivalent-sphere) radius used by the magnitude model [m]
    albedo     Bond albedo used by the diffuse-sphere magnitude model
    area_m2    SRP cross-section [m²] (bus + arrays, orientation-averaged)
    mass_kg    mass [kg]
    cr         radiation-pressure coefficient (1 = perfect absorber, 2 = perfect reflector)
    """

    radius_m: float
    albedo: float
    area_m2: float
    mass_kg: float
    cr: float = 1.3

    @property
    def cr_area_mass(self) -> float:
        """C_R · A/m [m²/kg] as consumed by :class:`selene.dynamics.ephemeris.EphemParams`."""
        return self.cr * self.area_m2 / self.mass_kg

    def as_dict(self) -> dict:
        d = asdict(self)
        d["cr_area_mass_m2_kg"] = self.cr_area_mass
        d["note"] = "notional design values (SIMULATED object)"
        return d


@dataclass(frozen=True)
class KeplerianMoon:
    """Moon-centred osculating elements at the demo epoch [km, deg].

    Reference plane: the instantaneous Earth-Moon orbital plane (x̂ toward the Moon from the
    Earth, i.e. the rotating-frame x̂, ẑ = lunar orbit-pole), which is the plane that governs the
    Earth third-body frozen condition of Ely (2005).  The resulting Moon-centred state is
    expressed in GCRF axes and treated as inertial from the epoch on (osculating elements).
    """

    a_km: float
    e: float
    i_deg: float
    raan_deg: float = 0.0
    argp_deg: float = 90.0
    nu_deg: float = 0.0
    reference_plane: str = "earth_moon_orbit_plane"

    @property
    def perilune_km(self) -> float:
        return self.a_km * (1.0 - self.e)

    @property
    def apolune_km(self) -> float:
        return self.a_km * (1.0 + self.e)

    @property
    def period_s(self) -> float:
        return 2.0 * np.pi * np.sqrt(self.a_km**3 / GM_MOON)

    def as_dict(self) -> dict:
        d = asdict(self)
        d.update(perilune_km=self.perilune_km, apolune_km=self.apolune_km, period_s=self.period_s,
                 perilune_alt_km=self.perilune_km - R_MOON)
        return d


@dataclass(frozen=True)
class NotionalObject:
    id: str
    name: str
    orbit_type: str                 # human label: 'DRO', '9:2 NRHO', 'L1 northern halo', 'ELFO', ...
    orbit_ref: str                  # library record id, or 'keplerian_moon'
    phase: float                    # fraction of the period at the demo epoch (library orbits)
    physical: Physical
    description: str
    role: str
    kind: str = "simulated"
    actor: str = NOTIONAL_ACTOR
    kepler: KeplerianMoon | None = None
    #: True: fit the epoch velocity (Gauss-Newton on the STM, see
    #: :func:`selene.objects.catalog.fit_epoch_velocity`) so the uncontrolled ephemeris-model
    #: truth shadows the CR3BP reference over the catalog window -- used for the strongly
    #: unstable libration-point orbits (halos, Lyapunov; stability index ≫ 1), whose raw CR3BP
    #: state departs the orbit (or impacts the Moon) within ~10 days in the real force model.
    #: False: raw CR3BP mapping (DROs, NRHOs, resonant orbits: stable or nearly so).
    ephem_fit: bool = True
    tags: tuple[str, ...] = ()

    @property
    def is_library_orbit(self) -> bool:
        return self.orbit_ref not in ("keplerian_moon",)

    def record(self) -> OrbitRecord:
        if not self.is_library_orbit:
            raise ValueError(f"{self.id} is not placed on a library orbit ({self.orbit_ref})")
        return get_library()[self.orbit_ref]

    def period_s(self) -> float:
        if self.is_library_orbit:
            return float(self.record().period_days) * 86400.0
        assert self.kepler is not None
        return float(self.kepler.period_s)

    def as_dict(self) -> dict:
        d = {
            "id": self.id,
            "name": self.name,
            "kind": self.kind,
            "label": SIMULATED_LABEL,
            "actor": self.actor,
            "orbit_type": self.orbit_type,
            "orbit_ref": self.orbit_ref,
            "phase": self.phase,
            "physical": self.physical.as_dict(),
            "description": self.description,
            "role": self.role,
            "tags": list(self.tags),
            "ephem_fit": self.ephem_fit,
        }
        if self.kepler is not None:
            d["kepler_moon"] = self.kepler.as_dict()
        return d


# ---------------------------------------------------------------------------
# the catalogue
# ---------------------------------------------------------------------------
_BUS_MEDIUM = Physical(radius_m=2.0, albedo=0.30, area_m2=12.0, mass_kg=1500.0, cr=1.3)
_BUS_RELAY = Physical(radius_m=1.5, albedo=0.30, area_m2=8.0, mass_kg=800.0, cr=1.3)
_BUS_SMALL = Physical(radius_m=1.2, albedo=0.25, area_m2=6.0, mass_kg=500.0, cr=1.3)
_BUS_MICRO = Physical(radius_m=0.8, albedo=0.25, area_m2=2.5, mass_kg=180.0, cr=1.3)
_BUS_LOGISTICS = Physical(radius_m=3.0, albedo=0.35, area_m2=30.0, mass_kg=6000.0, cr=1.2)
_DEBRIS = Physical(radius_m=0.3, albedo=0.10, area_m2=0.3, mass_kg=20.0, cr=1.1)

#: Ely (2005) frozen elliptical inclined lunar orbit (see module docstring for the citation).
ELY_2005_ELFO = KeplerianMoon(a_km=6541.4, e=0.60, i_deg=56.2, raan_deg=0.0, argp_deg=90.0, nu_deg=0.0)

NOTIONAL_OBJECTS: list[NotionalObject] = [
    NotionalObject(
        id="SIM-DRO-01",
        name="Notional DRO spacecraft (demo protagonist)",
        orbit_type="DRO",
        orbit_ref="DRO_019",          # period ≈ 12.5 d, 64 000-83 000 km from the Moon
        phase=0.15,
        physical=_BUS_MEDIUM,
        description="Medium bus (≈2 m radius) loitering in a 12.5-day distant retrograde orbit. "
                    "In the demo scenario it performs an unannounced burn.",
        role="unknown-purpose spacecraft",
        tags=("demo", "protagonist"),
        ephem_fit=False,
    ),
    NotionalObject(
        id="SIM-DRO-02",
        name="Notional DRO companion",
        orbit_type="DRO",
        orbit_ref="DRO_018",          # period ≈ 11.4 d
        phase=0.62,
        physical=_BUS_SMALL,
        description="Small bus in a neighbouring 11.4-day DRO; used as a background custody object.",
        role="unknown-purpose spacecraft",
        ephem_fit=False,
    ),
    NotionalObject(
        id="SIM-NRHO-RELAY-01",
        name="Notional allied relay (9:2 NRHO)",
        orbit_type="9:2 NRHO",
        orbit_ref="L2_halo_S_nrho_9_2",
        phase=0.80,
        physical=_BUS_RELAY,
        description="Communications relay on the 9:2 synodic-resonant L2 southern NRHO "
                    "(perilune ≈ 3250 km, apolune ≈ 71 000 km, period ≈ 6.56 d).",
        role="allied communications relay",
        actor=NOTIONAL_ALLY,
        tags=("demo", "high_value"),
        ephem_fit=False,
    ),
    NotionalObject(
        id="SIM-NRHO-02",
        name="Notional 4:1 NRHO loiterer",
        orbit_type="4:1 NRHO",
        orbit_ref="L2_halo_S_nrho_4_1",
        phase=0.35,
        physical=_BUS_SMALL,
        description="Small bus on the 4:1 synodic-resonant NRHO, a few thousand km from the relay "
                    "corridor at perilune.",
        role="unknown-purpose spacecraft",
        ephem_fit=False,
    ),
    NotionalObject(
        id="SIM-L1-HALO-01",
        name="Notional L1 halo observer",
        orbit_type="L1 northern halo",
        orbit_ref="L1_halo_N_018",    # Az ≈ 51 000 km, period ≈ 12.0 d
        phase=0.25,
        physical=_BUS_SMALL,
        description="Mid-amplitude L1 northern halo (Az ≈ 51 000 km); nominal Earth-Moon "
                    "corridor observation platform.",
        role="observation / comms satellite",
        actor=NOTIONAL_ALLY,
    ),
    NotionalObject(
        id="SIM-L2-HALO-01",
        name="Notional L2 halo comms satellite",
        orbit_type="L2 northern halo",
        orbit_ref="L2_halo_N_015",    # Az ≈ 56 000 km, period ≈ 13.9 d
        phase=0.40,
        physical=_BUS_SMALL,
        description="Mid-amplitude L2 northern halo (Az ≈ 56 000 km) providing far-side "
                    "communications; always beyond the Moon as seen from Earth.",
        role="observation / comms satellite",
        actor=NOTIONAL_ALLY,
    ),
    NotionalObject(
        id="SIM-L2-HALO-LOG-01",
        name="Notional L2 halo logistics vehicle",
        orbit_type="L2 southern halo",
        orbit_ref="L2_halo_S_021",    # Az ≈ 76 000 km, near the NRHO end of the family
        phase=0.55,
        physical=_BUS_LOGISTICS,
        description="Large (≈3 m radius, 6 t) logistics vehicle on a large-amplitude L2 southern "
                    "halo awaiting a rendezvous window with the NRHO relay.",
        role="logistics vehicle",
        actor=NOTIONAL_ALLY,
    ),
    NotionalObject(
        id="SIM-L2-LYAP-01",
        name="Notional L2 Lyapunov loiterer",
        orbit_type="L2 Lyapunov",
        orbit_ref="L2_lyap_009",      # planar, Ay ≈ 50 000 km, period ≈ 14.9 d
        phase=0.10,
        physical=_BUS_MICRO,
        description="Microsat on a planar L2 Lyapunov orbit (in the lunar orbit plane).",
        role="unknown-purpose spacecraft",
    ),
    NotionalObject(
        id="SIM-ELFO-01",
        name="Notional south-pole ELFO relay",
        orbit_type="ELFO",
        orbit_ref="keplerian_moon",
        phase=0.0,
        physical=_BUS_SMALL,
        kepler=ELY_2005_ELFO,
        description="Elliptical lunar frozen orbit (Ely 2005: a = 6541 km, e = 0.6, i = 56.2°, "
                    "ω = 90°; apolune over the south pole) for polar communications.",
        role="lunar communications relay",
        actor=NOTIONAL_ALLY,
        ephem_fit=False,
    ),
    NotionalObject(
        id="SIM-RES-31-01",
        name="Notional 3:1 resonant tour vehicle",
        orbit_type="3:1 resonant",
        orbit_ref="RES_31_009",       # perigee ≈ 94 000 km, 28.4 d period
        phase=0.30,
        physical=_BUS_RELAY,
        description="Spacecraft on a planar 3:1 Earth-Moon resonant orbit (three revolutions per "
                    "synodic month) touring the L1/L2 region and the GEO graveyard altitudes.",
        role="unknown-purpose spacecraft",
        ephem_fit=False,
    ),
    NotionalObject(
        id="SIM-XGEO-DEBRIS-01",
        name="Notional xGEO debris fragment",
        orbit_type="2:1 resonant (debris)",
        orbit_ref="RES_21_014",       # stable planar 2:1 resonant orbit, perigee ≈ 106 000 km
        phase=0.45,
        physical=_DEBRIS,
        description="Faint (0.3 m, albedo 0.1) tumbling fragment, e.g. a spent upper-stage part, "
                    "left on a 2:1 lunar-resonant orbit between GEO and the Moon.",
        role="debris",
        actor="notional actor (unattributed debris)",
        tags=("faint",),
        ephem_fit=False,
    ),
]

_BY_ID = {o.id: o for o in NOTIONAL_OBJECTS}


def notional_ids() -> list[str]:
    return [o.id for o in NOTIONAL_OBJECTS]


def get_notional(obj_id: str) -> NotionalObject:
    try:
        return _BY_ID[obj_id]
    except KeyError:
        raise KeyError(f"unknown notional object {obj_id!r}; known: {notional_ids()}") from None


# ---------------------------------------------------------------------------
# state construction
# ---------------------------------------------------------------------------
def cr3bp_phase_state(record: OrbitRecord | str, phase: float, rtol: float = 1e-12, atol: float = 1e-12) -> np.ndarray:
    """Rotating-frame nondimensional state of a library orbit at ``phase`` ∈ [0, 1) of its period
    (``phase`` is wrapped), from a tight CR3BP propagation of the stored initial condition."""
    rec = get_library()[record] if isinstance(record, str) else record
    ph = float(phase) % 1.0
    if ph == 0.0:
        return rec.ic_array.copy()
    sol = propagate_cr3bp(rec.ic_array, tf=ph * rec.period_nd, rtol=rtol, atol=atol)
    if not sol.success:
        raise RuntimeError(f"CR3BP phase propagation failed for {rec.id}: {sol.message}")
    return sol.y[:6, -1].copy()


def reference_rot_state(obj: NotionalObject, dt_s) -> np.ndarray:
    """CR3BP reference state(s) of a library-orbit object ``dt_s`` seconds after the demo epoch,
    rotating frame nd: phase advances by dt / period.  Shape (6,) or (N,6)."""
    rec = obj.record()
    dt = np.asarray(dt_s, dtype=np.float64)
    scalar = dt.ndim == 0
    dts = np.atleast_1d(dt)
    tau = (obj.phase + dts / (rec.period_days * 86400.0)) * rec.period_nd
    # propagate through enough periods with t_eval covering the requested phases (monotone)
    order = np.argsort(tau)
    t_eval = np.concatenate([[0.0], tau[order]]) if tau[order][0] > 0 else tau[order]
    sol = propagate_cr3bp(rec.ic_array, t_eval=t_eval, rtol=1e-12, atol=1e-12)
    if not sol.success:
        raise RuntimeError(f"CR3BP reference propagation failed for {rec.id}: {sol.message}")
    y = sol.y[:6].T
    if len(y) != len(tau):
        y = y[1:]
    out = np.empty_like(y)
    out[order] = y
    return out[0] if scalar else out


def _rot1(th: float) -> np.ndarray:
    c, s = np.cos(th), np.sin(th)
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])


def _rot3(th: float) -> np.ndarray:
    c, s = np.cos(th), np.sin(th)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def keplerian_moon_state_gcrf(k: KeplerianMoon, t_s: float) -> np.ndarray:
    """Moon-centred Keplerian elements -> Earth-centred GCRF state [km, km/s] at TDB ``t_s``.

    Perifocal state -> reference plane via R3(Ω)·R1(i)·R3(ω) -> GCRF axes with the instantaneous
    rotating-frame basis R (columns x̂, ŷ, ẑ) -> Earth-centred by adding the DE440s Moon state
    (:func:`selene.dynamics.frames.moon_centered_to_gcrf`).  Two-body elements about the Moon
    (GM from DE440); no lunar gravity-field harmonics (the ephemeris truth model is point-mass).
    """
    p = k.a_km * (1.0 - k.e**2)
    nu = np.radians(k.nu_deg)
    r = p / (1.0 + k.e * np.cos(nu))
    r_pf = np.array([r * np.cos(nu), r * np.sin(nu), 0.0])
    v_pf = np.sqrt(GM_MOON / p) * np.array([-np.sin(nu), k.e + np.cos(nu), 0.0])
    M = _rot3(np.radians(k.raan_deg)) @ _rot1(np.radians(k.i_deg)) @ _rot3(np.radians(k.argp_deg))
    if k.reference_plane == "earth_moon_orbit_plane":
        R = frames.rotating_frame(float(t_s)).R  # (3,3) columns x̂ ŷ ẑ in GCRF
    elif k.reference_plane == "gcrf_equator":
        R = np.eye(3)
    else:
        raise ValueError(f"unknown reference_plane {k.reference_plane!r}")
    s_moon = np.concatenate([R @ (M @ r_pf), R @ (M @ v_pf)])
    return frames.moon_centered_to_gcrf(s_moon, float(t_s))


def initial_state_gcrf(obj: NotionalObject, t_s: float | None = None) -> np.ndarray:
    """Raw (uncorrected) Earth-centred GCRF state [km, km/s] of ``obj`` at TDB ``t_s`` (default
    the demo epoch): library IC -> CR3BP phase state -> :func:`frames.rot_to_gcrf`, or the
    Keplerian element set for Moon-centred objects."""
    t = frames.DEMO_EPOCH_TDB_S if t_s is None else float(t_s)
    if obj.is_library_orbit:
        return frames.rot_to_gcrf(cr3bp_phase_state(obj.record(), obj.phase), t)
    assert obj.kepler is not None
    return keplerian_moon_state_gcrf(obj.kepler, t)
