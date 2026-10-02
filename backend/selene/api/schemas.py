"""Pydantic v2 models for the SELENE API contract (PLAN.md §5).

Conventions: positions km, velocities km/s (Earth-centred GCRF unless a field name says
``rot``/``nd`` = nondimensional Earth-Moon rotating frame, or ``moon`` = Moon-centred GCRF axes);
times are ISO-8601 UTC strings and/or TDB seconds past J2000 (``tdb_s``); Δv in m/s where the
name says ``_mps``; horizons in hours where the name says ``_h``.  Floats are rounded by the
routes (not here) to keep payloads small.

Models for routes owned by later tracks (sensors, coverage, OD, maneuver, reachability,
tasking, architecture, demo) are given here as reasonable stubs so those tracks extend them
rather than inventing parallel shapes; they are not yet enforced by any route.
"""
from __future__ import annotations

from typing import Any, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field

Vec3 = list[float]
State6 = list[float]


class _Base(BaseModel):
    model_config = ConfigDict(extra="allow")


# ---------------------------------------------------------------------------
# health
class Health(_Base):
    status: str
    version: str
    offline: bool
    ephemeris: Optional[str] = None


# ---------------------------------------------------------------------------
# catalog
ObjectKind = Literal["simulated", "horizons"]


class PhysicalOut(_Base):
    radius_m: float
    albedo: float
    area_m2: float
    mass_kg: float
    cr: float
    cr_area_mass_m2_kg: float
    note: str = "notional design values (SIMULATED object)"


class ObjectSummary(_Base):
    """One catalog object with its state at the catalog epoch (PLAN §5 ``/catalog/objects``)."""

    id: str
    name: str
    kind: ObjectKind
    is_real: bool
    label: str = Field(description="'SIMULATED' or 'REAL (JPL Horizons)'")
    actor: str = Field(description="'notional actor' / 'notional allied operator' for simulated objects")
    orbit_type: str
    orbit_ref: Optional[str] = Field(None, description="orbit-library record id or 'keplerian_moon'; None for Horizons")
    role: str
    description: str
    source: str
    epoch_utc: str
    epoch_tdb_s: float
    state_gcrf_km: State6
    state_rot_nd: State6
    geocentric_range_km: float
    selenocentric_range_km: float
    physical: Optional[PhysicalOut] = None
    radius_m: Optional[float] = None
    albedo: Optional[float] = None
    phase: Optional[float] = None
    period_s: Optional[float] = None
    tags: list[str] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)
    orbit_record: Optional[dict[str, Any]] = None
    kepler_moon: Optional[dict[str, Any]] = None
    truth: Optional[dict[str, Any]] = Field(None, description="simulated objects: truth-model window and insertion fit")
    horizons_id: Optional[int] = None
    span_utc: Optional[list[str]] = None
    regime: Optional[str] = None


class CatalogOut(_Base):
    epoch_utc: str
    n_simulated: int
    n_real: int
    disclaimer: str
    objects: list[ObjectSummary]


class TrajectoryOut(_Base):
    object_id: str
    kind: ObjectKind
    frame: Literal["gcrf_km", "rot_nd", "moon_km"]
    units: str
    t0_utc: str
    t1_utc: str
    n: int = Field(description="number of samples actually returned")
    epochs_utc: list[str]
    tdb_s: list[float]
    states: list[State6]
    meta: dict[str, Any] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# orbits
class MemberOut(_Base):
    id: str
    ic: State6 = Field(description="rotating frame, nondimensional")
    period: float = Field(description="nondimensional (× T* s)")
    period_days: float
    jacobi: float
    stability: float = Field(description="ν = ½(|λmax| + 1/|λmax|)")
    closure_error: float
    tags: list[str] = Field(default_factory=list)
    params: dict[str, Any] = Field(default_factory=dict, description="geometry in km (perilune_km, Az_km, ...)")
    samples_rot: list[Vec3] = Field(description="positions over one period, rotating frame, nondimensional")


class FamilyOut(_Base):
    name: str = Field(description="family key, e.g. 'L2_halo_S', 'DRO', 'resonant_3:1'")
    family: str
    branch: str
    n_total: int
    n_returned: int
    period_days_range: list[float]
    jacobi_range: list[float]
    references: list[str] = Field(default_factory=list)
    members: list[MemberOut]


class FamiliesOut(_Base):
    mu: float
    L_star_km: float
    T_star_s: float
    n_samples: int
    note: str
    families: list[FamilyOut]


class RecordOut(_Base):
    id: str
    family: str
    branch: str
    ic: State6
    period_nd: float
    period_days: float
    jacobi: float
    stability_index: float
    eigenvalues: list[float]
    closure_error: float
    params: dict[str, Any]
    tags: list[str]
    method: str
    index: int
    samples_rot: Optional[list[Vec3]] = None
    samples_gcrf_km: Optional[list[Vec3]] = None


# ---------------------------------------------------------------------------
# ephemeris
class RotFrameOut(_Base):
    """Rotating-frame landmarks in Earth-centred GCRF km, one Vec3 per epoch."""

    moon: list[Vec3]
    l1: list[Vec3]
    l2: list[Vec3]
    l3: list[Vec3]
    l4: list[Vec3]
    l5: list[Vec3]


class BasisOut(_Base):
    """Instantaneous rotating-frame basis per epoch so a client can convert exactly:
    r_rot_nd = Rᵀ (r_gcrf - r_bary) / d_km, with R given row-major as 9 floats (columns x̂ ŷ ẑ)."""

    R: list[list[float]]
    d_km: list[float]
    r_bary_km: list[Vec3]
    omega_rad_s: list[float]


class BodiesOut(_Base):
    source: str
    epochs: list[str]
    tdb_s: list[float]
    earth: list[Vec3]
    moon: list[Vec3]
    sun: list[Vec3]
    rot_frame: RotFrameOut
    basis: BasisOut
    lagrange_rot_nd: dict[str, Vec3]
    mu: float
    L_star_km: float
    T_star_s: float
    note: str


# ---------------------------------------------------------------------------
# stubs for later tracks (extend, do not fork)
class GroundSensorOut(_Base):
    id: str
    name: str
    lat_deg: float
    lon_deg: float
    alt_m: float
    limiting_mag: float
    fov_deg: float
    min_elevation_deg: float


class SpaceSensorOut(_Base):
    id: str
    name: str
    platform_orbit: str
    orbit_ref: Optional[str] = None
    phase: Optional[float] = None
    limiting_mag: float
    fov_deg: float
    sun_exclusion_deg: float
    moon_exclusion_deg: float
    earth_exclusion_deg: float


class SensorsOut(_Base):
    ground: list[GroundSensorOut]
    space: list[SpaceSensorOut]


class CoverageRequest(_Base):
    t0: str
    t1: Optional[str] = None
    n_t: int = 168
    grid: Any = "2d"


class OdRequest(_Base):
    object_id: str
    t0: str
    t1: str
    sensors: list[str]
    method: Literal["iod", "batch", "ukf", "all"] = "all"
    noise_arcsec: float = 1.0
    seed: int = 0


class FilterSeries(_Base):
    epochs: list[str]
    states: list[State6]
    covs: list[list[list[float]]]
    nis: list[float]


class OdOut(_Base):
    iod: Optional[dict[str, Any]] = None
    batch: Optional[dict[str, Any]] = None
    ukf: Optional[FilterSeries] = None
    particles: list[Any] = Field(default_factory=list)


class ManeuverDetection(_Base):
    t: str
    nis: float
    dv_est_mps: Optional[float] = None
    dir: Optional[Vec3] = None
    sigma: Optional[float] = None


class ManeuverOut(_Base):
    detections: list[ManeuverDetection]
    threshold: float
    alpha: float


class ReachabilityRequest(_Base):
    object_id: str
    dv_budget_mps: float
    horizon_h: float = Field(72.0, ge=24.0, le=168.0)
    n_samples: int = 2000
    seed: int = 0


class RegionHit(_Base):
    name: str
    fraction: float
    earliest_h: Optional[float] = None


class ReachabilityOut(_Base):
    points: list[Vec3]
    regions: list[RegionHit]


class TaskingRequest(_Base):
    objects: list[str]
    sensors: list[str]
    t0: str
    t1: str
    method: Literal["greedy", "milp", "random"] = "greedy"


class TaskingOut(_Base):
    schedule: list[dict[str, Any]]
    custody_pct: float
    mean_tslo_h: float
    trace_series: list[dict[str, Any]]


class ArchitectureSpec(_Base):
    name: str
    sensors: list[dict[str, Any]]


class ArchitectureRequest(_Base):
    architectures: list[ArchitectureSpec]
    n_mc: int = 20
    seed: int = 0


class ArchitectureScore(_Base):
    name: str
    coverage_pct: float
    custody_pct: float
    revisit_h: float
    detection_latency_h: float
    detection_latency_p95_h: Optional[float] = None


class ArchitectureOut(_Base):
    scores: list[ArchitectureScore]


class DemoEvent(_Base):
    t: str
    kind: str
    object_id: Optional[str] = None
    text: str


class DemoOut(_Base):
    meta: dict[str, Any]
    frames: list[dict[str, Any]]
    events: list[DemoEvent]
    brief: str
