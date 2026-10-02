/**
 * SELENE API contract — TypeScript mirror of PLAN.md §5 (pydantic models in backend/selene/api/schemas.py).
 *
 * Units (SI-derived, as served by the backend):
 *   positions km, velocities km/s, times seconds or ISO-8601 UTC strings, angles degrees unless noted,
 *   Δv in m/s where the field name says `_mps`, horizons in hours where the name says `_h`.
 * Nondimensional rotating-frame samples (`samples_rot`, `rot_frame`) are in Earth–Moon CR3BP units
 * (L* = 384 400 km) with the barycenter at the origin and the Moon at (1−μ, 0, 0).
 */

export type Vec3 = [number, number, number];
/** 6-state [x, y, z, vx, vy, vz] in km and km/s (GCRF) or nondimensional (rotating). */
export type State6 = [number, number, number, number, number, number];
/** Row-major 6×6 covariance (km², km²/s², ...). */
export type Cov6 = number[][];

// ---------------------------------------------------------------------------
// GET /api/health
export interface Health {
  status: string;
  version: string;
  offline: boolean;
}

// ---------------------------------------------------------------------------
// GET /api/catalog/objects
export type ObjectKind = 'simulated' | 'horizons';

export interface CatalogObject {
  id: string;
  name: string;
  kind: ObjectKind;
  /** e.g. "L2 southern halo", "9:2 NRHO", "DRO", "ELFO", "resonant 3:1" */
  orbit_type: string;
  /** Always "notional" for simulated objects; never a real country/operator. */
  actor: string;
  state_gcrf_km: State6;
  epoch_utc: string;
}

// ---------------------------------------------------------------------------
// GET /api/orbits/families
export interface OrbitMember {
  id: string;
  /** Nondimensional CR3BP initial condition in the rotating frame. */
  ic: State6;
  /** Period, nondimensional time units (multiply by T* ≈ 3.7519e5 s for seconds). */
  period: number;
  jacobi: number;
  /** Stability index ν = ½(λ_max + 1/λ_max); |ν| ≤ 1 is linearly stable. */
  stability: number;
  /** Sampled positions along one period, rotating frame, nondimensional. */
  samples_rot: Vec3[];
}

export interface OrbitFamily {
  name: string;
  members: OrbitMember[];
}

export interface OrbitFamilies {
  families: OrbitFamily[];
}

// ---------------------------------------------------------------------------
// GET /api/ephemeris/bodies?t0&t1&n
export interface EphemerisBodies {
  /** ISO UTC epochs, length n. */
  epochs: string[];
  /** GCRF positions, km, one Vec3 per epoch. Earth is the origin (all zeros) in GCRF. */
  earth: Vec3[];
  moon: Vec3[];
  sun: Vec3[];
  /** Instantaneous rotating-frame quantities, nondimensional, one per epoch. */
  rot_frame: {
    moon: Vec3[];
    l1: Vec3[];
    l2: Vec3[];
    l3: Vec3[];
    l4: Vec3[];
    l5: Vec3[];
  };
}

// ---------------------------------------------------------------------------
// GET /api/sensors
export interface GroundSensor {
  id: string;
  name: string;
  lat_deg: number;
  lon_deg: number;
  alt_km: number;
  /** Limiting visual magnitude. */
  limiting_mag: number;
  fov_deg: number;
  min_elevation_deg: number;
  sun_exclusion_deg: number;
  moon_exclusion_deg: number;
}

export type SpaceObserverOrbit = 'GEO' | 'L1_halo' | 'L2_halo' | 'DRO' | 'resonant' | string;

export interface SpaceSensor {
  id: string;
  name: string;
  orbit: SpaceObserverOrbit;
  /** Optional: id of the orbit-library member the observer rides. */
  orbit_member_id?: string;
  limiting_mag: number;
  fov_deg: number;
  sun_exclusion_deg: number;
  moon_exclusion_deg: number;
  earth_exclusion_deg: number;
  /** Max slew rate, deg/s. */
  slew_rate_dps?: number;
}

export interface Sensors {
  ground: GroundSensor[];
  space: SpaceSensor[];
}

// ---------------------------------------------------------------------------
// POST /api/coverage
export interface CoverageGridSpec {
  /** Axis-aligned box in the rotating frame, nondimensional. */
  xmin: number;
  xmax: number;
  ymin: number;
  ymax: number;
  zmin?: number;
  zmax?: number;
  nx: number;
  ny: number;
  nz?: number;
}

export interface CoverageRequest {
  t0: string;
  t1: string;
  n_t: number;
  grid: CoverageGridSpec;
  sensor_ids?: string[];
  /** Assumed target radius (m) and albedo for detectability. */
  target_radius_m?: number;
  target_albedo?: number;
}

export interface CoverageResponse {
  grid: CoverageGridSpec & { x: number[]; y: number[]; z?: number[] };
  /** values[t][i]: number of sensors able to detect a target at cell i, epoch t (0 = blind). */
  values: number[][];
  epochs: string[];
}

// ---------------------------------------------------------------------------
// POST /api/od/run
export type OdMethod = 'iod' | 'batch' | 'ukf' | 'all';

export interface OdRequest {
  object_id: string;
  t0: string;
  t1: string;
  sensors: string[];
  method: OdMethod;
  /** Measurement noise, arcsec (1σ). */
  noise_arcsec?: number;
  seed?: number;
}

export interface IodResult {
  epoch: string;
  state: State6;
  /** Position error vs. truth, km (synthetic runs only). */
  pos_err_km?: number;
}

export interface BatchResult {
  epoch: string;
  state: State6;
  cov: Cov6;
  iterations: number;
  rms_arcsec: number;
  converged: boolean;
}

export interface UkfResult {
  epochs: string[];
  states: State6[];
  covs: Cov6[];
  /** Normalized innovation squared per processed measurement. */
  nis: number[];
  /** Position-covariance trace sqrt (km), convenience series. */
  sigma_pos_km?: number[];
}

export interface ParticleCloud {
  epoch: string;
  /** N×3 positions in km (GCRF) — flattened server-side as rows. */
  positions_km: Vec3[];
  frame: 'gcrf' | 'rotating';
}

export interface OdResponse {
  iod?: IodResult;
  batch?: BatchResult;
  ukf?: UkfResult;
  particles?: ParticleCloud[];
}

// ---------------------------------------------------------------------------
// POST /api/maneuver/detect
export interface ManeuverDetectRequest {
  object_id: string;
  t0?: string;
  t1?: string;
  sensors?: string[];
  /** False-alarm probability α for the NIS χ² gate. */
  alpha?: number;
  seed?: number;
}

export interface ManeuverDetection {
  t: string;
  nis: number;
  /** Estimated Δv magnitude, m/s. */
  dv_est: number;
  /** Unit direction (GCRF) of the estimated Δv. */
  dir: Vec3;
  /** 1σ on dv_est, m/s. */
  sigma: number;
}

export interface ManeuverDetectResponse {
  detections: ManeuverDetection[];
  /** Threshold used, χ²_m(1−α). */
  threshold?: number;
}

// ---------------------------------------------------------------------------
// POST /api/reachability
export interface ReachabilityRequest {
  object_id: string;
  dv_budget_mps: number;
  horizon_h: number;
  n_samples?: number;
  seed?: number;
}

export interface ReachabilityRegion {
  /** e.g. "L1 gateway", "L2 gateway", "NRHO corridor", "Lunar south pole", "GEO return" */
  name: string;
  /** Fraction of samples that enter the region within the horizon. */
  fraction: number;
  /** Earliest arrival, hours after burn (null if unreachable). */
  earliest_h: number | null;
}

export interface ReachabilityResponse {
  /** Terminal (or sampled) positions, rotating frame nondimensional, for the point cloud. */
  points: Vec3[];
  regions: ReachabilityRegion[];
}

// ---------------------------------------------------------------------------
// POST /api/tasking/schedule
export type TaskingMethod = 'greedy' | 'milp' | 'random';

export interface TaskingRequest {
  objects: string[];
  sensors: string[];
  t0: string;
  t1: string;
  method: TaskingMethod;
  slot_s?: number;
}

export interface TaskingSlot {
  t: string;
  sensor_id: string;
  object_id: string;
  /** Expected information gain ½ ln(det P⁻/det P⁺). */
  info_gain: number;
}

export interface TaskingResponse {
  schedule: TaskingSlot[];
  custody_pct: number;
  /** Mean time since last observation, hours, over objects and time. */
  mean_tslo_h: number;
  /** Summed position-covariance trace per epoch (km²). */
  trace_series: { epochs: string[]; trace: number[] };
  per_object?: { object_id: string; custody_pct: number; mean_tslo_h: number }[];
}

// ---------------------------------------------------------------------------
// POST /api/architecture/evaluate
export interface ArchitectureSensorSpec {
  orbit: SpaceObserverOrbit;
  orbit_member_id?: string;
  limiting_mag?: number;
  fov_deg?: number;
  /** Phase offset along the orbit, fraction of period [0,1). */
  phase?: number;
}

export interface Architecture {
  name: string;
  sensors: ArchitectureSensorSpec[];
}

export interface ArchitectureEvaluateRequest {
  architectures: Architecture[];
  n_mc: number;
  seed?: number;
}

export interface ArchitectureScore {
  name: string;
  coverage_pct: number;
  custody_pct: number;
  /** Mean revisit time, hours. */
  revisit_h: number;
  /** Maneuver-detection latency, hours: mean and 95th percentile. */
  detect_latency_h_mean: number;
  detect_latency_h_p95: number;
  n_mc: number;
}

export interface ArchitectureEvaluateResponse {
  scores: ArchitectureScore[];
}

// ---------------------------------------------------------------------------
// GET /api/demo/scenario
export type EventSeverity = 'info' | 'ok' | 'warn' | 'alert';

export type EventKind =
  | 'maneuver_detected'
  | 'custody_lost'
  | 'custody_regained'
  | 'entered_region'
  | 'tasking'
  | 'observation'
  | 'brief'
  | 'info'
  | string;

export interface SeleneEvent {
  /** Seconds since scenario t0. */
  t: number;
  kind: EventKind;
  severity: EventSeverity;
  text: string;
  object_id?: string;
}

export interface FrameObject {
  id: string;
  /** Rotating-frame position, nondimensional. */
  pos_rot: Vec3;
  /** GCRF position, km (optional; backend may provide both). */
  pos_gcrf_km?: Vec3;
  custody: 'held' | 'degraded' | 'lost';
  /** sqrt(trace of position covariance), km. */
  sigma_pos_km?: number;
}

export interface FrameCloud {
  object_id: string;
  /** Particle positions, rotating frame, nondimensional. */
  points_rot: Vec3[];
}

export interface FrameSensor {
  id: string;
  pos_rot: Vec3;
  /** Boresight unit vector in the rotating frame, if tasked. */
  boresight_rot?: Vec3;
  /** Object currently tasked, if any. */
  target_id?: string | null;
  fov_deg?: number;
}

export interface DemoFrame {
  t: number;
  objects: FrameObject[];
  clouds: FrameCloud[];
  sensors: FrameSensor[];
  events: SeleneEvent[];
}

export interface DemoMeta {
  title: string;
  t0_utc: string;
  duration_s: number;
  /** Wall-clock playback length target, seconds (≈120 for the pitch). */
  playback_s?: number;
  disclaimer: string;
}

export interface DemoScenario {
  meta: DemoMeta;
  frames: DemoFrame[];
  events: SeleneEvent[];
  /** Plain-English analyst brief (markdown-ish paragraphs). */
  brief: string;
}
