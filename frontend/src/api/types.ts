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
  /** Ephemeris kernel in use, e.g. "de440s.bsp" (live backend). */
  ephemeris?: string;
}

// ---------------------------------------------------------------------------
// GET /api/catalog/objects
export type ObjectKind = 'simulated' | 'horizons';

/** Notional physical/photometric design values carried by SIMULATED objects (backend `physical`). */
export interface PhysicalAssumptions {
  radius_m: number;
  albedo: number;
  area_m2?: number;
  mass_kg?: number;
  cr?: number;
  cr_area_mass_m2_kg?: number;
  note?: string;
}

/** Summary of the periodic-orbit library record an object rides (backend `orbit_record`). */
export interface OrbitRecordSummary {
  id: string;
  family: string;
  branch?: string;
  period_days: number;
  jacobi: number;
  stability_index: number;
  tags?: string[];
  params?: Record<string, number>;
}

export interface CatalogObject {
  id: string;
  name: string;
  kind: ObjectKind;
  /** e.g. "L2 southern halo", "9:2 NRHO", "DRO", "ELFO", "3:1 resonant" */
  orbit_type: string;
  /** "notional actor" / "notional allied operator" for simulated objects; never a real country/operator. */
  actor: string;
  state_gcrf_km: State6;
  /** ISO UTC (the client normalises the backend's zone-less strings to a trailing 'Z'). */
  epoch_utc: string;
  /** Rotating-frame nondimensional state at `epoch_utc` (`state_rot_nd` from the backend, `ic_rot` in the mock). */
  ic_rot?: State6;
  state_rot_nd?: State6;
  /** Optional photometric parameters: characteristic radius [m] and Bond albedo. */
  radius_m?: number | null;
  albedo?: number | null;
  // ---- live backend extras (all optional) ----
  is_real?: boolean;
  /** "SIMULATED" | "REAL (JPL Horizons)" */
  label?: string;
  orbit_ref?: string | null;
  role?: string;
  description?: string;
  /** Provenance, e.g. "JPL Horizons id -1176 (CAPSTONE_merged), fetched …" or "SIMULATED: CR3BP library record …". */
  source?: string;
  physical?: PhysicalAssumptions | null;
  period_s?: number | null;
  phase?: number | null;
  tags?: string[];
  notes?: string[];
  orbit_record?: OrbitRecordSummary | null;
  horizons_id?: number | null;
  /** Horizons regime tag: 'nrho' | 'lunar_orbit' | 'xgeo_heo' | … */
  regime?: string | null;
  /** Horizons cached span [start, end] (ISO UTC, normalised). */
  span_utc?: [string, string] | null;
  geocentric_range_km?: number;
  selenocentric_range_km?: number;
  epoch_tdb_s?: number;
}

/** Live `GET /api/catalog/objects` envelope (the mock returns a bare array; the client unwraps both). */
export interface CatalogMeta {
  epoch_utc: string;
  n_simulated: number;
  n_real: number;
  disclaimer: string;
}
export interface CatalogResponse extends CatalogMeta {
  objects: CatalogObject[];
}

// ---------------------------------------------------------------------------
// GET /api/catalog/objects/{id}/trajectory?t0&t1&n&frame
export type TrajectoryFrame = 'gcrf_km' | 'rot_nd' | 'moon_km';

export interface TrajectoryResponse {
  object_id: string;
  kind: ObjectKind;
  frame: TrajectoryFrame;
  units: string;
  t0_utc: string;
  t1_utc: string;
  n: number;
  epochs_utc: string[];
  tdb_s: number[];
  /** 6-states per epoch: km & km/s (gcrf_km, moon_km) or nondimensional (rot_nd; length = instantaneous Earth–Moon distance, time T*). */
  states: State6[];
  meta?: Record<string, unknown>;
}

// ---------------------------------------------------------------------------
// GET /api/orbits/families
export interface OrbitMember {
  id: string;
  /** Nondimensional CR3BP initial condition in the rotating frame. */
  ic: State6;
  /** Period, nondimensional time units (multiply by T* ≈ 3.7519e5 s for seconds). */
  period: number;
  period_days?: number;
  jacobi: number;
  /** Stability index ν = ½(λ_max + 1/λ_max); |ν| ≤ 1 is linearly stable. */
  stability: number;
  /** Sampled positions along one period, rotating frame, nondimensional. */
  samples_rot: Vec3[];
  /** True when the member was corrected from an approximate (non-literature) seed; shown as "approx." in the UI. */
  approximate?: boolean;
  closure_error?: number;
  /** e.g. ["NRHO", "NRHO_9:2", "synodic_resonant"] */
  tags?: string[];
  params?: Record<string, number>;
}

export interface OrbitFamily {
  /** Unique name, e.g. "L2_halo_S", "DRO", "resonant_3:1" (live) or "L2 southern halo / NRHO" (mock). */
  name: string;
  members: OrbitMember[];
  family?: string;
  branch?: string;
  n_total?: number;
  n_returned?: number;
  period_days_range?: [number, number];
  jacobi_range?: [number, number];
  references?: string[];
}

export interface OrbitFamilies {
  families: OrbitFamily[];
  mu?: number;
  L_star_km?: number;
  T_star_s?: number;
  n_samples?: number;
  note?: string;
}

// GET /api/orbits/records/{id}
export interface OrbitRecord {
  id: string;
  family: string;
  branch?: string;
  ic: State6;
  period_nd: number;
  period_days: number;
  jacobi: number;
  stability_index: number;
  eigenvalues?: number[];
  closure_error?: number;
  params?: Record<string, number>;
  tags?: string[];
  method?: string;
  index?: number;
  /** One period, uniformly sampled in time from `ic`, rotating frame nondimensional. */
  samples_rot: Vec3[];
  samples_gcrf_km?: Vec3[] | null;
}

// ---------------------------------------------------------------------------
// GET /api/ephemeris/bodies?t0&t1&n
/**
 * Per-epoch Earth–Moon rotating-frame basis (live backend only).
 * `R[k]` is row-major 3×3 with columns x̂, ŷ, ẑ of the rotating frame expressed in GCRF, so
 *   r_gcrf = r_bary + d · R · r_rot_nd      and      r_rot_nd = Rᵀ (r_gcrf − r_bary) / d.
 */
export interface RotatingBasis {
  R: number[][];
  d_km: number[];
  r_bary_km: Vec3[];
  omega_rad_s: number[];
}

export interface EphemerisBodies {
  /** Provenance tag; the browser mock sets 'mock-mean-elements', the backend 'de440s'. */
  source?: string;
  /** ISO UTC epochs, length n (normalised to a trailing 'Z' by the client). */
  epochs: string[];
  tdb_s?: number[];
  /** GCRF positions, km, one Vec3 per epoch. Earth is the origin (all zeros) in GCRF. */
  earth: Vec3[];
  moon: Vec3[];
  sun: Vec3[];
  /** Moon and the CR3BP libration points of the instantaneous rotating frame, GCRF km (display aid). */
  rot_frame: {
    moon: Vec3[];
    l1: Vec3[];
    l2: Vec3[];
    l3: Vec3[];
    l4: Vec3[];
    l5: Vec3[];
  };
  basis?: RotatingBasis;
  lagrange_rot_nd?: Record<'l1' | 'l2' | 'l3' | 'l4' | 'l5', Vec3>;
  mu?: number;
  L_star_km?: number;
  T_star_s?: number;
  note?: string;
}

// ---------------------------------------------------------------------------
// GET /api/sensors
export interface GroundSensor {
  id: string;
  name: string;
  lat_deg: number;
  lon_deg: number;
  alt_km: number;
  /** Backend may serve metres instead; the client adapter fills alt_km. */
  alt_m?: number;
  /** Limiting visual magnitude. */
  limiting_mag: number;
  fov_deg: number;
  min_elevation_deg: number;
  sun_exclusion_deg?: number;
  moon_exclusion_deg?: number;
  /** Sun elevation must be below this for the site to operate (deg, e.g. −12). */
  sun_elev_max_deg?: number;
  aperture_m?: number;
  slew_rate_deg_s?: number;
  kind?: 'ground';
  notes?: string;
  spec_note?: string;
}

/** Live backend `orbit` summary for a space observer (the client moves it to `orbit_info`). */
export interface SpaceObserverOrbitInfo {
  /** 'geostationary_itrs' | 'library:<record id>' | 'custom' */
  source: string;
  period_days: number;
  ic_rot_nd?: State6;
  closure_residual?: number;
  length_unit_km?: number;
  radius_km?: number;
  longitude_deg?: number;
}

export type SpaceObserverOrbit = 'GEO' | 'L1_halo' | 'L2_halo' | 'DRO' | 'resonant' | string;

export interface SpaceSensor {
  id: string;
  name: string;
  /** Candidate orbit class (client adapter copies `platform_orbit` here when the backend uses that name). */
  orbit: SpaceObserverOrbit;
  platform_orbit?: string;
  /** Live backend orbit summary (host orbit record / GEO parameters). */
  orbit_info?: SpaceObserverOrbitInfo;
  /** Library record id the observer rides (parsed from orbit_info.source = "library:<id>"). */
  orbit_record_id?: string;
  /** Epoch of `phase` (TDB seconds past J2000) and its UTC ms equivalent (client-derived). */
  epoch_s?: number;
  epoch_ms?: number;
  /** Optional: id of the orbit-library member the observer rides (`orbit_ref` is the backend alias). */
  orbit_member_id?: string;
  orbit_ref?: string | null;
  /** Phase along the orbit at the epoch, fraction of a period. */
  phase?: number;
  /** GEO observers: sub-satellite longitude, deg east. */
  geo_longitude_deg?: number;
  notes?: string;
  limiting_mag: number;
  fov_deg: number;
  sun_exclusion_deg: number;
  moon_exclusion_deg: number;
  earth_exclusion_deg: number;
  /** Max slew rate, deg/s. */
  slew_rate_dps?: number;
  slew_rate_deg_s?: number;
  kind?: 'space';
  spec_note?: string;
}

export interface Sensors {
  ground: GroundSensor[];
  space: SpaceSensor[];
  note?: string;
  reason_bits?: Record<string, number>;
}

// GET /api/sensors/{id}/visibility?object_id&t0&t1&n
export interface VisibilityResponse {
  sensor_id: string;
  object_id: string;
  t0: string;
  t1: string;
  n: number;
  fraction_visible: number;
  t_s: number[];
  visible: boolean[];
  magnitude: (number | null)[];
  reasons: number[];
  reason_names: string[][];
  range_km: number[];
  phase_deg: number[];
}

// GET /api/coverage/presets
export type CoveragePresets = Record<string, { ground_ids: string[]; space: string[] }>;

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
  /** ≤ 12-word plain-English headline (served by the backend, else derived in demo/headline.ts at load). */
  headline?: string;
  object_id?: string;
  /**
   * Optional structured payload. Conventions used by the panels:
   *   observation: { sensor_id, residual_arcsec }
   *   maneuver_detected / maneuver_characterized: { dv_mps, dv_sigma_mps, direction: Vec3 (rotating-frame unit), confidence (0–1), nis }
   *   tasking: { sensor_id, target_id }
   *   any: { show_layers: string[] } — scene layers the demo driver switches on when the event is crossed
   */
  data?: Record<string, unknown>;
}

export interface FrameObject {
  id: string;
  /** Rotating-frame position, nondimensional. */
  pos_rot: Vec3;
  /** GCRF position, km (optional; backend may provide both). */
  pos_gcrf_km?: Vec3;
  /** 'unknown' = no OD/tasking has evaluated this object (idle catalog view): not a measured status.
   *  The backend bundle serves 'CUSTODY' | 'DEGRADED' | 'LOST' (+ `custody_ui`); demo/normalize.ts maps both. */
  custody: 'held' | 'degraded' | 'lost' | 'unknown';
  custody_ui?: string;
  /** sqrt(trace of position covariance), km (`sigma_km` is the backend alias). */
  sigma_pos_km?: number;
  sigma_km?: number;
}

export interface FrameCloud {
  object_id: string;
  /** Particle positions, rotating frame, nondimensional — either as rows ... */
  points_rot?: Vec3[];
  /** ... or as a flat xyz array (what the backend streams; Float32Array after normalisation). */
  points?: number[] | Float32Array;
  /** sqrt(trace of position covariance) of the cloud, km. */
  sigma_km?: number;
}

export interface FrameSensor {
  id: string;
  /** Observer position in the rotating frame (space observers; omitted for ground sites). */
  pos_rot?: Vec3;
  /** Boresight unit vector in the rotating frame, if tasked (`pointing_rot` is the backend alias). */
  boresight_rot?: Vec3;
  pointing_rot?: Vec3;
  /** Object currently tasked, if any. */
  target_id?: string | null;
  fov_deg?: number;
  /** Sensor is currently taking data. */
  active?: boolean;
}

export interface FrameReachable {
  /** Reachable-set sample positions, rotating frame nondimensional (rows or flat). */
  points: Vec3[] | number[];
  regions: ReachabilityRegion[];
  /** Δv budget and horizon the set was computed for. */
  dv_budget_mps?: number;
  horizon_h?: number;
}

/** One tracklet (or tasking look) taken in a frame. The backend lists every tracklet here even when the events
 *  feed thins space-observer follow-ups to one entry per 6 h; rows with `note` and no residual are linear-covariance
 *  tasking looks without a measurement realisation. */
export interface FrameObservation {
  sensor_id: string;
  object_id: string;
  residual_arcsec?: number | null;
  magnitude?: number | null;
  note?: string;
}

export interface DemoFrame {
  t: number;
  objects: FrameObject[];
  clouds: FrameCloud[];
  sensors: FrameSensor[];
  observations?: FrameObservation[];
  events?: SeleneEvent[];
  reachable?: FrameReachable;
}

export interface DemoMeta {
  title?: string;
  t0_utc: string;
  duration_s: number;
  /** Wall-clock playback length target, seconds (≈120 for the pitch). */
  playback_s?: number;
  /** Scripted playback speed (sim seconds per wall second); derived from playback_s when omitted. */
  playback_speed?: number;
  disclaimer?: string;
  /** Object the story follows (auto-selected by the demo driver). */
  protagonist_id?: string;
  /** UTC at which the analyst brief was generated (footer). */
  brief_generated_utc?: string;
  /** Custody thresholds on the particle-cloud σ_pos [km]: held below `custody_km`, lost at/above `lost_km`
   *  (backend: metrics.filter.custody_km / lost_km; mock: its CUSTODY constants). Filled by demo/normalize.ts. */
  custody_km?: number;
  lost_km?: number;
}

export interface DemoScenario {
  meta: DemoMeta;
  frames: DemoFrame[];
  events: SeleneEvent[];
  /** Plain-English analyst brief (markdown-ish paragraphs). */
  brief: string;
  /** Backend run metrics (opaque to the UI except `filter.custody_km` / `filter.lost_km`). */
  metrics?: Record<string, unknown>;
}
