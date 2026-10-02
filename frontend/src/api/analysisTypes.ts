/**
 * Live response shapes of the analysis routes, as served by the backend (verified against real responses of
 * backend/selene/api/routes/{od,maneuver,reachability,tasking}.py). Only the fields the panels read are typed;
 * everything else is carried through as unknown so nothing is invented client-side.
 *
 * Units: km, km/s, m/s where the field name says `_mps`, hours where it says `_h`, arcsec where it says `_arcsec`,
 * UTC ISO strings for `*_utc` / `epochs`, TDB seconds past J2000 for `t_s`.
 */
import type { State6, Vec3 } from './types';

// ---------------------------------------------------------------------------
// GET /api/od/presets · POST /api/od/run
export interface OdPreset {
  t1?: string | null;
  cadence_min?: number;
  sensors?: string | string[];
  method?: 'ukf' | 'batch' | 'iod' | 'all';
  particles?: { n?: number; t_grid_h?: number; step_h?: number; max_export?: number };
  note?: string;
}
export interface OdPresets {
  presets: Record<string, OdPreset>;
  defaults: Record<string, unknown>;
}

export interface OdRunRequest {
  object_id: string;
  t0: string;
  t1?: string | null;
  sensors?: string | string[];
  cadence_min?: number;
  sigma_arcsec?: number;
  method?: 'ukf' | 'batch' | 'iod' | 'all';
  filter?: 'ukf' | 'ekf';
  q_psd?: number;
  prior_sigma_pos_km?: number;
  prior_sigma_vel_m_s?: number;
  ukf_prior?: 'simulated' | 'iod';
  particles?: { n?: number; t_grid_h?: number; step_h?: number; max_export?: number } | null;
  seed?: number;
}

export interface OdObservationsSummary {
  n_used: number;
  n_measurements: number;
  n_candidates: number;
  fraction_observed: number;
  by_sensor: Record<string, number>;
  dropped_by_reason: Record<string, number>;
  dropped_by_sensor: Record<string, number>;
  n_dropped: number;
  n_epochs: number;
  span_h: number;
  sigma_arcsec: number;
}

export interface OdIodBest {
  epoch: string;
  state: State6;
  sigma_pos_km: number;
  rms_arcsec: number;
  truth_error_pos_km: number;
  truth_error_vel_m_s: number;
  converged: boolean;
  admissible: boolean;
}
export interface OdIod {
  method: string;
  n_obs: number;
  n_candidates: number;
  quality: string;
  note?: string;
  best: OdIodBest | null;
  usable_as_seed?: boolean;
}
export interface OdBatch {
  method: string;
  epoch: string;
  state: State6;
  sigma_pos_km: number;
  sigma_vel_km_s: number;
  rms_arcsec: number;
  weighted_rms: number;
  iterations: number;
  converged: boolean;
  n_obs: number;
  dof: number;
  truth_error_pos_km: number;
  truth_error_vel_m_s: number;
  truth_nees: number;
  initial_guess_source: string;
}
export interface OdUkf {
  n: number;
  epochs: string[];
  t_s: number[];
  sigma_pos_km: number[];
  sigma_pos_pred_km?: number[];
  nis: number[];
  sensor_ids: string[];
  prior_source: string;
  meta: { filter: string; mean_nis: number; n_updates: number; q_psd_km2_s3: number };
}
export interface OdParticleFrame {
  epoch: string;
  t_s: number;
  positions_km: Vec3[];
  positions_rot: Vec3[];
  mean_km: Vec3;
}
export interface OdParticles {
  n_particles: number;
  n_exported: number;
  epochs: string[];
  metrics: { hours: number[]; sigma_pos_km: number[]; linear_sigma_pos_km?: number[]; extent_km?: number[] };
  frames: OdParticleFrame[];
  source: string;
  truth_positions_km: Vec3[];
  custody: { sigma_pos_km_start: number; sigma_pos_km_end: number; growth_factor: number | null; hours_to_1000km: number | null; note: string };
}
export interface OdRunResponse {
  object_id: string;
  label: string;
  t0_utc: string;
  t1_utc: string;
  sensors: string[];
  method: string;
  caps_applied: string[];
  observations: OdObservationsSummary;
  force_model: { filter: string; truth: string };
  iod: OdIod | null;
  batch: OdBatch | null;
  ukf: OdUkf | null;
  particles: OdParticles | null;
  truth_error_km: number[] | null;
  truth_error_vel_m_s?: number[];
  nees?: number[];
  nees_bounds_95_single_epoch?: [number, number];
  notes: string[];
  timing: Record<string, number> & { total_s: number };
}

// ---------------------------------------------------------------------------
// POST /api/maneuver/detect
export const MANEUVER_DIRECTIONS = ['prograde', 'retrograde', 'radial_out', 'radial_in', 'normal', 'anti_normal', 'toward_moon', 'toward_earth', 'toward_l1', 'toward_l2'] as const;
export type ManeuverDirection = (typeof MANEUVER_DIRECTIONS)[number];

export interface ManeuverDetectRequest {
  object_id: string;
  t0: string;
  t1?: string;
  sensors?: string[];
  cadence_min?: number;
  sigma_arcsec?: number;
  alpha?: number;
  window?: number;
  injected?: { t_burn_utc: string; magnitude_mps: number; direction: ManeuverDirection } | null;
  seed?: number;
  estimate?: boolean;
  filter?: 'auto' | 'ukf' | 'ekf';
}

export interface ManeuverDetection {
  t_s: number;
  t_utc: string;
  test: 'nis' | 'nis_window' | 'gap_refit' | 'cusum' | 'nees' | string;
  statistic: number;
  threshold: number;
  p_value: number | null;
  confidence: number | null;
  index: number;
  sensor_id: string;
}
export interface ManeuverDetectResponse {
  object_id: string;
  label: string;
  filter: {
    name: string;
    n_updates: number;
    t_s: number[];
    epochs_utc: string[];
    sigma_pos_km: number[];
    nis: number[];
    sensor_ids: string[];
    meta: { mean_nis: number };
  };
  detections: ManeuverDetection[];
  declared: boolean;
  status: string;
  status_note: string;
  dv_estimate: {
    magnitude_mps: number;
    magnitude_sigma_mps: number;
    direction_gcrf: Vec3;
    direction_sigma_deg: number;
    dv_rtn_mps?: Vec3;
    t_burn_utc: string;
    t_burn_sigma_s: number;
    n_obs: number;
    converged: boolean;
    residual_rms_arcsec: number;
    reduced_chi2: number;
    classification?: { primary: string; heuristic?: boolean; note?: string };
  } | null;
  truth: { magnitude_mps: number; t_burn_utc: string; note?: string; estimate_error?: { magnitude_error_mps: number; magnitude_error_pct: number; direction_error_deg: number; t_burn_error_s: number }; detection_latency_s?: number | null; detected?: boolean; detected_after_n_post_burn_obs?: number | null } | null;
  summary: {
    status: string;
    threshold_nis: number;
    threshold_nis_familywise: number;
    threshold_window?: number;
    alpha: number;
    n_updates: number;
    n_tested: number;
    counts: Record<string, number>;
    mean_nis: number;
    max_nis: number;
    nees?: { mean_nees: number; n_epochs: number; dof: number; mean_bounds: [number, number]; consistent: boolean };
    nees_series?: number[];
    pos_err_km?: number[];
    filter_health?: { baseline_established: boolean; nees_consistent?: boolean; warnings: string[] };
  };
  timing: Record<string, number> & { total_s: number };
  first_detection_utc?: string | null;
  first_declared_utc?: string | null;
}

// ---------------------------------------------------------------------------
// POST /api/reachability · GET /api/reachability/regions
export interface ReachRequest {
  object_id: string;
  t0?: string;
  dv_budget_mps: number;
  horizon_h: number;
  n_samples?: number;
  burn_epochs_h?: number[];
  include_paths?: boolean;
  include_hints?: boolean;
  refine?: boolean;
}
export interface ReachRegionOut {
  key: string;
  name: string;
  kind: string;
  fraction: number;
  sample_fraction: number;
  n_hit: number;
  earliest_h: number | null;
  min_dv_mps: number | null;
  nominal_hits: boolean;
  newly_reachable: boolean;
  earliest_nominal_h?: number | null;
  refined_min_dv?: { dv_mps?: number; [k: string]: unknown } | null;
  why_it_matters?: string;
  description?: string;
}
export interface ReachSensorHintRow {
  sensor_id: string;
  kind: 'ground' | 'space';
  peak_visible_fraction: number;
  peak_fov_capture_fraction: number;
  first_opportunity_h: number | null;
  n_steps_visible: number;
}
export interface ReachResponse {
  object: { id: string; name: string; kind: string; label: string; orbit_type: string; is_real: boolean };
  t0_utc: string;
  config: { dv_budget_mps: number; horizon_h: number; n_dirs: number; burn_epochs_h: number[]; magnitudes_mps: number[]; n_samples_actual: number; caps_applied: string[] };
  nominal: { t_h: number[]; rot: Vec3[]; hit_regions: string[] };
  samples: {
    n: number;
    dv_mps: number[];
    burn_h: number[];
    end_rot: Vec3[];
    end_t_h: number[];
    terminated: boolean[];
    termination_reason: (string | null)[];
    hit_region: (string | null)[];
    hit_regions: string[][];
    /** Ray index (direction × burn epoch) of each sample; the regions' `fraction` is per ray, not per sample. */
    ray?: number[];
  };
  regions: ReachRegionOut[];
  envelope: { t_h: number; n_active: number; max_radius_km: number; rms_radius_km: number; centroid_rot_nd: Vec3 }[];
  sensor_hints: { steps: { t_h: number; best_single_field: string | null; best_any_pointing: string | null }[]; summary: Record<string, ReachSensorHintRow>; best_overall: string | null; note: string; photometry?: { radius_m: number; albedo: number } };
  timing: Record<string, number> & { route_total_s: number };
  disclaimer: string;
}
export interface ReachRegionsResponse {
  regions: { key: string; name: string; kind: string; description: string; why_it_matters: string; geometry: Record<string, unknown> }[];
  frame: string;
  disclaimer: string;
}

// ---------------------------------------------------------------------------
// GET /api/tasking/presets · POST /api/tasking/schedule
export type TaskingMethod = 'greedy' | 'milp' | 'random' | 'round_robin' | 'compare';
export const TASKING_METHODS: TaskingMethod[] = ['greedy', 'milp', 'random', 'round_robin', 'compare'];

export interface TaskingPresets {
  sensor_presets: Record<string, { ground_ids: string[]; space: string[] }>;
  default_object_ids: string[];
  methods: TaskingMethod[];
  milp_available: boolean;
  defaults: { slot_min: number; horizon_h: number; custody_threshold_km: number; q_psd: number; initial_sigma_km: number };
  assumptions: string[];
}
export interface TaskingRequest {
  object_ids: 'default' | 'all_simulated' | 'all' | string[];
  preset?: string;
  sensor_ids?: string[];
  t0: string;
  t1?: string;
  slot_min?: number;
  method: TaskingMethod;
  custody_threshold_km?: number;
  gain_kind?: 'logdet' | 'trace' | 'maxeig';
  time_budget_s?: number;
  include_series?: boolean;
  seed?: number;
}
export interface TaskingAssignment {
  slot: number;
  t_s: number;
  sensor_id: string;
  object_id: string;
  gain: number;
  p_acq: number;
  magnitude?: number | null;
  range_km?: number | null;
  sigma_before_km: number;
  sigma_after_km: number;
}
export interface TaskingPerObject {
  id: string;
  name: string;
  orbit_type: string;
  custody_pct: number;
  n_obs: number;
  never_observed: boolean;
  mean_tslo_h: number;
  max_tslo_h: number;
  initial_sigma_km: number;
  final_sigma_km: number;
  max_sigma_km: number;
  sigma_series_km?: number[];
  in_custody_series?: boolean[];
  observed_by: string[];
}
export interface TaskingComparisonRow {
  method: string;
  custody_pct: number;
  custody_pct_min_object: number;
  mean_tslo_h: number;
  max_tslo_h: number;
  n_observations: number;
  summed_trace_mean_km2: number;
  summed_trace_final_km2: number;
  mean_final_sigma_km: number;
  sensor_utilisation_pct: number;
  runtime_s: number;
  custody_pct_null?: number;
}
export interface TaskingResponse {
  method: string;
  config: { object_ids: string[]; sensor_ids: string[]; slot_min: number; n_slots: number; custody_threshold_km: number; t0_utc: string; t1_utc: string; gain_kind: string; acquisition: string };
  schedule: TaskingAssignment[];
  per_object: TaskingPerObject[];
  sensors: { id: string; n_obs: number; utilisation_pct: number; slots_with_candidates: number; blocked_reason_counts: Record<string, number>; objects_observed: string[] }[];
  overall: { custody_pct: number; custody_pct_null: number; mean_tslo_h: number; max_tslo_h: number; n_observations: number; n_objects: number; n_sensors: number; n_slots: number; horizon_h: number; summed_trace_mean_km2: number; summed_trace_series_km2?: number[]; budget_exhausted?: boolean; custody_note?: string };
  comparison: TaskingComparisonRow[];
  timing: Record<string, number> & { total_s: number };
  epochs_utc: string[];
  assumptions: string[];
  disclaimer: string;
}

// ---------------------------------------------------------------------------
// GET /api/architecture/presets (subset the studio uses)
export interface ArchitecturePresetSensor {
  platform: string;
  aperture_m: number | null;
  limiting_mag: number | null;
  fov_deg: number;
  slew_rate_deg_s: number;
  lon_deg?: number;
  phase?: number;
  label?: string;
}
export interface ArchitecturePresets {
  presets: { name: string; ground: boolean; notes?: string; sensors: ArchitecturePresetSensor[] }[];
  platforms: { id: string; platform_orbit: string; aliases?: string[]; [k: string]: unknown }[];
  defaults?: Record<string, unknown>;
  limits?: Record<string, unknown>;
  method_notes?: string[];
  metric_definitions?: Record<string, string>;
  disclaimer?: string;
}
