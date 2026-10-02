/**
 * Types local to the Coverage page and the Architecture Studio (frontend/src/studio/**).
 *
 * These mirror PLAN.md §5 for `POST /api/coverage` and `POST /api/architecture/evaluate`, extended
 * with the optional fields the mock generator produces (`reasons`) and the studio sends
 * (`ground_network`, `horizon_days`, sensor spec fields). Unknown extra fields are ignored by a
 * pydantic backend with default config, so sending them is safe.
 *
 * Units: rotating-frame grid coordinates are nondimensional Earth–Moon CR3BP units
 * (1 L* = 384 400 km, barycenter at the origin, Moon at (1−μ, 0, 0)); times are ISO-8601 UTC;
 * angles in degrees in API payloads (radians inside the model code); magnitudes are visual magnitudes.
 */

export type Vec3 = [number, number, number];

// ---------------------------------------------------------------------------------------------
// Coverage

/** Why a cell is blind. `covered` is used for cells with ≥1 sensor. */
export type BlindReason =
  | 'covered'
  | 'daylight' // every ground site that could point there has the Sun above −12° elevation
  | 'horizon' // night-side sites exist but the target is below the elevation mask
  | 'lunar_glare' // line of sight within the Moon exclusion angle
  | 'sun_exclusion' // space observer: line of sight within the Sun exclusion angle
  | 'earth_exclusion' // space observer: line of sight within the Earth-limb exclusion angle
  | 'too_faint' // apparent magnitude fainter than the limiting magnitude
  | 'shadow' // target inside the Earth or Moon umbra
  | 'out_of_fov'; // backend only: outside the pointed field of view

/** Stable numeric codes used in `CoverageResponse.reasons` (Uint8-friendly). */
export const REASON_CODES: BlindReason[] = [
  'covered',
  'daylight',
  'horizon',
  'lunar_glare',
  'sun_exclusion',
  'earth_exclusion',
  'too_faint',
  'shadow',
  'out_of_fov',
];

/** Backend reason names (backend/selene/sensors/reasons.py REASON_NAMES) → studio reasons. */
export const BACKEND_REASON_MAP: Record<string, BlindReason> = {
  covered: 'covered',
  daylight: 'daylight',
  low_elevation: 'horizon',
  sun_exclusion: 'sun_exclusion',
  moon_exclusion: 'lunar_glare',
  earth_exclusion: 'earth_exclusion',
  in_shadow: 'shadow',
  too_faint: 'too_faint',
  out_of_fov: 'out_of_fov',
};

export const REASON_LABELS: Record<BlindReason, string> = {
  covered: 'Covered (≥1 sensor)',
  daylight: 'Daylight (ground sites in twilight/day)',
  horizon: 'Below site elevation mask',
  lunar_glare: 'Lunar glare (Moon exclusion)',
  sun_exclusion: 'Sun exclusion (space observer)',
  earth_exclusion: 'Earth-limb exclusion (space observer)',
  too_faint: 'Too faint (m > m_lim)',
  shadow: 'Earth/Moon shadow',
  out_of_fov: 'Outside pointed field of view',
};

/** Studio preset → backend preset name (backend/selene/sensors/coverage.py NETWORK_PRESETS). */
export const BACKEND_PRESET: Record<NetworkPreset, string> = {
  ground: 'ground_only',
  'ground+geo': 'ground_plus_geo',
  'ground+l2': 'ground_plus_l2_halo',
  'ground+dro': 'ground_plus_dro',
  'ground+all': 'full',
};
/** Backend preset name → the closest studio (mock) preset, for the offline fallback. */
export function toStudioPreset(network: string | undefined): NetworkPreset {
  if (!network) return 'ground';
  if ((Object.keys(BACKEND_PRESET) as NetworkPreset[]).includes(network as NetworkPreset)) return network as NetworkPreset;
  const inv = (Object.entries(BACKEND_PRESET) as [NetworkPreset, string][]).find(([, b]) => b === network);
  if (inv) return inv[0];
  if (/geo/.test(network)) return 'ground+geo';
  if (/l2|l1|nrho/.test(network)) return 'ground+l2';
  if (/dro/.test(network)) return 'ground+dro';
  if (/full|space/.test(network)) return 'ground+all';
  return 'ground';
}
/** Human label for a backend preset id ("ground_plus_l2_halo" → "+ L2 halo"). */
export function backendPresetLabel(id: string): string {
  const known: Record<string, string> = { ground_only: 'Ground only', full: 'Full network', space_only: 'Space only', ground_plus_geo: '+ GEO ×2', ground_plus_l1_halo: '+ L1 halo', ground_plus_l2_halo: '+ L2 halo', ground_plus_dro: '+ DRO', ground_plus_nrho: '+ NRHO' };
  if (known[id]) return known[id];
  return id.replace(/^ground_plus_/, '+ ').replace(/_/g, ' ');
}

export type NetworkPreset = 'ground' | 'ground+geo' | 'ground+l2' | 'ground+dro' | 'ground+all';

/**
 * Preset descriptions. `hint` describes the browser MOCK network; `live_hint` the backend preset the
 * studio maps to (backend/selene/sensors/coverage.py NETWORK_PRESETS: 9 named ground sites, GEO
 * preset = two GEO observers, 'full' = GEO×2 + L1 halo + L2 halo + DRO). The page shows the hint
 * that matches the source of the last run.
 */
export const NETWORK_PRESETS: { id: NetworkPreset; label: string; hint: string; live_hint: string }[] = [
  { id: 'ground', label: 'Ground only', hint: 'mock: 3 notional 1-m class sites (equatorial approximation)', live_hint: 'backend ground_only: 9 named ground sites' },
  { id: 'ground+geo', label: '+ GEO observer', hint: 'mock: adds one GEO-hosted telescope', live_hint: 'backend ground_plus_geo: 9 sites + two GEO observers (west, east)' },
  { id: 'ground+l2', label: '+ L2 halo observer', hint: 'mock: adds one observer on an L2 southern halo', live_hint: 'backend ground_plus_l2_halo: 9 sites + one L2 halo observer' },
  { id: 'ground+dro', label: '+ DRO observer', hint: 'mock: adds one observer on a 70 000 km DRO', live_hint: 'backend ground_plus_dro: 9 sites + one DRO observer' },
  { id: 'ground+all', label: '+ GEO + L2 + DRO', hint: 'mock: all three space observers', live_hint: 'backend full: 9 sites + GEO×2 + L1 halo + L2 halo + DRO observers' },
];

export interface CoverageGridSpec {
  xmin: number;
  xmax: number;
  ymin: number;
  ymax: number;
  nx: number;
  ny: number;
  zmin?: number;
  zmax?: number;
  nz?: number;
}

export interface CoverageRequest {
  t0: string;
  t1: string;
  n_t: number;
  grid: CoverageGridSpec;
  sensor_ids?: string[];
  /** Studio preset id, or a backend preset name straight from GET /api/coverage/presets (live list). */
  network?: NetworkPreset | string;
  /** Reference target radius (m) and geometric albedo for detectability. */
  target_radius_m?: number;
  target_albedo?: number;
}

export interface CoverageResponse {
  grid: CoverageGridSpec & { x: number[]; y: number[]; z?: number[] };
  /**
   * values[t][i]: number of sensors able to detect the reference target at cell i, epoch t
   * (0 = blind). Cell index convention i = ix * ny + iy (x-major), matching src/api/mock.ts.
   */
  values: number[][];
  epochs: string[];
  /** Optional: dominant blind reason code per cell per epoch (REASON_CODES index). Mock only. */
  reasons?: number[][];
  /** Sensor ids used, for the HUD. */
  sensors_used?: string[];
  /**
   * Time-averaged product as served by the live backend (which does not return the per-epoch
   * field). When present and `values` is empty, the page runs in averaged-only mode.
   */
  averaged?: {
    /** Fraction of epochs with ≥1 sensor, per cell. */
    coverage: number[];
    /** Dominant blind reason code (REASON_CODES index) per cell over the window. */
    reasons: number[];
    /** Coverage % of the slice per epoch (aligned with `epochs`). */
    per_time_pct: number[];
    /** Mean number of sensors able to see each cell. */
    n_sensors_visible: number[];
    note?: string;
  };
}

/**
 * One per-epoch field fetched lazily from the live backend (the averaged product does not carry
 * per-epoch fields). Obtained by asking for a 1-second window with n_t = 2 around the epoch, so
 * `values` is the integer sensor count and `reasons` the dominant reason at that instant.
 */
export interface CoverageFrame {
  values: number[];
  reasons: number[];
}

/** Live backend request/response (backend/selene/api/routes/coverage.py). */
export interface LiveCoverageRequest {
  t0: string;
  t1: string;
  n_t: number;
  network: string;
  grid: '2d' | '3d';
  radius_m: number;
  albedo: number;
}
export interface LiveCoverageResponse {
  grid: { kind: string; xs: number[]; ys: number[]; shape: number[] };
  coverage: number[];
  reason_dominant: number[];
  reason_dominant_name: string[];
  reason_fraction: Record<string, number[]>;
  n_sensors_visible: number[];
  /** t = TDB seconds past J2000.0. */
  per_time: { t: number; pct: number }[];
  meta: { sensor_ids?: string[]; note?: string; mean_coverage_pct?: number; [k: string]: unknown };
}

export interface CoverageRunMeta {
  mock: boolean;
  elapsed_ms: number;
  note?: string;
}

// ---------------------------------------------------------------------------------------------
// Architecture studio

export type CandidateOrbit = 'GEO' | 'L1_halo' | 'L2_halo' | 'DRO' | 'resonant_3_1';

export interface CandidateOrbitInfo {
  id: CandidateOrbit;
  label: string;
  /** One-line description shown in the palette. */
  blurb: string;
  /** Default limiting magnitude for a sensor hosted there (visual mag). */
  default_limiting_mag: number;
}

export const CANDIDATE_ORBITS: CandidateOrbitInfo[] = [
  { id: 'GEO', label: 'GEO', blurb: 'Hosted payload on a geostationary bus, r = 42 164 km', default_limiting_mag: 18.0 },
  { id: 'L1_halo', label: 'L1 halo', blurb: 'Earth–Moon L1 halo (schematic, ~12 d period)', default_limiting_mag: 18.5 },
  { id: 'L2_halo', label: 'L2 halo (southern)', blurb: 'Earth–Moon L2 southern halo (schematic, ~14.5 d)', default_limiting_mag: 18.5 },
  { id: 'DRO', label: 'DRO', blurb: 'Distant retrograde orbit, ~70 000 km x-crossing (~13 d)', default_limiting_mag: 18.5 },
  { id: 'resonant_3_1', label: '3:1 resonant', blurb: 'Earth-centred 3:1 lunar-resonant ellipse (a ≈ 0.48 L*)', default_limiting_mag: 18.0 },
];

export interface SensorSpec {
  id: string;
  orbit: CandidateOrbit;
  /** Primary aperture, m (passed through; the mock scores on limiting magnitude). */
  aperture_m: number;
  limiting_mag: number;
  fov_deg: number;
  slew_rate_dps: number;
  /** Phase along the host orbit, fraction of period [0,1). */
  phase: number;
}

export interface ArchitectureDef {
  id: string;
  name: string;
  sensors: SensorSpec[];
  /** Include the 3 notional ground sites. */
  ground_network: boolean;
  /** Colour slot (ARCH_COLORS index) assigned at creation and kept for the architecture's lifetime. */
  slot: number;
}

export interface ArchitectureEvaluateRequest {
  architectures: {
    name: string;
    ground_network: boolean;
    sensors: {
      orbit: CandidateOrbit;
      aperture_m: number;
      limiting_mag: number;
      fov_deg: number;
      slew_rate_dps: number;
      phase: number;
    }[];
  }[];
  n_mc: number;
  horizon_days: number;
  seed?: number;
  /** Reference target population: diffuse-sphere radius (m) and geometric albedo. */
  target_radius_m?: number;
  target_albedo?: number;
}

export interface ArchitectureScore {
  name: string;
  coverage_pct: number;
  custody_pct: number;
  revisit_h: number;
  detect_latency_h_mean: number;
  detect_latency_h_p95: number;
  n_mc: number;
  /** Extra diagnostics the mock provides. */
  n_sensors?: number;
  undetected_pct?: number;
  never_observed_pct?: number;
}

export interface ArchitectureEvaluateResponse {
  scores: ArchitectureScore[];
  /** Present when generated by the browser-side mock. */
  mock?: boolean;
  method?: string;
}
