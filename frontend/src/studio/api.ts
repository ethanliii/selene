/**
 * Fetch helpers for the studio pages (PLAN.md §5 shapes) with browser-side MOCK fallbacks.
 * Independent of src/api/client.ts (owned by another track) but follows the same policy:
 * network failure / gateway errors / the Vite proxy's empty 500 => mock; real 4xx/5xx => thrown.
 *
 * A route that answered 404 / unreachable is remembered for UNAVAILABLE_TTL_MS so that repeated
 * mock runs do not keep producing "Failed to load resource" console errors; the route is re-probed
 * after the TTL (or when the user explicitly re-runs after the TTL) so a backend started later is
 * picked up.
 */
import { markEndpointLive } from '../api/client';
import { mockArchitectureEvaluate } from './mockArchitecture';
import { mockCoverage } from './mockCoverage';
import {
  BACKEND_PRESET,
  BACKEND_REASON_MAP,
  REASON_CODES,
  type ArchitectureEvaluateRequest,
  type ArchitectureEvaluateResponse,
  type ArchitectureScore,
  type CandidateOrbit,
  type CoverageFrame,
  type CoverageRequest,
  type CoverageResponse,
  type LiveCoverageRequest,
  type LiveCoverageResponse,
} from './types';

/**
 * TDB seconds past J2000.0 → UTC ISO string for display. J2000.0 = 2000-01-01T12:00:00 TDB; for epochs
 * after 2017-01-01 TT − UTC = 32.184 + 37 = 69.184 s (TDB − TT < 2 ms), so UTC = TDB − 69.184 s.
 * Exact only for post-2017 epochs (the demo era); earlier dates would be off by past leap seconds.
 */
function tdbSecondsToIso(t: number): string {
  const J2000_MS_AS_IF_UTC = Date.UTC(2000, 0, 1, 12, 0, 0, 0);
  return new Date(J2000_MS_AS_IF_UTC + (t - 69.184) * 1000).toISOString();
}

const codeOf = new Map(REASON_CODES.map((r, i) => [r, i] as const));
const reasonCode = (name: string, covered: boolean) => codeOf.get(covered ? 'covered' : BACKEND_REASON_MAP[name] ?? 'too_faint') ?? 0;

/** Adapt the live backend's time-averaged coverage product to the studio shape. */
export function adaptLiveCoverage(live: LiveCoverageResponse): CoverageResponse {
  const xs = live.grid.xs;
  const ys = live.grid.ys;
  return {
    grid: { xmin: xs[0], xmax: xs[xs.length - 1], ymin: ys[0], ymax: ys[ys.length - 1], nx: xs.length, ny: ys.length, x: xs, y: ys },
    values: [],
    epochs: live.per_time.map((p) => tdbSecondsToIso(p.t)),
    sensors_used: live.meta?.sensor_ids,
    averaged: {
      coverage: live.coverage,
      reasons: live.reason_dominant_name.map((n, i) => reasonCode(n, live.coverage[i] >= 1)),
      per_time_pct: live.per_time.map((p) => p.pct),
      n_sensors_visible: live.n_sensors_visible,
      note: typeof live.meta?.note === 'string' ? live.meta.note : undefined,
    },
  };
}

/**
 * Adapt a 1-second, n_t = 2 live response to a single per-epoch frame. Over one second the geometry
 * does not change, so coverage is exactly 0 or 1 and n_sensors_visible is the integer sensor count.
 */
export function adaptLiveFrame(live: LiveCoverageResponse): CoverageFrame {
  const values = live.n_sensors_visible.map((v, i) => (live.coverage[i] > 0 ? Math.max(1, Math.round(v)) : 0));
  const reasons = live.reason_dominant_name.map((n, i) => reasonCode(n, values[i] > 0));
  return { values, reasons };
}

export class StudioApiError extends Error {
  constructor(public status: number, public path: string, public body: string) {
    super(`API ${status} ${path}: ${body.slice(0, 200)}`);
    this.name = 'StudioApiError';
  }
}

export interface Result<T> {
  data: T;
  mock: boolean;
  elapsed_ms: number;
}

const UNAVAILABLE_TTL_MS = 60_000;
const unavailableUntil = new Map<string, number>();

/** True when the route recently failed as unreachable (skip the fetch, go straight to the mock). */
export function routeUnavailable(path: string): boolean {
  const until = unavailableUntil.get(path);
  return until !== undefined && performance.now() < until;
}

async function postWithFallback<T>(path: string, body: unknown, fallback: () => T, signal?: AbortSignal): Promise<Result<T>> {
  const started = performance.now();
  if (routeUnavailable(path)) {
    const data = fallback();
    return { data, mock: true, elapsed_ms: performance.now() - started };
  }
  let res: Response;
  try {
    res = await fetch('/api' + path, {
      method: 'POST',
      headers: { Accept: 'application/json', 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
      signal,
    });
  } catch (e) {
    if (signal?.aborted) throw e;
    unavailableUntil.set(path, performance.now() + UNAVAILABLE_TTL_MS);
    const data = fallback();
    return { data, mock: true, elapsed_ms: performance.now() - started };
  }
  if (!res.ok) {
    const text = await res.text().catch(() => '');
    // Only "no backend answered" (gateway 502–504, the Vite proxy's empty-body 500) is the offline fallback; a 404
    // (route missing on a stale backend) or any real error is shown inline, never masked by the schematic model.
    const unreachable = res.status === 502 || res.status === 503 || res.status === 504 || (res.status === 500 && text.trim() === '');
    if (unreachable) {
      unavailableUntil.set(path, performance.now() + UNAVAILABLE_TTL_MS);
      const data = fallback();
      return { data, mock: true, elapsed_ms: performance.now() - started };
    }
    throw new StudioApiError(res.status, path, text);
  }
  unavailableUntil.delete(path);
  const data = (await res.json()) as T;
  return { data, mock: false, elapsed_ms: performance.now() - started };
}

function toLive(req: CoverageRequest): LiveCoverageRequest {
  return {
    t0: req.t0,
    t1: req.t1,
    n_t: req.n_t,
    network: BACKEND_PRESET[(req.network ?? 'ground') as keyof typeof BACKEND_PRESET] ?? req.network ?? 'ground_only',
    grid: '2d',
    radius_m: req.target_radius_m ?? 1.0,
    albedo: req.target_albedo ?? 0.2,
  };
}

export const studioApi = {
  coverage: async (req: CoverageRequest): Promise<Result<CoverageResponse>> => {
    const r = await postWithFallback<LiveCoverageResponse | CoverageResponse>('/coverage', toLive(req), () => mockCoverage(req));
    if (r.mock) return r as Result<CoverageResponse>;
    const data = r.data as LiveCoverageResponse;
    if (!Array.isArray(data.coverage) || !data.grid?.xs) throw new StudioApiError(502, '/coverage', 'unexpected response shape from backend');
    markEndpointLive('coverage');
    return { ...r, data: adaptLiveCoverage(data) };
  },
  /**
   * Live-only: the per-epoch field at `epochIso` for the same network/target as `req`, computed by
   * the backend over a 1-second window (n_t = 2). Throws when the backend is unreachable (no mock:
   * mixing mock frames with a live averaged product would be misleading).
   */
  coverageFrame: async (req: CoverageRequest, epochIso: string, signal?: AbortSignal): Promise<CoverageFrame> => {
    const t0ms = Date.parse(epochIso);
    const live: LiveCoverageRequest = { ...toLive(req), t0: new Date(t0ms).toISOString(), t1: new Date(t0ms + 1000).toISOString(), n_t: 2 };
    const res = await fetch('/api/coverage', {
      method: 'POST',
      headers: { Accept: 'application/json', 'Content-Type': 'application/json' },
      body: JSON.stringify(live),
      signal,
    });
    if (!res.ok) throw new StudioApiError(res.status, '/coverage', await res.text().catch(() => ''));
    const data = (await res.json()) as LiveCoverageResponse;
    if (!Array.isArray(data.coverage)) throw new StudioApiError(502, '/coverage', 'unexpected response shape from backend');
    return adaptLiveFrame(data);
  },
  /**
   * POST /api/architecture/evaluate. The backend route owns a different contract (platform names, `ground`,
   * n_mc ≤ 16, horizon ≤ 7 d, a work-unit guard) and a richer score object; `toLiveArchitecture` /
   * `adaptLiveArchitecture` translate both ways. Offline / 404 → the browser schematic model (mock: true).
   */
  architectureEvaluate: async (req: ArchitectureEvaluateRequest): Promise<Result<ArchitectureEvaluateResponse>> => {
    const live = toLiveArchitecture(req);
    const r = await postWithFallback<LiveArchitectureResponse | ArchitectureEvaluateResponse>('/architecture/evaluate', live, () => mockArchitectureEvaluate(req));
    if (r.mock) return r as Result<ArchitectureEvaluateResponse>;
    const data = r.data as LiveArchitectureResponse;
    if (!Array.isArray(data.scores)) throw new StudioApiError(502, '/architecture/evaluate', 'unexpected response shape from backend');
    markEndpointLive('architecture');
    return { ...r, data: adaptLiveArchitecture(data, live) };
  },
};

// ---- architecture: live contract adapters -----------------------------------------------------
const LIVE_PLATFORM: Record<CandidateOrbit, string> = { GEO: 'geo', L1_halo: 'l1_halo', L2_halo: 'l2_halo_S', DRO: 'dro', resonant_3_1: 'resonant_3_1' };
/** Backend caps (backend/selene/api/routes/architecture.py): n_mc ≤ 16, horizon ≤ 7 d, architectures × draws × slots ≤ 12 000. */
const LIVE_MAX_N_MC = 16;
const LIVE_MAX_HORIZON_D = 7;
const LIVE_MAX_WORK_UNITS = 12_000;

export interface LiveArchitectureRequest {
  architectures: { name: string; ground: boolean; sensors: { platform: string; aperture_m: number; limiting_mag: number; fov_deg: number; slew_rate_deg_s: number; phase: number; lon_deg?: number }[] }[];
  n_mc: number;
  horizon_days: number;
  seed: number;
  slot_min: number;
  workers: number;
}

export interface LiveArchitectureScore {
  name: string;
  coverage_pct: number;
  custody_pct: number;
  custody_pct_null?: number;
  revisit_mean_h: number;
  revisit_p95_h?: number | null;
  detection_latency_mean_h?: number | null;
  detection_latency_p95_h?: number | null;
  detection_latency_censored_mean_h?: number | null;
  detection_latency_censored_p95_h?: number | null;
  detections_pct?: number | null;
  revisit_censored_objects_pct?: number;
  n_burns?: number;
  n_detected?: number;
  n_space_sensors?: number;
  notes?: string;
  coverage_pct_p05?: number;
  coverage_pct_p95?: number;
  custody_pct_p05?: number;
  custody_pct_p95?: number;
  revisit_mean_h_p05?: number;
  revisit_mean_h_p95?: number;
}

export interface LiveArchitectureResponse {
  scores: LiveArchitectureScore[];
  meta?: { n_mc?: number; timing?: { total_s?: number }; method_notes?: unknown; custody_threshold_km?: number; caps_applied?: string[] };
  disclaimer?: string;
  method?: string;
}

/** Clamp to the backend caps and pick the coarsest slot that satisfies the work-unit guard (20 → 240 min). */
export function toLiveArchitecture(req: ArchitectureEvaluateRequest): LiveArchitectureRequest {
  const n_mc = Math.max(1, Math.min(LIVE_MAX_N_MC, Math.round(req.n_mc)));
  const horizon_days = Math.max(0.5, Math.min(LIVE_MAX_HORIZON_D, req.horizon_days));
  const nArch = Math.max(1, req.architectures.length);
  const units = (slot: number) => nArch * n_mc * Math.floor((horizon_days * 1440) / slot);
  let slot_min = 20;
  while (units(slot_min) > LIVE_MAX_WORK_UNITS && slot_min < 240) slot_min += 10;
  return {
    architectures: req.architectures.map((a) => ({
      name: a.name,
      ground: a.ground_network,
      sensors: a.sensors.map((s) => ({ platform: LIVE_PLATFORM[s.orbit] ?? 'dro', aperture_m: s.aperture_m, limiting_mag: s.limiting_mag, fov_deg: s.fov_deg, slew_rate_deg_s: s.slew_rate_dps, phase: Math.min(0.99, Math.max(0, s.phase)), ...(typeof s.lon_deg === 'number' ? { lon_deg: s.lon_deg } : {}) })),
    })),
    n_mc,
    horizon_days,
    seed: req.seed ?? 0,
    slot_min: Math.min(240, slot_min),
    workers: 2,
  };
}

/** Backend score → studio score (latency = censored statistics, matching the explainer's definition). */
export function adaptLiveArchitecture(live: LiveArchitectureResponse, req: LiveArchitectureRequest): ArchitectureEvaluateResponse {
  const horizonH = req.horizon_days * 24;
  const scores: ArchitectureScore[] = live.scores.map((s) => ({
    name: s.name,
    coverage_pct: s.coverage_pct,
    custody_pct: s.custody_pct,
    revisit_h: s.revisit_mean_h,
    detect_latency_h_mean: s.detection_latency_censored_mean_h ?? s.detection_latency_mean_h ?? horizonH,
    detect_latency_h_p95: s.detection_latency_censored_p95_h ?? s.detection_latency_p95_h ?? horizonH,
    n_mc: live.meta?.n_mc ?? req.n_mc,
    n_sensors: s.n_space_sensors,
    n_space_sensors: s.n_space_sensors,
    undetected_pct: typeof s.detections_pct === 'number' ? Math.max(0, 100 - s.detections_pct) : undefined,
    never_observed_pct: s.revisit_censored_objects_pct,
    coverage_pct_p05: s.coverage_pct_p05,
    coverage_pct_p95: s.coverage_pct_p95,
    custody_pct_p05: s.custody_pct_p05,
    custody_pct_p95: s.custody_pct_p95,
    revisit_h_p05: s.revisit_mean_h_p05,
    revisit_h_p95: s.revisit_mean_h_p95,
    custody_pct_null: s.custody_pct_null,
    notes: s.notes,
  }));
  const notes = Array.isArray(live.meta?.method_notes) ? (live.meta!.method_notes as unknown[]).filter((x): x is string => typeof x === 'string') : undefined;
  return {
    scores,
    mock: false,
    method: live.method ?? `backend Monte Carlo · ${live.meta?.n_mc ?? req.n_mc} draws · ${req.horizon_days} d · ${req.slot_min}-min slots · custody threshold ${live.meta?.custody_threshold_km ?? 100} km`,
    method_notes: notes,
    caps_applied: live.meta?.caps_applied,
    disclaimer: live.disclaimer,
    custody_threshold_km: live.meta?.custody_threshold_km,
  };
}
