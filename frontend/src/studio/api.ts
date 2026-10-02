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
import { mockArchitectureEvaluate } from './mockArchitecture';
import { mockCoverage } from './mockCoverage';
import {
  BACKEND_PRESET,
  BACKEND_REASON_MAP,
  REASON_CODES,
  type ArchitectureEvaluateRequest,
  type ArchitectureEvaluateResponse,
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

/** True when the route recently failed as unreachable/404 (skip the fetch, go straight to the mock). */
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
    const unreachable = res.status === 502 || res.status === 503 || res.status === 504 || (res.status === 500 && text.trim() === '') || res.status === 404;
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
  architectureEvaluate: (req: ArchitectureEvaluateRequest) =>
    postWithFallback<ArchitectureEvaluateResponse>('/architecture/evaluate', req, () => mockArchitectureEvaluate(req)),
};
