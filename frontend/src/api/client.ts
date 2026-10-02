/**
 * Typed fetch client for the SELENE API (PLAN.md §5). Base path '/api' is proxied by Vite in dev
 * and served by uvicorn in production.
 *
 * Mock mode: if a request fails at the network level (backend down) or the health probe fails,
 * the client flips `backendStatus` to 'offline' and every call returns data from ./mock.ts.
 * The UI watches `useBackendStatus()` and shows "BACKEND OFFLINE — MOCK DATA".
 * HTTP errors with a response (4xx/5xx) are thrown as ApiError, NOT mocked, so real backend bugs stay visible.
 *
 * Per-endpoint liveness: every call records whether its data came from the backend ('live') or the browser
 * mock ('mock') under a short endpoint name (`useEndpointStatus()`), so the banner can say exactly which parts
 * of the screen are live. Routes that are not implemented yet are detected cheaply by POSTing an invalid body
 * (`[]`): FastAPI answers 422 (pydantic rejects the body before the handler runs) or 405 (GET-only route) for an
 * existing route and 404 for a missing one. (OPTIONS cannot be used: the Vite dev proxy answers it itself.)
 */
import { useSyncExternalStore } from 'react';
import * as mock from './mock';
import type {
  ArchitecturePresets,
  ManeuverDetectRequest,
  ManeuverDetectResponse,
  OdPresets,
  OdRunRequest,
  OdRunResponse,
  ReachRegionsResponse,
  ReachRequest,
  ReachResponse,
  TaskingPresets,
  TaskingRequest,
  TaskingResponse,
} from './analysisTypes';
import type {
  CatalogMeta,
  CatalogObject,
  CatalogResponse,
  CoveragePresets,
  CoverageRequest,
  CoverageResponse,
  DemoScenario,
  EphemerisBodies,
  Health,
  OrbitFamilies,
  OrbitRecord,
  Sensors,
  TrajectoryFrame,
  TrajectoryResponse,
  VisibilityResponse,
} from './types';

export const API_BASE = '/api';

/** 'partial' = backend reachable but some endpoints are not implemented yet (404) and were served from mock. */
export type BackendStatus = 'unknown' | 'online' | 'partial' | 'offline';

export class ApiError extends Error {
  /**
   * `unreachable` is true when no backend answered at all: a network failure, a 502/503/504 gateway answer, or
   * the Vite dev proxy's empty-body 500 when uvicorn is down. Callers with an EXPLICIT offline fallback (the demo
   * scenario) key on this flag, never on the status alone (the dev proxy says 500, not 502).
   */
  constructor(public status: number, public path: string, public body: string, public unreachable = false) {
    super(`API ${status} ${path}: ${body.slice(0, 200)}`);
    this.name = 'ApiError';
  }
}

// ---- tiny external store for backend status (no zustand dependency in the API layer) ----------
let status: BackendStatus = 'unknown';
const listeners = new Set<() => void>();
function setStatus(s: BackendStatus) {
  // 'online' never downgrades an already-'partial' session (the missing endpoints are still missing).
  if (s === 'online' && status === 'partial') return;
  if (s === status) return;
  status = s;
  listeners.forEach((l) => l());
}
export function getBackendStatus(): BackendStatus {
  return status;
}
export function useBackendStatus(): BackendStatus {
  return useSyncExternalStore(
    (l) => {
      listeners.add(l);
      return () => listeners.delete(l);
    },
    () => status,
  );
}

// ---- per-endpoint liveness registry -----------------------------------------------------------
/** Short names used in the banner, in display order. */
export const ENDPOINTS = ['catalog', 'trajectory', 'orbits', 'ephemeris', 'sensors', 'coverage', 'OD', 'maneuver', 'reachability', 'tasking', 'architecture', 'demo'] as const;
export type EndpointName = (typeof ENDPOINTS)[number];
/**
 * 'live' = the UI fetched data from this route in this session; 'available' = the route exists on the backend
 * (OpenAPI probe) but nothing on screen has consumed it yet; 'mock' = served by the browser fallback.
 */
export type EndpointState = 'unknown' | 'live' | 'available' | 'mock';
export type EndpointStatus = Readonly<Record<EndpointName, EndpointState>>;

let endpointStatus: EndpointStatus = Object.fromEntries(ENDPOINTS.map((e) => [e, 'unknown'])) as Record<EndpointName, EndpointState>;
const epListeners = new Set<() => void>();
function setEndpoint(name: EndpointName, st: EndpointState) {
  if (endpointStatus[name] === st) return;
  endpointStatus = { ...endpointStatus, [name]: st };
  epListeners.forEach((l) => l());
}
export function getEndpointStatus(): EndpointStatus {
  return endpointStatus;
}
/** For callers that fetch outside this client (the studio pages): record a successful live fetch of a route. */
export function markEndpointLive(name: EndpointName): void {
  setStatus('online');
  setEndpoint(name, 'live');
}
export function useEndpointStatus(): EndpointStatus {
  return useSyncExternalStore(
    (l) => {
      epListeners.add(l);
      return () => epListeners.delete(l);
    },
    () => endpointStatus,
  );
}
/** Routes that may not exist yet (written by other tracks). Probed with an invalid POST: 422/405/400 = implemented, 404 = missing. */
const PROBES: { name: EndpointName; path: string }[] = [
  { name: 'coverage', path: '/coverage' },
  { name: 'OD', path: '/od/run' },
  { name: 'maneuver', path: '/maneuver/detect' },
  { name: 'reachability', path: '/reachability' },
  { name: 'tasking', path: '/tasking/schedule' },
  { name: 'architecture', path: '/architecture/evaluate' },
  { name: 'demo', path: '/demo/scenario' },
];
let probed = false;
/**
 * Probe the optional routes once. Preferred: read the backend's OpenAPI document (/openapi.json, proxied by the Vite
 * dev server) and look the paths up — one 200 request, no console noise. Fallback: an invalid POST per route
 * (validation fails before any handler runs). Safe to call repeatedly.
 */
export async function probeOptionalRoutes(): Promise<void> {
  if (probed) return;
  probed = true;
  try {
    const res = await fetch('/openapi.json', { headers: { Accept: 'application/json' } });
    if (res.ok) {
      const doc = (await res.json()) as { paths?: Record<string, unknown> };
      const paths = Object.keys(doc.paths ?? {});
      if (paths.length > 0) {
        // A route that exists is 'available', not 'live': only an actual successful fetch by the UI promotes it.
        for (const p of PROBES) if (endpointStatus[p.name] !== 'live') setEndpoint(p.name, paths.some((x) => x === API_BASE + p.path || x.startsWith(API_BASE + p.path + '/')) ? 'available' : 'mock');
        return;
      }
    }
  } catch {
    /* fall through to the per-route probe */
  }
  await Promise.all(
    PROBES.map(async (p) => {
      try {
        const res = await fetch(API_BASE + p.path, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: '[]' });
        const text = await res.text().catch(() => '');
        if (res.status === 404 && /"detail"\s*:\s*"Not Found"/.test(text)) setEndpoint(p.name, 'mock');
        else if ((res.status === 422 || res.status === 405 || res.status === 400 || res.ok) && endpointStatus[p.name] !== 'live') setEndpoint(p.name, 'available');
      } catch {
        /* backend down: the health call marks the session offline */
      }
    }),
  );
}

// ---- helpers ----------------------------------------------------------------------------------
/** The backend serialises UTC epochs without a zone designator ("2026-03-01T00:00:00.000"); JS would parse that as
 *  LOCAL time. Append 'Z' when no offset is present. */
export function utcIso(s: string): string {
  if (!s) return s;
  return /(Z|[+-]\d\d:?\d\d)$/.test(s) ? s : `${s}Z`;
}
/** TDB seconds past J2000.0 → UTC ms (TT−UTC = 69.184 s for epochs after 2017-01-01; TDB−TT < 2 ms). */
export function tdbSecondsToUtcMs(t: number): number {
  return Date.UTC(2000, 0, 1, 12, 0, 0, 0) + (t - 69.184) * 1000;
}

// ---- core request with mock fallback ---------------------------------------------------------
async function request<T>(name: EndpointName, path: string, init: RequestInit | undefined, fallback: () => T): Promise<T> {
  let res: Response;
  try {
    res = await fetch(API_BASE + path, {
      ...init,
      headers: { Accept: 'application/json', ...(init?.body ? { 'Content-Type': 'application/json' } : {}), ...init?.headers },
    });
  } catch (err) {
    // Network failure => backend down => mock mode.
    setStatus('offline');
    setEndpoint(name, 'mock');
    return fallback();
  }
  if (!res.ok) {
    const text = await res.text().catch(() => '');
    // Upstream unreachable: gateway 502/503/504, or the Vite dev proxy's ECONNREFUSED response, which is
    // "500 text/plain" with an EMPTY body (verified: `curl -i :5173/api/health` with uvicorn down).
    // A real FastAPI 500 carries a body ("Internal Server Error" / JSON detail) and is thrown instead.
    const gateway = res.status === 502 || res.status === 503 || res.status === 504;
    const proxyRefused = res.status === 500 && text.trim() === '';
    if (gateway || proxyRefused) {
      setStatus('offline');
      setEndpoint(name, 'mock');
      return fallback();
    }
    // Route not implemented yet (FastAPI's generic 404 body) → mock for that endpoint, flag the session 'partial'.
    if (res.status === 404 && /"detail"\s*:\s*"Not Found"/.test(text)) {
      console.warn(`API ${path} not implemented (404) — using mock data for this endpoint`);
      setStatus('partial');
      setEndpoint(name, 'mock');
      return fallback();
    }
    throw new ApiError(res.status, path, text);
  }
  setStatus('online');
  setEndpoint(name, 'live');
  return (await res.json()) as T;
}

const get = <T,>(name: EndpointName, path: string, fallback: () => T) => request<T>(name, path, undefined, fallback);
const post = <T,>(name: EndpointName, path: string, body: unknown, fallback: () => T) =>
  request<T>(name, path, { method: 'POST', body: JSON.stringify(body) }, fallback);

/**
 * Live-only request: NEVER falls back to the browser mock. A network failure marks the session offline and the
 * endpoint 'mock' (so the data-sources chip stays honest) but the error is thrown to the caller, which shows it
 * inline. Used by the analysis panels and the demo bundle: analysis results are either the backend's or absent.
 */
async function requestLive<T>(name: EndpointName, path: string, init?: RequestInit): Promise<T> {
  let res: Response;
  try {
    res = await fetch(API_BASE + path, {
      ...init,
      headers: { Accept: 'application/json', ...(init?.body ? { 'Content-Type': 'application/json' } : {}), ...init?.headers },
    });
  } catch (err) {
    if (init?.signal?.aborted) throw err;
    setStatus('offline');
    setEndpoint(name, 'mock');
    throw new ApiError(0, path, `backend unreachable (${err instanceof Error ? err.message : String(err)})`, true);
  }
  if (!res.ok) {
    const text = await res.text().catch(() => '');
    const gateway = res.status === 502 || res.status === 503 || res.status === 504 || (res.status === 500 && text.trim() === '');
    if (gateway) {
      setStatus('offline');
      setEndpoint(name, 'mock');
      throw new ApiError(res.status, path, 'backend unreachable (gateway error)', true);
    }
    // A FastAPI generic 404 (route missing on a stale backend) is thrown like any other error and shown inline;
    // the endpoint is NOT marked 'mock' (nothing mock is rendered), so the data-sources chip stays truthful.
    // FastAPI puts the message in {"detail": ...}; surface it verbatim (pydantic 422 details are a list).
    let detail = text;
    try {
      const j = JSON.parse(text) as { detail?: unknown };
      if (typeof j.detail === 'string') detail = j.detail;
      else if (Array.isArray(j.detail)) detail = j.detail.map((d) => (typeof d === 'object' && d && 'msg' in d ? `${(d as { loc?: unknown[] }).loc?.slice(1).join('.') ?? ''}: ${(d as { msg: string }).msg}` : JSON.stringify(d))).join('; ');
    } catch {
      /* not JSON */
    }
    throw new ApiError(res.status, path, detail);
  }
  setStatus('online');
  setEndpoint(name, 'live');
  return (await res.json()) as T;
}
const getLive = <T,>(name: EndpointName, path: string, signal?: AbortSignal) => requestLive<T>(name, path, signal ? { signal } : undefined);
const postLive = <T,>(name: EndpointName, path: string, body: unknown, signal?: AbortSignal) => requestLive<T>(name, path, { method: 'POST', body: JSON.stringify(body), signal });

function qs(params: Record<string, string | number | undefined>): string {
  const u = new URLSearchParams();
  for (const [k, v] of Object.entries(params)) if (v !== undefined) u.set(k, String(v));
  const s = u.toString();
  return s ? `?${s}` : '';
}

// ---- endpoints -------------------------------------------------------------------------------
/** Catalog envelope + objects (the mock has no envelope: `meta` is null then). */
export interface CatalogResult {
  objects: CatalogObject[];
  meta: CatalogMeta | null;
}

export const api = {
  health: () =>
    get<Health>('catalog', '/health', () => mock.mockHealth).then((h) => {
      if (!h.offline || h.status === 'ok') void probeOptionalRoutes();
      return h;
    }),

  /** Objects plus the live envelope (epoch, counts, disclaimer). */
  catalog: (): Promise<CatalogResult> =>
    get<CatalogResponse | CatalogObject[]>('catalog', '/catalog/objects', () => mock.mockCatalog()).then((raw) => {
      if (Array.isArray(raw)) return { objects: raw.map(adaptCatalogObject), meta: null };
      const { objects, ...meta } = raw;
      return { objects: (objects ?? []).map(adaptCatalogObject), meta: { ...meta, epoch_utc: utcIso(meta.epoch_utc) } };
    }),
  /** Back-compat: objects only. */
  catalogObjects: () => api.catalog().then((r) => r.objects),

  catalogObject: (id: string) => get<CatalogObject>('catalog', `/catalog/objects/${encodeURIComponent(id)}`, () => mock.mockCatalog().find((c) => c.id === id)!).then(adaptCatalogObject),

  /** Trajectory of one object over [t0, t1] with n samples (n ≤ 5000) in the requested frame. No mock: throws offline. */
  trajectory: (id: string, t0: string, t1: string, n: number, frame: TrajectoryFrame = 'rot_nd') =>
    get<TrajectoryResponse>('trajectory', `/catalog/objects/${encodeURIComponent(id)}/trajectory${qs({ t0, t1, n: Math.min(5000, Math.max(2, Math.round(n))), frame })}`, () => {
      throw new ApiError(0, `/catalog/objects/${id}/trajectory`, 'trajectory endpoint unavailable (no mock)');
    }).then((t) => ({ ...t, epochs_utc: t.epochs_utc.map(utcIso), t0_utc: utcIso(t.t0_utc), t1_utc: utcIso(t.t1_utc) })),

  orbitFamilies: (opts: { n_samples?: number; max_members?: number } = {}) => get<OrbitFamilies>('orbits', `/orbits/families${qs(opts)}`, () => mock.mockFamilies()),

  orbitRecord: (id: string) =>
    get<OrbitRecord>('orbits', `/orbits/records/${encodeURIComponent(id)}`, () => {
      throw new ApiError(0, `/orbits/records/${id}`, 'orbit record endpoint unavailable (no mock)');
    }),

  ephemerisBodies: (t0: string, t1: string, n: number) =>
    get<EphemerisBodies>('ephemeris', `/ephemeris/bodies${qs({ t0, t1, n })}`, () => mock.mockEphemeris(t0, t1, n)).then((e) => ({ ...e, epochs: e.epochs.map(utcIso) })),

  sensors: () => get<Sensors>('sensors', '/sensors', () => mock.mockSensors).then(adaptSensors),

  sensorVisibility: (sensorId: string, objectId: string, t0: string, t1: string, n: number) =>
    get<VisibilityResponse>('sensors', `/sensors/${encodeURIComponent(sensorId)}/visibility${qs({ object_id: objectId, t0, t1, n })}`, () => {
      throw new ApiError(0, `/sensors/${sensorId}/visibility`, 'visibility endpoint unavailable (no mock)');
    }),

  coverage: (req: CoverageRequest) => post<CoverageResponse>('coverage', '/coverage', req, () => mock.mockCoverage(req)),

  coveragePresets: () => get<CoveragePresets>('coverage', '/coverage/presets', () => ({}) as CoveragePresets),

  // ---- analysis routes: LIVE ONLY (errors are thrown and shown inline; nothing is ever mocked) ----
  odPresets: () => getLive<OdPresets>('OD', '/od/presets'),
  odRun: (req: OdRunRequest, signal?: AbortSignal) => postLive<OdRunResponse>('OD', '/od/run', req, signal),

  maneuverDetect: (req: ManeuverDetectRequest, signal?: AbortSignal) => postLive<ManeuverDetectResponse>('maneuver', '/maneuver/detect', req, signal),

  reachability: (req: ReachRequest, signal?: AbortSignal) => postLive<ReachResponse>('reachability', '/reachability', req, signal),
  reachabilityRegions: () => getLive<ReachRegionsResponse>('reachability', '/reachability/regions'),

  taskingPresets: () => getLive<TaskingPresets>('tasking', '/tasking/presets'),
  taskingSchedule: (req: TaskingRequest, signal?: AbortSignal) => postLive<TaskingResponse>('tasking', '/tasking/schedule', req, signal),

  architecturePresets: () => getLive<ArchitecturePresets>('architecture', '/architecture/presets'),

  /** The precomputed scenario bundle. LIVE ONLY: the demo driver decides explicitly whether to fall back. */
  demoScenario: () => getLive<DemoScenario>('demo', '/demo/scenario'),
  /** The browser-side mock story (CR3BP in the browser). Only the demo driver calls this, and labels the result. */
  demoScenarioMock: (): DemoScenario => {
    setEndpoint('demo', 'mock');
    return mock.mockDemo();
  },
};

export type Api = typeof api;

// ---- adapters --------------------------------------------------------------------------------
/** Normalise epochs and alias the live `state_rot_nd` to `ic_rot` (what the idle propagation reads). */
export function adaptCatalogObject(c: CatalogObject): CatalogObject {
  const out: CatalogObject = { ...c, epoch_utc: utcIso(c.epoch_utc) };
  if (!out.ic_rot && c.state_rot_nd) out.ic_rot = c.state_rot_nd;
  if (c.span_utc && c.span_utc.length === 2) out.span_utc = [utcIso(c.span_utc[0]), utcIso(c.span_utc[1])];
  if (out.radius_m === null) out.radius_m = undefined;
  if (out.albedo === null) out.albedo = undefined;
  return out;
}

/** Tolerate the backend's field names (platform_orbit, orbit_ref, alt_m, orbit as an object) alongside the §5 names. */
export function adaptSensors(raw: Sensors): Sensors {
  const anyRaw = raw as unknown as { ground?: Record<string, unknown>[]; space?: Record<string, unknown>[]; note?: string; reason_bits?: Record<string, number> };
  const ground = (anyRaw.ground ?? []).map((g) => ({
    ...g,
    alt_km: typeof g.alt_km === 'number' ? g.alt_km : typeof g.alt_m === 'number' ? (g.alt_m as number) / 1000 : 0,
    moon_exclusion_deg: typeof g.moon_exclusion_deg === 'number' ? g.moon_exclusion_deg : 15,
    sun_exclusion_deg: typeof g.sun_exclusion_deg === 'number' ? g.sun_exclusion_deg : 90,
  })) as unknown as Sensors['ground'];
  const space = (anyRaw.space ?? []).map((s) => {
    const orbitInfo = s.orbit && typeof s.orbit === 'object' ? (s.orbit as Record<string, unknown>) : undefined;
    const orbitClass = String(typeof s.orbit === 'string' ? s.orbit : (s.platform_orbit ?? 'unknown'));
    const src = typeof orbitInfo?.source === 'string' ? (orbitInfo.source as string) : '';
    const recordId = src.startsWith('library:') ? src.slice('library:'.length) : (s.orbit_member_id ?? s.orbit_ref ?? undefined);
    const epochS = typeof s.epoch_s === 'number' ? (s.epoch_s as number) : undefined;
    return {
      ...s,
      orbit: normaliseOrbitClass(orbitClass),
      platform_orbit: orbitClass,
      orbit_info: orbitInfo,
      orbit_member_id: recordId as string | undefined,
      orbit_record_id: recordId as string | undefined,
      epoch_s: epochS,
      epoch_ms: epochS !== undefined ? tdbSecondsToUtcMs(epochS) : undefined,
      slew_rate_dps: typeof s.slew_rate_dps === 'number' ? s.slew_rate_dps : typeof s.slew_rate_deg_s === 'number' ? s.slew_rate_deg_s : undefined,
    };
  }) as unknown as Sensors['space'];
  return { ground, space, note: anyRaw.note, reason_bits: anyRaw.reason_bits };
}

/** 'geo' → 'GEO', 'l1_halo' → 'L1_halo', 'nrho' → 'NRHO', 'dro' → 'DRO'. */
function normaliseOrbitClass(s: string): string {
  const k = s.toLowerCase();
  if (k === 'geo') return 'GEO';
  if (k === 'dro') return 'DRO';
  if (k === 'nrho') return 'NRHO';
  if (k === 'l1_halo') return 'L1_halo';
  if (k === 'l2_halo') return 'L2_halo';
  return s;
}
