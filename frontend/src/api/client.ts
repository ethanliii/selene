/**
 * Typed fetch client for the SELENE API (PLAN.md §5). Base path '/api' is proxied by Vite in dev
 * and served by uvicorn in production.
 *
 * Mock mode: if a request fails at the network level (backend down) or the health probe fails,
 * the client flips `backendStatus` to 'offline' and every call returns data from ./mock.ts.
 * The UI watches `useBackendStatus()` and shows "BACKEND OFFLINE — MOCK DATA".
 * HTTP errors with a response (4xx/5xx) are thrown as ApiError, NOT mocked, so real backend bugs stay visible.
 */
import { useSyncExternalStore } from 'react';
import * as mock from './mock';
import type {
  ArchitectureEvaluateRequest,
  ArchitectureEvaluateResponse,
  CatalogObject,
  CoverageRequest,
  CoverageResponse,
  DemoScenario,
  EphemerisBodies,
  Health,
  ManeuverDetectRequest,
  ManeuverDetectResponse,
  OdRequest,
  OdResponse,
  OrbitFamilies,
  ReachabilityRequest,
  ReachabilityResponse,
  Sensors,
  TaskingRequest,
  TaskingResponse,
} from './types';

export const API_BASE = '/api';

export type BackendStatus = 'unknown' | 'online' | 'offline';

export class ApiError extends Error {
  constructor(public status: number, public path: string, public body: string) {
    super(`API ${status} ${path}: ${body.slice(0, 200)}`);
    this.name = 'ApiError';
  }
}

// ---- tiny external store for backend status (no zustand dependency in the API layer) ----------
let status: BackendStatus = 'unknown';
const listeners = new Set<() => void>();
function setStatus(s: BackendStatus) {
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

// ---- core request with mock fallback ---------------------------------------------------------
async function request<T>(path: string, init: RequestInit | undefined, fallback: () => T): Promise<T> {
  let res: Response;
  try {
    res = await fetch(API_BASE + path, {
      ...init,
      headers: { Accept: 'application/json', ...(init?.body ? { 'Content-Type': 'application/json' } : {}), ...init?.headers },
    });
  } catch (err) {
    // Network failure => backend down => mock mode.
    setStatus('offline');
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
      return fallback();
    }
    throw new ApiError(res.status, path, text);
  }
  setStatus('online');
  return (await res.json()) as T;
}

const get = <T,>(path: string, fallback: () => T) => request<T>(path, undefined, fallback);
const post = <T,>(path: string, body: unknown, fallback: () => T) =>
  request<T>(path, { method: 'POST', body: JSON.stringify(body) }, fallback);

function qs(params: Record<string, string | number | undefined>): string {
  const u = new URLSearchParams();
  for (const [k, v] of Object.entries(params)) if (v !== undefined) u.set(k, String(v));
  const s = u.toString();
  return s ? `?${s}` : '';
}

// ---- endpoints -------------------------------------------------------------------------------
export const api = {
  health: () => get<Health>('/health', () => mock.mockHealth),

  catalogObjects: () => get<CatalogObject[]>('/catalog/objects', () => mock.mockCatalog),

  orbitFamilies: () => get<OrbitFamilies>('/orbits/families', () => mock.mockFamilies),

  ephemerisBodies: (t0: string, t1: string, n: number) =>
    get<EphemerisBodies>(`/ephemeris/bodies${qs({ t0, t1, n })}`, () => mock.mockEphemeris(n)),

  sensors: () => get<Sensors>('/sensors', () => mock.mockSensors),

  coverage: (req: CoverageRequest) => post<CoverageResponse>('/coverage', req, () => mock.mockCoverage(req)),

  odRun: (req: OdRequest) => post<OdResponse>('/od/run', req, () => mock.mockOd),

  maneuverDetect: (req: ManeuverDetectRequest) =>
    post<ManeuverDetectResponse>('/maneuver/detect', req, () => mock.mockManeuver),

  reachability: (req: ReachabilityRequest) =>
    post<ReachabilityResponse>('/reachability', req, () => mock.mockReachability(req)),

  taskingSchedule: (req: TaskingRequest) => post<TaskingResponse>('/tasking/schedule', req, () => mock.mockTasking(req)),

  architectureEvaluate: (req: ArchitectureEvaluateRequest) =>
    post<ArchitectureEvaluateResponse>('/architecture/evaluate', req, () => mock.mockArchitecture(req)),

  demoScenario: () => get<DemoScenario>('/demo/scenario', () => mock.mockDemo),
};

export type Api = typeof api;
