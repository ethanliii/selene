/**
 * Mock-mode fallback. Used ONLY when the backend is unreachable so the UI still renders.
 * Everything here is minimal, obviously synthetic, and labelled SIMULATED / notional.
 * Numbers are placeholders for layout, NOT physics results — never present them as real.
 */
import type {
  ArchitectureEvaluateRequest,
  ArchitectureEvaluateResponse,
  CatalogObject,
  CoverageRequest,
  CoverageResponse,
  DemoScenario,
  EphemerisBodies,
  Health,
  ManeuverDetectResponse,
  OdResponse,
  OrbitFamilies,
  ReachabilityRequest,
  ReachabilityResponse,
  SeleneEvent,
  Sensors,
  TaskingRequest,
  TaskingResponse,
  Vec3,
} from './types';

/** CR3BP Earth–Moon mass ratio (DE440 GMs), see PLAN.md §2.1. */
export const MU = 0.0121505;
/** Collinear/triangular libration points, rotating frame, nondimensional (PLAN.md §2.1). */
export const LAGRANGE_ROT: Record<'L1' | 'L2' | 'L3' | 'L4' | 'L5', Vec3> = {
  L1: [0.836915, 0, 0],
  L2: [1.155682, 0, 0],
  L3: [-1.005063, 0, 0],
  L4: [0.5 - MU, 0.866025, 0],
  L5: [0.5 - MU, -0.866025, 0],
};

const T0 = '2026-10-01T00:00:00Z';

export const mockHealth: Health = { status: 'mock', version: '0.0.0-mock', offline: true };

export const mockCatalog: CatalogObject[] = [
  {
    id: 'SIM-DRO-01',
    name: 'NOTIONAL DRO LOITERER',
    kind: 'simulated',
    orbit_type: 'DRO',
    actor: 'notional',
    state_gcrf_km: [-311000, 180000, 12000, -0.55, -0.92, 0.01],
    epoch_utc: T0,
  },
  {
    id: 'SIM-NRHO-01',
    name: 'NOTIONAL ALLIED RELAY',
    kind: 'simulated',
    orbit_type: '9:2 NRHO',
    actor: 'notional',
    state_gcrf_km: [380000, -40000, -60000, 0.1, 1.0, 0.3],
    epoch_utc: T0,
  },
];

/** Circle-ish placeholder for a planar family member (NOT a corrected periodic orbit). */
function ring(cx: number, r: number, n = 90, retro = false): Vec3[] {
  const out: Vec3[] = [];
  for (let i = 0; i <= n; i++) {
    const a = (retro ? -1 : 1) * (2 * Math.PI * i) / n;
    out.push([cx + r * Math.cos(a), r * Math.sin(a), 0]);
  }
  return out;
}

export const mockFamilies: OrbitFamilies = {
  families: [
    {
      name: 'DRO (mock)',
      members: [
        { id: 'mock-dro-1', ic: [0.8, 0, 0, 0, 0.5, 0], period: 3.5, jacobi: 2.9, stability: 1, samples_rot: ring(1 - MU, 0.18, 90, true) },
      ],
    },
    {
      name: 'L1 Lyapunov (mock)',
      members: [
        { id: 'mock-l1-1', ic: [0.8234, 0, 0, 0, 0.1263, 0], period: 2.743, jacobi: 3.17, stability: 500, samples_rot: ring(LAGRANGE_ROT.L1[0], 0.03) },
      ],
    },
  ],
};

export function mockEphemeris(n = 24): EphemerisBodies {
  const epochs: string[] = [];
  const moon: Vec3[] = [];
  const sun: Vec3[] = [];
  const earth: Vec3[] = [];
  const base = Date.parse(T0);
  for (let i = 0; i < n; i++) {
    const t = i * 3600;
    epochs.push(new Date(base + t * 1000).toISOString());
    const a = (2 * Math.PI * t) / (27.321661 * 86400);
    moon.push([384400 * Math.cos(a), 384400 * Math.sin(a), 0]);
    sun.push([1.496e8, 0, 0]);
    earth.push([0, 0, 0]);
  }
  const rep = (v: Vec3): Vec3[] => Array.from({ length: n }, () => v);
  return {
    epochs,
    earth,
    moon,
    sun,
    rot_frame: {
      moon: rep([1 - MU, 0, 0]),
      l1: rep(LAGRANGE_ROT.L1),
      l2: rep(LAGRANGE_ROT.L2),
      l3: rep(LAGRANGE_ROT.L3),
      l4: rep(LAGRANGE_ROT.L4),
      l5: rep(LAGRANGE_ROT.L5),
    },
  };
}

export const mockSensors: Sensors = {
  ground: [
    { id: 'GND-MAUI', name: 'Maui (mock site)', lat_deg: 20.71, lon_deg: -156.26, alt_km: 3.06, limiting_mag: 19.5, fov_deg: 1.0, min_elevation_deg: 20, sun_exclusion_deg: 90, moon_exclusion_deg: 15 },
  ],
  space: [
    { id: 'SPC-DRO-A', name: 'DRO observer (mock)', orbit: 'DRO', limiting_mag: 18.0, fov_deg: 4.0, sun_exclusion_deg: 40, moon_exclusion_deg: 10, earth_exclusion_deg: 10 },
  ],
};

export function mockCoverage(req: CoverageRequest): CoverageResponse {
  const { nx, ny, xmin, xmax, ymin, ymax } = req.grid;
  const x = Array.from({ length: nx }, (_, i) => xmin + ((xmax - xmin) * i) / Math.max(1, nx - 1));
  const y = Array.from({ length: ny }, (_, j) => ymin + ((ymax - ymin) * j) / Math.max(1, ny - 1));
  const epochs = Array.from({ length: req.n_t }, (_, k) => new Date(Date.parse(req.t0) + k * 3600e3).toISOString());
  const values = epochs.map(() => x.flatMap((xi) => y.map((yj) => (Math.hypot(xi - (1 - MU), yj) < 0.08 ? 0 : 1))));
  return { grid: { ...req.grid, x, y }, values, epochs };
}

export const mockOd: OdResponse = {
  ukf: { epochs: [T0], states: [mockCatalog[0].state_gcrf_km], covs: [identity6(100)], nis: [1.0] },
  particles: [],
};

export const mockManeuver: ManeuverDetectResponse = { detections: [], threshold: 5.99 };

export function mockReachability(req: ReachabilityRequest): ReachabilityResponse {
  const pts: Vec3[] = [];
  const r = 0.05 + 0.002 * req.dv_budget_mps * (req.horizon_h / 24);
  for (let i = 0; i < 200; i++) {
    const a = (i / 200) * 2 * Math.PI;
    pts.push([1 - MU + r * Math.cos(a) * (0.5 + 0.5 * Math.random()), r * Math.sin(a) * (0.5 + 0.5 * Math.random()), 0]);
  }
  return {
    points: pts,
    regions: [
      { name: 'L1 gateway', fraction: 0, earliest_h: null },
      { name: 'L2 gateway', fraction: 0, earliest_h: null },
      { name: 'NRHO corridor', fraction: 0, earliest_h: null },
      { name: 'Lunar south pole', fraction: 0, earliest_h: null },
      { name: 'GEO return', fraction: 0, earliest_h: null },
    ],
  };
}

export function mockTasking(req: TaskingRequest): TaskingResponse {
  return {
    schedule: [],
    custody_pct: 0,
    mean_tslo_h: 0,
    trace_series: { epochs: [req.t0, req.t1], trace: [0, 0] },
  };
}

export function mockArchitecture(req: ArchitectureEvaluateRequest): ArchitectureEvaluateResponse {
  return {
    scores: req.architectures.map((a) => ({
      name: a.name,
      coverage_pct: 0,
      custody_pct: 0,
      revisit_h: 0,
      detect_latency_h_mean: 0,
      detect_latency_h_p95: 0,
      n_mc: 0,
    })),
  };
}

const mockEvents: SeleneEvent[] = [
  { t: 0, kind: 'info', severity: 'info', text: 'MOCK: backend offline. Scenario events are layout placeholders only.' },
  { t: 20 * 3600, kind: 'maneuver_detected', severity: 'alert', text: 'MOCK: notional DRO object — NIS gate exceeded (placeholder).', object_id: 'SIM-DRO-01' },
  { t: 26 * 3600, kind: 'custody_lost', severity: 'warn', text: 'MOCK: custody degraded in lunar glare (placeholder).', object_id: 'SIM-DRO-01' },
  { t: 40 * 3600, kind: 'custody_regained', severity: 'ok', text: 'MOCK: custody regained by DRO observer (placeholder).', object_id: 'SIM-DRO-01' },
];

export const mockDemo: DemoScenario = {
  meta: {
    title: 'MOCK scenario (backend offline)',
    t0_utc: T0,
    duration_s: 48 * 3600,
    playback_s: 120,
    disclaimer: 'All objects and events are SIMULATED and attributed to a notional actor.',
  },
  frames: [],
  events: mockEvents,
  brief:
    '## Analyst brief (MOCK)\n\nThe backend is offline; this brief is a layout placeholder and contains no analysis results.\n\n- All objects and events shown are SIMULATED.\n- Start the API (make dev) to load the real scripted scenario.',
};

function identity6(scale: number): number[][] {
  return Array.from({ length: 6 }, (_, i) => Array.from({ length: 6 }, (_, j) => (i === j ? scale : 0)));
}
