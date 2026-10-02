/**
 * Mock-mode fallback. Used ONLY when the backend is unreachable so the UI stays fully demonstrable.
 *
 * What is physically meaningful here and what is not:
 *  - Orbit families, object motion and uncertainty clouds come from a browser CR3BP integrator
 *    (src/lib/cr3bp.ts): real three-body dynamics, literature initial conditions, differential correction.
 *  - Ephemeris angles use mean-element formulas (src/lib/ephem.ts), not DE440s.
 *  - Coverage / OD / tasking / architecture numbers are LAYOUT PLACEHOLDERS (zeros), never physics results.
 * Everything is labelled SIMULATED / notional; nothing here refers to a real spacecraft or country.
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
  Sensors,
  TaskingRequest,
  TaskingResponse,
  Vec3,
} from './types';
import { MOCK_GROUND, MOCK_SPACE } from '../demo/mockNetwork';
import { buildMockScenario, MOCK_OBJECTS, mockObjectGcrfState, SCENARIO_T0 } from '../demo/mockScenario';
import { L_STAR_KM, lagrangePoints, mockFamilies as cr3bpFamilies, MU as MU_CR3BP } from '../lib/cr3bp';
import { meanMoonAngle, meanSunAngle } from '../lib/ephem';

/** CR3BP Earth–Moon mass ratio (DE440 GMs), see PLAN.md §2.1. */
export const MU = MU_CR3BP;
/** Collinear/triangular libration points, rotating frame, nondimensional (PLAN.md §2.1). */
export const LAGRANGE_ROT: Record<'L1' | 'L2' | 'L3' | 'L4' | 'L5', Vec3> = lagrangePoints();

const T0 = SCENARIO_T0;

export const mockHealth: Health = { status: 'mock', version: '0.0.0-mock', offline: true };

let catalogCache: CatalogObject[] | null = null;
export function mockCatalog(): CatalogObject[] {
  if (catalogCache) return catalogCache;
  catalogCache = MOCK_OBJECTS.map((d) => {
    const { rot, gcrf } = mockObjectGcrfState(d);
    return {
      id: d.id,
      name: d.name,
      kind: 'simulated' as const,
      orbit_type: d.orbit_type,
      actor: 'notional',
      state_gcrf_km: gcrf,
      epoch_utc: T0,
      ic_rot: rot,
      radius_m: d.radius_m,
      albedo: d.albedo,
    };
  });
  return catalogCache;
}

export function mockFamilies(): OrbitFamilies {
  return {
    families: cr3bpFamilies().map((f) => ({
      name: f.name,
      members: f.members.map((m) => ({ id: m.id, ic: m.ic, period: m.period, jacobi: m.jacobi, stability: m.stability, samples_rot: m.samples_rot, approximate: m.approximate })),
    })),
  };
}

/** Mean-element Moon/Sun positions in a GCRF-like frame (xy-plane only; see lib/ephem.ts for sources). */
export function mockEphemeris(t0: string = T0, t1?: string, n = 24): EphemerisBodies {
  const base = Date.parse(t0);
  const end = t1 ? Date.parse(t1) : base + (n - 1) * 3600e3;
  const epochs: string[] = [];
  const moon: Vec3[] = [];
  const sun: Vec3[] = [];
  const earth: Vec3[] = [];
  for (let i = 0; i < n; i++) {
    const ms = base + ((end - base) * i) / Math.max(1, n - 1);
    epochs.push(new Date(ms).toISOString());
    const a = meanMoonAngle(ms);
    const s = meanSunAngle(ms);
    moon.push([L_STAR_KM * Math.cos(a), L_STAR_KM * Math.sin(a), 0]);
    sun.push([1.496e8 * Math.cos(s), 1.496e8 * Math.sin(s), 0]);
    earth.push([0, 0, 0]);
  }
  const rep = (v: Vec3): Vec3[] => Array.from({ length: n }, () => v);
  return {
    source: 'mock-mean-elements',
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

/** Notional sensor network (see demo/mockNetwork.ts for the specification disclaimer). */
export const mockSensors: Sensors = { ground: MOCK_GROUND, space: MOCK_SPACE };

export function mockCoverage(req: CoverageRequest): CoverageResponse {
  const { nx, ny, xmin, xmax, ymin, ymax } = req.grid;
  const x = Array.from({ length: nx }, (_, i) => xmin + ((xmax - xmin) * i) / Math.max(1, nx - 1));
  const y = Array.from({ length: ny }, (_, j) => ymin + ((ymax - ymin) * j) / Math.max(1, ny - 1));
  const epochs = Array.from({ length: req.n_t }, (_, k) => new Date(Date.parse(req.t0) + k * 3600e3).toISOString());
  const values = epochs.map(() => x.flatMap((xi) => y.map((yj) => (Math.hypot(xi - (1 - MU), yj) < 0.08 ? 0 : 1))));
  return { grid: { ...req.grid, x, y }, values, epochs };
}

export function mockOd(): OdResponse {
  const c = mockCatalog()[0];
  return { ukf: { epochs: [T0], states: [c.state_gcrf_km], covs: [identity6(100)], nis: [1.0] }, particles: [] };
}

export const mockManeuver: ManeuverDetectResponse = { detections: [], threshold: 5.99 };

export function mockReachability(req: ReachabilityRequest): ReachabilityResponse {
  // Placeholder ring (layout only). The scripted mock scenario carries a real CR3BP reachable set instead.
  const pts: Vec3[] = [];
  const r = 0.05 + 0.002 * req.dv_budget_mps * (req.horizon_h / 24);
  for (let i = 0; i < 200; i++) {
    const a = (i / 200) * 2 * Math.PI;
    pts.push([1 - MU + r * Math.cos(a) * (0.5 + 0.5 * ((i * 7919) % 97) / 97), r * Math.sin(a) * (0.5 + 0.5 * ((i * 104729) % 89) / 89), 0]);
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

/** Scripted story generated in the browser from CR3BP dynamics (see demo/mockScenario.ts). */
export function mockDemo(): DemoScenario {
  return buildMockScenario();
}

function identity6(scale: number): number[][] {
  return Array.from({ length: 6 }, (_, i) => Array.from({ length: 6 }, (_, j) => (i === j ? scale : 0)));
}
