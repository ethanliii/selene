/**
 * Mock-mode fallback. Used ONLY when the backend is unreachable so the UI stays fully demonstrable.
 *
 * What is physically meaningful here and what is not:
 *  - Orbit families, object motion and uncertainty clouds come from a browser CR3BP integrator
 *    (src/lib/cr3bp.ts): real three-body dynamics, literature initial conditions, differential correction.
 *  - Ephemeris angles use mean-element formulas (src/lib/ephem.ts), not DE440s.
 *  - Coverage numbers are LAYOUT PLACEHOLDERS, never physics results. OD / maneuver / reachability / tasking /
 *    architecture have NO browser placeholder at all: those panels show the backend's answer or an inline error.
 * Everything is labelled SIMULATED / notional; nothing here refers to a real spacecraft or country.
 */
import type { CatalogObject, CoverageRequest, CoverageResponse, DemoScenario, EphemerisBodies, Health, OrbitFamilies, Sensors, Vec3 } from './types';
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

/** Scripted story generated in the browser from CR3BP dynamics (see demo/mockScenario.ts). */
export function mockDemo(): DemoScenario {
  return buildMockScenario();
}
