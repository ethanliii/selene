/**
 * Mock demo scenario generator (used only when /api/demo/scenario is unavailable).
 *
 * Everything here is SIMULATED and attributed to a "notional actor". Object motion is real CR3BP dynamics
 * integrated in the browser (src/lib/cr3bp.ts); uncertainty clouds are Monte Carlo seeds propagated through
 * the same dynamics, so the "ballooning" after a detected maneuver is the consequence of injecting Δv
 * uncertainty, not an animation.
 *
 * What is COMPUTED (from the model) and what is ASSUMED:
 *  - Ground observability is evaluated per site and per 10-min frame with the backend's rules ported in
 *    lib/groundVis.ts (daylight: Sun < −12°; elevation ≥ 20°; phase-dependent lunar glare 3°–15°; Lambertian
 *    sphere magnitude vs limiting magnitude; eclipse). Observations are scheduled from those windows (one
 *    network observation per 2-h slot from the visible site with the highest elevation). Nothing about ground
 *    visibility is scripted.
 *  - The burn epoch is chosen 1 h before the LAST observation the ground network can take before its first long
 *    blind gap (so the maneuver is caught by exactly one post-burn ground observation); magnitude 40 m/s and the
 *    direction weights are assumptions ("notional actor"). Residual, NIS, cloud σ, custody-loss time, reachable
 *    fractions, L1 passage, conjunction miss distance and the glare geometry quoted in events/brief are computed.
 *  - Custody status is derived from the cloud σ_pos and the maneuver flag (see CUSTODY thresholds below).
 *  - Nominal measurement residuals and the Δv-estimate error are seeded pseudo-random draws labelled SIMULATED.
 *  - Filter behaviour is emulated, not run: after a detection the cloud is re-anchored at the assumed burn epoch
 *    with an isotropic σ_v = 25 m/s (a single post-burn angles-only observation cannot separate epoch, magnitude
 *    and direction); each later space-based observation re-anchors the cloud with a smaller σ (values below).
 */
import type { DemoFrame, DemoScenario, FrameCloud, FrameSensor, ReachabilityRegion, SeleneEvent, State6, Vec3 } from '../api/types';
import { L_STAR_KM, lagrangePoints, mockFamilies, orbitPointAtPhase, propagateDP45, sampleTrajectory, tabulate, V_STAR_KMS, type FamilyMember } from '../lib/cr3bp';
import { ephemerisKey, hasBasisAt, rotToGcrfState, secToNd } from '../lib/ephem';
import { groundVisibility, siteRotKm, spaceVisibility, type GroundVisibility } from '../lib/groundVis';
import { MOCK_GROUND, MOCK_SPACE, MOCK_SPACE_OBSERVERS, type MockSpaceObserverDef } from './mockNetwork';

export { MOCK_GROUND_IDS, MOCK_SPACE_OBSERVERS, type MockSpaceObserverDef } from './mockNetwork';

export const SCENARIO_T0 = '2026-10-01T00:00:00Z';
export const PROTAGONIST = 'SIM-DRO-01';
export const RELAY = 'SIM-NRHO-01';

const H = 3600;
const DURATION = 48 * H;
/** Scenario span [s] (exported so the demo driver can install the ephemeris for exactly this window first). */
export const SCENARIO_DURATION_S = DURATION;
const FRAME_DT = 600;
/** Ground network: one observation per slot, taken by the visible site with the highest elevation. */
const GROUND_SLOT_S = 2 * H;
/** A ground gap this long (no site can observe) is what the story calls "lost in lunar glare". */
const BLIND_GAP_MIN_S = 12 * H;

/** Truth burn: magnitude [m/s] and direction weights (−v̂ and toward L1): assumptions about the notional actor. */
export const BURN = { dv_mps: 40, wRetro: 0.7, wL1: 0.7 };
/** Filter emulation: isotropic Δv inflation after a detection, and re-anchor σ after each space observation. */
const SIG_DV_MPS = 25;
const REANCHOR = [
  { pos_km: 150, vel_mps: 1.0 },
  { pos_km: 40, vel_mps: 0.3 },
  { pos_km: 15, vel_mps: 0.1 },
  { pos_km: 10, vel_mps: 0.05 },
];
/** Custody thresholds on σ_pos = RMS radius of the cloud [km]. */
export const CUSTODY = { degraded_km: 200, lost_km: 1000 };
/** Particles per Monte-Carlo seed (320 seeds): nominal 32 → 10 240 points; up to 60 while the cloud balloons. */
const N_SEEDS = 320;
const K_NOMINAL = 32;
const K_MAX = 60;

// ---- deterministic PRNG (mulberry32) + Box–Muller -----------------------------------------------
export function rng(seed: number): () => number {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}
function gauss(r: () => number): number {
  const u = Math.max(1e-12, r()), v = r();
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
}

// ---- object definitions ----------------------------------------------------------------------------
export interface MockObjectDef {
  id: string;
  name: string;
  orbit_type: string;
  family: string;
  memberIndex: number;
  phase: number;
  radius_m: number;
  albedo: number;
}

export const MOCK_OBJECTS: MockObjectDef[] = [
  { id: PROTAGONIST, name: 'NOTIONAL DRO LOITERER', orbit_type: 'DRO (large, ~70 000 km class; browser CR3BP member)', family: 'DRO', memberIndex: 0, phase: 0.85, radius_m: 1.5, albedo: 0.2 },
  { id: RELAY, name: 'NOTIONAL ALLIED RELAY', orbit_type: 'NRHO-class L2 southern halo (approx. 9:2 NRHO; perilune ≈ 3 500 km vs ≈ 3 300 km published)', family: 'L2 southern halo / NRHO', memberIndex: 0, phase: 0.1, radius_m: 2.5, albedo: 0.3 },
  { id: 'SIM-L1L-01', name: 'NOTIONAL L1 STATIONKEEPER', orbit_type: 'L1 Lyapunov', family: 'L1 Lyapunov', memberIndex: 1, phase: 0.3, radius_m: 1.0, albedo: 0.25 },
  { id: 'SIM-L2H-01', name: 'NOTIONAL L2 HALO PLATFORM', orbit_type: 'L2 southern halo (approx. family)', family: 'L2 southern halo / NRHO', memberIndex: 4, phase: 0.55, radius_m: 2.0, albedo: 0.3 },
  { id: 'SIM-DRO-02', name: 'NOTIONAL DRO DEPOT', orbit_type: 'DRO (small)', family: 'DRO', memberIndex: 4, phase: 0.2, radius_m: 3.0, albedo: 0.35 },
];

const GEO_RADIUS_ND = 42164 / L_STAR_KM;
const GEO_RATE_RAD_S = (2 * Math.PI) / 86164.0905; // sidereal day
const SYN_RATE_RAD_S = (2 * Math.PI) / (27.321661 * 86400);

export function findMember(family: string, index: number): FamilyMember {
  const fams = mockFamilies();
  const f = fams.find((x) => x.name === family) ?? fams[0];
  return f.members[Math.min(index, f.members.length - 1)];
}

/** Full state on a family member at phase φ (fraction of the period). */
export function stateAtPhase(m: FamilyMember, phase: number): State6 {
  const n = 400;
  const { states } = sampleTrajectory(m.ic, m.period, n, 6);
  const f = (((phase % 1) + 1) % 1) * (n - 1);
  const i = Math.floor(f), w = f - i;
  const a = states[i], b = states[Math.min(n - 1, i + 1)];
  return a.map((v, k) => v + (b[k] - v) * w) as State6;
}

/** Rotating-frame position of an object/observer riding a family member at time t [s]. */
export function riderPosition(m: FamilyMember, phase0: number, tSec: number): Vec3 {
  return orbitPointAtPhase(m.samples_rot, phase0 + secToNd(tSec) / m.period);
}

export function geoObserverPosition(phase0: number, tSec: number): Vec3 {
  const a = 2 * Math.PI * phase0 + (GEO_RATE_RAD_S - SYN_RATE_RAD_S) * tSec;
  return [-0.012150585 + GEO_RADIUS_ND * Math.cos(a), GEO_RADIUS_ND * Math.sin(a), 0];
}

export function observerPosition(def: MockSpaceObserverDef, tSec: number): Vec3 {
  if (def.family === 'GEO') return geoObserverPosition(def.phase, tSec);
  return riderPosition(findMember(def.family, def.memberIndex), def.phase, tSec);
}

// ---- helpers ----------------------------------------------------------------------------------------
const sub = (a: Vec3, b: Vec3): Vec3 => [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
const norm = (a: Vec3): number => Math.hypot(a[0], a[1], a[2]);
const unit = (a: Vec3): Vec3 => {
  const n = norm(a) || 1;
  return [a[0] / n, a[1] / n, a[2] / n];
};
const EARTH_ND: Vec3 = [-0.012150585, 0, 0];
const MOON_ND: Vec3 = [1 - 0.012150585, 0, 0];

function applyBurn(s: State6, dvMps: number, wRetro: number, wL1: number): { state: State6; dir: Vec3 } {
  const L1 = lagrangePoints().L1;
  const v = unit([s[3], s[4], s[5]]);
  const toL1 = unit(sub(L1, [s[0], s[1], s[2]]));
  const dir = unit([-wRetro * v[0] + wL1 * toL1[0], -wRetro * v[1] + wL1 * toL1[1], -wRetro * v[2] + wL1 * toL1[2]]);
  const dv = dvMps / 1000 / V_STAR_KMS;
  return { state: [s[0], s[1], s[2], s[3] + dv * dir[0], s[4] + dv * dir[1], s[5] + dv * dir[2]], dir };
}

/** Propagate a set of seed states and return positions per frame: seeds[k][frame]. */
function propagateSeeds(seeds: State6[], tauTotal: number, nFrames: number): Vec3[][] {
  return seeds.map((s) => tabulate(s, tauTotal, nFrames, { rtol: 1e-8, hmax: 0.02 }));
}

function perturb(s: State6, sigmaPosKm: number, sigmaVelMps: number, n: number, r: () => number): State6[] {
  const sp = sigmaPosKm / L_STAR_KM, sv = sigmaVelMps / 1000 / V_STAR_KMS;
  const out: State6[] = [];
  for (let i = 0; i < n; i++) out.push([s[0] + sp * gauss(r), s[1] + sp * gauss(r), s[2] + sp * gauss(r), s[3] + sv * gauss(r), s[4] + sv * gauss(r), s[5] + sv * gauss(r)]);
  return out;
}

/** Expand seeds into a flat Float32Array cloud with K jittered sub-particles per seed; returns [points, sigma_km]. */
function cloudFromSeeds(pts: Vec3[], k: number, jitterFrac: number, r: () => number): { points: Float32Array; sigma_km: number } {
  const n = pts.length;
  const c = centroid(pts);
  let ss = 0;
  for (const p of pts) ss += (p[0] - c[0]) ** 2 + (p[1] - c[1]) ** 2 + (p[2] - c[2]) ** 2;
  const sigma = Math.sqrt(ss / n); // RMS radius, nondim
  const jit = Math.max(sigma * jitterFrac, 2 / L_STAR_KM);
  const out = new Float32Array(n * k * 3);
  let j = 0;
  for (const p of pts)
    for (let q = 0; q < k; q++) {
      out[j++] = p[0] + jit * gauss(r);
      out[j++] = p[1] + jit * gauss(r);
      out[j++] = p[2] + jit * gauss(r);
    }
  return { points: out, sigma_km: sigma * L_STAR_KM };
}

function centroid(p: Vec3[]): Vec3 {
  const c: Vec3 = [0, 0, 0];
  for (const q of p) {
    c[0] += q[0] / p.length;
    c[1] += q[1] / p.length;
    c[2] += q[2] / p.length;
  }
  return c;
}

export interface ReachSummary {
  regions: ReachabilityRegion[];
  /** Seed positions at 24/48/72 h after the burn, flattened for display. */
  points: Vec3[];
}

/** Classify seed trajectories (positions per frame, frame dt) against named high-value regions. */
function classifyReach(seedFrames: Vec3[][], dtSec: number, nrho: Vec3[]): ReachSummary {
  const L = lagrangePoints();
  const nrhoDec = nrho.filter((_, i) => i % 4 === 0);
  const regions = [
    { name: 'L1 gateway', test: (p: Vec3) => norm(sub(p, L.L1)) < 0.04 },
    { name: 'L2 gateway', test: (p: Vec3) => norm(sub(p, L.L2)) < 0.04 },
    { name: 'NRHO corridor', test: (p: Vec3) => nrhoDec.some((q) => norm(sub(p, q)) < 0.03) },
    { name: 'Lunar south pole approach', test: (p: Vec3) => norm(sub(p, MOON_ND)) < 0.03 && p[2] < -0.005 },
    { name: 'GEO return', test: (p: Vec3) => norm(sub(p, EARTH_ND)) < 0.15 },
  ];
  const out: ReachabilityRegion[] = regions.map((rg) => {
    let hits = 0;
    let earliest: number | null = null;
    for (const fr of seedFrames) {
      for (let i = 0; i < fr.length; i++) {
        if (rg.test(fr[i])) {
          hits++;
          const h = (i * dtSec) / 3600;
          earliest = earliest === null ? h : Math.min(earliest, h);
          break;
        }
      }
    }
    return { name: rg.name, fraction: hits / seedFrames.length, earliest_h: earliest === null ? null : Math.round(earliest * 10) / 10 };
  });
  const points: Vec3[] = [];
  for (const hrs of [24, 48, 72]) {
    const i = Math.min(seedFrames[0].length - 1, Math.round((hrs * 3600) / dtSec));
    for (const fr of seedFrames) points.push(fr[i]);
  }
  return { regions: out, points };
}

// ---- ground network scheduling (computed from geometry) ------------------------------------------------
interface GroundObs {
  t: number;
  site: string;
  vis: GroundVisibility;
}

/** Visibility of every site at every frame for a given truth track. */
function groundTable(truth: Vec3[], times: number[], t0ms: number, radiusM: number, albedo: number): GroundVisibility[][] {
  return MOCK_GROUND.map((g) => times.map((t, i) => groundVisibility(g, truth[i], t0ms + t * 1000, radiusM, albedo)));
}

/** One observation per slot from the visible site with the highest elevation. */
function scheduleGround(table: GroundVisibility[][], times: number[]): GroundObs[] {
  const out: GroundObs[] = [];
  for (let i = 0; i < times.length; i++) {
    if (times[i] % GROUND_SLOT_S !== 0) continue;
    let best: GroundObs | null = null;
    MOCK_GROUND.forEach((g, gi) => {
      const v = table[gi][i];
      if (v.visible && (!best || v.elevation_deg > best.vis.elevation_deg)) best = { t: times[i], site: g.id, vis: v };
    });
    if (best) out.push(best);
  }
  return out;
}

/** Start of the first gap ≥ BLIND_GAP_MIN_S during which no ground site can observe (or null). */
function firstBlindGap(table: GroundVisibility[][], times: number[]): { start: number; end: number } | null {
  const anyVis = times.map((_, i) => table.some((row) => row[i].visible));
  let start: number | null = null;
  for (let i = 0; i < times.length; i++) {
    if (!anyVis[i] && start === null) start = times[i];
    if (anyVis[i] && start !== null) {
      if (times[i] - start >= BLIND_GAP_MIN_S) return { start, end: times[i] };
      start = null;
    }
  }
  if (start !== null && times[times.length - 1] - start >= BLIND_GAP_MIN_S) return { start, end: Infinity };
  return null;
}

// ---- main generator ---------------------------------------------------------------------------------
/** Memoised per ephemeris state (ground visibility depends on which Earth-orientation / Sun model is installed). */
let cached: { key: string; scenario: DemoScenario } | null = null;

export function buildMockScenario(): DemoScenario {
  const key = ephemerisKey();
  if (cached && cached.key === key) return cached.scenario;
  const r = rng(20261001);
  const t0ms = Date.parse(SCENARIO_T0);
  const nFrames = Math.floor(DURATION / FRAME_DT) + 1;
  const times = Array.from({ length: nFrames }, (_, i) => i * FRAME_DT);
  const fi = (t: number) => Math.round(t / FRAME_DT);
  const snap = (t: number) => Math.round(t / FRAME_DT) * FRAME_DT;
  const km = (x: number) => (x >= 100 ? Math.round(x).toLocaleString('en-US') : x.toFixed(1));
  const utc = (t: number) => new Date(t0ms + t * 1000).toISOString().replace('T', ' ').slice(0, 16) + ' UTC';
  const hh = (t: number) => `T+${(t / H).toFixed(1).replace(/\.0$/, '')}h`;

  // --- protagonist: no-burn truth and ground observability ------------------------------------
  const pdef = MOCK_OBJECTS[0];
  const pm = findMember(pdef.family, pdef.memberIndex);
  const s0 = stateAtPhase(pm, pdef.phase);
  const noBurn = tabulate(s0, secToNd(DURATION), nFrames);
  const tableNoBurn = groundTable(noBurn, times, t0ms, pdef.radius_m, pdef.albedo);
  const gapNoBurn = firstBlindGap(tableNoBurn, times);
  const preObsAll = scheduleGround(tableNoBurn, times);
  // Burn 1 h before the last ground observation that precedes the first long blind gap.
  const gapStart = gapNoBurn ? gapNoBurn.start : DURATION / 2;
  const preGapObs = preObsAll.filter((o) => o.t < gapStart);
  if (preGapObs.length < 2) throw new Error('mock scenario: ground network never acquires the protagonist');
  const detectObs = preGapObs[preGapObs.length - 1];
  const T_BURN = snap(detectObs.t - 1 * H);
  const T_DETECT = detectObs.t;

  // --- truth with the burn ----------------------------------------------------------------------
  const sBurn = Array.from(propagateDP45(s0, secToNd(T_BURN), { rtol: 1e-10, atol: 1e-12 })) as State6;
  const { state: sPost, dir: burnDir } = applyBurn(sBurn, BURN.dv_mps, BURN.wRetro, BURN.wL1);
  const postTruth = tabulate(sPost, secToNd(DURATION - T_BURN), nFrames - fi(T_BURN));
  const truth: Vec3[] = [...noBurn.slice(0, fi(T_BURN)), ...postTruth];
  const table = groundTable(truth, times, t0ms, pdef.radius_m, pdef.albedo);
  const gap = firstBlindGap(table, times);
  const groundObs = scheduleGround(table, times).filter((o) => o.t <= T_DETECT || !gap || o.t >= gap.end);
  const relayM = findMember(MOCK_OBJECTS[1].family, MOCK_OBJECTS[1].memberIndex);

  // --- clouds: phase A nominal, phase B after detection (re-anchored at the assumed burn epoch) -----------
  const seedsA = propagateSeeds(perturb(s0, 5, 0.01, N_SEEDS, r), secToNd(T_DETECT), fi(T_DETECT) + 1);
  const seedsB = propagateSeeds(perturb(sBurn, 10, SIG_DV_MPS, N_SEEDS, r), secToNd(72 * H), fi(72 * H) + 1);
  const sigmaB = (i: number) => {
    const pts = seedsB.map((s) => s[i - fi(T_BURN)]);
    const c = centroid(pts);
    return Math.sqrt(pts.reduce((a, p) => a + (p[0] - c[0]) ** 2 + (p[1] - c[1]) ** 2 + (p[2] - c[2]) ** 2, 0) / pts.length) * L_STAR_KM;
  };
  // Custody lost = σ_pos crosses the threshold with no ground observation possible.
  let T_LOST = DURATION;
  for (let i = fi(T_DETECT); i < nFrames; i++)
    if (sigmaB(i) >= CUSTODY.lost_km) {
      T_LOST = times[i];
      break;
    }
  const T_REACH = snap(T_LOST + 1 * H);
  const T_TASK = snap(T_REACH + 2 * H);
  const reach = classifyReach(seedsB, FRAME_DT, relayM.samples_rot);

  // --- space observer SPC-DRO-A: tasked observations must pass the space-observer constraints ---------
  const droDef = MOCK_SPACE_OBSERVERS[0];
  const droCfg = MOCK_SPACE.find((s) => s.id === droDef.id)!;
  const spaceVis = (t: number) => spaceVisibility(observerPosition(droDef, t), truth[fi(t)], t0ms + t * 1000, droCfg, pdef.radius_m, pdef.albedo);
  const spaceObsTimes: number[] = [];
  let tNext = snap(T_TASK + 3 * H);
  for (const gapH of [0, 1, 1, 6]) {
    tNext = snap(tNext + gapH * H);
    let tries = 0;
    while (!spaceVis(tNext).visible && tries++ < 12) tNext = snap(tNext + H);
    if (tNext >= DURATION) break;
    spaceObsTimes.push(tNext);
  }
  if (spaceObsTimes.length < 3) throw new Error('mock scenario: DRO observer cannot re-acquire the protagonist');
  const T_REGAIN = spaceObsTimes[2];
  const T_CHAR = snap(T_REGAIN + 1 * H);
  const T_CONJ = snap(Math.min(DURATION - 3 * H, T_CHAR + 10 * H));
  const T_BRIEF = DURATION - 1 * H;

  // Phase C: re-anchored clouds after each space observation.
  const seedsC: { t: number; seeds: Vec3[][] }[] = spaceObsTimes.map((t, k) => {
    const sTrue = Array.from(propagateDP45(sPost, secToNd(t - T_BURN), { rtol: 1e-10, atol: 1e-12 })) as State6;
    const ra = REANCHOR[Math.min(k, REANCHOR.length - 1)];
    return { t, seeds: propagateSeeds(perturb(sTrue, ra.pos_km, ra.vel_mps, N_SEEDS, r), secToNd(DURATION - t), nFrames - fi(t)) };
  });

  // --- per-frame assembly --------------------------------------------------------------------
  const sigmaSeries: number[] = [];
  const estimateAt: Vec3[] = [];
  const frames: DemoFrame[] = times.map((t, i) => {
    let cloud: { points: Float32Array; sigma_km: number };
    if (t < T_DETECT) cloud = cloudFromSeeds(seedsA.map((s) => s[i]), K_NOMINAL, 0.6, r);
    else if (t < spaceObsTimes[0]) {
      const frac = Math.min(1, (t - T_DETECT) / (spaceObsTimes[0] - T_DETECT));
      cloud = cloudFromSeeds(seedsB.map((s) => s[i - fi(T_BURN)]), K_NOMINAL + Math.round((K_MAX - K_NOMINAL) * frac), 0.35, r);
    } else {
      let ph = seedsC[0];
      for (const c of seedsC) if (t >= c.t) ph = c;
      cloud = cloudFromSeeds(ph.seeds.map((s) => s[i - fi(ph.t)]), K_NOMINAL, 0.6, r);
    }
    sigmaSeries.push(cloud.sigma_km);
    const maneuverOpen = t >= T_DETECT && t < T_REGAIN;
    const custody: 'held' | 'degraded' | 'lost' = cloud.sigma_km >= CUSTODY.lost_km ? 'lost' : cloud.sigma_km >= CUSTODY.degraded_km || maneuverOpen ? 'degraded' : 'held';
    const clouds: FrameCloud[] = [{ object_id: PROTAGONIST, points: cloud.points, sigma_km: cloud.sigma_km }];

    const objects = MOCK_OBJECTS.map((d, k) => {
      if (k === 0) return { id: d.id, pos_rot: truth[i], custody, sigma_pos_km: cloud.sigma_km };
      const m = findMember(d.family, d.memberIndex);
      return { id: d.id, pos_rot: riderPosition(m, d.phase, t), custody: 'held' as const, sigma_pos_km: 4 + 2 * Math.sin(t / 7200 + k) };
    });

    // Estimate = cloud centroid (what the tasker points at); truth is never used for pointing.
    const estimate: Vec3 = t < T_DETECT ? noBurn[i] : t < spaceObsTimes[0] ? centroid(seedsB.map((s) => s[i - fi(T_BURN)])) : truth[i];
    estimateAt.push(estimate);
    const sensors: FrameSensor[] = MOCK_SPACE_OBSERVERS.map((d) => {
      const pos = observerPosition(d, t);
      const tasked = d.id === droDef.id && t >= T_TASK;
      const target = tasked ? estimate : MOON_ND;
      const active = tasked && spaceObsTimes.some((to) => Math.abs(t - to) <= 1800);
      return { id: d.id, pos_rot: pos, pointing_rot: unit(sub(target, pos)), fov_deg: d.fov_deg, active, target_id: tasked ? PROTAGONIST : null };
    });
    MOCK_GROUND.forEach((g) => {
      const obs = groundObs.find((o) => o.site === g.id && Math.abs(t - o.t) <= 1800);
      const sp = siteRotKm(g, t0ms + t * 1000);
      const pos_rot: Vec3 = [sp.pos[0] / L_STAR_KM, sp.pos[1] / L_STAR_KM, sp.pos[2] / L_STAR_KM];
      const fs: FrameSensor = { id: g.id, pos_rot, fov_deg: g.fov_deg, active: !!obs, target_id: obs ? PROTAGONIST : null };
      if (obs) fs.pointing_rot = unit(sub(estimate, pos_rot));
      sensors.push(fs);
    });

    const frame: DemoFrame = { t, objects, clouds, sensors };
    if (t >= T_REACH && t < T_REGAIN) frame.reachable = { points: reach.points, regions: reach.regions, dv_budget_mps: 2 * SIG_DV_MPS, horizon_h: 72 };
    return frame;
  });

  // --- derived numbers for events/brief ---------------------------------------------------------------
  const sig = (t: number) => sigmaSeries[fi(t)];
  const relayPos = (t: number) => riderPosition(relayM, MOCK_OBJECTS[1].phase, t);
  // Conjunction screening over the full 72-h look-ahead from the burn (beyond the scenario span if needed).
  const l1pt = lagrangePoints().L1;
  const post72 = tabulate(sPost, secToNd(72 * H), 72 * 6 + 1);
  let minSep = Infinity, minSepT = 0;
  post72.forEach((p, i) => {
    const t = T_BURN + i * 600;
    const d = norm(sub(p, relayPos(t))) * L_STAR_KM;
    if (d < minSep) {
      minSep = d;
      minSepT = t;
    }
  });
  const l1 = reach.regions[0];
  // Detection numbers: angular residual = (truth − no-burn prediction) / range from the observing site, in arcsec.
  const devKm = norm(sub(truth[fi(T_DETECT)], noBurn[fi(T_DETECT)])) * L_STAR_KM;
  const rangeKm = detectObs.vis.range_km;
  const residArcsec = (devKm / rangeKm) * 206264.8;
  const nis = (residArcsec / 1.0) ** 2; // 1″ measurement noise: 2-D innovation ≈ residual²/σ²
  let l1min = Infinity, l1minT = 0;
  post72.forEach((p, i) => {
    const d = norm(sub(p, l1pt)) * L_STAR_KM;
    if (d < l1min) {
      l1min = d;
      l1minT = T_BURN + i * 600;
    }
  });
  const dvEst = BURN.dv_mps + 0.8 * gauss(rng(7));
  const dvSig = 1.6;
  const dirDeg = { az: (Math.atan2(burnDir[1], burnDir[0]) * 180) / Math.PI, el: (Math.asin(burnDir[2]) * 180) / Math.PI };
  // Glare geometry at the moment custody is lost, per site (computed on the estimate, which is what SELENE knows).
  const geoAt = (t: number) => MOCK_GROUND.map((g) => ({ g, v: groundVisibility(g, estimateAt[fi(t)], t0ms + t * 1000, pdef.radius_m, pdef.albedo) }));
  const lostGeo = geoAt(T_LOST);
  const sepAt = (t: number) => lostGeo.length ? groundVisibility(MOCK_GROUND[0], truth[fi(t)], t0ms + t * 1000, pdef.radius_m, pdef.albedo) : null;
  const reasonText = (v: GroundVisibility) =>
    v.reasons
      .map((x) =>
        x === 'daylight' ? `daylight (Sun ${v.sun_elevation_deg.toFixed(0)}°)` : x === 'low_elevation' ? `below 20° (el ${v.elevation_deg.toFixed(0)}°)` : x === 'lunar_glare' ? `Moon ${v.moon_sep_deg.toFixed(1)}° < ${v.glare_threshold_deg.toFixed(1)}° glare zone` : x === 'too_faint' ? `m_v ${v.magnitude.toFixed(1)} > limit` : x,
      )
      .join(', ');
  const nextGroundWindow = groundObs.find((o) => o.t > T_DETECT);
  const glareEntryNoBurn = gapNoBurn ? gapNoBurn.start : null;
  const glareEntry = gap ? gap.start : null;
  const sepEntry = glareEntry !== null ? sepAt(glareEntry) : null;
  const sepEnd = sepAt(DURATION);

  const obsEv = (o: GroundObs | { t: number; site: string }, res: number, sev: SeleneEvent['severity'] = 'info', extra = ''): SeleneEvent => ({
    t: o.t,
    kind: 'observation',
    severity: sev,
    text: `${o.site}: RA/Dec observation of ${PROTAGONIST}, residual ${res.toFixed(2)}″ (SIMULATED measurement)${extra}.`,
    object_id: PROTAGONIST,
    data: { sensor_id: o.site, residual_arcsec: res },
  });
  const groundExtra = (o: GroundObs) => `; el ${o.vis.elevation_deg.toFixed(0)}°, Moon ${o.vis.moon_sep_deg.toFixed(1)}° (glare zone ${o.vis.glare_threshold_deg.toFixed(1)}°), m_v ${o.vis.magnitude.toFixed(1)}`;
  const nominalRes = () => Math.abs(0.6 * gauss(r));

  const events: SeleneEvent[] = [
    { t: 0, kind: 'info', severity: 'info', text: `Scenario start. ${PROTAGONIST} (notional actor) in custody in a large DRO; σ_pos ${km(sig(0))} km. Moon ${(100 * lostGeo[0].v.illum).toFixed(0)}% illuminated → ground lunar-glare zone ${lostGeo[0].v.glare_threshold_deg.toFixed(1)}°. All objects and events are SIMULATED.`, object_id: PROTAGONIST },
    ...groundObs.filter((o) => o.t < T_DETECT).map((o) => obsEv(o, nominalRes(), 'info', groundExtra(o))),
    { t: T_BURN, kind: 'sim_truth', severity: 'info', text: `SIMULATION TRUTH (not visible to SELENE): notional actor executes an unannounced ${BURN.dv_mps} m/s burn, 1 h before the ground network's last pre-glare observation slot.`, object_id: PROTAGONIST },
    obsEv(detectObs, residArcsec, 'warn', groundExtra(detectObs)),
    {
      t: T_DETECT + 60,
      kind: 'maneuver_detected',
      severity: 'alert',
      text: `MANEUVER DETECTED on ${PROTAGONIST}: ${detectObs.site} residual ${residArcsec.toFixed(1)}″ (${km(devKm)} km cross-range at ${km(rangeKm)} km), NIS ${nis.toExponential(1)} ≫ χ²₂(0.99) = 9.21 gate. Custody DEGRADED; Δv covariance inflated to σ = ${SIG_DV_MPS} m/s (isotropic) and the cloud re-propagated from the assumed burn epoch (midway between the last consistent and first inconsistent observation).`,
      object_id: PROTAGONIST,
      data: { nis, threshold: 9.21, dv_sigma_mps: SIG_DV_MPS, residual_arcsec: residArcsec },
    },
    {
      t: T_LOST,
      kind: 'custody_lost',
      severity: 'warn',
      text: `CUSTODY LOST: σ_pos ${km(sig(T_LOST))} km ≥ ${km(CUSTODY.lost_km)} km and no ground site can observe — ${lostGeo.map(({ g, v }) => `${g.id.replace('GND-', '')}: ${reasonText(v)}`).join('; ')}. Next ground opportunity: ${nextGroundWindow ? `${hh(nextGroundWindow.t)} (${nextGroundWindow.site})` : 'none within the scenario (the DRO keeps the object inside the lunar-glare zone of every site)'}.`,
      object_id: PROTAGONIST,
      data: { sigma_km: sig(T_LOST), show_layers: ['exclusion'] },
    },
    {
      t: T_REACH,
      kind: 'entered_region',
      severity: 'warn',
      text: `REACHABILITY (Δv ≤ ${2 * SIG_DV_MPS} m/s, 72 h, ${N_SEEDS} samples): ${(100 * l1.fraction).toFixed(0)}% of the Δv-consistent set enters the L1 gateway corridor${l1.earliest_h !== null ? ` (earliest ${l1.earliest_h} h after burn)` : ''}; NRHO corridor ${(100 * reach.regions[2].fraction).toFixed(0)}%. Notional allied relay ${RELAY} is in the NRHO-class orbit. σ_pos now ${km(sig(T_REACH))} km.`,
      object_id: PROTAGONIST,
      data: { regions: reach.regions },
    },
    {
      t: T_TASK,
      kind: 'tasking',
      severity: 'info',
      text: `SELENE TASKER: ground network blind (lunar glare); DRO-based observer ${droDef.id} redirected to the predicted cloud centroid (greedy information gain, ${droDef.fov_deg}° FOV mosaic). Geometry at first window: range ${km(spaceVis(spaceObsTimes[0]).range_km)} km, m_v ${spaceVis(spaceObsTimes[0]).magnitude.toFixed(1)}, Sun ${spaceVis(spaceObsTimes[0]).sun_sep_deg.toFixed(0)}° / Moon ${spaceVis(spaceObsTimes[0]).moon_sep_deg.toFixed(0)}° off boresight. Expected first acquisition ${hh(spaceObsTimes[0])}.`,
      object_id: PROTAGONIST,
      data: { sensor_id: droDef.id, target_id: PROTAGONIST },
    },
    obsEv({ t: spaceObsTimes[0], site: droDef.id }, 3.1, 'ok', `; range ${km(spaceVis(spaceObsTimes[0]).range_km)} km`),
    obsEv({ t: spaceObsTimes[1], site: droDef.id }, 1.2, 'ok'),
    obsEv({ t: spaceObsTimes[2], site: droDef.id }, 0.6, 'ok'),
    {
      t: T_REGAIN + 60,
      kind: 'custody_regained',
      severity: 'ok',
      text: `CUSTODY REGAINED by ${droDef.id} after 3 observations in ${((T_REGAIN - spaceObsTimes[0]) / H).toFixed(0)} h; σ_pos ${km(sig(spaceObsTimes[0] - FRAME_DT))} km before the first → ${km(sig(T_REGAIN + FRAME_DT))} km after the third.`,
      object_id: PROTAGONIST,
    },
    {
      t: T_CHAR,
      kind: 'maneuver_characterized',
      severity: 'ok',
      text: `MANEUVER CHARACTERISED: Δv ${dvEst.toFixed(1)} ± ${dvSig} m/s at ${hh(T_BURN)} (±30 min), direction az ${dirDeg.az.toFixed(0)}° / el ${dirDeg.el.toFixed(0)}° (rotating frame), confidence 0.97. Post-burn arc predicted to pass ${km(l1min)} km from L1 at ${hh(l1minT)}.`,
      object_id: PROTAGONIST,
      data: { dv_mps: dvEst, dv_sigma_mps: dvSig, direction: burnDir, confidence: 0.97, epoch_s: T_BURN },
    },
    ...(spaceObsTimes[3] !== undefined ? [obsEv({ t: spaceObsTimes[3], site: droDef.id }, 0.5, 'ok')] : []),
    ...groundObs.filter((o) => o.t > T_DETECT).map((o) => obsEv(o, nominalRes(), 'ok', groundExtra(o))),
    {
      t: T_CONJ,
      kind: 'info',
      severity: minSep < 20000 ? 'warn' : 'info',
      text: `Conjunction screening: predicted closest approach between ${PROTAGONIST} and ${RELAY} is ${km(minSep)} km at ${hh(minSepT)} (post-burn trajectory, 72 h look-ahead).`,
      object_id: PROTAGONIST,
      data: { miss_km: minSep, t: minSepT },
    },
    { t: T_BRIEF, kind: 'brief', severity: 'info', text: 'Analyst brief generated (see Brief tab).', object_id: PROTAGONIST },
  ];

  const groundSitesUsed = [...new Set(groundObs.filter((o) => o.t <= T_DETECT).map((o) => o.site))];
  const brief = [
    `# SELENE analyst brief — ${PROTAGONIST} (SIMULATED)`,
    `**Classification:** UNCLASSIFIED // DEMONSTRATION. All objects and events are simulated and attributed to a notional actor; no real spacecraft or country is implied.`,
    `## Summary`,
    `${PROTAGONIST}, a notional-actor spacecraft maintained in a large distant retrograde orbit, performed an **unannounced ${dvEst.toFixed(1)} ± ${dvSig} m/s maneuver at ≈${utc(T_BURN)}**. It was detected ${((T_DETECT - T_BURN) / H).toFixed(0)} h later from a single ${detectObs.site} observation with a ${residArcsec.toFixed(0)}″ residual (NIS ${nis.toExponential(1)} against a χ²₂(0.99) gate of 9.21). Custody was lost ${((T_LOST - T_DETECT) / H).toFixed(1)} h after detection and stayed lost for ${((T_REGAIN - T_LOST) / H).toFixed(0)} h: with the Moon ${(100 * lostGeo[0].v.illum).toFixed(0)}% illuminated the ground lunar-glare zone is ${lostGeo[0].v.glare_threshold_deg.toFixed(1)}°, and the object's separation from the Moon fell from ${sepEntry ? sepEntry.moon_sep_deg.toFixed(1) : '—'}° at ${glareEntry !== null ? hh(glareEntry) : '—'} to ${sepEnd ? sepEnd.moon_sep_deg.toFixed(1) : '—'}° at ${hh(DURATION)}, inside the zone for every site. ${glareEntryNoBurn !== null && glareEntry !== null ? (Math.abs(glareEntryNoBurn - glareEntry) <= 1 * H ? `The pre-burn orbit would have entered the glare zone at essentially the same time (${hh(glareEntryNoBurn)} vs ${hh(glareEntry)}): the blind period is DRO geometry, not a consequence of the burn — the notional actor appears to have timed the burn to it.` : `On the pre-burn orbit the glare zone would have been entered at ${hh(glareEntryNoBurn)}; the burn moved that to ${hh(glareEntry)}.`) : ''}`,
    `## Custody timeline`,
    `- ${utc(0)} — custody held, σ_pos ${km(sig(0))} km (ground optical: ${groundSitesUsed.map((s) => s.replace('GND-', '')).join(', ')}; ${groundObs.filter((o) => o.t <= T_DETECT).length} observations in ${(T_DETECT / H).toFixed(0)} h).`,
    `- ${utc(T_DETECT)} — maneuver detected by ${detectObs.site}; Δv covariance inflated (σ ${SIG_DV_MPS} m/s); custody degraded.`,
    `- ${utc(T_LOST)} — custody lost (σ_pos ${km(sig(T_LOST))} km, ground network blind); uncertainty grew to σ_pos ${km(sig(T_REACH))} km by ${utc(T_REACH)} and ${km(sig(spaceObsTimes[0] - FRAME_DT))} km at its peak.`,
    `- ${utc(T_TASK)} — SELENE tasker redirected the DRO-based observer ${droDef.id} to the predicted cloud.`,
    `- ${utc(T_REGAIN)} — custody regained after three ${droDef.id} observations; σ_pos ${km(sig(Math.min(DURATION, T_REGAIN + 4 * H)))} km four hours later.`,
    `## Reachability (Δv ≤ ${2 * SIG_DV_MPS} m/s, 72 h, ${N_SEEDS} samples)`,
    ...reach.regions.map((rg) => `- ${rg.name}: ${(100 * rg.fraction).toFixed(0)}% of samples${rg.earliest_h !== null ? `, earliest ${rg.earliest_h} h after burn` : ''}.`),
    `## Assessment`,
    `The characterised post-burn trajectory departs the DRO and is predicted to pass **${km(l1min)} km from L1 at ${utc(l1minT)}** (the L1 gateway corridor, 15 000 km radius, ${l1.fraction > 0 ? `is entered by ${(100 * l1.fraction).toFixed(0)}% of the Δv-consistent reachable set` : 'is not entered by the sampled reachable set'}). Predicted closest approach to the notional allied relay ${RELAY} (NRHO-class orbit) is **${km(minSep)} km at ${utc(minSepT)}** — ${minSep < 20000 ? 'inside the 20 000 km screening threshold; recommend continued priority tasking and a notification to the relay operator.' : 'outside the 20 000 km screening threshold; no safety action required at this time.'} Intent is not assessed; this is a traffic-safety and custody product.`,
    `## Sensor performance`,
    `- Ground optical: ${groundObs.filter((o) => o.t <= T_DETECT).length} observations before the blind period (${MOCK_GROUND.map((g) => `${g.id.replace('GND-', '')} ${groundObs.filter((o) => o.site === g.id).length}`).join(', ')}), none for the remaining ${((DURATION - T_DETECT) / H).toFixed(0)} h. Rules applied per site and epoch: Sun below −12°, elevation ≥ 20°, Moon separation above the phase-dependent 3°–15° glare zone, m_v ≤ site limit (target: ${pdef.radius_m * 2} m diameter, albedo ${pdef.albedo}, m_v ${detectObs.vis.magnitude.toFixed(1)} at detection). This gap is the structural weakness of ground-only cislunar SDA.`,
    `- ${droDef.id} (DRO observer): acquired on the first tasked window at ${km(spaceVis(spaceObsTimes[0]).range_km)} km; three observations collapsed σ_pos from ${km(sig(spaceObsTimes[0] - FRAME_DT))} km to ${km(sig(T_REGAIN + FRAME_DT))} km.`,
    `## Recommended actions`,
    `- Keep ${droDef.id} tasked on ${PROTAGONIST} at ≥ 1 observation / 4 h until the L1 passage.`,
    `- Pre-position ground observations for the next window outside the lunar-glare zone (none before ${hh(DURATION)} on the current orbit).`,
    `- Share the post-burn ephemeris and covariance with the ${RELAY} operator for conjunction screening.`,
    `## Model notes`,
    `Browser mock: CR3BP dynamics (μ = 0.012150585), ${hasBasisAt(t0ms) ? 'Earth orientation and Sun direction from the backend DE440s rotating-frame basis (GMST spin)' : 'mean-element Moon/Sun/GMST with the J2000 obliquity (lunar inclination neglected)'}, Lambertian-sphere photometry (m☉ = −26.74). The lunar-glare zone (3° new Moon → 15° full Moon) is a modelling assumption shared with the backend. Filter behaviour is emulated: isotropic σ_v = ${SIG_DV_MPS} m/s after detection; re-anchor σ = ${REANCHOR.map((x) => `${x.pos_km} km / ${x.vel_mps} m/s`).join(', ')} after successive observations. Nominal residuals and the Δv-estimate error are seeded random draws.`,
  ].join('\n\n');

  const scenario: DemoScenario = {
    meta: {
      title: 'Unannounced DRO departure (mock, browser CR3BP)',
      t0_utc: SCENARIO_T0,
      duration_s: DURATION,
      playback_s: 120,
      playback_speed: DURATION / 120,
      disclaimer: 'All objects and events are SIMULATED and attributed to a notional actor. Mock scenario generated in the browser; the backend scenario uses DE440s dynamics and the full OD/tasking stack.',
      protagonist_id: PROTAGONIST,
      brief_generated_utc: new Date(t0ms + T_BRIEF * 1000).toISOString(),
    },
    frames,
    events: events.sort((a, b) => a.t - b.t),
    brief,
  };
  cached = { key, scenario };
  return scenario;
}

/** GCRF state (km, km/s) of a mock object at the scenario epoch, for the catalog. */
export function mockObjectGcrfState(d: MockObjectDef): { rot: State6; gcrf: State6 } {
  const m = findMember(d.family, d.memberIndex);
  const rot = stateAtPhase(m, d.phase);
  return { rot, gcrf: rotToGcrfState(rot, Date.parse(SCENARIO_T0)) };
}
