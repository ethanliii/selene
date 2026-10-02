/**
 * "Idle" frames: when no demo scenario is loaded, catalog objects and space observers are still shown moving.
 *
 * LIVE mode (backend catalog): object positions come from backend trajectories (`useLiveTracks`: DE440s N-body +
 * SRP truth for SIMULATED objects, JPL Horizons splines for real ones), evaluated on a 10-minute frame grid here
 * and per render tick in the scene. Space observers ride their orbit-library record (`/api/orbits/records/{id}`,
 * same phase convention as the backend: τ = ((t − epoch)/T* + phase·P) mod P) or, for GEO hosts, the exact
 * Earth-fixed geostationary point carried by Rᵀ·R_z(GMST).
 *
 * MOCK mode (backend down): each object's rotating-frame state (`ic_rot`, or `state_gcrf_km` converted with the
 * display-level frame model in lib/ephem.ts) is propagated in the browser with the CR3BP integrator. This is a
 * DISPLAY approximation (CR3BP, no Sun/SRP, planar frame model); it exists so the Ops view is never empty.
 */
import { useMemo } from 'react';
import type { CatalogObject, DemoFrame, FrameSensor, OrbitFamilies, OrbitMember, OrbitRecord, Sensors, Vec3 } from '../api/types';
import { propagateDP45, tabulate, orbitPointAtPhase } from '../lib/cr3bp';
import { ecefToRot, gcrfToRotState, moonDistanceKm, secToNd } from '../lib/ephem';
import { trackCovers, trackEval, type Track } from '../lib/tracks';
import { geoObserverPosition } from './mockScenario';
import { EARTH_ROT, kmToScene, MOON_ROT } from '../scene/constants';
import { useSelene } from '../store/useSelene';
import { useLiveTracks, type TrackSet } from './useLiveTracks';

const GEO_RADIUS_KM = 42164;

/** GEO observer at a fixed sub-satellite longitude: Earth-fixed equatorial point carried by the Earth's orientation. */
function geoAtLongitude(lonDeg: number, ms: number, radiusKm = GEO_RADIUS_KM): Vec3 {
  const lo = (lonDeg * Math.PI) / 180;
  const u = ecefToRot([Math.cos(lo), Math.sin(lo), 0], ms);
  const r = radiusKm / moonDistanceKm(ms);
  return [EARTH_ROT[0] + r * u[0], r * u[1], r * u[2]];
}

const IDLE_DT_S = 600;
const MAX_EPOCH_OFFSET_S = 60 * 86400;

function unitTo(a: Vec3, b: Vec3): Vec3 {
  const d: Vec3 = [b[0] - a[0], b[1] - a[1], b[2] - a[2]];
  const n = Math.hypot(d[0], d[1], d[2]) || 1;
  return [d[0] / n, d[1] / n, d[2] / n];
}

/** Pick an orbit-library member for a space observer: explicit member id, else a family matched by keyword. */
export function memberForObserver(orbit: string, memberId: string | undefined, families: OrbitFamilies | null): OrbitMember | null {
  if (!families) return null;
  if (memberId) for (const f of families.families) for (const m of f.members) if (m.id === memberId) return m;
  const key = String(orbit ?? '').toLowerCase();
  const want = key.includes('dro') ? 'dro' : key.includes('nrho') ? 'nrho' : key.includes('l1') ? 'l1' : key.includes('l2') ? 'l2' : key.includes('reson') ? 'reson' : null;
  if (!want) return null;
  const fams = families.families.filter((x) => x.members.length > 0);
  const lname = (x: { name: string }) => x.name.toLowerCase();
  let f = fams.find((x) => lname(x).includes(want));
  if (want === 'l2') f = fams.find((x) => lname(x).includes('l2') && (lname(x).includes('halo') || lname(x).includes('nrho'))) ?? f;
  if (want === 'nrho' && !f) f = fams.find((x) => lname(x).includes('l2') && lname(x).includes('halo'));
  if (!f) return null;
  // NRHO: the most NRHO-like member is the first of the mock family (largest |z|, shortest period).
  return want === 'nrho' ? f.members[0] : f.members[Math.floor(f.members.length / 2)];
}

function hashPhase(id: string): number {
  let h = 0;
  for (let i = 0; i < id.length; i++) h = (h * 31 + id.charCodeAt(i)) >>> 0;
  return (h % 1000) / 1000;
}

export interface IdleInputs {
  catalog: CatalogObject[];
  sensors: Sensors | null;
  families: OrbitFamilies | null;
  t0Iso: string;
  t0Sec: number;
  t1Sec: number;
  /** Live backend tracks (empty map in mock mode). */
  tracks: Map<string, Track>;
  /** Live orbit records for space observers (sensor id → record). */
  observerOrbits: Record<string, OrbitRecord>;
  /** True when the catalog is live: never fall back to browser CR3BP propagation for missing tracks. */
  live: boolean;
}

export function buildIdleFrames(inp: IdleInputs): DemoFrame[] {
  const { catalog, sensors, families, t0Iso, t0Sec, t1Sec, tracks, observerOrbits, live } = inp;
  const t0Ms = Date.parse(t0Iso);
  const n = Math.max(2, Math.floor((t1Sec - t0Sec) / IDLE_DT_S) + 1);
  const span = (n - 1) * IDLE_DT_S;
  const trackTracks: { id: string; track: Track }[] = [];
  const propTracks: { id: string; pos: Vec3[] }[] = [];
  for (const c of catalog) {
    const tr = tracks.get(c.id);
    if (tr) {
      trackTracks.push({ id: c.id, track: tr });
      continue;
    }
    if (live) continue; // track still loading or unavailable in this window: not drawn rather than approximated
    try {
      const epochMs = Date.parse(c.epoch_utc);
      if (!Number.isFinite(epochMs)) continue;
      const dt0 = (t0Ms + t0Sec * 1000 - epochMs) / 1000;
      if (Math.abs(dt0) > MAX_EPOCH_OFFSET_S) continue;
      const ic = c.ic_rot ?? gcrfToRotState(c.state_gcrf_km, epochMs);
      if (!ic.every(Number.isFinite) || Math.hypot(ic[0], ic[1], ic[2]) > 6) continue;
      const s0 = Math.abs(dt0) > 1 ? Array.from(propagateDP45(ic, secToNd(dt0), { rtol: 1e-9, atol: 1e-11, hmax: 0.02 })) : ic;
      const pos = tabulate(s0, secToNd(span), n, { rtol: 1e-8, hmax: 0.02 });
      if (pos.every((p) => p.every(Number.isFinite))) propTracks.push({ id: c.id, pos });
    } catch (e) {
      console.warn('idle propagation failed for', c.id, e);
    }
  }
  const observers = (sensors?.space ?? []).map((s) => ({
    s,
    record: observerOrbits[s.id] ?? null,
    member: memberForObserver(s.orbit, s.orbit_member_id, families),
    phase: typeof s.phase === 'number' ? s.phase : hashPhase(s.id),
    epochMs: typeof s.epoch_ms === 'number' ? s.epoch_ms : t0Ms,
  }));
  const frames: DemoFrame[] = [];
  const tmp: Vec3 = [0, 0, 0];
  for (let i = 0; i < n; i++) {
    const t = t0Sec + i * IDLE_DT_S;
    const ms = t0Ms + t * 1000;
    const sens: FrameSensor[] = observers.map(({ s, record, member, phase, epochMs }) => {
      const isGeo = String(s.orbit ?? '').toLowerCase().includes('geo');
      let pos: Vec3;
      if (isGeo) {
        pos = typeof s.geo_longitude_deg === 'number' ? geoAtLongitude(s.geo_longitude_deg, ms, s.orbit_info?.radius_km ?? GEO_RADIUS_KM) : geoObserverPosition(phase, t);
      } else if (record) {
        pos = orbitPointAtPhase(record.samples_rot, phase + (ms - epochMs) / (record.period_days * 86400e3));
      } else if (member) {
        pos = orbitPointAtPhase(member.samples_rot, phase + secToNd(t) / member.period);
      } else pos = geoObserverPosition(phase, t);
      const bs = unitTo(pos, MOON_ROT);
      return { id: s.id, pos_rot: pos, boresight_rot: bs, pointing_rot: bs, fov_deg: s.fov_deg, active: false, target_id: null };
    });
    const objects: DemoFrame['objects'] = [];
    for (const tr of trackTracks) {
      if (!trackCovers(tr.track, t)) continue;
      trackEval(tr.track, t, tmp);
      // No OD/tasking has run in idle mode: custody is NOT measured, so it is reported as 'unknown', never 'held'.
      objects.push({ id: tr.id, pos_rot: [tmp[0], tmp[1], tmp[2]], custody: 'unknown' });
    }
    for (const tr of propTracks) objects.push({ id: tr.id, pos_rot: tr.pos[i], custody: 'unknown' });
    frames.push({ t, objects, clouds: [], sensors: sens });
  }
  return frames;
}

export interface IdleResult {
  frames: DemoFrame[];
  tracks: TrackSet;
  live: boolean;
}

/** Idle frames memoised on the catalog/sensors/families/time range; empty when a scenario is loaded. */
export function useIdleFrames(): IdleResult {
  const catalog = useSelene((s) => s.catalog);
  const live = useSelene((s) => s.catalogMeta !== null);
  const sensors = useSelene((s) => s.sensors);
  const observerOrbits = useSelene((s) => s.observerOrbits);
  const families = useSelene((s) => s.families);
  const t0Iso = useSelene((s) => s.t0Iso);
  const t0Sec = useSelene((s) => s.t0Sec);
  const t1Sec = useSelene((s) => s.t1Sec);
  const hasScenario = useSelene((s) => !!s.scenario);
  const tracks = useLiveTracks(live && !hasScenario);
  const frames = useMemo(
    () => (hasScenario ? [] : buildIdleFrames({ catalog, sensors, families, t0Iso, t0Sec, t1Sec, tracks: tracks.tracks, observerOrbits, live })),
    [catalog, sensors, families, t0Iso, t0Sec, t1Sec, hasScenario, tracks, observerOrbits, live],
  );
  return { frames, tracks, live };
}

export { kmToScene };
