/**
 * "Idle" frames: when no demo scenario is loaded, catalog objects and space observers are still shown moving.
 * Each object's rotating-frame state (`ic_rot`, or `state_gcrf_km` converted with the display-level frame model
 * in lib/ephem.ts) is propagated in the browser with the CR3BP integrator and tabulated on a 10-minute grid over
 * the timeline span. This is a DISPLAY approximation (CR3BP, no Sun/SRP, planar frame model), clearly weaker than
 * the backend's DE440s propagation; it exists so the Ops view is never empty.
 */
import { useMemo } from 'react';
import type { CatalogObject, DemoFrame, FrameSensor, OrbitFamilies, OrbitMember, Sensors, Vec3 } from '../api/types';
import { propagateDP45, tabulate, orbitPointAtPhase } from '../lib/cr3bp';
import { ecefToRot, gcrfToRotState, secToNd } from '../lib/ephem';
import { geoObserverPosition } from './mockScenario';
import { EARTH_ROT, kmToScene } from '../scene/constants';

const GEO_RADIUS = kmToScene(42164);

/** GEO observer at a fixed sub-satellite longitude: Earth-fixed equatorial point carried by the Earth's orientation. */
function geoAtLongitude(lonDeg: number, ms: number): Vec3 {
  const lo = (lonDeg * Math.PI) / 180;
  const u = ecefToRot([Math.cos(lo), Math.sin(lo), 0], ms);
  return [EARTH_ROT[0] + GEO_RADIUS * u[0], GEO_RADIUS * u[1], GEO_RADIUS * u[2]];
}
import { useSelene } from '../store/useSelene';
import { MOON_ROT } from '../scene/constants';

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

export function buildIdleFrames(catalog: CatalogObject[], sensors: Sensors | null, families: OrbitFamilies | null, t0Iso: string, t0Sec: number, t1Sec: number): DemoFrame[] {
  const t0Ms = Date.parse(t0Iso);
  const n = Math.max(2, Math.floor((t1Sec - t0Sec) / IDLE_DT_S) + 1);
  const span = (n - 1) * IDLE_DT_S;
  const tracks: { id: string; pos: Vec3[] }[] = [];
  for (const c of catalog) {
    try {
      const epochMs = Date.parse(c.epoch_utc);
      if (!Number.isFinite(epochMs)) continue;
      const dt0 = (t0Ms + t0Sec * 1000 - epochMs) / 1000;
      if (Math.abs(dt0) > MAX_EPOCH_OFFSET_S) continue;
      const ic = c.ic_rot ?? gcrfToRotState(c.state_gcrf_km, epochMs);
      if (!ic.every(Number.isFinite) || Math.hypot(ic[0], ic[1], ic[2]) > 6) continue;
      const s0 = Math.abs(dt0) > 1 ? Array.from(propagateDP45(ic, secToNd(dt0), { rtol: 1e-9, atol: 1e-11, hmax: 0.02 })) : ic;
      const pos = tabulate(s0, secToNd(span), n, { rtol: 1e-8, hmax: 0.02 });
      if (pos.every((p) => p.every(Number.isFinite))) tracks.push({ id: c.id, pos });
    } catch (e) {
      console.warn('idle propagation failed for', c.id, e);
    }
  }
  const observers = (sensors?.space ?? []).map((s) => ({ s, member: memberForObserver(s.orbit, s.orbit_member_id, families), phase: typeof s.phase === 'number' ? s.phase : hashPhase(s.id) }));
  const frames: DemoFrame[] = [];
  for (let i = 0; i < n; i++) {
    const t = t0Sec + i * IDLE_DT_S;
    const sens: FrameSensor[] = observers.map(({ s, member, phase }) => {
      const isGeo = String(s.orbit ?? '').toLowerCase().includes('geo');
      const pos: Vec3 = isGeo
        ? typeof s.geo_longitude_deg === 'number'
          ? geoAtLongitude(s.geo_longitude_deg, t0Ms + t * 1000)
          : geoObserverPosition(phase, t)
        : member
          ? orbitPointAtPhase(member.samples_rot, phase + secToNd(t) / member.period)
          : geoObserverPosition(phase, t);
      const bs = unitTo(pos, MOON_ROT);
      return { id: s.id, pos_rot: pos, boresight_rot: bs, pointing_rot: bs, fov_deg: s.fov_deg, active: false, target_id: null };
    });
    frames.push({ t, objects: tracks.map((tr) => ({ id: tr.id, pos_rot: tr.pos[i], custody: 'held' as const })), clouds: [], sensors: sens });
  }
  return frames;
}

/** Idle frames memoised on the catalog/sensors/families/time range; empty when a scenario is loaded. */
export function useIdleFrames(): DemoFrame[] {
  const catalog = useSelene((s) => s.catalog);
  const sensors = useSelene((s) => s.sensors);
  const families = useSelene((s) => s.families);
  const t0Iso = useSelene((s) => s.t0Iso);
  const t0Sec = useSelene((s) => s.t0Sec);
  const t1Sec = useSelene((s) => s.t1Sec);
  const hasScenario = useSelene((s) => !!s.scenario);
  return useMemo(() => (hasScenario ? [] : buildIdleFrames(catalog, sensors, families, t0Iso, t0Sec, t1Sec)), [catalog, sensors, families, t0Iso, t0Sec, t1Sec, hasScenario]);
}
