/**
 * Frame lookup at the time cursor → scene-ready structures.
 * Source of frames: the loaded demo scenario, else the idle frames (live backend trajectories, or catalog objects
 * CR3BP-propagated in the browser when the backend is down). Objects carry the bracketing frame positions so the
 * scene can interpolate per render tick without re-rendering React on every tSec change; live objects additionally
 * carry their `track` so the scene samples the Hermite-interpolated trajectory exactly at the cursor.
 * Trails/clouds/sensors change only when the frame index changes.
 */
import { useMemo } from 'react';
import type { DemoFrame, FrameReachable, FrameSensor, Vec3 } from '../api/types';
import { trackSlice, type Track } from '../lib/tracks';
import { useSelene } from '../store/useSelene';
import { useIdleFrames } from './useIdleObjects';
import { trailSpanFor } from './useLiveTracks';

/** Binary search: index of the last frame with frame.t <= t (or 0). */
export function frameIndexAt(frames: DemoFrame[], t: number): number {
  let lo = 0;
  let hi = frames.length - 1;
  if (hi < 0) return -1;
  while (lo < hi) {
    const mid = (lo + hi + 1) >> 1;
    if (frames[mid].t <= t) lo = mid;
    else hi = mid - 1;
  }
  return lo;
}

export interface SceneObject {
  id: string;
  name?: string;
  simulated: boolean;
  pos: Vec3;
  /** Position at the next frame (for interpolation) and the bracketing frame times [s]. */
  posNext: Vec3;
  tFrame: number;
  tNext: number;
  trail: Vec3[];
  custody: 'held' | 'degraded' | 'lost' | 'unknown';
  sigmaKm?: number;
  /** Live backend track (rotating frame): when present the scene samples it directly at the cursor. */
  track?: Track;
}

export interface SceneCloud {
  objectId: string;
  positions: Float32Array;
  sigmaKm?: number;
}

export interface SceneFrame {
  frame: DemoFrame | null;
  idx: number;
  source: 'scenario' | 'idle' | 'none';
  objects: SceneObject[];
  clouds: SceneCloud[];
  sensors: FrameSensor[];
  reachable?: FrameReachable;
  /** Live trajectory loading progress (idle live mode). */
  tracksLoaded: number;
  tracksTotal: number;
  trackErrors: Map<string, string>;
  live: boolean;
}

const EMPTY: SceneFrame = { frame: null, idx: -1, source: 'none', objects: [], clouds: [], sensors: [], tracksLoaded: 0, tracksTotal: 0, trackErrors: new Map(), live: false };
const TRAIL = 72;

export function useScenarioFrame(): SceneFrame {
  const scenario = useSelene((s) => s.scenario);
  const catalog = useSelene((s) => s.catalog);
  const idle = useIdleFrames();
  const frames = scenario?.frames ?? idle.frames;
  // Quantise the time cursor to the frame grid so we only recompute when the frame changes.
  const idx = useSelene((s) => frameIndexAt(frames, s.tSec));

  return useMemo(() => {
    const base = { tracksLoaded: idle.tracks.loaded, tracksTotal: idle.tracks.total, trackErrors: idle.tracks.errors, live: idle.live };
    if (idx < 0 || idx >= frames.length) return { ...EMPTY, ...base };
    const frame = frames[idx];
    const next = frames[Math.min(idx + 1, frames.length - 1)];
    const byId = new Map(catalog.map((c) => [c.id, c]));
    const objects: SceneObject[] = frame.objects.map((o) => {
      const cat = byId.get(o.id);
      const track = scenario ? undefined : idle.tracks.tracks.get(o.id);
      let trail: Vec3[];
      if (track) {
        trail = trackSlice(track, frame.t - trailSpanFor(cat), frame.t, 160);
      } else {
        trail = [];
        for (let k = Math.max(0, idx - TRAIL); k <= idx; k++) {
          const p = frames[k].objects.find((q) => q.id === o.id);
          if (p) trail.push(p.pos_rot);
        }
      }
      const nx = next.objects.find((q) => q.id === o.id);
      return {
        id: o.id,
        name: cat?.name,
        simulated: cat ? cat.kind === 'simulated' : true,
        pos: o.pos_rot,
        posNext: nx?.pos_rot ?? o.pos_rot,
        tFrame: frame.t,
        tNext: next.t > frame.t ? next.t : frame.t + 1,
        trail,
        custody: o.custody,
        sigmaKm: o.sigma_pos_km,
        track,
      };
    });
    const clouds: SceneCloud[] = frame.clouds
      .filter((c) => c.points instanceof Float32Array && c.points.length >= 3)
      .map((c) => ({ objectId: c.object_id, positions: c.points as Float32Array, sigmaKm: c.sigma_km }));
    return { frame, idx, source: scenario ? 'scenario' : 'idle', objects, clouds, sensors: frame.sensors, reachable: frame.reachable, ...base };
  }, [frames, catalog, idx, scenario, idle]);
}
