/**
 * Demo-scenario driver helpers: pick the frame at (or just before) the current time cursor and
 * derive scene-ready structures from it.
 * TODO(later agent, M10): linear interpolation of object positions between frames; trail accumulation
 *   from past frames; FOV cone props from frame.sensors; exclusion cones from ephemeris.
 */
import { useMemo } from 'react';
import type { DemoFrame } from '../api/types';
import type { SceneObject } from '../scene/Objects';
import { packPoints } from '../scene/ParticleCloud';
import { useSelene } from '../store/useSelene';

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

export interface SceneFrame {
  frame: DemoFrame | null;
  objects: SceneObject[];
  clouds: { objectId: string; positions: Float32Array }[];
}

export function useScenarioFrame(): SceneFrame {
  const scenario = useSelene((s) => s.scenario);
  const catalog = useSelene((s) => s.catalog);
  // Quantize the time cursor to the frame grid so we only recompute when the frame changes.
  const idx = useSelene((s) => (s.scenario ? frameIndexAt(s.scenario.frames, s.tSec) : -1));

  return useMemo(() => {
    const frames = scenario?.frames ?? [];
    if (idx < 0 || idx >= frames.length) return { frame: null, objects: [], clouds: [] };
    const frame = frames[idx];
    const byId = new Map(catalog.map((c) => [c.id, c]));
    // Trail: positions from the previous frames (bounded), oldest first.
    const TRAIL = 60;
    const objects: SceneObject[] = frame.objects.map((o) => {
      const trail = [];
      for (let k = Math.max(0, idx - TRAIL); k <= idx; k++) {
        const p = frames[k].objects.find((q) => q.id === o.id);
        if (p) trail.push(p.pos_rot);
      }
      const cat = byId.get(o.id);
      return { id: o.id, name: cat?.name, simulated: cat ? cat.kind === 'simulated' : true, pos: o.pos_rot, trail, custody: o.custody };
    });
    const clouds = frame.clouds.map((c) => ({ objectId: c.object_id, positions: packPoints(c.points_rot) }));
    return { frame, objects, clouds };
  }, [scenario, catalog, idx]);
}
