/**
 * Observation flashes: when the scenario frame at the cursor carries realised tracklets (`frame.observations`
 * rows with a residual), a thin line from the observing sensor to its target and a small ring at the target flash
 * on and fade over ~1.2 s of wall-clock time. Tasking "looks" without a measurement realisation are not flashed.
 * The sensor position comes from the frame's sensors (`pos_rot`), the target from the interpolated object map.
 */
import { useFrame } from '@react-three/fiber';
import { useMemo, useRef } from 'react';
import * as THREE from 'three';
import type { DemoFrame, FrameSensor, Vec3 } from '../api/types';
import { COLORS } from './constants';

const FADE_MS = 1200;
const MAX_LINES = 64;

export function ObservationFlashes({ frame, sensors, targets }: { frame: DemoFrame | null; sensors: FrameSensor[]; targets: Map<string, Vec3> }) {
  const { geom, posAttr, mat } = useMemo(() => {
    const g = new THREE.BufferGeometry();
    const pa = new THREE.BufferAttribute(new Float32Array(MAX_LINES * 6), 3);
    pa.setUsage(THREE.DynamicDrawUsage);
    g.setAttribute('position', pa);
    g.setDrawRange(0, 0);
    const m = new THREE.LineBasicMaterial({ color: COLORS.ok, transparent: true, opacity: 0, depthWrite: false, blending: THREE.AdditiveBlending });
    return { geom: g, posAttr: pa, mat: m };
  }, []);
  const born = useRef(0);
  const lastT = useRef(-1);
  const count = useRef(0);

  // Rebuild the segment list when the frame changes (not per tick).
  const segs = useMemo(() => {
    if (!frame?.observations?.length) return [] as { a: Vec3; b: Vec3 }[];
    const byId = new Map(sensors.map((s) => [s.id, s]));
    const out: { a: Vec3; b: Vec3 }[] = [];
    for (const o of frame.observations) {
      if (typeof o.residual_arcsec !== 'number') continue; // tasking look without a measurement: no flash
      const s = byId.get(o.sensor_id);
      const t = targets.get(o.object_id);
      if (!s?.pos_rot || !t) continue;
      out.push({ a: s.pos_rot, b: t });
      if (out.length >= MAX_LINES) break;
    }
    return out;
  }, [frame, sensors, targets]);

  useFrame(() => {
    if (frame && frame.t !== lastT.current) {
      lastT.current = frame.t;
      born.current = performance.now();
      const arr = posAttr.array as Float32Array;
      segs.forEach((s, i) => arr.set([s.a[0], s.a[1], s.a[2], s.b[0], s.b[1], s.b[2]], 6 * i));
      count.current = segs.length;
      posAttr.needsUpdate = true;
      geom.setDrawRange(0, segs.length * 2);
    }
    const age = performance.now() - born.current;
    mat.opacity = count.current && age < FADE_MS ? 0.9 * (1 - age / FADE_MS) : 0;
  });
  return <lineSegments geometry={geom} material={mat} frustumCulled={false} renderOrder={2} />;
}
