/**
 * Reachable-set point cloud (rotating frame): sampled positions the object could occupy under the assumed Δv
 * budget/horizon, drawn in --accent-2 (violet) with additive blending so it reads as "possible futures", distinct
 * from the amber uncertainty cloud. A faint convex-hull outline (xy projection at the set's mean z) frames the set.
 * Data comes from frame.reachable (rows or flat array).
 */
import { Line } from '@react-three/drei';
import { useMemo } from 'react';
import * as THREE from 'three';
import type { FrameReachable, Vec3 } from '../api/types';
import { toFloat32 } from '../demo/normalize';
import { COLORS } from './constants';

/** Andrew's monotone chain on (x, y); returns the hull as a closed loop at height z. */
function hullXY(pts: Float32Array, z: number): Vec3[] {
  const n = Math.floor(pts.length / 3);
  if (n < 3) return [];
  const idx = Array.from({ length: n }, (_, i) => i).sort((a, b) => pts[3 * a] - pts[3 * b] || pts[3 * a + 1] - pts[3 * b + 1]);
  const cross = (o: number, a: number, b: number) => (pts[3 * a] - pts[3 * o]) * (pts[3 * b + 1] - pts[3 * o + 1]) - (pts[3 * a + 1] - pts[3 * o + 1]) * (pts[3 * b] - pts[3 * o]);
  const lower: number[] = [];
  for (const i of idx) {
    while (lower.length >= 2 && cross(lower[lower.length - 2], lower[lower.length - 1], i) <= 0) lower.pop();
    lower.push(i);
  }
  const upper: number[] = [];
  for (let k = idx.length - 1; k >= 0; k--) {
    const i = idx[k];
    while (upper.length >= 2 && cross(upper[upper.length - 2], upper[upper.length - 1], i) <= 0) upper.pop();
    upper.push(i);
  }
  const hull = lower.slice(0, -1).concat(upper.slice(0, -1));
  const out: Vec3[] = hull.map((i) => [pts[3 * i], pts[3 * i + 1], z]);
  if (out.length) out.push(out[0]);
  return out;
}

export function Reachable({ data, visible = true }: { data?: FrameReachable; visible?: boolean }) {
  const { geom, hull } = useMemo(() => {
    if (!data) return { geom: null, hull: [] as Vec3[] };
    const pts = Array.isArray(data.points) && data.points.length && Array.isArray(data.points[0]) ? toFloat32(undefined, data.points as [number, number, number][]) : toFloat32(data.points as number[]);
    if (pts.length < 3) return { geom: null, hull: [] as Vec3[] };
    const g = new THREE.BufferGeometry();
    g.setAttribute('position', new THREE.BufferAttribute(pts, 3));
    g.computeBoundingSphere();
    let z = 0;
    const n = Math.floor(pts.length / 3);
    for (let i = 0; i < n; i++) z += pts[3 * i + 2];
    z /= n || 1;
    return { geom: g, hull: hullXY(pts, z) };
  }, [data]);
  if (!visible || !geom) return null;
  return (
    <group>
      <points geometry={geom} frustumCulled={false}>
        <pointsMaterial color={COLORS.accent2} size={0.0035} sizeAttenuation transparent opacity={0.6} depthWrite={false} blending={THREE.AdditiveBlending} />
      </points>
      {hull.length > 3 && <Line points={hull} color={COLORS.accent2} lineWidth={1} transparent opacity={0.35} depthWrite={false} />}
    </group>
  );
}
