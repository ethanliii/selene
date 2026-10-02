/**
 * Reachable-set point cloud (rotating frame): sampled positions the object could occupy under the assumed Δv
 * budget/horizon, drawn in --warn with additive blending so it reads as "possible futures", distinct from the
 * uncertainty cloud (--sim). Data comes from frame.reachable (rows or flat array).
 */
import { useMemo } from 'react';
import * as THREE from 'three';
import type { FrameReachable } from '../api/types';
import { toFloat32 } from '../demo/normalize';
import { COLORS } from './constants';

export function Reachable({ data, visible = true }: { data?: FrameReachable; visible?: boolean }) {
  const geom = useMemo(() => {
    if (!data) return null;
    const pts = Array.isArray(data.points) && data.points.length && Array.isArray(data.points[0]) ? toFloat32(undefined, data.points as [number, number, number][]) : toFloat32(data.points as number[]);
    if (pts.length < 3) return null;
    const g = new THREE.BufferGeometry();
    g.setAttribute('position', new THREE.BufferAttribute(pts, 3));
    g.computeBoundingSphere();
    return g;
  }, [data]);
  if (!visible || !geom) return null;
  return (
    <points geometry={geom} frustumCulled={false}>
      <pointsMaterial color={COLORS.warn} size={0.0035} sizeAttenuation transparent opacity={0.55} depthWrite={false} blending={THREE.AdditiveBlending} />
    </points>
  );
}
