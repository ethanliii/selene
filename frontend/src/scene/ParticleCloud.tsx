/**
 * Uncertainty particle cloud: renders N points from a Float32Array (xyz triplets, rotating frame, nondim).
 * Minimal working rendering: THREE.Points with a size-attenuated PointsMaterial.
 * TODO(later agent, M10): animate between consecutive demo frames' clouds (interpolate or crossfade),
 *   color by particle weight / age, and grow the cloud visibly as custody decays.
 */
import { useMemo } from 'react';
import * as THREE from 'three';
import type { Vec3 } from '../api/types';
import { COLORS } from './constants';

interface Props {
  /** Flat xyz triplets. */
  positions: Float32Array;
  color?: string;
  size?: number;
  opacity?: number;
  visible?: boolean;
}

export function ParticleCloud({ positions, color = COLORS.sim, size = 0.004, opacity = 0.75, visible = true }: Props) {
  const geom = useMemo(() => {
    const g = new THREE.BufferGeometry();
    g.setAttribute('position', new THREE.BufferAttribute(positions, 3));
    g.computeBoundingSphere();
    return g;
  }, [positions]);
  if (!visible || positions.length < 3) return null;
  return (
    <points geometry={geom}>
      <pointsMaterial color={color} size={size} sizeAttenuation transparent opacity={opacity} depthWrite={false} />
    </points>
  );
}

/** Helper: pack Vec3[] into a Float32Array for ParticleCloud. */
export function packPoints(pts: Vec3[]): Float32Array {
  const out = new Float32Array(pts.length * 3);
  for (let i = 0; i < pts.length; i++) {
    out[3 * i] = pts[i][0];
    out[3 * i + 1] = pts[i][1];
    out[3 * i + 2] = pts[i][2];
  }
  return out;
}
