/**
 * Uncertainty particle cloud: THREE.Points over a PRE-ALLOCATED Float32Array (MAX_POINTS) with additive blending.
 * On a new frame the incoming positions are copied into the buffer, the draw range is set, and per-vertex colours
 * are computed from the Mahalanobis-like radius r = sqrt(Σ((p−c)/σ_axis)²) (diagonal covariance of the sample):
 * core particles (r < 1) are bright/white-hot, the 1–2σ shell is the object colour (amber, distinct from the violet
 * reachable set), the tail (r > 2) is dim. A "σ = X km" tag hangs at the centroid when the cloud's σ is known.
 * Nothing is allocated per render tick; per frame update is O(N) with no new typed arrays.
 */
import { useEffect, useMemo, useState } from 'react';
import * as THREE from 'three';
import type { Vec3 } from '../api/types';
import { Label } from './Label';

export const MAX_POINTS = 32768;

interface Props {
  /** Flat xyz triplets (rotating frame, nondim). Only the first MAX_POINTS points are drawn. */
  positions: Float32Array;
  color?: string;
  size?: number;
  opacity?: number;
  visible?: boolean;
  /** Object the cloud belongs to (label id) and its σ_pos [km] for the centroid tag. */
  objectId?: string;
  sigmaKm?: number;
  showLabel?: boolean;
}

const CORE = new THREE.Color('#ffffff');
const SIGMA_OFFSET = { dx: -98, dy: 12 };
const TAIL = new THREE.Color('#3a2a10');

export function ParticleCloud({ positions, color = '#ffb86b', size = 0.003, opacity = 0.85, visible = true, objectId, sigmaKm, showLabel = true }: Props) {
  const { geom, posAttr, colAttr } = useMemo(() => {
    const g = new THREE.BufferGeometry();
    const pa = new THREE.BufferAttribute(new Float32Array(MAX_POINTS * 3), 3);
    const ca = new THREE.BufferAttribute(new Float32Array(MAX_POINTS * 3), 3);
    pa.setUsage(THREE.DynamicDrawUsage);
    ca.setUsage(THREE.DynamicDrawUsage);
    g.setAttribute('position', pa);
    g.setAttribute('color', ca);
    g.setDrawRange(0, 0);
    return { geom: g, posAttr: pa, colAttr: ca };
  }, []);
  const mid = useMemo(() => new THREE.Color(color), [color]);
  const [centroid, setCentroid] = useState<Vec3 | null>(null);

  useEffect(() => {
    const n = Math.min(MAX_POINTS, Math.floor(positions.length / 3));
    const dst = posAttr.array as Float32Array;
    dst.set(positions.subarray(0, n * 3));
    // Sample mean and per-axis std (diagonal covariance).
    let cx = 0, cy = 0, cz = 0;
    for (let i = 0; i < n; i++) {
      cx += dst[3 * i];
      cy += dst[3 * i + 1];
      cz += dst[3 * i + 2];
    }
    cx /= n || 1;
    cy /= n || 1;
    cz /= n || 1;
    let sx = 0, sy = 0, sz = 0;
    for (let i = 0; i < n; i++) {
      sx += (dst[3 * i] - cx) ** 2;
      sy += (dst[3 * i + 1] - cy) ** 2;
      sz += (dst[3 * i + 2] - cz) ** 2;
    }
    const eps = 1e-12;
    sx = Math.sqrt(sx / (n || 1)) + eps;
    sy = Math.sqrt(sy / (n || 1)) + eps;
    sz = Math.sqrt(sz / (n || 1)) + eps;
    const col = colAttr.array as Float32Array;
    for (let i = 0; i < n; i++) {
      const r = Math.sqrt(((dst[3 * i] - cx) / sx) ** 2 + ((dst[3 * i + 1] - cy) / sy) ** 2 + ((dst[3 * i + 2] - cz) / sz) ** 2);
      let cr: number, cg: number, cb: number;
      if (r < 1) {
        const w = r; // white-hot core → object colour at 1σ
        cr = CORE.r + (mid.r - CORE.r) * w;
        cg = CORE.g + (mid.g - CORE.g) * w;
        cb = CORE.b + (mid.b - CORE.b) * w;
      } else {
        const w = Math.min(1, (r - 1) / 2); // 1σ → 3σ fades to the tail colour
        cr = mid.r + (TAIL.r - mid.r) * w;
        cg = mid.g + (TAIL.g - mid.g) * w;
        cb = mid.b + (TAIL.b - mid.b) * w;
      }
      col[3 * i] = cr;
      col[3 * i + 1] = cg;
      col[3 * i + 2] = cb;
    }
    posAttr.needsUpdate = true;
    colAttr.needsUpdate = true;
    geom.setDrawRange(0, n);
    geom.computeBoundingSphere();
    // Hang the σ tag at the −σ_y (camera-side) edge of the cloud; the label is then offset further down-left in
    // screen space so it never shares the object label's box (which sits up-right of the marker).
    setCentroid(n > 0 ? [cx, cy - sy, cz - sz] : null);
  }, [positions, geom, posAttr, colAttr, mid]);

  useEffect(() => () => geom.dispose(), [geom]);

  if (!visible || positions.length < 3) return null;
  const sigmaText = sigmaKm !== undefined && Number.isFinite(sigmaKm) ? `σ = ${sigmaKm >= 100 ? Math.round(sigmaKm).toLocaleString('en-US') : sigmaKm.toFixed(1)} km` : null;
  return (
    <group>
      <points geometry={geom} frustumCulled={false}>
        <pointsMaterial vertexColors size={size} sizeAttenuation transparent opacity={opacity} depthWrite={false} blending={THREE.AdditiveBlending} />
      </points>
      {showLabel && centroid && sigmaText && <Label position={centroid} text={sigmaText} variant="tag" className="sigma" id={`cloud-${objectId ?? 'x'}`} priority={69} color="#ffb86b" offset={SIGMA_OFFSET} />}
    </group>
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
