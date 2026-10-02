/** Faint xy-plane reference grid (barycentric, rotating frame). The axis triad lives in the corner gizmo (SceneRoot). */
import { useMemo } from 'react';
import * as THREE from 'three';
import { COLORS } from './constants';

interface Props {
  visible?: boolean;
  /** Half-extent in scene units (1 = L*). */
  extent?: number;
  step?: number;
}

export function ReferenceGrid({ visible = true, extent = 1.5, step = 0.25 }: Props) {
  const geom = useMemo(() => {
    const pts: number[] = [];
    for (let v = -extent; v <= extent + 1e-9; v += step) {
      pts.push(-extent, v, 0, extent, v, 0);
      pts.push(v, -extent, 0, v, extent, 0);
    }
    const g = new THREE.BufferGeometry();
    g.setAttribute('position', new THREE.Float32BufferAttribute(pts, 3));
    return g;
  }, [extent, step]);

  if (!visible) return null;
  return (
    <group>
      <lineSegments geometry={geom}>
        <lineBasicMaterial color={COLORS.grid} transparent opacity={0.55} depthWrite={false} />
      </lineSegments>
    </group>
  );
}
