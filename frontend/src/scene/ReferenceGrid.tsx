/** Faint xy-plane reference grid (barycentric, rotating frame) and a screen-constant axis triad at the barycentre. */
import { useMemo, useRef } from 'react';
import * as THREE from 'three';
import { COLORS } from './constants';
import { useScreenScale } from './screenScale';

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

  const axes = useMemo(() => {
    const mk = (dir: [number, number, number]) => {
      const g = new THREE.BufferGeometry();
      g.setAttribute('position', new THREE.Float32BufferAttribute([0, 0, 0, ...dir], 3));
      return g;
    };
    return { x: mk([1, 0, 0]), y: mk([0, 1, 0]), z: mk([0, 0, 1]) };
  }, []);
  // Unit-length axes scaled to a constant 48 px on screen (capped at 0.12 L*), so the triad never sprawls across
  // a close-up nor vanishes in the overview.
  const triad = useRef<THREE.Group>(null);
  useScreenScale(triad, 48, 0, 0.12);

  if (!visible) return null;
  return (
    <group>
      <lineSegments geometry={geom}>
        <lineBasicMaterial color={COLORS.grid} transparent opacity={0.55} depthWrite={false} />
      </lineSegments>
      {/* Axis triad at the barycenter: x red-ish (toward Moon), y green-ish (prograde), z blue-ish (pole). */}
      <group ref={triad}>
        <lineSegments geometry={axes.x}>
          <lineBasicMaterial color="#c0504d" transparent opacity={0.7} />
        </lineSegments>
        <lineSegments geometry={axes.y}>
          <lineBasicMaterial color="#5aa469" transparent opacity={0.7} />
        </lineSegments>
        <lineSegments geometry={axes.z}>
          <lineBasicMaterial color="#4f81bd" transparent opacity={0.7} />
        </lineSegments>
      </group>
    </group>
  );
}
