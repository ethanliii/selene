/**
 * Tracked objects + trails (rotating frame, nondimensional).
 * Each object is a constant-pixel-size marker (core dot + soft halo, selection ring) coloured by custody status,
 * with an HTML label carrying a SIMULATED tag (--sim) or a REAL/HORIZONS tag (--accent). Markers are scaled per
 * render tick to a fixed screen size (scene/screenScale.ts) so they never out-size the Moon in close-ups.
 * Positions are interpolated between the bracketing scenario frames every render tick (imperatively, no React
 * re-render per tick). Trails are the last N frame positions drawn as a fading line. Click (generous hit sphere)
 * selects.
 */
import { Html, Line } from '@react-three/drei';
import { useFrame } from '@react-three/fiber';
import { useMemo, useRef } from 'react';
import * as THREE from 'three';
import type { SceneObject } from '../demo/useScenarioFrame';
import { useSelene } from '../store/useSelene';
import { COLORS } from './constants';
import { useScreenScale } from './screenScale';

export type { SceneObject } from '../demo/useScenarioFrame';

export const CUSTODY_COLOR: Record<SceneObject['custody'], string> = {
  held: COLORS.ok,
  degraded: COLORS.warn,
  lost: COLORS.alert,
};

/** Marker radii in screen pixels. */
const PX = { core: 3.5, coreSel: 4.5, halo: 8, haloSel: 10, ringIn: 12, ringOut: 13.5, hit: 14 };

interface Props {
  objects: SceneObject[];
  showTrails?: boolean;
}

function ObjectMarker({ o, selected, onSelect, showLabel }: { o: SceneObject; selected: boolean; onSelect: () => void; showLabel: boolean }) {
  const ref = useRef<THREE.Group>(null);
  const markerRef = useRef<THREE.Group>(null);
  const color = CUSTODY_COLOR[o.custody];
  useFrame(() => {
    if (!ref.current) return;
    const t = useSelene.getState().tSec;
    const w = Math.min(1, Math.max(0, (t - o.tFrame) / (o.tNext - o.tFrame)));
    ref.current.position.set(o.pos[0] + (o.posNext[0] - o.pos[0]) * w, o.pos[1] + (o.posNext[1] - o.pos[1]) * w, o.pos[2] + (o.posNext[2] - o.pos[2]) * w);
  });
  // Unit-radius geometry scaled to 1 px; children sizes below are then in pixels.
  useScreenScale(markerRef, 1);
  return (
    <group ref={ref} position={o.pos}>
      <group ref={markerRef}>
        <mesh
          onClick={(e) => {
            e.stopPropagation();
            onSelect();
          }}
        >
          <sphereGeometry args={[PX.hit, 8, 6]} />
          <meshBasicMaterial transparent opacity={0} depthWrite={false} />
        </mesh>
        <mesh renderOrder={3}>
          <sphereGeometry args={[selected ? PX.coreSel : PX.core, 12, 8]} />
          <meshBasicMaterial color={color} depthTest={false} />
        </mesh>
        <mesh renderOrder={2}>
          <sphereGeometry args={[selected ? PX.haloSel : PX.halo, 12, 8]} />
          <meshBasicMaterial color={color} transparent opacity={0.16} depthWrite={false} depthTest={false} />
        </mesh>
        {selected && (
          <mesh renderOrder={3}>
            <ringGeometry args={[PX.ringIn, PX.ringOut, 40]} />
            <meshBasicMaterial color={color} side={THREE.DoubleSide} transparent opacity={0.85} depthWrite={false} depthTest={false} />
          </mesh>
        )}
      </group>
      {showLabel && (
        <Html zIndexRange={[20, 0]} style={{ pointerEvents: 'none' }}>
          <div className={`obj-label${selected ? ' selected' : ''}`} style={{ borderColor: color }}>
            <span className="id">{o.id}</span>
            <span className={`tag ${o.simulated ? 'sim' : 'accent'}`}>{o.simulated ? 'SIMULATED' : 'HORIZONS'}</span>
            {o.custody !== 'held' && <span className={`tag ${o.custody === 'lost' ? 'alert' : 'warn'}`}>{o.custody.toUpperCase()}</span>}
          </div>
        </Html>
      )}
    </group>
  );
}

function Trail({ points, color }: { points: THREE.Vector3[] | [number, number, number][]; color: string }) {
  const colors = useMemo(() => {
    const c = new THREE.Color(color);
    const bg = new THREE.Color('#05070b');
    const n = points.length;
    return Array.from({ length: n }, (_, i) => {
      const w = n > 1 ? Math.pow(i / (n - 1), 1.5) : 1;
      return bg.clone().lerp(c, 0.08 + 0.92 * w);
    }).map((cc) => [cc.r, cc.g, cc.b] as [number, number, number]);
  }, [points, color]);
  return <Line points={points} vertexColors={colors} lineWidth={1.2} transparent opacity={0.9} depthWrite={false} />;
}

export function Objects({ objects, showTrails = true }: Props) {
  const selected = useSelene((s) => s.selectedObjectId);
  const select = useSelene((s) => s.selectObject);
  const showLabels = useSelene((s) => s.layers.labels);
  return (
    <group>
      {objects.map((o) => (
        <group key={o.id}>
          <ObjectMarker o={o} selected={o.id === selected} onSelect={() => select(o.id, 'user')} showLabel={showLabels || o.id === selected} />
          {showTrails && o.trail.length > 1 && <Trail points={o.trail} color={CUSTODY_COLOR[o.custody]} />}
        </group>
      ))}
    </group>
  );
}
