/**
 * Tracked objects + trails (rotating frame, nondimensional).
 * Each object is a constant-pixel-size marker (core dot + soft halo, selection ring) coloured by custody status,
 * with an HTML label carrying a SIMULATED tag (--sim) or a REAL · JPL HORIZONS tag (--accent). Markers are scaled
 * per render tick to a fixed screen size (scene/screenScale.ts) so they never out-size the Moon in close-ups.
 * Positions: live objects sample their backend track (Hermite) every render tick; scenario objects interpolate
 * between the bracketing frames. Trails are a fading line over the last hours of the track (or frames).
 * Labels register with the scene-wide declutter (scene/labels.ts, LabelDeclutter in SceneRoot): the selected object
 * always wins, simulated objects outrank real ones, nearer ones win ties; a label whose box would overlap an
 * already-placed one is hidden (so the five lunar orbiters do not pile up on the Moon at the overview zoom; zoom in
 * or select to reveal). Click (generous hit sphere) selects.
 */
import { Html, Line } from '@react-three/drei';
import { useFrame } from '@react-three/fiber';
import { useCallback, useMemo, useRef } from 'react';
import * as THREE from 'three';
import type { SceneObject } from '../demo/useScenarioFrame';
import { trackEval } from '../lib/tracks';
import { useSelene } from '../store/useSelene';
import { COLORS, MOON_RADIUS, MOON_ROT } from './constants';
import { registerLabel, unregisterLabel } from './labels';
import { useScreenScale, worldPerPixel } from './screenScale';

const wTmp = new THREE.Vector3(), sTmp = new THREE.Vector3();
/** Uniform world scale of an object's parent chain (the synodic group is scaled by d(t)/L* in the inertial view). */
function parentScale(o: THREE.Object3D): number {
  const p = o.parent;
  if (!p) return 1;
  p.getWorldScale(sTmp);
  return sTmp.x || 1;
}

export type { SceneObject } from '../demo/useScenarioFrame';

export const CUSTODY_COLOR: Record<SceneObject['custody'], string> = {
  held: COLORS.ok,
  degraded: COLORS.warn,
  lost: COLORS.alert,
  /** No OD has evaluated the object (idle catalog view): neutral, not the green of a measured 'held'. */
  unknown: '#a9bbd1',
};

/** Marker radii in screen pixels. */
const PX = { core: 3.5, coreSel: 4.5, halo: 8, haloSel: 10, ringIn: 12, ringOut: 13.5, hit: 14 };
/**
 * Near-Moon declutter: when the Moon's disc is small on screen (overview zoom) and a marker sits within a few
 * pixels of it (lunar orbiters), the halo is dropped and the core shrunk so the five orbiters do not bury the Moon
 * under a cyan blob. Both thresholds are in screen pixels; the selected object keeps its full marker.
 */
const NEAR_MOON = { moonPxMax: 12, distPx: 16, coreSmall: 2.2 };

interface Props {
  objects: SceneObject[];
  showTrails?: boolean;
}

function ObjectMarker({ o, selected, onSelect, showLabel }: { o: SceneObject; selected: boolean; onSelect: () => void; showLabel: boolean }) {
  const ref = useRef<THREE.Group>(null);
  const labelNode = useRef<HTMLDivElement | null>(null);
  const markerRef = useRef<THREE.Group>(null);
  const color = CUSTODY_COLOR[o.custody];
  const coreRef = useRef<THREE.Mesh>(null);
  const haloRef = useRef<THREE.Mesh>(null);
  const tmp = useMemo(() => [0, 0, 0] as [number, number, number], []);
  useFrame(({ camera, size }) => {
    if (!ref.current) return;
    const t = useSelene.getState().tSec;
    if (o.track) {
      trackEval(o.track, t, tmp);
      ref.current.position.set(tmp[0], tmp[1], tmp[2]);
    } else {
      const w = Math.min(1, Math.max(0, (t - o.tFrame) / (o.tNext - o.tFrame)));
      ref.current.position.set(o.pos[0] + (o.posNext[0] - o.pos[0]) * w, o.pos[1] + (o.posNext[1] - o.pos[1]) * w, o.pos[2] + (o.posNext[2] - o.pos[2]) * w);
    }
    // Near-Moon declutter (see NEAR_MOON). Distances in the parent's (synodic) units, converted to pixels at the
    // marker's camera distance; the Moon radius is MOON_RADIUS · bodyScale ≈ MOON_RADIUS at any epoch.
    const p = ref.current.position;
    const dMoon = Math.hypot(p.x - MOON_ROT[0], p.y - MOON_ROT[1], p.z - MOON_ROT[2]);
    let crowd = false;
    if (!selected && dMoon < 0.08) {
      ref.current.getWorldPosition(wTmp);
      const wpp = worldPerPixel(camera, camera.position.distanceTo(wTmp), size.height) / parentScale(ref.current);
      crowd = MOON_RADIUS / wpp < NEAR_MOON.moonPxMax && dMoon / wpp < NEAR_MOON.distPx;
    }
    if (haloRef.current) haloRef.current.visible = !crowd;
    if (coreRef.current) {
      const s = crowd ? NEAR_MOON.coreSmall / (selected ? PX.coreSel : PX.core) : 1;
      coreRef.current.scale.setScalar(s);
    }
  });
  // Unit-radius geometry scaled to 1 px; children sizes below are then in pixels.
  useScreenScale(markerRef, 1);
  // Callback ref: drei's <Html> mounts its children in a separate root after this component's effects run.
  const setLabel = useCallback(
    (el: HTMLDivElement | null) => {
      if (el && ref.current) {
        labelNode.current = el;
        registerLabel({ id: o.id, priority: selected ? 100 : o.simulated ? 70 : 60, el, anchor: ref.current, dx: 10, dy: -10 });
      } else if (!el && labelNode.current) {
        unregisterLabel(o.id, labelNode.current);
        labelNode.current = null;
      }
    },
    [o.id, o.simulated, selected],
  );
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
        <mesh renderOrder={3} ref={coreRef}>
          <sphereGeometry args={[selected ? PX.coreSel : PX.core, 12, 8]} />
          <meshBasicMaterial color={color} depthTest={false} />
        </mesh>
        <mesh renderOrder={2} ref={haloRef}>
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
          <div ref={setLabel} className={`obj-label${selected ? ' selected' : ''}${o.simulated ? '' : ' real'}`} style={{ borderColor: color }} data-id={o.id}>
            <span className="id">{o.id}</span>
            <span className={`tag ${o.simulated ? 'sim' : 'accent'}`}>{o.simulated ? 'SIMULATED' : 'REAL · JPL HORIZONS'}</span>
            {(o.custody === 'lost' || o.custody === 'degraded') && <span className={`tag ${o.custody === 'lost' ? 'alert' : 'warn'}`}>{o.custody.toUpperCase()}</span>}
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
      const w = n > 1 ? Math.pow(i / (n - 1), 1.4) : 1;
      return bg.clone().lerp(c, 0.1 + 0.9 * w);
    }).map((cc) => [cc.r, cc.g, cc.b] as [number, number, number]);
  }, [points, color]);
  return <Line points={points} vertexColors={colors} lineWidth={1.4} transparent opacity={0.95} depthWrite={false} />;
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
