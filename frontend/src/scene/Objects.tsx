/**
 * Tracked objects + trails (rotating frame, nondimensional).
 * Minimal working rendering: a small sphere per object colored by custody status, optional trail polyline,
 * label with id and a SIM marker, click to select.
 * TODO(later agent, M10): feed `objects` from the demo scenario frames interpolated at store.tSec;
 *   accumulate trails from past frames; covariance ellipsoid (from UKF covs) around the selected object;
 *   Horizons objects (kind='horizons') styled differently from SIMULATED ones.
 */
import { Line } from '@react-three/drei';
import type { Vec3 } from '../api/types';
import { useSelene } from '../store/useSelene';
import { COLORS } from './constants';
import { Label } from './Label';

export interface SceneObject {
  id: string;
  name?: string;
  simulated: boolean;
  pos: Vec3;
  trail?: Vec3[];
  custody: 'held' | 'degraded' | 'lost';
}

export const CUSTODY_COLOR: Record<SceneObject['custody'], string> = {
  held: COLORS.ok,
  degraded: COLORS.warn,
  lost: COLORS.alert,
};

interface Props {
  objects: SceneObject[];
  showTrails?: boolean;
}

export function Objects({ objects, showTrails = true }: Props) {
  const selected = useSelene((s) => s.selectedObjectId);
  const select = useSelene((s) => s.selectObject);
  return (
    <group>
      {objects.map((o) => {
        const color = CUSTODY_COLOR[o.custody];
        const isSel = o.id === selected;
        return (
          <group key={o.id}>
            <mesh
              position={o.pos}
              onClick={(e) => {
                e.stopPropagation();
                select(o.id);
              }}
            >
              <sphereGeometry args={[isSel ? 0.009 : 0.006, 12, 8]} />
              <meshBasicMaterial color={color} />
            </mesh>
            {isSel && (
              <mesh position={o.pos}>
                <ringGeometry args={[0.014, 0.016, 32]} />
                <meshBasicMaterial color={color} side={2} transparent opacity={0.8} />
              </mesh>
            )}
            {showTrails && o.trail && o.trail.length > 1 && (
              <Line points={o.trail} color={color} lineWidth={1} transparent opacity={0.6} depthWrite={false} />
            )}
            <Label position={o.pos} text={`${o.id}${o.simulated ? ' [SIM]' : ''}`} color={o.simulated ? COLORS.sim : COLORS.text} />
          </group>
        );
      })}
    </group>
  );
}
