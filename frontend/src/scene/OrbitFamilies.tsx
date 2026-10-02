/**
 * Periodic-orbit family polylines (rotating frame, nondimensional).
 * Minimal working rendering: one drei <Line> per member from `samples_rot`.
 * TODO(later agent, M10): color by family, highlight hovered member, show period/Jacobi tooltip,
 *   LOD/decimation for large families, and transform to inertial frame when store.frame === 'inertial'
 *   (currently the whole rotating group is rotated by FrameGroup, which is a placeholder).
 */
import { Line } from '@react-three/drei';
import type { OrbitFamilies as OrbitFamiliesData } from '../api/types';
import { COLORS } from './constants';

const FAMILY_COLORS = [COLORS.accent2, COLORS.accent, COLORS.ok, COLORS.warn, '#e07a5f', '#81b29a', '#f2cc8f'];

interface Props {
  data: OrbitFamiliesData | null;
  visible?: boolean;
  /** Draw at most this many members per family (evenly strided) to keep the scene light. */
  maxPerFamily?: number;
  opacity?: number;
}

export function OrbitFamilies({ data, visible = true, maxPerFamily = 12, opacity = 0.45 }: Props) {
  if (!visible || !data) return null;
  return (
    <group>
      {data.families.map((fam, fi) => {
        const stride = Math.max(1, Math.ceil(fam.members.length / maxPerFamily));
        const color = FAMILY_COLORS[fi % FAMILY_COLORS.length];
        return fam.members
          .filter((_, i) => i % stride === 0)
          .map((m) =>
            m.samples_rot.length > 1 ? (
              <Line key={m.id} points={m.samples_rot} color={color} lineWidth={1} transparent opacity={opacity} depthWrite={false} />
            ) : null,
          );
      })}
    </group>
  );
}
