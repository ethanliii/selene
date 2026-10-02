/**
 * Periodic-orbit family polylines (rotating frame, nondimensional): one line per member, colour per family
 * (constants.familyColor, shared with the TopBar legend). A highlighted family (store.highlightFamily) is drawn
 * bright; the others are dimmed. Members are strided to `maxPerFamily` to keep the draw count small.
 */
import { Line } from '@react-three/drei';
import { useMemo } from 'react';
import type { OrbitFamilies as OrbitFamiliesData } from '../api/types';
import { useSelene } from '../store/useSelene';
import { familyColor } from './constants';

interface Props {
  data: OrbitFamiliesData | null;
  visible?: boolean;
  /** Draw at most this many members per family (evenly strided) to keep the scene light. */
  maxPerFamily?: number;
}

export function OrbitFamilies({ data, visible = true, maxPerFamily = 12 }: Props) {
  const highlight = useSelene((s) => s.highlightFamily);
  const lines = useMemo(() => {
    if (!data) return [];
    return data.families.flatMap((fam, fi) => {
      const stride = Math.max(1, Math.ceil(fam.members.length / maxPerFamily));
      const color = familyColor(fam.name, fi);
      return fam.members
        .filter((_, i) => i % stride === 0 || i === fam.members.length - 1)
        .filter((m) => m.samples_rot.length > 1)
        .map((m) => ({ key: m.id, family: fam.name, color, points: m.samples_rot }));
    });
  }, [data, maxPerFamily]);
  if (!visible || lines.length === 0) return null;
  return (
    <group>
      {lines.map((l) => {
        const hl = highlight === l.family;
        const dim = highlight !== null && !hl;
        return <Line key={l.key} points={l.points} color={l.color} lineWidth={hl ? 1.8 : 1} transparent opacity={hl ? 0.95 : dim ? 0.12 : 0.45} depthWrite={false} />;
      })}
    </group>
  );
}
