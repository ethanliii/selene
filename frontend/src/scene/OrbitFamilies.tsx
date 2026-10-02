/**
 * Periodic-orbit family polylines (rotating frame, nondimensional): one line per member, colour per family
 * (constants.familyColor, shared with the TopBar legend). Families are thinned to a legible subset (evenly strided
 * by member index, `maxPerFamily`); members tagged NRHO are always kept and the 9:2 synodic-resonant NRHO is drawn
 * bright with its own label (at apolune). The legend can hide families (store.hiddenFamilies) and highlight one
 * (store.highlightFamily: it is drawn bright, the others dimmed).
 */
import { Line } from '@react-three/drei';
import { useMemo } from 'react';
import type { OrbitFamilies as OrbitFamiliesData, OrbitMember, Vec3 } from '../api/types';
import { useSelene } from '../store/useSelene';
import { familyColor, MOON_ROT, NRHO_COLOR } from './constants';
import { Label } from './Label';

interface Props {
  data: OrbitFamiliesData | null;
  visible?: boolean;
  /** Draw at most this many members per family (evenly strided) to keep the scene light. */
  maxPerFamily?: number;
}

export function isNrho92(m: OrbitMember): boolean {
  return (m.tags ?? []).some((t) => /nrho_?9[:_]2/i.test(t)) || /nrho_9_2/i.test(m.id);
}
export function isNrho(m: OrbitMember): boolean {
  return (m.tags ?? []).some((t) => /^nrho/i.test(t)) || /nrho/i.test(m.id);
}

export const DEFAULT_MAX_PER_FAMILY = 5;

/**
 * Indices of the members drawn for a family: evenly strided to `maxPerFamily` (3 for the huge resonant families),
 * NRHO-tagged members always kept, plus the last member. Shared with the legend so its counts describe what is drawn.
 */
export function thinnedMembers(fam: { name: string; members: OrbitMember[] }, maxPerFamily = DEFAULT_MAX_PER_FAMILY): Set<number> {
  const n = fam.members.length;
  const resonant = /reson/i.test(fam.name);
  const keep = new Set<number>();
  fam.members.forEach((m, i) => {
    if (isNrho(m)) keep.add(i);
  });
  const budget = Math.max(1, (resonant ? Math.min(3, maxPerFamily) : maxPerFamily) - keep.size);
  const stride = Math.max(1, Math.ceil(n / budget));
  for (let i = 0; i < n; i += stride) keep.add(i);
  if (n > 0) keep.add(n - 1);
  return keep;
}

/** Point of the sampled orbit farthest from the Moon (apolune) — where the NRHO label hangs. */
function apolune(samples: Vec3[]): Vec3 {
  let best = samples[0], bd = -1;
  for (const p of samples) {
    const d = Math.hypot(p[0] - MOON_ROT[0], p[1] - MOON_ROT[1], p[2] - MOON_ROT[2]);
    if (d > bd) {
      bd = d;
      best = p;
    }
  }
  return best;
}

export function OrbitFamilies({ data, visible = true, maxPerFamily = DEFAULT_MAX_PER_FAMILY }: Props) {
  const highlight = useSelene((s) => s.highlightFamily);
  const hidden = useSelene((s) => s.hiddenFamilies);
  const showLabels = useSelene((s) => s.layers.labels);
  const lines = useMemo(() => {
    if (!data) return [];
    return data.families.flatMap((fam, fi) => {
      const color = familyColor(fam.name, fi);
      const resonant = /reson/i.test(fam.name);
      // Resonant families are huge (they tour the whole Earth–Moon system): draw fewer, fainter members.
      const keep = thinnedMembers(fam, maxPerFamily);
      return fam.members
        .filter((m, i) => keep.has(i) && m.samples_rot.length > 1)
        .map((m) => ({ key: m.id, family: fam.name, color, points: m.samples_rot, nrho92: isNrho92(m), nrho: isNrho(m), resonant, periodDays: m.period_days ?? m.period * 4.3425 }));
    });
  }, [data, maxPerFamily]);
  if (!visible || lines.length === 0) return null;
  return (
    <group>
      {lines
        .filter((l) => !hidden.includes(l.family))
        .map((l) => {
          const hl = highlight === l.family;
          const dim = highlight !== null && !hl;
          const color = l.nrho92 ? NRHO_COLOR : l.color;
          const width = l.nrho92 ? 2.4 : hl ? 1.8 : l.nrho ? 1.3 : 1;
          // Other NRHO-tagged members stay quieter than the highlighted 9:2 so the Moon is not buried under them.
          const opacity = dim ? 0.1 : l.nrho92 ? 1 : hl ? 0.95 : l.nrho ? 0.42 : l.resonant ? 0.28 : 0.45;
          return (
            <group key={l.key}>
              <Line points={l.points} color={color} lineWidth={width} transparent opacity={opacity} depthWrite={false} />
              {l.nrho92 && showLabels && !dim && <Label position={apolune(l.points)} text={`9:2 NRHO · ${l.periodDays.toFixed(2)} d`} color={NRHO_COLOR} id="nrho-9-2" priority={85} />}
            </group>
          );
        })}
    </group>
  );
}
