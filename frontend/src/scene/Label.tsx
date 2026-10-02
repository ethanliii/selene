/**
 * Small HTML label anchored to a 3D position (drei Html). Registers with the declutter registry (scene/labels.ts)
 * when given an `id`, so overlapping labels are hidden by priority instead of piling up.
 */
import { Html } from '@react-three/drei';
import { useCallback, useRef } from 'react';
import type * as THREE from 'three';
import type { Vec3 } from '../api/types';
import { registerLabel, unregisterLabel } from './labels';

interface Props {
  position: Vec3;
  text: string;
  accent?: boolean;
  color?: string;
  /** 'body' = Earth/Moon style (centred below the anchor); 'lpoint' = libration point; default = small tag. */
  variant?: 'tag' | 'body' | 'lpoint';
  /** Declutter id (omit to always show). */
  id?: string;
  priority?: number;
}

const OFFSET: Record<NonNullable<Props['variant']>, { dx: number; dy: number }> = {
  tag: { dx: 6, dy: -6 },
  body: { dx: 0, dy: 10 },
  lpoint: { dx: 8, dy: -8 },
};

export function Label({ position, text, accent, color, variant = 'tag', id, priority = 50 }: Props) {
  const anchor = useRef<THREE.Group>(null);
  const node = useRef<HTMLDivElement | null>(null);
  // drei's <Html> renders its children into a separate React root after the parent commits, so a plain useEffect
  // would run before the element exists; a callback ref fires exactly when the div mounts/unmounts.
  const setEl = useCallback(
    (el: HTMLDivElement | null) => {
      if (!id) return;
      if (el && anchor.current) {
        node.current = el;
        const off = OFFSET[variant];
        registerLabel({ id, priority, el, anchor: anchor.current, dx: variant === 'body' ? -el.offsetWidth / 2 : off.dx, dy: off.dy });
      } else if (!el && node.current) {
        unregisterLabel(id, node.current);
        node.current = null;
      }
    },
    [id, priority, variant],
  );
  return (
    <group ref={anchor} position={position}>
      <Html zIndexRange={[10, 0]} style={{ pointerEvents: 'none' }}>
        <div ref={setEl} className={`scene-label ${variant}${accent ? ' accent' : ''}`} style={color ? { color } : undefined}>
          {text}
        </div>
      </Html>
    </group>
  );
}
