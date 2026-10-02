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
  /** Extra class on the label div ('plain' = no chip border, e.g. sensor markers, line-of-sight hints). */
  className?: string;
  /** Screen offset of the label's top-left corner from the anchor, px (overrides the variant default). */
  offset?: { dx: number; dy: number };
}

const OFFSET: Record<NonNullable<Props['variant']>, { dx: number; dy: number }> = {
  tag: { dx: 6, dy: -6 },
  body: { dx: 0, dy: 10 },
  lpoint: { dx: 8, dy: -8 },
};

export function Label({ position, text, accent, color, variant = 'tag', id, priority = 50, className, offset }: Props) {
  const anchor = useRef<THREE.Group>(null);
  const node = useRef<HTMLDivElement | null>(null);
  // drei's <Html> renders its children into a separate React root after the parent commits, so a plain useEffect
  // would run before the element exists; a callback ref fires exactly when the div mounts/unmounts.
  const setEl = useCallback(
    (el: HTMLDivElement | null) => {
      if (!id) return;
      if (el && anchor.current) {
        node.current = el;
        const off = offset ?? OFFSET[variant];
        registerLabel({ id, priority, el, anchor: anchor.current, dx: variant === 'body' && !offset ? -el.offsetWidth / 2 : off.dx, dy: off.dy });
      } else if (!el && node.current) {
        unregisterLabel(id, node.current);
        node.current = null;
      }
    },
    [id, priority, variant, offset?.dx, offset?.dy],
  );
  return (
    <group ref={anchor} position={position}>
      <Html zIndexRange={[10, 0]} style={{ pointerEvents: 'none' }}>
        <div ref={setEl} className={`scene-label ${variant}${accent ? ' accent' : ''}${className ? ` ${className}` : ''}`} style={offset ? { color, ['--dx' as string]: `${offset.dx}px`, ['--dy' as string]: `${offset.dy}px` } : color ? { color } : undefined}>
          {text}
        </div>
      </Html>
    </group>
  );
}
