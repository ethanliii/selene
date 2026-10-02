/** Small HTML label anchored to a 3D position (drei Html). */
import { Html } from '@react-three/drei';
import type { Vec3 } from '../api/types';

interface Props {
  position: Vec3;
  text: string;
  accent?: boolean;
  color?: string;
}

export function Label({ position, text, accent, color }: Props) {
  return (
    <Html position={position} zIndexRange={[10, 0]} style={{ pointerEvents: 'none' }}>
      <div className={`scene-label${accent ? ' accent' : ''}`} style={color ? { color } : undefined}>
        {text}
      </div>
    </Html>
  );
}
