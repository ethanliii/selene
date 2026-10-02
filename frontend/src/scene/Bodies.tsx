/** Earth, Moon, Sun light and Lagrange markers. Positions are rotating-frame (nondimensional). */
import { useMemo } from 'react';
import * as THREE from 'three';
import type { Vec3 } from '../api/types';
import { COLORS, EARTH_RADIUS, EARTH_ROT, LAGRANGE_ROT, MOON_RADIUS, MOON_ROT } from './constants';
import { Label } from './Label';

export function Earth() {
  return (
    <group position={EARTH_ROT}>
      <mesh>
        <sphereGeometry args={[EARTH_RADIUS, 48, 32]} />
        <meshStandardMaterial color={COLORS.earth} emissive={COLORS.earthEmissive} emissiveIntensity={0.7} roughness={0.9} metalness={0.05} />
      </mesh>
      {/* Faint atmosphere ring: a slightly larger, back-side-lit transparent shell. */}
      <mesh>
        <sphereGeometry args={[EARTH_RADIUS * 1.06, 48, 32]} />
        <meshBasicMaterial color={COLORS.atmosphere} transparent opacity={0.22} side={THREE.BackSide} depthWrite={false} />
      </mesh>
      <Label position={[0, 0, EARTH_RADIUS * 1.6]} text="EARTH" />
    </group>
  );
}

export function Moon() {
  return (
    <group position={MOON_ROT}>
      <mesh>
        <sphereGeometry args={[MOON_RADIUS, 32, 24]} />
        <meshStandardMaterial color={COLORS.moon} emissive="#3a3d44" emissiveIntensity={0.6} roughness={1} metalness={0} />
      </mesh>
      <Label position={[0, 0, MOON_RADIUS * 2.2]} text="MOON" />
    </group>
  );
}

interface SunProps {
  /** Unit vector FROM the barycenter TOWARD the Sun, in the displayed frame. Default +x. */
  direction?: Vec3;
  intensity?: number;
}

/**
 * Sun as a distant directional light. In the rotating frame the true Sun direction rotates at
 * −(synodic rate); a later agent should drive `direction` from /api/ephemeris/bodies.
 */
export function SunLight({ direction = [1, 0, 0], intensity = 2.2 }: SunProps) {
  const pos = useMemo(() => new THREE.Vector3(...direction).normalize().multiplyScalar(20), [direction]);
  return (
    <>
      <directionalLight position={pos} intensity={intensity} color="#fff4e0" />
      <ambientLight intensity={0.3} />
    </>
  );
}

export function LagrangePoints({ visible = true }: { visible?: boolean }) {
  if (!visible) return null;
  return (
    <group>
      {LAGRANGE_ROT.map(({ name, pos }) => (
        <group key={name} position={pos}>
          {/* Diamond marker: small octahedron. */}
          <mesh>
            <octahedronGeometry args={[0.012, 0]} />
            <meshBasicMaterial color={COLORS.accent} wireframe />
          </mesh>
          <Label position={[0, 0, 0]} text={name} accent />
        </group>
      ))}
    </group>
  );
}
