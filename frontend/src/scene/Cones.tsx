/**
 * Sensor field-of-view cones and Sun/Moon/Earth exclusion cones.
 * Both are rendered as an open cone from an apex along a boresight unit vector with a given half-angle.
 * Minimal working rendering only.
 * TODO(later agent, M10):
 *   - SensorFOVCone: drive apex/boresight/fov from demo frame `sensors[]`; pulse when an observation is taken.
 *   - ExclusionCone: compute boresight toward Sun/Moon/Earth from the ephemeris for each sensor;
 *     use true exclusion half-angles from /api/sensors; render as hatched/wireframe red.
 */
import { useMemo } from 'react';
import * as THREE from 'three';
import type { Vec3 } from '../api/types';
import { COLORS } from './constants';

interface ConeProps {
  apex: Vec3;
  /** Unit vector from the apex along the cone axis (displayed frame). */
  boresight: Vec3;
  /** Half-angle, degrees. */
  halfAngleDeg: number;
  /** Cone length, scene units (1 = L*). */
  length?: number;
  color?: string;
  opacity?: number;
  wireframe?: boolean;
  visible?: boolean;
}

/** Shared cone mesh: THREE.ConeGeometry points along +y; we rotate it to the boresight. */
function Cone({ apex, boresight, halfAngleDeg, length = 0.5, color = COLORS.accent, opacity = 0.12, wireframe = false, visible = true }: ConeProps) {
  const { quaternion, position } = useMemo(() => {
    const dir = new THREE.Vector3(...boresight).normalize();
    const q = new THREE.Quaternion().setFromUnitVectors(new THREE.Vector3(0, -1, 0), dir);
    // ConeGeometry is centered at half-height; shift so the apex sits at `apex`.
    const pos = new THREE.Vector3(...apex).add(dir.clone().multiplyScalar(length / 2));
    return { quaternion: q, position: pos };
  }, [apex, boresight, length]);
  const radius = length * Math.tan((halfAngleDeg * Math.PI) / 180);
  if (!visible) return null;
  return (
    <mesh position={position} quaternion={quaternion}>
      <coneGeometry args={[radius, length, 32, 1, true]} />
      <meshBasicMaterial color={color} transparent opacity={opacity} side={THREE.DoubleSide} depthWrite={false} wireframe={wireframe} />
    </mesh>
  );
}

export interface SensorFOVConeProps extends Omit<ConeProps, 'color' | 'wireframe'> {
  sensorId: string;
  /** Highlight when the sensor is currently tasked on an object. */
  active?: boolean;
}

export function SensorFOVCone({ active, sensorId: _sensorId, ...rest }: SensorFOVConeProps) {
  return <Cone {...rest} color={active ? COLORS.ok : COLORS.accent} opacity={active ? 0.22 : 0.1} />;
}

export interface ExclusionConeProps extends Omit<ConeProps, 'color' | 'wireframe'> {
  /** Which body the exclusion is about; affects color only. */
  body: 'sun' | 'moon' | 'earth';
}

export function ExclusionCone({ body, ...rest }: ExclusionConeProps) {
  const color = body === 'sun' ? COLORS.warn : body === 'moon' ? COLORS.muted : COLORS.accent2;
  return <Cone {...rest} color={color} opacity={0.18} wireframe />;
}
