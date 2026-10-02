/**
 * Sensor field-of-view cones and Sun/Moon/Earth exclusion cones.
 * A cone is drawn from an apex along a boresight unit vector with a given half-angle (FOV/2 or the exclusion
 * angle). FOV cones are translucent (brighter when the sensor is actively observing); exclusion cones are
 * low-opacity solids in the body's colour (Sun: --warn, Moon: muted grey, Earth: --accent-2).
 * `ExclusionCones` derives the Sun direction from lib/ephem.ts and the Moon/Earth directions from geometry; the
 * ground network's lunar-glare cone uses the phase-dependent half-angle of lib/groundVis.ts (3°–15°), updated as
 * the cursor moves, so what is drawn is the rule the mock scenario and the backend apply.
 */
import { useFrame } from '@react-three/fiber';
import { useMemo, useRef } from 'react';
import * as THREE from 'three';
import type { FrameSensor, Sensors, Vec3 } from '../api/types';
import { sunDirRot } from '../lib/ephem';
import { lunarGlareHalfAngleDeg } from '../lib/groundVis';
import { cursorMs } from '../store/useSelene';
import { COLORS, EARTH_ROT, MOON_ROT } from './constants';

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

const UP = new THREE.Vector3(0, -1, 0);

/** Shared cone mesh: THREE.ConeGeometry points along −y (apex at +y/2); we rotate so the apex sits at `apex`. */
function Cone({ apex, boresight, halfAngleDeg, length = 0.5, color = COLORS.accent, opacity = 0.12, wireframe = false, visible = true }: ConeProps) {
  const { quaternion, position } = useMemo(() => {
    const dir = new THREE.Vector3(...boresight).normalize();
    const q = new THREE.Quaternion().setFromUnitVectors(UP, dir);
    const pos = new THREE.Vector3(...apex).add(dir.clone().multiplyScalar(length / 2));
    return { quaternion: q, position: pos };
  }, [apex, boresight, length]);
  const radius = length * Math.tan(Math.min(89, halfAngleDeg) * (Math.PI / 180));
  if (!visible) return null;
  return (
    <mesh position={position} quaternion={quaternion}>
      <coneGeometry args={[radius, length, 24, 1, true]} />
      <meshBasicMaterial color={color} transparent opacity={opacity} side={THREE.DoubleSide} depthWrite={false} wireframe={wireframe} />
    </mesh>
  );
}

export interface SensorFOVConeProps extends Omit<ConeProps, 'color' | 'wireframe'> {
  sensorId: string;
  /** Highlight when the sensor is currently taking data. */
  active?: boolean;
}

export function SensorFOVCone({ active, sensorId: _sensorId, ...rest }: SensorFOVConeProps) {
  return <Cone {...rest} color={active ? COLORS.ok : COLORS.accent} opacity={active ? 0.26 : 0.1} />;
}

export interface ExclusionConeProps extends Omit<ConeProps, 'color' | 'wireframe'> {
  /** Which body the exclusion is about; affects color only. */
  body: 'sun' | 'moon' | 'earth';
}

export function ExclusionCone({ body, ...rest }: ExclusionConeProps) {
  const color = body === 'sun' ? COLORS.warn : body === 'moon' ? COLORS.muted : COLORS.accent2;
  return <Cone {...rest} color={color} opacity={body === 'sun' ? 0.07 : 0.12} />;
}

const sub = (a: Vec3, b: Vec3): Vec3 => [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
const norm = (a: Vec3) => Math.hypot(a[0], a[1], a[2]);
const unit = (a: Vec3): Vec3 => {
  const n = norm(a) || 1;
  return [a[0] / n, a[1] / n, a[2] / n];
};

/**
 * FOV cones for every frame sensor that has a position and a boresight (space observers always; ground sites
 * while they are observing, since the scenario only supplies a pointing then). Length reaches the target if known.
 * Ground needles get a minimum half-angle so a 1° FOV stays visible at the Earth–Moon scale.
 */
export function SensorFOVCones({ sensors, targets }: { sensors: FrameSensor[]; targets: Map<string, Vec3> }) {
  return (
    <group>
      {sensors
        .filter((s) => s.pos_rot && (s.boresight_rot ?? s.pointing_rot))
        .map((s) => {
          const bs = (s.boresight_rot ?? s.pointing_rot)!;
          const tgt = s.target_id ? targets.get(s.target_id) : undefined;
          const length = tgt ? Math.max(0.05, norm(sub(tgt, s.pos_rot!)) * 1.05) : 0.45;
          const ground = s.id.startsWith('GND');
          const half = Math.max(ground ? 0.35 : 0, (s.fov_deg ?? 2) / 2);
          return <SensorFOVCone key={s.id} sensorId={s.id} apex={s.pos_rot!} boresight={bs} halfAngleDeg={half} length={length} active={!!s.active} />;
        })}
    </group>
  );
}

/** Ground-network lunar-glare cone from the Earth centre toward the Moon; half-angle follows the Moon's phase. */
function GroundGlareCone() {
  const ref = useRef<THREE.Mesh>(null);
  const BASE_DEG = 10;
  const length = 1.0 - 0.012;
  const baseRadius = length * Math.tan(BASE_DEG * (Math.PI / 180));
  useFrame(() => {
    if (!ref.current) return;
    const k = Math.tan(lunarGlareHalfAngleDeg(cursorMs()) * (Math.PI / 180)) / Math.tan(BASE_DEG * (Math.PI / 180));
    ref.current.scale.set(k, 1, k);
  });
  // Cone axis is −y in geometry space; rotate so it points along +x from the Earth.
  const q = useMemo(() => new THREE.Quaternion().setFromUnitVectors(UP, new THREE.Vector3(1, 0, 0)), []);
  return (
    <mesh ref={ref} position={[EARTH_ROT[0] + length / 2, 0, 0]} quaternion={q}>
      <coneGeometry args={[baseRadius, length, 32, 1, true]} />
      <meshBasicMaterial color={COLORS.muted} transparent opacity={0.12} side={THREE.DoubleSide} depthWrite={false} />
    </mesh>
  );
}

/**
 * Exclusion cones for space observers (Sun + Moon + Earth, half-angles from /api/sensors, defaults 40°/10°/10°)
 * and the phase-dependent lunar-glare cone for the ground network from the Earth centre. The Sun cones follow the
 * rotating-frame Sun direction each tick.
 */
export function ExclusionCones({ sensors, config }: { sensors: FrameSensor[]; config: Sensors | null }) {
  const sunRef = useRef<THREE.Group>(null);
  useFrame(() => {
    // Rotate the Sun-cone group so its +x axis points at the Sun (cones inside are authored along +x).
    if (!sunRef.current) return;
    const d = sunDirRot(cursorMs());
    sunRef.current.rotation.z = Math.atan2(d[1], d[0]);
  });
  const space = sensors.filter((s) => s.pos_rot && !s.id.startsWith('GND'));
  const cfg = (id: string) => config?.space.find((x) => x.id === id);
  return (
    <group>
      {space.map((s) => {
        const c = cfg(s.id);
        const p = s.pos_rot!;
        return (
          <group key={s.id}>
            <ExclusionCone body="moon" apex={p} boresight={unit(sub(MOON_ROT, p))} halfAngleDeg={c?.moon_exclusion_deg ?? 10} length={Math.min(0.35, norm(sub(MOON_ROT, p)))} />
            <ExclusionCone body="earth" apex={p} boresight={unit(sub(EARTH_ROT, p))} halfAngleDeg={c?.earth_exclusion_deg ?? 10} length={Math.min(0.35, norm(sub(EARTH_ROT, p)))} />
          </group>
        );
      })}
      {/* Sun cones: authored along +x inside a group that is rotated toward the Sun every tick. */}
      <group ref={sunRef}>
        {space.map((s) => {
          const c = cfg(s.id);
          return <SunCone key={s.id} pos={s.pos_rot!} halfAngleDeg={c?.sun_exclusion_deg ?? 40} />;
        })}
      </group>
      <GroundGlareCone />
    </group>
  );
}

/** A Sun exclusion cone whose apex tracks a rotating-frame position while the parent group spins toward the Sun. */
function SunCone({ pos, halfAngleDeg }: { pos: Vec3; halfAngleDeg: number }) {
  const ref = useRef<THREE.Group>(null);
  useFrame(() => {
    if (!ref.current) return;
    const d = sunDirRot(cursorMs());
    const a = -Math.atan2(d[1], d[0]);
    ref.current.position.set(pos[0] * Math.cos(a) - pos[1] * Math.sin(a), pos[0] * Math.sin(a) + pos[1] * Math.cos(a), pos[2]);
  });
  return (
    <group ref={ref}>
      <ExclusionCone body="sun" apex={[0, 0, 0]} boresight={[1, 0, 0]} halfAngleDeg={halfAngleDeg} length={0.3} />
    </group>
  );
}
