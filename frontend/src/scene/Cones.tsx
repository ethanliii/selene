/**
 * Sensor field-of-view cones, space-observer markers and Sun/Moon/Earth exclusion cones.
 *
 * A cone is drawn from an apex along a boresight unit vector with a given half-angle (FOV/2 or the exclusion
 * angle) as a thin wireframe edge plus a very faint fill, so several cones can overlap without turning the scene
 * into mud. FOV cones are drawn only for sensors that are tasked or taking data (idle observers staring at the Moon
 * are not drawn: their position is shown by a small marker instead); an active cone also gets a dashed line of
 * sight to its target. When a tasking event has just fired for a sensor its cone pulses for a few sim-hours.
 * Exclusion cones: Sun amber, Moon grey, Earth blue. `ExclusionCones` derives the Sun direction from
 * lib/ephem.ts and the Moon/Earth directions from geometry; the ground network's lunar-glare cone uses the
 * phase-dependent half-angle of lib/groundVis.ts (3°–15°), updated as the cursor moves.
 */
import { Line } from '@react-three/drei';
import { useFrame } from '@react-three/fiber';
import { useEffect, useMemo, useRef } from 'react';
import * as THREE from 'three';
import type { FrameSensor, Sensors, Vec3 } from '../api/types';
import { beatOfEvent, sensorName } from '../demo/headline';
import { sunDirRot } from '../lib/ephem';
import { lunarGlareHalfAngleDeg } from '../lib/groundVis';
import { cursorMs, useSelene } from '../store/useSelene';
import { COLORS, EARTH_ROT, MOON_ROT } from './constants';
import { Label } from './Label';
import { useScreenScale } from './screenScale';

interface ConeProps {
  apex: Vec3;
  /** Unit vector from the apex along the cone axis (displayed frame). */
  boresight: Vec3;
  /** Half-angle, degrees. */
  halfAngleDeg: number;
  /** Cone length, scene units (1 = L*). */
  length?: number;
  color?: string;
  /** Edge (wireframe) opacity and fill opacity. */
  edgeOpacity?: number;
  fillOpacity?: number;
  visible?: boolean;
  /** Pulse phase source: when set, the edge opacity breathes (tasking just fired). */
  pulse?: boolean;
}

const UP = new THREE.Vector3(0, -1, 0);
/** Cones never reach across the whole system: cap at 1.2 L*. */
const MAX_CONE_LENGTH = 1.2;
/** How long (sim seconds) a sensor's cone pulses after a tasking event fires for it. */
const PULSE_SIM_S = 3 * 3600;

/**
 * Shared cone: a translucent wedge (fill) plus its far rim (EdgesGeometry keeps only the base circle, not the side
 * triangles, so overlapping cones never read as a bundle of laser lines). THREE.ConeGeometry points along −y
 * (apex at +y/2); we rotate so the apex sits at `apex`.
 */
function Cone({ apex, boresight, halfAngleDeg, length = 0.5, color = COLORS.accent, edgeOpacity = 0.35, fillOpacity = 0.06, visible = true, pulse = false }: ConeProps) {
  const len = Math.min(MAX_CONE_LENGTH, length);
  const { quaternion, position } = useMemo(() => {
    const dir = new THREE.Vector3(...boresight).normalize();
    const q = new THREE.Quaternion().setFromUnitVectors(UP, dir);
    const pos = new THREE.Vector3(...apex).add(dir.clone().multiplyScalar(len / 2));
    return { quaternion: q, position: pos };
  }, [apex, boresight, len]);
  const radius = len * Math.tan(Math.min(89, halfAngleDeg) * (Math.PI / 180));
  const { geom, edges } = useMemo(() => {
    const g = new THREE.ConeGeometry(radius, len, 32, 1, true);
    const e = new THREE.EdgesGeometry(g, 40);
    return { geom: g, edges: e };
  }, [radius, len]);
  useEffect(
    () => () => {
      geom.dispose();
      edges.dispose();
    },
    [geom, edges],
  );
  const edgeMat = useRef<THREE.LineBasicMaterial>(null);
  const fillMat = useRef<THREE.MeshBasicMaterial>(null);
  useFrame(() => {
    if (!pulse) {
      if (edgeMat.current && edgeMat.current.opacity !== edgeOpacity) edgeMat.current.opacity = edgeOpacity;
      if (fillMat.current && fillMat.current.opacity !== fillOpacity) fillMat.current.opacity = fillOpacity;
      return;
    }
    const w = 0.5 + 0.5 * Math.sin((performance.now() / 1000) * 2 * Math.PI * 1.1);
    if (edgeMat.current) edgeMat.current.opacity = Math.min(1, edgeOpacity + 0.55 * w);
    if (fillMat.current) fillMat.current.opacity = Math.min(0.4, fillOpacity + 0.2 * w);
  });
  if (!visible) return null;
  return (
    <group position={position} quaternion={quaternion}>
      <mesh geometry={geom} renderOrder={1}>
        <meshBasicMaterial ref={fillMat} color={color} transparent opacity={fillOpacity} side={THREE.DoubleSide} depthWrite={false} />
      </mesh>
      <lineSegments geometry={edges} renderOrder={1}>
        <lineBasicMaterial ref={edgeMat} color={color} transparent opacity={edgeOpacity} depthWrite={false} />
      </lineSegments>
    </group>
  );
}

export interface SensorFOVConeProps extends Omit<ConeProps, 'color' | 'edgeOpacity' | 'fillOpacity'> {
  sensorId: string;
  /** Highlight when the sensor is currently taking data. */
  active?: boolean;
}

export function SensorFOVCone({ active, sensorId: _sensorId, ...rest }: SensorFOVConeProps) {
  return <Cone {...rest} color={active ? COLORS.ok : COLORS.accent} edgeOpacity={active ? 0.45 : 0.3} fillOpacity={active ? 0.14 : 0.07} />;
}

export interface ExclusionConeProps extends Omit<ConeProps, 'color' | 'edgeOpacity' | 'fillOpacity'> {
  /** Which body the exclusion is about; affects color only. */
  body: 'sun' | 'moon' | 'earth';
}

export function ExclusionCone({ body, ...rest }: ExclusionConeProps) {
  const color = body === 'sun' ? COLORS.warn : body === 'moon' ? '#9aa4b2' : COLORS.accent;
  return <Cone {...rest} color={color} edgeOpacity={body === 'sun' ? 0.3 : 0.35} fillOpacity={0.04} />;
}

const sub = (a: Vec3, b: Vec3): Vec3 => [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
const norm = (a: Vec3) => Math.hypot(a[0], a[1], a[2]);
const unit = (a: Vec3): Vec3 => {
  const n = norm(a) || 1;
  return [a[0] / n, a[1] / n, a[2] / n];
};

/** Sensor ids named by a tasking event (`sensor_id`, `sensor_ids`, or the id quoted in the text). */
function taskedSensorIds(e: { kind: string; text: string; data?: Record<string, unknown> }): string[] {
  const out: string[] = [];
  const d = e.data ?? {};
  if (typeof d.sensor_id === 'string') out.push(d.sensor_id);
  if (Array.isArray(d.sensor_ids)) for (const x of d.sensor_ids) if (typeof x === 'string') out.push(x);
  return out;
}

/** Sensors whose tasking event fired within PULSE_SIM_S before the cursor (evaluated at the cursor). */
function pulsingSensorIds(tSec: number, events: { t: number; kind: string; text: string; data?: Record<string, unknown> }[], candidates: string[]): Set<string> {
  const out = new Set<string>();
  for (const e of events) {
    if (e.t > tSec) break;
    if (tSec - e.t > PULSE_SIM_S || beatOfEvent(e) !== 'tasked') continue;
    const ids = taskedSensorIds(e);
    if (ids.length === 0) for (const c of candidates) if (e.text.includes(c)) ids.push(c);
    for (const id of ids) out.add(id);
  }
  return out;
}

/**
 * FOV wedges for frame sensors that are tasked or observing. Sensors pointed at the selected object (or at the
 * protagonist when nothing is selected) get the full wedge + a dashed line of sight; other busy sensors get only a
 * faint dashed line of sight, so a network of six observers does not bury the Moon under cones. Length reaches the
 * target if known. Ground needles get a minimum half-angle so a 1° FOV stays visible at the Earth–Moon scale.
 */
export function SensorFOVCones({ sensors, targets, groundIds }: { sensors: FrameSensor[]; targets: Map<string, Vec3>; groundIds?: Set<string> }) {
  const focus = useSelene((s) => s.selectedObjectId ?? s.scenario?.meta.protagonist_id ?? null);
  const shown = useMemo(() => sensors.filter((s) => s.pos_rot && (s.boresight_rot ?? s.pointing_rot) && (s.active || s.target_id)), [sensors]);
  const shownKey = useMemo(() => shown.map((x) => x.id).join(','), [shown]);
  // Pulse set is re-evaluated on a coarse cursor (one minute). The selector runs on every store update (every
  // animation tick while playing) so it allocates nothing unless the minute, the events or the sensor set changed.
  const pulseCache = useRef<{ tq: number; events: unknown; key: string; out: string }>({ tq: -1, events: null, key: '', out: '' });
  const pulsing = useSelene((s) => {
    if (!s.scenario) return '';
    const tq = Math.floor(s.tSec / 60) * 60;
    const c = pulseCache.current;
    if (c.tq === tq && c.events === s.events && c.key === shownKey) return c.out;
    const out = [...pulsingSensorIds(tq, s.events, shown.map((x) => x.id))].sort().join(',');
    pulseCache.current = { tq, events: s.events, key: shownKey, out };
    return out;
  });
  const pulseSet = useMemo(() => new Set(pulsing ? pulsing.split(',') : []), [pulsing]);
  return (
    <group>
      {shown.map((s) => {
        const bs = (s.boresight_rot ?? s.pointing_rot)!;
        const tgt = s.target_id ? targets.get(s.target_id) : undefined;
        const length = tgt ? Math.max(0.05, norm(sub(tgt, s.pos_rot!)) * 1.02) : Math.min(0.45, norm(sub(MOON_ROT, s.pos_rot!)));
        const ground = s.id.startsWith('GND') || !!groundIds?.has(s.id);
        const half = Math.max(ground ? 0.35 : 0, (s.fov_deg ?? 2) / 2);
        const active = !!s.active;
        const emphasis = !focus || !s.target_id || s.target_id === focus || pulseSet.has(s.id);
        const color = active ? COLORS.ok : COLORS.accent;
        return (
          <group key={s.id}>
            {emphasis && <SensorFOVCone sensorId={s.id} apex={s.pos_rot!} boresight={bs} halfAngleDeg={half} length={length} active={active} pulse={pulseSet.has(s.id)} />}
            {tgt && <Line points={[s.pos_rot!, tgt]} color={color} lineWidth={1} dashed dashSize={0.012} gapSize={0.009} transparent opacity={emphasis ? 0.5 : 0.18} depthWrite={false} />}
          </group>
        );
      })}
    </group>
  );
}

/** Small markers (+ labels) at the space observers' positions, so idle observers are visible without their cones. */
export function SensorMarkers({ sensors, groundIds, showLabels = true }: { sensors: FrameSensor[]; groundIds?: Set<string>; showLabels?: boolean }) {
  const space = sensors.filter((s) => s.pos_rot && !s.id.startsWith('GND') && !groundIds?.has(s.id) && (s as { kind?: string }).kind !== 'ground');
  return (
    <group>
      {space.map((s) => (
        <SensorMarker key={s.id} id={s.id} pos={s.pos_rot!} active={!!s.active} showLabel={showLabels} />
      ))}
    </group>
  );
}

function SensorMarker({ id, pos, active, showLabel }: { id: string; pos: Vec3; active: boolean; showLabel: boolean }) {
  const ref = useRef<THREE.Group>(null);
  useScreenScale(ref, 1);
  const color = active ? COLORS.ok : COLORS.accent;
  return (
    <group position={pos}>
      <group ref={ref}>
        <mesh renderOrder={3}>
          <octahedronGeometry args={[4.5, 0]} />
          <meshBasicMaterial color={color} transparent opacity={0.95} depthTest={false} />
        </mesh>
        <mesh renderOrder={2}>
          <octahedronGeometry args={[8, 0]} />
          <meshBasicMaterial color={color} transparent opacity={0.18} depthWrite={false} depthTest={false} />
        </mesh>
      </group>
      {showLabel && <Label position={[0, 0, 0]} text={sensorName(id)} variant="tag" className={`plain sensor${active ? ' active' : ''}`} id={`sensor-${id}`} priority={active ? 72 : 35} color={color} />}
    </group>
  );
}

/** Ground-network lunar-glare cone from the Earth centre toward the Moon; half-angle follows the Moon's phase. */
function GroundGlareCone() {
  const ref = useRef<THREE.Group>(null);
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
    <group ref={ref} position={[EARTH_ROT[0] + length / 2, 0, 0]} quaternion={q}>
      <mesh>
        <coneGeometry args={[baseRadius, length, 12, 1, true]} />
        <meshBasicMaterial color="#9aa4b2" transparent opacity={0.28} wireframe depthWrite={false} />
      </mesh>
      <mesh>
        <coneGeometry args={[baseRadius, length, 32, 1, true]} />
        <meshBasicMaterial color="#9aa4b2" transparent opacity={0.035} side={THREE.DoubleSide} depthWrite={false} />
      </mesh>
    </group>
  );
}

/**
 * Exclusion cones for space observers (Sun + Moon + Earth, half-angles from /api/sensors, defaults 40°/10°/10°)
 * and the phase-dependent lunar-glare cone for the ground network from the Earth centre. The Sun cones follow the
 * rotating-frame Sun direction each tick.
 */
export function ExclusionCones({ sensors, config, groundIds }: { sensors: FrameSensor[]; config: Sensors | null; groundIds?: Set<string> }) {
  const sunRef = useRef<THREE.Group>(null);
  useFrame(() => {
    // Rotate the Sun-cone group so its +x axis points at the Sun (cones inside are authored along +x).
    if (!sunRef.current) return;
    const d = sunDirRot(cursorMs());
    sunRef.current.rotation.z = Math.atan2(d[1], d[0]);
  });
  const space = sensors.filter((s) => s.pos_rot && !s.id.startsWith('GND') && !groundIds?.has(s.id) && (s as { kind?: string }).kind !== 'ground');
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
