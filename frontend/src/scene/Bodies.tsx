/**
 * Earth, Moon, Sun light and Lagrange markers. Positions are rotating-frame (nondimensional).
 * Earth and Moon use a small day/night shader driven by the Sun direction in the rotating frame
 * (lib/ephem.ts: ephemeris when loaded, mean solar longitude otherwise), so the terminator and lunar phase
 * move with the time cursor. The Earth is tilted by the J2000 obliquity relative to the Earth–Moon (ecliptic)
 * plane, consistent with the ground-site placement in GroundSites.tsx and the mock scenario's visibility model.
 * The Sun light sits inside the frame group and follows the same direction.
 */
import { useFrame } from '@react-three/fiber';
import { useMemo, useRef } from 'react';
import * as THREE from 'three';
import { moonDistanceKm, sunDirRot } from '../lib/ephem';
import { cursorMs, useSelene } from '../store/useSelene';
import { COLORS, EARTH_RADIUS, EARTH_ROT, L_STAR_KM, LAGRANGE_ROT, MOON_RADIUS, MOON_ROT } from './constants';
import { earthQuaternion } from './earthOrientation';
import { Label } from './Label';
import { useScreenScale, worldPerPixel } from './screenScale';

const DAYNIGHT_VERT = /* glsl */ `
  varying vec3 vN;
  varying vec3 vP;
  void main() {
    vN = normalize(normal);        // object space; uSun is supplied in the same space
    vP = position;
    gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
  }
`;
const DAYNIGHT_FRAG = /* glsl */ `
  uniform vec3 uSun;      // unit vector toward the Sun, object space
  uniform vec3 uDay;
  uniform vec3 uNight;
  uniform vec3 uRim;
  uniform float uBands;   // >0: faint latitude bands (Earth)
  varying vec3 vN;
  varying vec3 vP;
  void main() {
    float d = dot(normalize(vN), normalize(uSun));
    float lit = smoothstep(-0.12, 0.18, d);
    vec3 base = mix(uNight, uDay, lit);
    if (uBands > 0.5) {
      float lat = asin(clamp(normalize(vP).z, -1.0, 1.0));
      float band = 0.5 + 0.5 * sin(lat * 9.0);
      base += vec3(0.02, 0.05, 0.06) * band * lit;
    }
    // Soft limb brightening on the day side.
    float rim = pow(1.0 - abs(dot(normalize(vN), vec3(0.0, 0.0, 1.0))), 2.0) * 0.08 * lit;
    gl_FragColor = vec4(base + uRim * rim, 1.0);
  }
`;

function useDayNightMaterial(day: string, night: string, rim: string, bands: boolean) {
  return useMemo(
    () =>
      new THREE.ShaderMaterial({
        uniforms: {
          uSun: { value: new THREE.Vector3(1, 0, 0) },
          uDay: { value: new THREE.Color(day) },
          uNight: { value: new THREE.Color(night) },
          uRim: { value: new THREE.Color(rim) },
          uBands: { value: bands ? 1 : 0 },
        },
        vertexShader: DAYNIGHT_VERT,
        fragmentShader: DAYNIGHT_FRAG,
      }),
    [day, night, rim, bands],
  );
}

/** Shared per-frame Sun direction (rotating frame) so all consumers agree within a tick. */
const sunTmp = new THREE.Vector3(1, 0, 0);
function updateSun(): THREE.Vector3 {
  const d = sunDirRot(cursorMs());
  return sunTmp.set(d[0], d[1], d[2]);
}

const qA = new THREE.Quaternion(), qInv = new THREE.Quaternion();

/** Rotating-frame unit = instantaneous Earth–Moon distance, so a body of radius R km is R/d(t) units across. */
function bodyScale(ms: number): number {
  return L_STAR_KM / moonDistanceKm(ms);
}

export function Earth() {
  const mat = useDayNightMaterial('#1f6f9f', '#0a1a2e', '#7fc8ff', true);
  const tilt = useRef<THREE.Group>(null);
  const outer = useRef<THREE.Group>(null);
  const sunObj = useMemo(() => new THREE.Vector3(), []);
  useFrame(() => {
    const ms = cursorMs();
    if (outer.current) outer.current.scale.setScalar(bodyScale(ms));
    if (tilt.current) tilt.current.quaternion.copy(earthQuaternion(qA, ms, false));
    // Sun direction into the tilted object space: apply the inverse orientation.
    sunObj.copy(updateSun()).applyQuaternion(qInv.copy(qA).invert());
    (mat.uniforms.uSun.value as THREE.Vector3).copy(sunObj);
  });
  return (
    <group position={EARTH_ROT} ref={outer}>
      <group ref={tilt}>
        <mesh material={mat}>
          <sphereGeometry args={[EARTH_RADIUS, 48, 32]} />
        </mesh>
      </group>
      {/* Faint atmosphere: a slightly larger, back-side-lit transparent shell. */}
      <mesh>
        <sphereGeometry args={[EARTH_RADIUS * 1.06, 48, 32]} />
        <meshBasicMaterial color={COLORS.atmosphere} transparent opacity={0.2} side={THREE.BackSide} depthWrite={false} />
      </mesh>
      <Label position={[0, 0, -EARTH_RADIUS * 1.3]} text="EARTH" variant="body" id="body-earth" priority={95} />
    </group>
  );
}

export function Moon() {
  // Night side kept visibly above the background so the disc reads in the Moon close-up.
  const mat = useDayNightMaterial('#c4c7ce', '#2c3038', '#ffffff', false);
  const outer = useRef<THREE.Group>(null);
  useFrame(() => {
    (mat.uniforms.uSun.value as THREE.Vector3).copy(updateSun());
    if (outer.current) outer.current.scale.setScalar(bodyScale(cursorMs()));
  });
  return (
    <group position={MOON_ROT} ref={outer}>
      <mesh material={mat}>
        <sphereGeometry args={[MOON_RADIUS, 48, 32]} />
      </mesh>
      {/* Thin limb so the disc is outlined even when the night side faces the camera. */}
      <mesh>
        <sphereGeometry args={[MOON_RADIUS * 1.02, 48, 32]} />
        <meshBasicMaterial color="#9aa0ad" transparent opacity={0.12} side={THREE.BackSide} depthWrite={false} />
      </mesh>
      <MoonLimbRing />
      <Label position={[0, 0, -MOON_RADIUS * 1.4]} text="MOON" variant="body" id="body-moon" priority={95} />
    </group>
  );
}

/**
 * Camera-facing limb ring with a MINIMUM on-screen radius: at the overview zoom the true Moon disc is ~3–4 px and
 * disappears under the lunar-orbiter markers; this ring (never smaller than LIMB_MIN_PX) keeps the Moon legible as
 * a body without drawing the disc itself larger than life. At close-ups it hugs the true limb (×1.08).
 */
const LIMB_MIN_PX = 10;
function MoonLimbRing() {
  const ref = useRef<THREE.Mesh>(null);
  const wp = useMemo(() => new THREE.Vector3(), []);
  useFrame(({ camera, size }) => {
    const m = ref.current;
    if (!m) return;
    m.getWorldPosition(wp);
    const wpp = worldPerPixel(camera, camera.position.distanceTo(wp), size.height);
    // The parent group is scaled by bodyScale: express the ring radius in the parent's units.
    const parent = m.parent;
    const ps = parent ? parent.getWorldScale(wp).x || 1 : 1;
    const r = Math.max(MOON_RADIUS * 1.08, (LIMB_MIN_PX * wpp) / ps);
    m.scale.setScalar(r);
    m.quaternion.copy(camera.quaternion);
  });
  return (
    <mesh ref={ref} renderOrder={4}>
      <ringGeometry args={[0.84, 1, 48]} />
      <meshBasicMaterial color="#e6e9ef" transparent opacity={0.9} side={THREE.DoubleSide} depthWrite={false} depthTest={false} />
    </mesh>
  );
}

/** Sun as a distant directional light that follows the rotating-frame Sun direction. */
export function SunLight({ intensity = 2.0 }: { intensity?: number }) {
  const ref = useRef<THREE.DirectionalLight>(null);
  useFrame(() => {
    if (!ref.current) return;
    const d = updateSun();
    ref.current.position.set(d.x * 20, d.y * 20, d.z * 20 + 2);
  });
  return <directionalLight ref={ref} position={[20, 0, 2]} intensity={intensity} color="#fff4e0" />;
}

function LagrangeMarker({ name, pos, showLabel }: { name: string; pos: [number, number, number]; showLabel: boolean }) {
  const ref = useRef<THREE.Group>(null);
  useScreenScale(ref, 6);
  return (
    <group position={pos}>
      <group ref={ref}>
        <mesh>
          <octahedronGeometry args={[1, 0]} />
          <meshBasicMaterial color={COLORS.accent} wireframe depthTest={false} />
        </mesh>
      </group>
      {showLabel && <Label position={[0, 0, 0]} text={name} accent variant="lpoint" id={`lpoint-${name}`} priority={name === 'L1' || name === 'L2' ? 90 : 40} />}
    </group>
  );
}

export function LagrangePoints({ visible = true }: { visible?: boolean }) {
  const showLabels = useSelene((s) => s.layers.labels);
  if (!visible) return null;
  return (
    <group>
      {LAGRANGE_ROT.map(({ name, pos }) => (
        <LagrangeMarker key={name} name={name} pos={pos} showLabel={showLabels} />
      ))}
    </group>
  );
}
