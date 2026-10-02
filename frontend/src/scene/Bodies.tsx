/**
 * Earth, Moon, Sun light and Lagrange markers. Positions are rotating-frame (nondimensional).
 * Earth and Moon use a small day/night shader driven by the Sun direction in the rotating frame
 * (lib/ephem.ts: ephemeris when loaded, mean solar longitude otherwise), so the terminator and lunar phase
 * move with the time cursor. The Earth adds a procedural night-side city-light speckle (no land mask: a seeded
 * cluster noise weighted toward northern mid-latitudes — decorative, not geographic) and an atmosphere glow with a
 * minimum on-screen size so the planet reads as a disc at the overview zoom; the Moon has a sharper terminator with
 * cosine shading. The Earth is tilted by the J2000 obliquity relative to the Earth–Moon (ecliptic) plane,
 * consistent with GroundSites.tsx and the mock scenario's visibility model. The Sun light sits inside the frame
 * group and follows the same direction. `MoonInertialOrbit` (inertial view only) draws the Moon's mean orbit and
 * ghost Moon positions ±3 d so the frame switch is visibly different even at t0.
 */
import { Line } from '@react-three/drei';
import { useFrame } from '@react-three/fiber';
import { useMemo, useRef } from 'react';
import * as THREE from 'three';
import type { Vec3 } from '../api/types';
import { emAngleAt, inertialDisplayMatrix, moonDistanceKm, sunDirRot } from '../lib/ephem';
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
  uniform float uLights;  // >0: night-side city-light speckle (Earth)
  uniform vec2 uTerm;     // terminator smoothstep edges (soft for an atmosphere, sharp for an airless body)
  uniform float uShade;   // 0..1: how strongly day-side brightness follows cos(incidence)
  varying vec3 vN;
  varying vec3 vP;
  float hash13(vec3 p) {
    return fract(sin(dot(p, vec3(127.1, 311.7, 74.7))) * 43758.5453);
  }
  void main() {
    vec3 n = normalize(vN);
    float d = dot(n, normalize(uSun));
    float lit = smoothstep(uTerm.x, uTerm.y, d);
    vec3 day = uDay * mix(1.0, 0.45 + 0.55 * max(d, 0.0), uShade);
    vec3 base = mix(uNight, day, lit);
    if (uBands > 0.5) {
      float lat = asin(clamp(normalize(vP).z, -1.0, 1.0));
      float band = 0.5 + 0.5 * sin(lat * 9.0);
      base += vec3(0.02, 0.05, 0.06) * band * lit;
    }
    if (uLights > 0.5) {
      vec3 u = normalize(vP);
      float lat = asin(clamp(u.z, -1.0, 1.0));
      // More lights at northern mid-latitudes, few near the poles (decorative, not a land mask).
      float latW = smoothstep(-0.95, -0.35, lat) * (1.0 - smoothstep(1.05, 1.35, lat)) * (0.55 + 0.45 * smoothstep(0.0, 0.5, lat));
      float cluster = step(0.58, hash13(floor(u * 36.0)));
      float dots = step(0.86, hash13(floor(u * 200.0) + 3.0)) * cluster * latW;
      float night = 1.0 - smoothstep(-0.25, 0.05, d);
      base += vec3(1.0, 0.78, 0.45) * dots * night * 0.75;
    }
    // Soft limb brightening on the day side.
    float rim = pow(1.0 - abs(dot(n, vec3(0.0, 0.0, 1.0))), 2.0) * 0.08 * lit;
    gl_FragColor = vec4(base + uRim * rim, 1.0);
  }
`;

interface MatOpts {
  bands?: boolean;
  lights?: boolean;
  term?: [number, number];
  shade?: number;
}

function useDayNightMaterial(day: string, night: string, rim: string, opts: MatOpts = {}) {
  const { bands = false, lights = false, term = [-0.12, 0.18], shade = 0 } = opts;
  return useMemo(
    () =>
      new THREE.ShaderMaterial({
        uniforms: {
          uSun: { value: new THREE.Vector3(1, 0, 0) },
          uDay: { value: new THREE.Color(day) },
          uNight: { value: new THREE.Color(night) },
          uRim: { value: new THREE.Color(rim) },
          uBands: { value: bands ? 1 : 0 },
          uLights: { value: lights ? 1 : 0 },
          uTerm: { value: new THREE.Vector2(term[0], term[1]) },
          uShade: { value: shade },
        },
        vertexShader: DAYNIGHT_VERT,
        fragmentShader: DAYNIGHT_FRAG,
      }),
    [day, night, rim, bands, lights, term, shade],
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

/** Radial-gradient sprite texture for the atmosphere glow. */
function useGlowTexture(inner: string, outer: string): THREE.Texture {
  return useMemo(() => {
    const c = document.createElement('canvas');
    c.width = c.height = 128;
    const ctx = c.getContext('2d')!;
    const g = ctx.createRadialGradient(64, 64, 18, 64, 64, 64);
    g.addColorStop(0, inner);
    g.addColorStop(0.45, outer);
    g.addColorStop(1, 'rgba(0,0,0,0)');
    ctx.fillStyle = g;
    ctx.fillRect(0, 0, 128, 128);
    const t = new THREE.CanvasTexture(c);
    t.needsUpdate = true;
    return t;
  }, [inner, outer]);
}

const EARTH_TERM: [number, number] = [-0.12, 0.18];
const MOON_TERM: [number, number] = [-0.03, 0.09];
/** Minimum on-screen radius (px) of the Earth disc + glow at the overview zoom. */
const EARTH_MIN_PX = 26;

/** Camera-facing atmosphere glow with a minimum on-screen radius so the Earth never shrinks to a dot. */
function EarthGlow() {
  const ref = useRef<THREE.Sprite>(null);
  const tex = useGlowTexture('rgba(110, 190, 245, 0.85)', 'rgba(50, 135, 215, 0.28)');
  const wp = useMemo(() => new THREE.Vector3(), []);
  useFrame(({ camera, size }) => {
    const s = ref.current;
    if (!s) return;
    s.getWorldPosition(wp);
    const wpp = worldPerPixel(camera, camera.position.distanceTo(wp), size.height);
    const ps = s.parent ? s.parent.getWorldScale(wp).x || 1 : 1;
    // Glow radius: 1.9 Earth radii, but at least EARTH_MIN_PX on screen (expressed in the scaled parent's units).
    const r = Math.max(EARTH_RADIUS * 1.9, (EARTH_MIN_PX * wpp) / ps);
    s.scale.set(2 * r, 2 * r, 1);
  });
  return (
    <sprite ref={ref} renderOrder={0}>
      <spriteMaterial map={tex} transparent depthWrite={false} blending={THREE.AdditiveBlending} opacity={0.9} />
    </sprite>
  );
}

export function Earth() {
  const mat = useDayNightMaterial('#2a7fb5', '#0a1a2e', '#7fc8ff', { bands: true, lights: true, term: EARTH_TERM, shade: 0.35 });
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
      <EarthGlow />
      <group ref={tilt}>
        <mesh material={mat}>
          <sphereGeometry args={[EARTH_RADIUS, 48, 32]} />
        </mesh>
      </group>
      {/* Faint atmosphere: a slightly larger, back-side-lit transparent shell. */}
      <mesh>
        <sphereGeometry args={[EARTH_RADIUS * 1.06, 48, 32]} />
        <meshBasicMaterial color={COLORS.atmosphere} transparent opacity={0.22} side={THREE.BackSide} depthWrite={false} />
      </mesh>
      <Label position={[0, 0, -EARTH_RADIUS * 1.3]} text="EARTH" variant="body" id="body-earth" priority={97} />
      {/* Invisible spacer over the Earth disc so nearby labels (GEO observers) are placed beside it, not across it. */}
      <Label position={[0, 0, 0]} text="" variant="tag" className="spacer" id="disc-earth" priority={96} offset={DISC_SPACER} />
    </group>
  );
}

export function Moon() {
  // Night side kept visibly above the background so the disc reads in the Moon close-up; sharp terminator.
  const mat = useDayNightMaterial('#cfd2d8', '#1e2128', '#ffffff', { term: MOON_TERM, shade: 0.75 });
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
      <Label position={[0, 0, -MOON_RADIUS * 1.4]} text="MOON" variant="body" id="body-moon" priority={97} />
      <Label position={[0, 0, 0]} text="" variant="tag" className="spacer" id="disc-moon" priority={96} offset={DISC_SPACER} />
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

const GHOST_OFFSETS_D = [-3, 3];
/** 24 px box centred on the body (the glow disc) that lower-priority labels must avoid; the body's own label
 *  (priority 97) is placed first so the spacer never displaces it. */
const DISC_SPACER = { dx: -12, dy: -12 };
const mTmp = new THREE.Matrix4();
const vTmp = new THREE.Vector3();

/** Display-frame position of the Moon at `ms` (exact basis when available, planar mean-element fallback). */
function moonDisplayAt(ms: number, t0Ms: number, out: THREE.Vector3): THREE.Vector3 {
  if (inertialDisplayMatrix(ms, t0Ms, mTmp)) return out.set(MOON_ROT[0], MOON_ROT[1], MOON_ROT[2]).applyMatrix4(mTmp);
  const a = emAngleAt(ms) - emAngleAt(t0Ms);
  return out.set(MOON_ROT[0] * Math.cos(a), MOON_ROT[0] * Math.sin(a), 0);
}

/**
 * Inertial view only (mounted OUTSIDE the synodic group): the Moon's mean orbit about the Earth (circle of radius
 * 1 L* in the t0 orbit plane, faint) and ghost Moons at ±3 d along the real ephemeris, so the inertial frame is
 * visibly different from the rotating one even at t0.
 */
export function MoonInertialOrbit() {
  const frame = useSelene((s) => s.frame);
  const showLabels = useSelene((s) => s.layers.labels);
  const ghosts = useRef<(THREE.Group | null)[]>([]);
  const circle = useMemo(() => {
    const pts: Vec3[] = [];
    for (let i = 0; i <= 180; i++) {
      const a = (i / 180) * 2 * Math.PI;
      pts.push([EARTH_ROT[0] + Math.cos(a), Math.sin(a), 0]);
    }
    return pts;
  }, []);
  // Span start epoch as a number (store-derived, so no Date.parse in the render loop).
  const t0Ms = useSelene((s) => Date.parse(s.t0Iso) + s.t0Sec * 1000);
  const t0Ref = useRef(t0Ms);
  t0Ref.current = t0Ms;
  useFrame(() => {
    if (useSelene.getState().frame !== 'inertial') return;
    const ms = cursorMs();
    const sc = bodyScale(ms);
    for (let i = 0; i < GHOST_OFFSETS_D.length; i++) {
      const g = ghosts.current[i];
      if (!g) continue;
      moonDisplayAt(ms + GHOST_OFFSETS_D[i] * 86400e3, t0Ref.current, vTmp);
      g.position.copy(vTmp);
      g.scale.setScalar(sc);
    }
  });
  if (frame !== 'inertial') return null;
  return (
    <group>
      <Line points={circle} color="#9aa4b2" lineWidth={1} dashed dashSize={0.03} gapSize={0.02} transparent opacity={0.35} depthWrite={false} />
      {showLabels && <Label position={[EARTH_ROT[0] + Math.SQRT1_2, -Math.SQRT1_2, 0]} text="Moon orbit · 27.3 d (mean)" variant="tag" className="plain" id="moon-orbit" priority={25} />}
      {GHOST_OFFSETS_D.map((dd, i) => (
        <group
          key={dd}
          ref={(el) => {
            ghosts.current[i] = el;
          }}
        >
          <mesh>
            <sphereGeometry args={[MOON_RADIUS * 1.6, 16, 12]} />
            <meshBasicMaterial color="#9aa4b2" wireframe transparent opacity={0.35} depthWrite={false} />
          </mesh>
          {showLabels && <Label position={[0, 0, -MOON_RADIUS * 2.5]} text={`Moon ${dd > 0 ? '+' : '−'}${Math.abs(dd)} d`} variant="tag" className="plain" id={`moon-ghost-${dd}`} priority={26} />}
        </group>
      ))}
    </group>
  );
}
