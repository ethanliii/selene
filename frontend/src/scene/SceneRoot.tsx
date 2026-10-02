/**
 * The 3D viewport. Scene unit = 1 L* (384 400 km). z is "up" (rotating-frame angular-momentum axis).
 *
 * Frame handling: everything physical is authored in the Earth–Moon ROTATING frame (nondimensional, Moon pinned
 * at (1−μ, 0, 0)) and placed under <FrameGroup>. In 'inertial' mode the group is given the exact display transform
 * from the DE440s rotating-frame basis (lib/ephem.ts `inertialDisplayMatrix`: uniform scale d(t)/L*, rotation
 * R(t0)ᵀR(t), Earth fixed at (−μ,0,0)), so the Earth–Moon line sweeps around and the Moon rides its true
 * ephemeris track at its true distance. Without a basis (backend down) the group is rotated about +z by
 * θ(t) − θ(t0), the mean-element Earth–Moon line angle (planar approximation; the HUD says which one is active).
 * The axis triad is a 48-px corner gizmo (drei GizmoHelper) instead of a world-space triad at the barycentre.
 */
import { Canvas, useFrame, useThree } from '@react-three/fiber';
import { GizmoHelper, GizmoViewport, OrbitControls, Stars } from '@react-three/drei';
import { useRef, type ReactNode } from 'react';
import * as THREE from 'three';
import { emAngleAt, inertialDisplayMatrix } from '../lib/ephem';
import { cursorMs, useSelene } from '../store/useSelene';
import { Earth, LagrangePoints, Moon, MoonInertialOrbit, SunLight } from './Bodies';
import { CameraRig } from './CameraRig';
import { CAMERA } from './constants';
import { FRAME_INFO, FRAME_MATRIX, publishWorldPerPixel } from './frameBus';
import { LabelDeclutter } from './LabelDeclutter';
import { ReferenceGrid } from './ReferenceGrid';
import { worldPerPixel } from './screenScale';

function FrameGroup({ children }: { children: ReactNode }) {
  const ref = useRef<THREE.Group>(null);
  useFrame(() => {
    const g = ref.current;
    if (!g) return;
    const { frame, t0Iso, t0Sec } = useSelene.getState();
    const ms = cursorMs();
    const t0Ms = Date.parse(t0Iso) + t0Sec * 1000;
    if (frame === 'inertial' && inertialDisplayMatrix(ms, t0Ms, g.matrix)) {
      g.matrixAutoUpdate = false;
      g.matrixWorldNeedsUpdate = true;
      FRAME_MATRIX.copy(g.matrix);
      FRAME_INFO.mode = 'inertial-exact';
      FRAME_INFO.scale = g.matrix.getMaxScaleOnAxis();
      return;
    }
    g.matrixAutoUpdate = true;
    g.position.set(0, 0, 0);
    g.scale.setScalar(1);
    g.rotation.set(0, 0, frame === 'inertial' ? emAngleAt(ms) - emAngleAt(t0Ms) : 0);
    g.updateMatrix();
    FRAME_MATRIX.copy(g.matrix);
    FRAME_INFO.mode = frame === 'inertial' ? 'inertial-planar' : 'rotating';
    FRAME_INFO.scale = 1;
  });
  return <group ref={ref}>{children}</group>;
}

/** Publishes the world size of one screen pixel at the orbit target (for the DOM scale bar). */
function ScaleProbe() {
  const controls = useThree((s) => s.controls) as unknown as { target: THREE.Vector3 } | null;
  const tick = useRef(0);
  useFrame(({ camera, size }) => {
    if (++tick.current % 6) return;
    const d = controls ? camera.position.distanceTo(controls.target) : camera.position.length();
    publishWorldPerPixel(worldPerPixel(camera, d, size.height));
  });
  return null;
}

export interface SceneRootProps {
  /** Extra layers authored in rotating-frame coordinates (families, objects, clouds, cones...). */
  children?: ReactNode;
}

export function SceneRoot({ children }: SceneRootProps) {
  const layers = useSelene((s) => s.layers);
  const select = useSelene((s) => s.selectObject);
  return (
    <Canvas
      dpr={[1, 2]}
      gl={{ antialias: true, alpha: false, powerPreference: 'high-performance' }}
      camera={{ position: CAMERA.position, near: CAMERA.near, far: CAMERA.far, fov: CAMERA.fov, up: [0, 0, 1] }}
      onPointerMissed={() => select(null)}
      style={{ background: '#05070b' }}
    >
      <color attach="background" args={['#05070b']} />
      <Stars radius={18} depth={14} count={6000} factor={2.6} saturation={0.15} fade speed={0} />
      <ambientLight intensity={0.18} />
      <FrameGroup>
        <SunLight />
        <ReferenceGrid visible={layers.grid} />
        <Earth />
        <Moon />
        <LagrangePoints visible={layers.lagrange} />
        {children}
      </FrameGroup>
      <MoonInertialOrbit />
      <OrbitControls makeDefault enableDamping dampingFactor={0.08} minDistance={0.02} maxDistance={12} target={CAMERA.target} />
      <GizmoHelper alignment="bottom-left" margin={[62, 112]} renderPriority={1}>
        <GizmoViewport axisColors={['#c0504d', '#5aa469', '#4f81bd']} labels={['x', 'y', 'z']} labelColor="#d6e2f0" axisHeadScale={0.8} hideNegativeAxes />
      </GizmoHelper>
      <CameraRig />
      <ScaleProbe />
      <LabelDeclutter />
    </Canvas>
  );
}
