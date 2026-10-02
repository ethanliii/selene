/**
 * The 3D viewport. Scene unit = 1 L* (384 400 km). z is "up" (rotating-frame angular-momentum axis).
 *
 * Frame handling: everything physical is authored in the Earth–Moon ROTATING frame and placed under
 * <FrameGroup>. In 'inertial' mode the group is rotated about +z by θ(t) − θ(t0), where θ is the angle of the
 * Earth–Moon line in the GCRF xy-plane (from /api/ephemeris/bodies when loaded, else the mean lunar longitude —
 * see lib/ephem.ts). The view is therefore inertial in the sense that the Earth–Moon line sweeps around while
 * the camera stays fixed; the lunar orbit inclination is not represented (HUD says "xy-projection").
 */
import { Canvas, useFrame } from '@react-three/fiber';
import { OrbitControls, Stars } from '@react-three/drei';
import { useRef, type ReactNode } from 'react';
import type { Group } from 'three';
import { emAngleAt } from '../lib/ephem';
import { cursorMs, useSelene } from '../store/useSelene';
import { Earth, LagrangePoints, Moon, SunLight } from './Bodies';
import { CameraRig } from './CameraRig';
import { CAMERA } from './constants';
import { ReferenceGrid } from './ReferenceGrid';

function FrameGroup({ children }: { children: ReactNode }) {
  const ref = useRef<Group>(null);
  useFrame(() => {
    const { frame, t0Iso } = useSelene.getState();
    if (!ref.current) return;
    ref.current.rotation.z = frame === 'inertial' ? emAngleAt(cursorMs()) - emAngleAt(Date.parse(t0Iso)) : 0;
  });
  return <group ref={ref}>{children}</group>;
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
      <Stars radius={18} depth={14} count={3500} factor={1.6} saturation={0} fade speed={0} />
      <ambientLight intensity={0.18} />
      <FrameGroup>
        <SunLight />
        <ReferenceGrid visible={layers.grid} />
        <Earth />
        <Moon />
        <LagrangePoints visible={layers.lagrange} />
        {children}
      </FrameGroup>
      <OrbitControls makeDefault enableDamping dampingFactor={0.08} minDistance={0.02} maxDistance={12} target={[0.5, 0, 0]} />
      <CameraRig />
    </Canvas>
  );
}
