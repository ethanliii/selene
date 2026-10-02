/**
 * The 3D viewport. Scene unit = 1 L* (384 400 km). z is "up" (rotating-frame angular-momentum axis).
 *
 * Frame handling: everything physical is authored in the Earth–Moon ROTATING frame and placed under
 * <FrameGroup>. In 'inertial' mode the group is rotated about +z by θ = ω·tSec so the Earth–Moon line
 * sweeps around while the camera stays fixed. This is a PLACEHOLDER for a true GCRF view (which would
 * place Earth at the origin and use the DE440s Moon position); a later agent wires /api/ephemeris/bodies.
 */
import { Canvas, useFrame } from '@react-three/fiber';
import { OrbitControls, Stars } from '@react-three/drei';
import { useRef, type ReactNode } from 'react';
import type { Group } from 'three';
import type { Vec3 } from '../api/types';
import { useSelene } from '../store/useSelene';
import { Earth, LagrangePoints, Moon, SunLight } from './Bodies';
import { CAMERA, SYNODIC_RATE_RAD_S } from './constants';
import { ReferenceGrid } from './ReferenceGrid';

function FrameGroup({ children }: { children: ReactNode }) {
  const ref = useRef<Group>(null);
  useFrame(() => {
    const { frame, tSec } = useSelene.getState();
    if (ref.current) ref.current.rotation.z = frame === 'inertial' ? SYNODIC_RATE_RAD_S * tSec : 0;
  });
  return <group ref={ref}>{children}</group>;
}

export interface SceneRootProps {
  /** Sun direction (unit vector toward the Sun) in the rotating frame. Default +x. */
  sunDirection?: Vec3;
  /** Extra layers authored in rotating-frame coordinates (families, objects, clouds, cones...). */
  children?: ReactNode;
}

export function SceneRoot({ sunDirection = [1, 0, 0], children }: SceneRootProps) {
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
      <SunLight direction={sunDirection} />
      <FrameGroup>
        <ReferenceGrid visible={layers.grid} />
        <Earth />
        <Moon />
        <LagrangePoints visible={layers.lagrange} />
        {children}
      </FrameGroup>
      <OrbitControls makeDefault enableDamping dampingFactor={0.08} minDistance={0.03} maxDistance={12} target={[0.5, 0, 0]} />
    </Canvas>
  );
}
