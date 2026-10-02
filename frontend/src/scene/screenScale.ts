/**
 * Constant-pixel-size markers. Object/L-point markers are abstract symbols, not physical bodies, so they must not
 * grow larger than the Moon when the camera zooms in (a 0.006 L* sphere would be 2 300 km across). The hook sets
 * the group's scale every render tick so that a unit-radius child appears `px` pixels in radius on screen.
 */
import { useFrame } from '@react-three/fiber';
import type { RefObject } from 'react';
import * as THREE from 'three';

const tmp = new THREE.Vector3();

/** World units per screen pixel at distance `d` from a perspective camera. */
export function worldPerPixel(camera: THREE.Camera, d: number, viewportHeightPx: number): number {
  const fov = (camera as THREE.PerspectiveCamera).fov ?? 40;
  return (2 * d * Math.tan((fov * Math.PI) / 360)) / Math.max(1, viewportHeightPx);
}

export function useScreenScale(ref: RefObject<THREE.Object3D | null>, px: number, minWorld = 0, maxWorld = Infinity): void {
  useFrame(({ camera, size }) => {
    const o = ref.current;
    if (!o) return;
    o.getWorldPosition(tmp);
    const d = camera.position.distanceTo(tmp);
    const s = Math.min(maxWorld, Math.max(minWorld, px * worldPerPixel(camera, d, size.height)));
    o.scale.setScalar(s);
  });
}
