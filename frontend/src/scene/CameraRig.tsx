/**
 * Flies the camera to a preset (smooth ease over ~0.9 s) when a CameraPresets button is pressed.
 * Presets are authored in rotating-frame coordinates; in the inertial view they are rotated by the current
 * frame angle so "Moon close-up" still looks at the Moon.
 */
import { useFrame, useThree } from '@react-three/fiber';
import { useEffect, useRef } from 'react';
import * as THREE from 'three';
import { emAngleAt } from '../lib/ephem';
import { cursorMs, useSelene } from '../store/useSelene';
import { onCameraPreset } from './cameraBus';
import { CAMERA_PRESETS } from './constants';

interface Flight {
  p0: THREE.Vector3;
  p1: THREE.Vector3;
  t0: THREE.Vector3;
  t1: THREE.Vector3;
  start: number;
  dur: number;
}

/** Current inertial-view rotation angle about +z (0 in the rotating frame). */
export function frameAngleNow(): number {
  const { frame, t0Iso } = useSelene.getState();
  if (frame !== 'inertial') return 0;
  return emAngleAt(cursorMs()) - emAngleAt(Date.parse(t0Iso));
}

export function CameraRig() {
  const camera = useThree((s) => s.camera);
  const controls = useThree((s) => s.controls) as unknown as { target: THREE.Vector3; update: () => void } | null;
  const flight = useRef<Flight | null>(null);

  useEffect(
    () =>
      onCameraPreset((name) => {
        const p = CAMERA_PRESETS[name];
        const a = frameAngleNow();
        const rot = (v: [number, number, number]) => new THREE.Vector3(v[0], v[1], v[2]).applyAxisAngle(new THREE.Vector3(0, 0, 1), a);
        flight.current = {
          p0: camera.position.clone(),
          p1: rot(p.position),
          t0: controls ? controls.target.clone() : new THREE.Vector3(),
          t1: rot(p.target),
          start: performance.now(),
          dur: 900,
        };
      }),
    [camera, controls],
  );

  useFrame(() => {
    const f = flight.current;
    if (!f) return;
    const u = Math.min(1, (performance.now() - f.start) / f.dur);
    const e = u < 0.5 ? 2 * u * u : 1 - Math.pow(-2 * u + 2, 2) / 2; // ease-in-out
    camera.position.lerpVectors(f.p0, f.p1, e);
    if (controls) {
      controls.target.lerpVectors(f.t0, f.t1, e);
      controls.update();
    } else camera.lookAt(f.t1);
    if (u >= 1) flight.current = null;
  });
  return null;
}
