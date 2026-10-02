/**
 * Flies the camera to a preset (smooth ease over ~0.9 s) when a CameraPresets button is pressed.
 * Presets are authored in rotating-frame coordinates; they are mapped through the synodic group's current display
 * matrix (scene/frameBus.ts) so "Moon close-up" still looks at the Moon in the inertial view.
 */
import { useFrame, useThree } from '@react-three/fiber';
import { useEffect, useRef } from 'react';
import * as THREE from 'three';
import { onCameraPreset } from './cameraBus';
import { CAMERA_PRESETS } from './constants';
import { FRAME_MATRIX } from './frameBus';

interface Flight {
  p0: THREE.Vector3;
  p1: THREE.Vector3;
  t0: THREE.Vector3;
  t1: THREE.Vector3;
  start: number;
  dur: number;
}

export function CameraRig() {
  const camera = useThree((s) => s.camera);
  const controls = useThree((s) => s.controls) as unknown as { target: THREE.Vector3; update: () => void } | null;
  const flight = useRef<Flight | null>(null);

  useEffect(
    () =>
      onCameraPreset((name) => {
        const p = CAMERA_PRESETS[name];
        // Rotation/translation of the display frame, without its uniform scale for the camera offset (the offset is
        // a viewing distance, the target is a scene point).
        const target = new THREE.Vector3(...p.target).applyMatrix4(FRAME_MATRIX);
        const offset = new THREE.Vector3(p.position[0] - p.target[0], p.position[1] - p.target[1], p.position[2] - p.target[2]);
        const rot = new THREE.Quaternion();
        FRAME_MATRIX.decompose(new THREE.Vector3(), rot, new THREE.Vector3());
        offset.applyQuaternion(rot);
        flight.current = {
          p0: camera.position.clone(),
          p1: target.clone().add(offset),
          t0: controls ? controls.target.clone() : new THREE.Vector3(),
          t1: target,
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
