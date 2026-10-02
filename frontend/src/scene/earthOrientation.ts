/**
 * Earth orientation in the rotating frame as a three.js quaternion: R_z(−θ_EM) · R_x(−ε) [· R_z(GMST) when `spin`]
 * (lib/ephem.ts frame model). Shared by the Earth mesh (tilt only) and the ground sites (tilt + spin), and identical
 * to lib/ephem.ts `ecefToRot`, which the mock scenario uses to place sites.
 */
import * as THREE from 'three';
import { earthOrientationRot } from '../lib/ephem';

const Z = new THREE.Vector3(0, 0, 1);
const X = new THREE.Vector3(1, 0, 0);
const qB = new THREE.Quaternion(), qC = new THREE.Quaternion();

export function earthQuaternion(out: THREE.Quaternion, ms: number, spin: boolean): THREE.Quaternion {
  const { thetaEM, obliquity, gmst } = earthOrientationRot(ms);
  out.setFromAxisAngle(Z, -thetaEM).multiply(qB.setFromAxisAngle(X, -obliquity));
  if (spin) out.multiply(qC.setFromAxisAngle(Z, gmst));
  return out;
}
