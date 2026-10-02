/**
 * Earth orientation in the rotating frame as a three.js quaternion.
 * Exact when the DE440s basis is loaded: Rᵀ · R_z(GMST) (lib/ephem.ts `earthQuaternionRot`); planar
 * R_z(−θ_EM) · R_x(−ε) [· R_z(GMST) when `spin`] otherwise. Shared by the Earth mesh (tilt only) and the ground
 * sites (tilt + spin), and identical to lib/ephem.ts `ecefToRot`, which the mock scenario uses to place sites.
 */
import * as THREE from 'three';
import { earthQuaternionRot } from '../lib/ephem';

export function earthQuaternion(out: THREE.Quaternion, ms: number, spin: boolean): THREE.Quaternion {
  return earthQuaternionRot(out, ms, spin);
}
