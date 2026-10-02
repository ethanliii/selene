/**
 * Ground optical sites as small markers on the Earth's surface at (lat, lon), carried by the Earth's orientation
 * in the rotating frame (R_z(−θ_EM)·R_x(−ε)·R_z(GMST), lib/ephem.ts) — the same model the mock scenario uses to
 * compute site visibility, so the glowing site and its FOV needle agree. Sites that are actively observing in the
 * current frame (frame.sensors[].active) glow in --ok; others are muted.
 */
import { useFrame } from '@react-three/fiber';
import { useMemo, useRef, useState } from 'react';
import { Quaternion, Vector3, type Group } from 'three';
import type { Vec3 } from '../api/types';
import { moonDistanceKm } from '../lib/ephem';
import { cursorMs, useSelene } from '../store/useSelene';
import { earthQuaternion } from './earthOrientation';
import { COLORS, EARTH_RADIUS, EARTH_ROT, L_STAR_KM } from './constants';
import { Label } from './Label';

const tmpV = new Vector3();
const tmpQ = new Quaternion();

interface Props {
  visible?: boolean;
  /** Ids of sites currently observing. */
  activeIds?: Set<string>;
}

/** Earth-fixed unit vector of a site (spherical Earth). */
export function siteUnitVector(latDeg: number, lonDeg: number): Vec3 {
  const la = (latDeg * Math.PI) / 180, lo = (lonDeg * Math.PI) / 180;
  return [Math.cos(la) * Math.cos(lo), Math.cos(la) * Math.sin(lo), Math.sin(la)];
}

export function GroundSites({ visible = true, activeIds }: Props) {
  const sensors = useSelene((s) => s.sensors);
  const showLabels = useSelene((s) => s.layers.labels);
  const ref = useRef<Group>(null);
  const sites = useMemo(
    () =>
      (sensors?.ground ?? []).map((g) => {
        const u = siteUnitVector(g.lat_deg, g.lon_deg);
        const r = EARTH_RADIUS * 1.01;
        return { id: g.id, name: g.name, short: g.id.replace(/^GND-/, '').replace(/_/g, ' ').toUpperCase(), pos: [u[0] * r, u[1] * r, u[2] * r] as Vec3 };
      }),
    [sensors],
  );
  // Site labels only when the camera is close enough to the Earth to read them (state flips rarely).
  const [near, setNear] = useState(false);
  useFrame(({ camera }) => {
    if (ref.current) {
      const ms = cursorMs();
      ref.current.quaternion.copy(earthQuaternion(tmpQ, ms, true));
      ref.current.scale.setScalar(L_STAR_KM / moonDistanceKm(ms));
    }
    const d = camera.position.distanceTo(ref.current ? ref.current.getWorldPosition(tmpV) : tmpV.set(EARTH_ROT[0], 0, 0));
    const n = d < 0.45;
    if (n !== near) setNear(n);
  });
  if (!visible || sites.length === 0) return null;
  return (
    <group position={EARTH_ROT} ref={ref}>
      {sites.map((s) => {
        const active = activeIds?.has(s.id) ?? false;
        return (
          <group key={s.id} position={s.pos}>
            <mesh>
              <sphereGeometry args={[EARTH_RADIUS * (active ? 0.09 : 0.06), 8, 6]} />
              <meshBasicMaterial color={active ? COLORS.ok : COLORS.sim} />
            </mesh>
            {showLabels && (near || active) && <Label position={[0, 0, 0]} text={s.short} color={active ? COLORS.ok : COLORS.muted} id={`site-${s.id}`} priority={active ? 75 : 30} />}
          </group>
        );
      })}
    </group>
  );
}
