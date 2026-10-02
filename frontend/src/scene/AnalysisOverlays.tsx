/**
 * Scene overlays for the Analysis panels (rotating frame, nondimensional):
 *  - OD custody-decay particle cloud at the slider epoch (blue, distinct from the scenario's amber cloud),
 *  - reachable endpoints coloured by the first region each sample enters (fixed palette slot per region; grey =
 *    no region), plus the no-burn nominal path as a faint line.
 * Data comes straight from the live responses in store/useAnalysis.ts; nothing is drawn when a run has not happened.
 */
import { Line } from '@react-three/drei';
import { useEffect, useMemo } from 'react';
import * as THREE from 'three';
import type { Vec3 } from '../api/types';
import { packRows } from '../demo/normalize';
import { useAnalysisTarget } from '../panels/analysis/common';
import { regionColor } from '../panels/analysis/palette';
import { useAnalysis } from '../store/useAnalysis';
import { ParticleCloud } from './ParticleCloud';

const OD_COLOR = '#4cc9f0';

function OdCloud() {
  const od = useAnalysis((s) => s.od.data);
  const forId = useAnalysis((s) => s.od.objectId);
  const target = useAnalysisTarget().objectId;
  const idx = useAnalysis((s) => s.odFrame);
  const show = useAnalysis((s) => s.showOdCloud);
  const part = od?.particles ?? null;
  const frame = part ? part.frames[Math.min(idx, part.frames.length - 1)] : null;
  const positions = useMemo(() => (frame ? packRows(frame.positions_rot) : null), [frame]);
  // Never draw a result over another object: the cloud is hidden when the analysis target has changed (the panel
  // shows a STALE notice with the object/epoch the run was for).
  if (!show || !part || !positions || positions.length < 3 || forId !== target) return null;
  const sigma = part.metrics.sigma_pos_km[Math.min(idx, part.metrics.sigma_pos_km.length - 1)];
  return <ParticleCloud positions={positions} color={OD_COLOR} size={0.0035} objectId={`od-${od?.object_id ?? 'x'}`} sigmaKm={sigma} />;
}

function ReachEndpoints() {
  const reach = useAnalysis((s) => s.reach.data);
  const forId = useAnalysis((s) => s.reach.objectId);
  const target = useAnalysisTarget().objectId;
  const show = useAnalysis((s) => s.showReach);
  const geom = useMemo(() => {
    if (!reach) return null;
    const pts = reach.samples.end_rot;
    const n = pts.length;
    const pos = new Float32Array(n * 3);
    const col = new Float32Array(n * 3);
    const c = new THREE.Color();
    for (let i = 0; i < n; i++) {
      pos[3 * i] = pts[i][0];
      pos[3 * i + 1] = pts[i][1];
      pos[3 * i + 2] = pts[i][2];
      c.set(regionColor(reach.samples.hit_region[i]));
      if (!reach.samples.hit_region[i]) c.multiplyScalar(0.75);
      col[3 * i] = c.r;
      col[3 * i + 1] = c.g;
      col[3 * i + 2] = c.b;
    }
    const g = new THREE.BufferGeometry();
    g.setAttribute('position', new THREE.BufferAttribute(pos, 3));
    g.setAttribute('color', new THREE.BufferAttribute(col, 3));
    g.computeBoundingSphere();
    return g;
  }, [reach]);
  useEffect(() => () => geom?.dispose(), [geom]);
  if (!show || !reach || !geom || forId !== target) return null;
  const nominal = reach.nominal.rot as Vec3[];
  return (
    <group>
      <points geometry={geom} frustumCulled={false}>
        <pointsMaterial vertexColors size={0.006} sizeAttenuation transparent opacity={0.9} depthWrite={false} />
      </points>
      {nominal.length > 1 && <Line points={nominal} color="#8d9db3" lineWidth={1} dashed dashSize={0.01} gapSize={0.008} transparent opacity={0.6} depthWrite={false} />}
    </group>
  );
}

export function AnalysisOverlays() {
  return (
    <group>
      <OdCloud />
      <ReachEndpoints />
    </group>
  );
}
