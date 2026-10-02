/** Ops page: top bar, 3D viewport, right dock, timeline. Loads catalog + orbit families on mount. */
import { useEffect, useState } from 'react';
import { api, useBackendStatus } from '../api/client';
import type { OrbitFamilies as OrbitFamiliesData } from '../api/types';
import { useScenarioFrame } from '../demo/useScenarioFrame';
import { usePlayback } from '../hooks/usePlayback';
import { Dock } from '../panels/Dock';
import { Timeline } from '../panels/Timeline';
import { TopBar } from '../panels/TopBar';
import { Objects } from '../scene/Objects';
import { OrbitFamilies } from '../scene/OrbitFamilies';
import { ParticleCloud } from '../scene/ParticleCloud';
import { SceneRoot } from '../scene/SceneRoot';
import { SensorFOVCone } from '../scene/Cones';
import { useSelene } from '../store/useSelene';

function Hud() {
  const frame = useSelene((s) => s.frame);
  const n = useSelene((s) => s.catalog.length);
  const scenario = useSelene((s) => s.scenario);
  return (
    <div className="hud">
      <div>
        FRAME <b>{frame === 'rotating' ? 'EARTH–MOON ROTATING (SYNODIC)' : 'INERTIAL (PLACEHOLDER ROTATION)'}</b>
      </div>
      <div>
        UNIT <b>1 L* = 384 400 km</b> · CATALOG <b>{n}</b>
      </div>
      {scenario && (
        <div>
          SCENARIO <b>{scenario.meta.title}</b>
        </div>
      )}
    </div>
  );
}

export function OpsPage() {
  usePlayback();
  const backend = useBackendStatus();
  const layers = useSelene((s) => s.layers);
  const setCatalog = useSelene((s) => s.setCatalog);
  const [families, setFamilies] = useState<OrbitFamiliesData | null>(null);
  const { frame, objects, clouds } = useScenarioFrame();

  useEffect(() => {
    let alive = true;
    api.health().catch(() => undefined);
    api
      .catalogObjects()
      .then((c) => alive && setCatalog(c))
      .catch((e) => console.warn('catalog failed', e));
    api
      .orbitFamilies()
      .then((f) => alive && setFamilies(f))
      .catch((e) => console.warn('families failed', e));
    return () => {
      alive = false;
    };
  }, [setCatalog]);

  return (
    <div className="ops">
      <TopBar />
      <div className="viewport">
        <SceneRoot>
          <OrbitFamilies data={families} visible={layers.families} />
          <Objects objects={objects} showTrails={layers.trails} />
          {layers.clouds && clouds.map((c) => <ParticleCloud key={c.objectId} positions={c.positions} />)}
          {layers.fov &&
            frame?.sensors
              .filter((s) => s.boresight_rot)
              .map((s) => (
                <SensorFOVCone key={s.id} sensorId={s.id} apex={s.pos_rot} boresight={s.boresight_rot!} halfAngleDeg={(s.fov_deg ?? 2) / 2} length={0.6} active={!!s.target_id} />
              ))}
          {/* TODO(M10): <ExclusionCone> per sensor from ephemeris Sun/Moon/Earth directions when layers.exclusion. */}
        </SceneRoot>
        <Hud />
        {backend === 'offline' && <div className="mock-badge">BACKEND OFFLINE — MOCK DATA</div>}
        <div className="hud-br">
          drag: orbit · wheel: zoom · right-drag: pan
          <br />
          all SIMULATED objects/events are notional
        </div>
      </div>
      <aside className="dock">
        <Dock />
      </aside>
      <footer className="timeline">
        <Timeline />
      </footer>
    </div>
  );
}
