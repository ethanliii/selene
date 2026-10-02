/** Ops page: top bar, 3D viewport, right dock, timeline. Loads catalog, families, sensors and ephemeris on mount. */
import { useEffect, useMemo } from 'react';
import { api, useBackendStatus } from '../api/client';
import type { Vec3 } from '../api/types';
import { startDemo, useDemoDriver } from '../demo/useDemoDriver';
import { useScenarioFrame } from '../demo/useScenarioFrame';
import { usePlayback } from '../hooks/usePlayback';
import { ephemerisSource } from '../lib/ephem';
import { CameraPresets } from '../panels/CameraPresets';
import { requestCameraPreset, type PresetName } from '../scene/cameraBus';
import { CAMERA_PRESETS } from '../scene/constants';
import { Dock } from '../panels/Dock';
import { Timeline } from '../panels/Timeline';
import { Toasts } from '../panels/Toasts';
import { TopBar } from '../panels/TopBar';
import { ExclusionCones, SensorFOVCones } from '../scene/Cones';
import { GroundSites } from '../scene/GroundSites';
import { Objects } from '../scene/Objects';
import { OrbitFamilies } from '../scene/OrbitFamilies';
import { ParticleCloud } from '../scene/ParticleCloud';
import { Reachable } from '../scene/Reachable';
import { SceneRoot } from '../scene/SceneRoot';
import { DEFAULT_SPAN_S, DEFAULT_T0_ISO, useSelene, type Layers } from '../store/useSelene';

let deepLinkApplied = false;

function Hud({ source, nParticles }: { source: string; nParticles: number }) {
  const frame = useSelene((s) => s.frame);
  const n = useSelene((s) => s.catalog.length);
  const scenario = useSelene((s) => s.scenario);
  const eph = useSelene((s) => s.ephemeris);
  const ephSrc = eph && ephemerisSource() === 'ephemeris' && eph.source !== 'mock-mean-elements' ? 'ephemeris' : 'mean';
  return (
    <div className="hud">
      <div>
        FRAME <b>{frame === 'rotating' ? 'EARTH–MOON ROTATING (SYNODIC)' : `INERTIAL (ecliptic-plane rotation, ${ephSrc === 'ephemeris' ? 'ephemeris Earth–Moon line' : 'mean-element Earth–Moon line'})`}</b>
      </div>
      <div>
        UNIT <b>1 L* = 384 400 km</b> · CATALOG <b>{n}</b> · MOTION <b>{source === 'scenario' ? 'SCENARIO FRAMES' : source === 'idle' ? 'CR3BP DISPLAY PROPAGATION' : '—'}</b>
      </div>
      {nParticles > 0 && (
        <div>
          CLOUD <b>{nParticles.toLocaleString('en-US')} particles</b>
        </div>
      )}
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
  useDemoDriver();
  const backend = useBackendStatus();
  const layers = useSelene((s) => s.layers);
  const families = useSelene((s) => s.families);
  const sensorsCfg = useSelene((s) => s.sensors);
  const setCatalog = useSelene((s) => s.setCatalog);
  const setFamilies = useSelene((s) => s.setFamilies);
  const setSensors = useSelene((s) => s.setSensors);
  const setEphemeris = useSelene((s) => s.setEphemeris);
  const hasEphemeris = useSelene((s) => !!s.ephemeris);
  const { objects, clouds, sensors, reachable, source } = useScenarioFrame();

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
    api
      .sensors()
      .then((s) => alive && setSensors(s))
      .catch((e) => console.warn('sensors failed', e));
    if (!hasEphemeris) {
      const t1 = new Date(Date.parse(DEFAULT_T0_ISO) + DEFAULT_SPAN_S * 1000).toISOString();
      api
        .ephemerisBodies(DEFAULT_T0_ISO, t1, 24 * 7)
        .then((e) => alive && setEphemeris(e))
        .catch((e) => console.warn('ephemeris failed', e));
    }
    return () => {
      alive = false;
    };
  }, [setCatalog, setFamilies, setSensors, setEphemeris, hasEphemeris]);

  // Deep links for the pitch / smoke tests: ?demo=1 auto-plays the story; &t=<sec> jumps there and pauses;
  // &frame=inertial switches the view; &view=<preset>; &layers=a,b; &tab=object|events|brief; &select=<id>.
  // Guarded so React.StrictMode's double-mount in dev starts the demo once.
  useEffect(() => {
    const q = new URLSearchParams(window.location.search);
    if (deepLinkApplied) return;
    deepLinkApplied = true;
    if (q.get('frame') === 'inertial') useSelene.getState().setFrame('inertial');
    const lay = q.get('layers');
    if (lay) for (const k of lay.split(',')) if (k in useSelene.getState().layers) useSelene.getState().setLayer(k as keyof Layers, true);
    const view = q.get('view');
    if (view && view in CAMERA_PRESETS) setTimeout(() => requestCameraPreset(view as PresetName), 300);
    const tab = q.get('tab');
    const sel = q.get('select');
    if (q.get('demo') === null) {
      if (tab === 'object' || tab === 'events' || tab === 'brief') useSelene.getState().setDockTab(tab);
      return;
    }
    startDemo()
      .then(() => {
        const t = Number(q.get('t'));
        if (Number.isFinite(t) && q.get('t') !== null) {
          useSelene.getState().setPlaying(false);
          useSelene.getState().setT(t);
        }
        if (sel) useSelene.getState().selectObject(sel, 'user');
        if (tab === 'object' || tab === 'events' || tab === 'brief') useSelene.getState().setDockTab(tab);
      })
      .catch((e) => console.warn('auto demo failed', e));
  }, []);

  const targets = useMemo(() => new Map<string, Vec3>(objects.map((o) => [o.id, o.pos])), [objects]);
  // Ground sites = ids in the sensor config's ground list (fallback: GND- prefix); they now carry pos_rot too.
  const activeSites = useMemo(() => {
    const groundIds = new Set((sensorsCfg?.ground ?? []).map((g) => g.id));
    return new Set(sensors.filter((s) => s.active && (groundIds.has(s.id) || s.id.startsWith('GND'))).map((s) => s.id));
  }, [sensors, sensorsCfg]);
  const nParticles = clouds.reduce((a, c) => a + c.positions.length / 3, 0);

  return (
    <div className="ops">
      <TopBar />
      <div className="viewport">
        <SceneRoot>
          <GroundSites visible={layers.sites} activeIds={activeSites} />
          <OrbitFamilies data={families} visible={layers.families} />
          <Objects objects={objects} showTrails={layers.trails} />
          {layers.clouds && clouds.map((c) => <ParticleCloud key={c.objectId} positions={c.positions} />)}
          {layers.reach && <Reachable data={reachable} />}
          {layers.fov && <SensorFOVCones sensors={sensors} targets={targets} />}
          {layers.exclusion && <ExclusionCones sensors={sensors} config={sensorsCfg} />}
        </SceneRoot>
        <Hud source={source} nParticles={nParticles} />
        {backend === 'offline' && <div className="mock-badge">BACKEND OFFLINE — MOCK DATA (browser CR3BP)</div>}
        {backend === 'partial' && <div className="mock-badge warn">API PARTIAL — MOCK FALLBACK (browser CR3BP) FOR UNIMPLEMENTED ENDPOINTS</div>}
        <Toasts />
        <CameraPresets />
        <div className="hud-br">
          drag: orbit · wheel: zoom · right-drag: pan · click object: select
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
