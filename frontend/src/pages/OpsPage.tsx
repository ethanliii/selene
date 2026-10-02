/**
 * Ops page: top bar, 3D viewport, right dock, timeline.
 * On mount: /api/health → /api/catalog/objects (envelope sets the idle timeline to the catalog epoch),
 * /api/orbits/families, /api/sensors (+ /api/orbits/records/{id} for each space observer's host orbit).
 * The DE440s ephemeris basis is (re)fetched whenever the timeline span changes. Object motion comes from
 * /api/catalog/objects/{id}/trajectory (demo/useLiveTracks.ts). Anything served from the browser mock is named
 * in the banner's LIVE/MOCK strip.
 */
import { useEffect, useMemo, useSyncExternalStore } from 'react';
import { api, useBackendStatus } from '../api/client';
import type { OrbitRecord, Vec3 } from '../api/types';
import { startDemo, useDemoDriver } from '../demo/useDemoDriver';
import { useScenarioFrame } from '../demo/useScenarioFrame';
import { usePlayback } from '../hooks/usePlayback';
import { ephemerisCovers, hasBasis, hasBasisAt } from '../lib/ephem';
import { CameraPresets } from '../panels/CameraPresets';
import { useLiveMockLists } from '../panels/LiveStatus';
import { requestCameraPreset, type PresetName } from '../scene/cameraBus';
import { CAMERA_PRESETS, L_STAR_KM } from '../scene/constants';
import { getWorldPerPixel, subscribeScale } from '../scene/frameBus';
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
import { useSelene, type Layers } from '../store/useSelene';

let deepLinkApplied = false;

function Hud({ source, nParticles, tracksLoaded, tracksTotal }: { source: string; nParticles: number; tracksLoaded: number; tracksTotal: number }) {
  const frame = useSelene((s) => s.frame);
  const n = useSelene((s) => s.catalog.length);
  const meta = useSelene((s) => s.catalogMeta);
  const scenario = useSelene((s) => s.scenario);
  const eph = useSelene((s) => s.ephemeris);
  // Coverage of the CURRENT cursor by the loaded tables (re-evaluated on the throttled cursor): outside the loaded
  // span lib/ephem.ts falls back to the mean model and the HUD must say so instead of claiming 'exact'.
  const coverage = useSelene((s) => {
    const ms = Date.parse(s.t0Iso) + Math.floor(s.tSec / 600) * 600 * 1000;
    return hasBasis() ? (hasBasisAt(ms) ? 'exact' : 'outside') : ephemerisCovers(ms) ? 'table' : 'mean';
  });
  const exact = !!eph && coverage === 'exact';
  const frameText =
    frame === 'rotating'
      ? 'EARTH–MOON ROTATING (SYNODIC) · Moon pinned at (1−μ, 0, 0)'
      : exact
        ? 'INERTIAL · Earth-centred, axes = rotating frame at t0 · DE440s basis'
        : coverage === 'outside'
          ? 'INERTIAL · cursor outside the loaded ephemeris span → planar mean-element fallback'
          : 'INERTIAL · planar mean-element Earth–Moon line (no ephemeris basis)';
  const nObjects = useSelene((s) => (s.scenario ? new Set(s.scenario.frames.flatMap((f) => f.objects.map((o) => o.id))).size : 0));
  return (
    <div className="hud" data-label-obstacle>
      <div className="hud-frame">
        <span className={`frame-pill${frame === 'inertial' ? ' inertial' : ''}`}>{frame === 'rotating' ? 'ROT' : 'INR'}</span>
        <b>{frameText}</b>
      </div>
      <div>
        CATALOG <b>{n}</b>
        {meta && (
          <>
            {' '}
            (<b>{meta.n_simulated}</b> simulated · <b>{meta.n_real}</b> real)
          </>
        )}{' '}
        · MOTION{' '}
        <b>
          {source === 'scenario'
            ? 'SCENARIO FRAMES'
            : source === 'idle' && meta
              ? `BACKEND TRAJECTORIES${tracksTotal && tracksLoaded < tracksTotal ? ` (${tracksLoaded}/${tracksTotal} loaded)` : ''}`
              : source === 'idle'
                ? 'CR3BP DISPLAY PROPAGATION'
                : '—'}
        </b>
      </div>
      <div>
        EPHEMERIS{' '}
        <b>
          {eph
            ? exact
              ? `${eph.source ?? 'de440s'} · exact rotating basis`
              : coverage === 'outside'
                ? `${eph.source ?? 'de440s'} loaded for another span · cursor uses mean elements`
                : (eph.source ?? 'mean elements')
            : 'loading…'}
        </b>
      </div>
      {nParticles > 0 && (
        <div>
          CLOUD <b>{nParticles.toLocaleString('en-US')} particles</b>
        </div>
      )}
      {scenario && (
        <div>
          SCENARIO <b>{scenario.meta.title}</b>
          <span className="muted">
            {' '}
            · {nObjects} scenario object{nObjects === 1 ? '' : 's'} drawn; the {n}-object catalog is hidden while the scenario plays
          </span>
        </div>
      )}
      <MockLine />
    </div>
  );
}

/** Dynamic scale bar (bottom-right) + the fixed unit hint. */
function ScaleBar() {
  const wpp = useSyncExternalStore(subscribeScale, getWorldPerPixel);
  if (!(wpp > 0)) return null;
  const kmPerPx = wpp * L_STAR_KM;
  const target = 140 * kmPerPx;
  const exp = Math.pow(10, Math.floor(Math.log10(target)));
  const mant = target / exp;
  const nice = (mant >= 5 ? 5 : mant >= 2 ? 2 : 1) * exp;
  const px = nice / kmPerPx;
  return (
    <div className="scalebar" data-label-obstacle>
      <div className="bar" style={{ width: `${px.toFixed(0)}px` }} />
      <div className="txt">
        <span>{nice.toLocaleString('en-US')} km</span>
        <span className="muted">scale: 1 unit = 384,400 km (L*)</span>
      </div>
    </div>
  );
}

function MockLine() {
  const backend = useBackendStatus();
  const { mock } = useLiveMockLists();
  if (backend === 'offline') return <div className="hud-mock offline">BACKEND OFFLINE — MOCK DATA (browser CR3BP)</div>;
  if (mock.length === 0) return null;
  return <div className="hud-mock">MOCK FALLBACK <b>{mock.join(', ')}</b></div>;
}

export function OpsPage() {
  usePlayback();
  useDemoDriver();
  const layers = useSelene((s) => s.layers);
  const families = useSelene((s) => s.families);
  const sensorsCfg = useSelene((s) => s.sensors);
  const setCatalog = useSelene((s) => s.setCatalog);
  const setBaseEpoch = useSelene((s) => s.setBaseEpoch);
  const setFamilies = useSelene((s) => s.setFamilies);
  const setSensors = useSelene((s) => s.setSensors);
  const setObserverOrbits = useSelene((s) => s.setObserverOrbits);
  const setEphemeris = useSelene((s) => s.setEphemeris);
  const t0Iso = useSelene((s) => s.t0Iso);
  const t0Sec = useSelene((s) => s.t0Sec);
  const t1Sec = useSelene((s) => s.t1Sec);
  const { objects, clouds, sensors, reachable, source, tracksLoaded, tracksTotal } = useScenarioFrame();

  useEffect(() => {
    let alive = true;
    api.health().catch(() => undefined);
    api
      .catalog()
      .then(({ objects, meta }) => {
        if (!alive) return;
        setCatalog(objects, meta);
        if (meta?.epoch_utc) setBaseEpoch(meta.epoch_utc);
      })
      .catch((e) => console.warn('catalog failed', e));
    api
      .orbitFamilies()
      .then((f) => alive && setFamilies(f))
      .catch((e) => console.warn('families failed', e));
    api
      .sensors()
      .then(async (s) => {
        if (!alive) return;
        setSensors(s);
        // Host orbits of the space observers (library records referenced by `orbit_info.source`).
        const ids = [...new Set(s.space.map((x) => x.orbit_record_id).filter((x): x is string => !!x))];
        if (ids.length === 0) return;
        const recs = await Promise.all(ids.map((id) => api.orbitRecord(id).catch((e) => (console.warn('orbit record failed', id, e), null))));
        if (!alive) return;
        const byId = new Map(ids.map((id, i) => [id, recs[i]]));
        const out: Record<string, OrbitRecord> = {};
        for (const x of s.space) {
          const r = x.orbit_record_id ? byId.get(x.orbit_record_id) : null;
          if (r) out[x.id] = r;
        }
        setObserverOrbits(out);
      })
      .catch((e) => console.warn('sensors failed', e));
    return () => {
      alive = false;
    };
  }, [setCatalog, setBaseEpoch, setFamilies, setSensors, setObserverOrbits]);

  // Ephemeris basis over the current timeline span (hourly, ≤ 400 samples); re-fetched when the span changes.
  useEffect(() => {
    let alive = true;
    const base = Date.parse(t0Iso);
    const t0 = new Date(base + t0Sec * 1000).toISOString();
    const t1 = new Date(base + t1Sec * 1000).toISOString();
    const n = Math.min(400, Math.max(24, Math.round((t1Sec - t0Sec) / 3600) + 1));
    api
      .ephemerisBodies(t0, t1, n)
      .then((e) => alive && setEphemeris(e))
      .catch((e) => console.warn('ephemeris failed', e));
    return () => {
      alive = false;
    };
  }, [t0Iso, t0Sec, t1Sec, setEphemeris]);

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
    const t = Number(q.get('t'));
    if (q.get('demo') === null) {
      if (tab === 'object' || tab === 'events' || tab === 'brief') useSelene.getState().setDockTab(tab);
      if (sel) useSelene.getState().selectObject(sel, 'user');
      if (Number.isFinite(t) && q.get('t') !== null) setTimeout(() => useSelene.getState().setT(t), 50);
      return;
    }
    startDemo()
      .then(() => {
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
  const groundIds = useMemo(() => new Set((sensorsCfg?.ground ?? []).map((g) => g.id)), [sensorsCfg]);
  // Ground sites = ids in the sensor config's ground list (fallback: GND- prefix); they now carry pos_rot too.
  const activeSites = useMemo(() => new Set(sensors.filter((s) => s.active && (groundIds.has(s.id) || s.id.startsWith('GND'))).map((s) => s.id)), [sensors, groundIds]);
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
          {layers.fov && <SensorFOVCones sensors={sensors} targets={targets} groundIds={groundIds} />}
          {layers.exclusion && <ExclusionCones sensors={sensors} config={sensorsCfg} groundIds={groundIds} />}
        </SceneRoot>
        <Hud source={source} nParticles={nParticles} tracksLoaded={tracksLoaded} tracksTotal={tracksTotal} />
        <Toasts />
        <CameraPresets />
        <ScaleBar />
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
