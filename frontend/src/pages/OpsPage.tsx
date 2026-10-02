/**
 * Ops page: top bar, 3D viewport, right dock, timeline.
 * On mount: /api/health → /api/catalog/objects (envelope sets the idle timeline to the catalog epoch),
 * /api/orbits/families, /api/sensors (+ /api/orbits/records/{id} for each space observer's host orbit).
 * The DE440s ephemeris basis is (re)fetched whenever the timeline span changes. Object motion comes from
 * /api/catalog/objects/{id}/trajectory (demo/useLiveTracks.ts). Anything served from the browser mock is named
 * in the top bar's DATA SOURCES popover; the HUD says so only when the backend is offline.
 *
 * Viewport overlays: two-line HUD with an ⓘ expander, colour keys (custody / exclusion / clouds), loading bar,
 * presenter caption (current story headline), toasts (bottom-right), camera presets, scale bar, axis gizmo.
 */
import { useEffect, useMemo, useState, useSyncExternalStore } from 'react';
import { api, useBackendStatus } from '../api/client';
import type { OrbitRecord, SeleneEvent, Vec3 } from '../api/types';
import { deriveHeadline, kindLabel } from '../demo/headline';
import { startDemo, useDemoDriver } from '../demo/useDemoDriver';
import { useScenarioFrame } from '../demo/useScenarioFrame';
import { usePlayback } from '../hooks/usePlayback';
import { ephemerisCovers, hasBasis, hasBasisAt } from '../lib/ephem';
import { CameraPresets } from '../panels/CameraPresets';
import { severityColor } from '../panels/severity';
import { requestCameraPreset, type PresetName } from '../scene/cameraBus';
import { CAMERA_PRESETS, L_STAR_KM } from '../scene/constants';
import { getWorldPerPixel, subscribeScale } from '../scene/frameBus';
import { Dock } from '../panels/Dock';
import { Timeline } from '../panels/Timeline';
import { Toasts } from '../panels/Toasts';
import { TopBar } from '../panels/TopBar';
import { ExclusionCones, SensorFOVCones, SensorMarkers } from '../scene/Cones';
import { GroundSites } from '../scene/GroundSites';
import { Objects } from '../scene/Objects';
import { OrbitFamilies } from '../scene/OrbitFamilies';
import { ParticleCloud } from '../scene/ParticleCloud';
import { Reachable } from '../scene/Reachable';
import { SceneRoot } from '../scene/SceneRoot';
import { fmtElapsed, useSelene, type Layers } from '../store/useSelene';

let deepLinkApplied = false;

function Hud({ source, nParticles, tracksLoaded, tracksTotal }: { source: string; nParticles: number; tracksLoaded: number; tracksTotal: number }) {
  const frame = useSelene((s) => s.frame);
  const n = useSelene((s) => s.catalog.length);
  const meta = useSelene((s) => s.catalogMeta);
  const scenario = useSelene((s) => s.scenario);
  const eph = useSelene((s) => s.ephemeris);
  const tMin = useSelene((s) => Math.floor(s.tSec / 60) * 60);
  const [more, setMore] = useState(false);
  // Coverage of the CURRENT cursor by the loaded tables (re-evaluated on the throttled cursor): outside the loaded
  // span lib/ephem.ts falls back to the mean model and the HUD must say so instead of claiming 'exact'.
  const coverage = useSelene((s) => {
    const ms = Date.parse(s.t0Iso) + Math.floor(s.tSec / 600) * 600 * 1000;
    return hasBasis() ? (hasBasisAt(ms) ? 'exact' : 'outside') : ephemerisCovers(ms) ? 'table' : 'mean';
  });
  const exact = !!eph && coverage === 'exact';
  const frameShort = frame === 'rotating' ? 'Earth–Moon rotating frame' : exact ? 'Inertial · Earth-centred · DE440s basis' : coverage === 'outside' ? 'Inertial · mean-element fallback (cursor outside ephemeris span)' : 'Inertial · planar mean-element fallback';
  const frameLong =
    frame === 'rotating'
      ? 'Synodic frame: Earth–Moon line fixed on +x, Moon pinned at (1−μ, 0, 0); 1 unit = instantaneous Earth–Moon distance.'
      : exact
        ? 'Earth-centred view, axes = rotating frame at t0; the Earth–Moon line sweeps along the DE440s ephemeris.'
        : 'Earth-centred view with the planar mean-element Earth–Moon line (no ephemeris basis for this cursor).';
  const nObjects = useSelene((s) => (s.scenario ? new Set(s.scenario.frames.flatMap((f) => f.objects.map((o) => o.id))).size : 0));
  const motion = source === 'scenario' ? 'scenario frames' : source === 'idle' && meta ? `backend trajectories${tracksTotal && tracksLoaded < tracksTotal ? ` (${tracksLoaded}/${tracksTotal} loaded)` : ''}` : source === 'idle' ? 'browser CR3BP display propagation' : '—';
  const ephText = eph ? (exact ? `${eph.source ?? 'de440s'} · exact rotating basis` : coverage === 'outside' ? `${eph.source ?? 'de440s'} loaded for another span · cursor uses mean elements` : (eph.source ?? 'mean elements')) : 'loading…';
  return (
    <div className="hud" data-label-obstacle>
      <div className="hud-line">
        <span className={`frame-pill${frame === 'inertial' ? ' inertial' : ''}`}>{frame === 'rotating' ? 'ROT' : 'INR'}</span>
        <b className="trunc" title={frameLong}>
          {frameShort}
        </b>
        <button className={`hud-i${more ? ' open' : ''}`} onClick={() => setMore((v) => !v)} title={more ? 'Hide details' : 'Catalog, motion source and ephemeris details'} aria-expanded={more}>
          i
        </button>
      </div>
      <div className="hud-line">
        {scenario ? (
          <>
            <span className="muted">SCENARIO</span>
            <b className="trunc" title={scenario.meta.title}>
              {scenario.meta.title}
            </b>
            <span className="muted">T{fmtElapsed(tMin)}</span>
          </>
        ) : (
          <>
            <span className="muted">CATALOG</span>
            <b>{n}</b>
            {meta && (
              <span className="muted">
                {meta.n_simulated} simulated · {meta.n_real} real
              </span>
            )}
            <span className="muted">·</span>
            <b>{eph ? (exact ? (eph.source ?? 'de440s').toUpperCase() : 'mean elements') : 'ephemeris loading…'}</b>
          </>
        )}
      </div>
      {more && (
        <div className="hud-more">
          <div>{frameLong}</div>
          <div>
            Catalog <b>{n}</b>
            {meta ? ` (${meta.n_simulated} simulated · ${meta.n_real} real)` : ''} · motion <b>{motion}</b>
          </div>
          <div>
            Ephemeris <b>{ephText}</b>
          </div>
          {nParticles > 0 && (
            <div>
              Cloud <b>{nParticles.toLocaleString('en-US')} particles</b>
            </div>
          )}
          {scenario && (
            <div>
              {nObjects} scenario object{nObjects === 1 ? '' : 's'} drawn; the {n}-object catalog is hidden while the scenario plays.
            </div>
          )}
        </div>
      )}
      <MockLine />
    </div>
  );
}

/** Colour keys for what is on screen: custody (scenario), exclusion cones, clouds/reachable sets. */
function Keys({ hasClouds, hasReach }: { hasClouds: boolean; hasReach: boolean }) {
  const scenario = useSelene((s) => s.scenario);
  const excl = useSelene((s) => s.layers.exclusion);
  const fov = useSelene((s) => s.layers.fov);
  if (!scenario && !excl) return null;
  const ck = scenario?.meta.custody_km, lk = scenario?.meta.lost_km;
  const custodyTitle = ck && lk ? `Custody from the particle-cloud σ_pos: held < ${ck.toLocaleString('en-US')} km · degraded ${ck.toLocaleString('en-US')}–${lk.toLocaleString('en-US')} km · lost ≥ ${lk.toLocaleString('en-US')} km (thresholds from the scenario)` : 'Custody status per scenario frame (σ thresholds set by the scenario engine)';
  return (
    <div className="hud-keys" data-label-obstacle>
      {scenario && (
        <div className="key" title={custodyTitle}>
          <span className="lbl">custody</span>
          <span>
            <i style={{ background: 'var(--held)' }} />
            held
          </span>
          <span>
            <i style={{ background: 'var(--degraded)' }} />
            degraded
          </span>
          <span>
            <i style={{ background: 'var(--lost)' }} />
            lost
          </span>
        </div>
      )}
      {excl && (
        <div className="key" title="Exclusion cones: a sensor cannot observe inside these half-angles (Sun 40°, Moon/Earth 10° default; ground lunar glare 3–15° with phase)">
          <span className="lbl">exclusion</span>
          <span>
            <i className="sq" style={{ background: '#f5b700' }} />
            Sun
          </span>
          <span>
            <i className="sq" style={{ background: '#9aa4b2' }} />
            Moon
          </span>
          <span>
            <i className="sq" style={{ background: '#4cc9f0' }} />
            Earth
          </span>
        </div>
      )}
      {scenario && fov && (
        <div className="key" title="Teal wedge = field of view of a sensor observing the selected object; dashed line = line of sight of a tasked sensor; ◇ = space observer">
          <span className="lbl">sensors</span>
          <span>
            <i className="sq" style={{ background: 'rgba(45,212,191,0.35)', border: '1px solid #2dd4bf' }} />
            FOV wedge
          </span>
          <span>
            <i className="sq" style={{ background: 'transparent', borderTop: '2px dashed #2dd4bf', height: 0, width: 14, borderRadius: 0 }} />
            line of sight
          </span>
        </div>
      )}
      {scenario && (hasClouds || hasReach) && (
        <div className="key" title="Amber: uncertainty particle cloud (where the object probably is). Violet: reachable set under the assumed Δv budget (where it could go).">
          {hasClouds && (
            <span>
              <i style={{ background: '#ffb86b' }} />
              uncertainty cloud
            </span>
          )}
          {hasReach && (
            <span>
              <i style={{ background: '#9b5de5' }} />
              reachable set
            </span>
          )}
        </div>
      )}
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
        <span className="muted">1 unit = 384,400 km (L*)</span>
      </div>
    </div>
  );
}

function MockLine() {
  const backend = useBackendStatus();
  if (backend === 'offline') return <div className="hud-mock offline">BACKEND OFFLINE — MOCK DATA (browser CR3BP)</div>;
  return null;
}

/**
 * Presenter caption: the story event currently being narrated (store.caption — one event at a time, with a
 * wall-clock dwell each, fed by the demo driver as the cursor crosses events; see useDemoDriver.ts). Identical for
 * the backend bundle and the browser mock.
 */
function PresenterCaption() {
  const scenario = useSelene((s) => s.scenario);
  const ev: SeleneEvent | null = useSelene((s) => s.caption);
  if (!scenario || !ev) return null;
  return (
    <div className="caption" style={{ ['--sev' as string]: severityColor(ev.severity) }} data-label-obstacle key={`${ev.t}-${ev.kind}`}>
      <span className="kind">{kindLabel(ev.kind)}</span>
      <span className="h">{ev.headline ?? deriveHeadline(ev)}</span>
      <span className="t">T{fmtElapsed(ev.t)}</span>
    </div>
  );
}

export function OpsPage() {
  usePlayback();
  useDemoDriver();
  const layers = useSelene((s) => s.layers);
  const families = useSelene((s) => s.families);
  const sensorsCfg = useSelene((s) => s.sensors);
  const catalogN = useSelene((s) => s.catalog.length);
  const eph = useSelene((s) => s.ephemeris);
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
  const [hintFaded, setHintFaded] = useState(false);

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

  // Controls hint fades after 8 s or on the first interaction with the viewport.
  useEffect(() => {
    const id = window.setTimeout(() => setHintFaded(true), 8000);
    return () => window.clearTimeout(id);
  }, []);

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
  const loading = !eph || catalogN === 0 || (source === 'idle' && tracksTotal > 0 && tracksLoaded < tracksTotal);

  return (
    <div className="ops">
      <TopBar />
      <div className="viewport" onPointerDown={() => setHintFaded(true)}>
        <SceneRoot>
          <GroundSites visible={layers.sites} activeIds={activeSites} />
          <OrbitFamilies data={families} visible={layers.families} />
          <Objects objects={objects} showTrails={layers.trails} />
          {layers.clouds && clouds.map((c) => <ParticleCloud key={c.objectId} objectId={c.objectId} positions={c.positions} sigmaKm={c.sigmaKm} showLabel={layers.labels} />)}
          {layers.reach && <Reachable data={reachable} />}
          {layers.fov && <SensorFOVCones sensors={sensors} targets={targets} groundIds={groundIds} />}
          {layers.fov && <SensorMarkers sensors={sensors} groundIds={groundIds} showLabels={layers.labels} />}
          {layers.exclusion && <ExclusionCones sensors={sensors} config={sensorsCfg} groundIds={groundIds} />}
        </SceneRoot>
        {loading && <div className="loadbar" role="progressbar" aria-label="Loading scene data" />}
        <Hud source={source} nParticles={nParticles} tracksLoaded={tracksLoaded} tracksTotal={tracksTotal} />
        <Keys hasClouds={layers.clouds && clouds.length > 0} hasReach={layers.reach && !!reachable} />
        <PresenterCaption />
        <Toasts />
        <CameraPresets />
        <ScaleBar />
        <div className={`hud-br${hintFaded ? ' fade' : ''}`}>
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
