/**
 * Top navigation bar: wordmark, routes, frame toggle, layer toggles (collapsed into a "Layers ▾" menu below
 * 1500 px), family legend, a quiet "DATA SOURCES ▾" chip whose popover lists which endpoints are LIVE / MOCK,
 * and Demo Scenario / Reset. The Demo button stays disabled until the ephemeris basis has loaded.
 */
import { useEffect, useRef, useState } from 'react';
import { NavLink } from 'react-router-dom';
import { useBackendStatus } from '../api/client';
import { resetDemo, startDemo } from '../demo/useDemoDriver';
import { familyColor, familyLabel, NRHO_COLOR } from '../scene/constants';
import { isNrho92, thinnedMembers } from '../scene/OrbitFamilies';
import { useSelene, type Layers } from '../store/useSelene';
import { Icon } from './icons';
import { useLiveMockLists } from './LiveStatus';

const LAYER_LABELS: { key: keyof Layers; label: string; title: string }[] = [
  { key: 'families', label: 'Families', title: 'Periodic-orbit families' },
  { key: 'lagrange', label: 'L-pts', title: 'Lagrange points' },
  { key: 'trails', label: 'Trails', title: 'Object trails' },
  { key: 'clouds', label: 'Clouds', title: 'Uncertainty particle clouds' },
  { key: 'reach', label: 'Reach', title: 'Reachable set' },
  { key: 'fov', label: 'FOV', title: 'Sensor fields of view (tasked sensors) and space-observer markers' },
  { key: 'exclusion', label: 'Excl.', title: 'Sun/Moon/Earth exclusion cones' },
  { key: 'sites', label: 'Sites', title: 'Ground sites' },
  { key: 'labels', label: 'Labels', title: 'Scene labels' },
  { key: 'grid', label: 'Grid', title: 'Reference grid' },
];

export function Wordmark() {
  return (
    <div className="wordmark">
      <span className="name">SELENE</span>
      <span className="sub">Cislunar Space Domain Awareness</span>
    </div>
  );
}

export function NavLinks() {
  return (
    <nav>
      <NavLink to="/ops">Ops</NavLink>
      <NavLink to="/architecture">Architecture</NavLink>
      <NavLink to="/coverage">Coverage</NavLink>
    </nav>
  );
}

/** Closes a popover on outside click / Escape. */
function useDismiss(open: boolean, close: () => void) {
  const ref = useRef<HTMLDivElement>(null);
  useEffect(() => {
    if (!open) return;
    const onDown = (e: MouseEvent) => {
      if (ref.current && !ref.current.contains(e.target as Node)) close();
    };
    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape') close();
    };
    document.addEventListener('mousedown', onDown);
    document.addEventListener('keydown', onKey);
    return () => {
      document.removeEventListener('mousedown', onDown);
      document.removeEventListener('keydown', onKey);
    };
  }, [open, close]);
  return ref;
}

function FamilyLegend() {
  const families = useSelene((s) => s.families);
  const highlight = useSelene((s) => s.highlightFamily);
  const hidden = useSelene((s) => s.hiddenFamilies);
  const setHighlight = useSelene((s) => s.setHighlightFamily);
  const toggleVisible = useSelene((s) => s.toggleFamilyVisible);
  const setVisible = useSelene((s) => s.setFamilyVisible);
  const [open, setOpen] = useState(false);
  if (!families || families.families.length === 0) return null;
  const nrho = families.families.flatMap((f) => f.members).find(isNrho92);
  const nShown = families.families.filter((f) => !hidden.includes(f.name)).length;
  return (
    <div className="legend-chip" onMouseLeave={() => setOpen(false)}>
      <button className={`toggle${open || highlight ? ' active' : ''}`} onClick={() => setOpen((v) => !v)} title="Orbit-family legend: show/hide families, click a name to highlight">
        <span className="swatches">
          {families.families.slice(0, 9).map((f, i) => (
            <i key={f.name} style={{ background: familyColor(f.name, i), opacity: hidden.includes(f.name) ? 0.25 : 1 }} />
          ))}
        </span>
        <span className="txt">{highlight ? familyLabel(highlight) : `Families (${nShown}/${families.families.length}) ▾`}</span>
      </button>
      {open && (
        <div className="legend-pop">
          <div className="legend-head">
            <span className="muted" title="drawn / in family — the scene draws a thinned, evenly strided subset per family (NRHO members always)">
              {families.families.length} families · {families.families.reduce((a, f) => a + thinnedMembers(f).size, 0)} drawn / {families.families.reduce((a, f) => a + (f.n_total ?? f.members.length), 0)} orbits
            </span>
            <span>
              <button className="toggle" onClick={() => families.families.forEach((f) => setVisible(f.name, true))}>
                all
              </button>
              <button className="toggle" onClick={() => families.families.forEach((f) => setVisible(f.name, false))}>
                none
              </button>
            </span>
          </div>
          {families.families.map((f, i) => {
            const off = hidden.includes(f.name);
            return (
              <div key={f.name} className={`row${highlight === f.name ? ' active' : ''}${off ? ' off' : ''}`}>
                <button className={`eye${off ? '' : ' on'}`} onClick={() => toggleVisible(f.name)} title={off ? 'Show family' : 'Hide family'} aria-pressed={!off}>
                  {off ? '○' : '●'}
                </button>
                <i style={{ background: familyColor(f.name, i) }} />
                <span className="name" onClick={() => setHighlight(highlight === f.name ? null : f.name)} title={(f.references ?? []).join('\n') || 'Click to highlight'}>
                  {familyLabel(f.name)}
                  {f.members.some((m) => m.approximate) && (
                    <span className="tag warn" title="Members corrected from an approximate (non-literature) seed — shapes are right, parameters are not reference values" style={{ marginLeft: 6 }}>
                      approx.
                    </span>
                  )}
                </span>
                <span
                  className="mono muted"
                  title={`${thinnedMembers(f).size} drawn (thinned for legibility) · ${f.members.length} loaded · ${f.n_total ?? f.members.length} in the family${f.period_days_range ? ` · period ${f.period_days_range[0].toFixed(1)}–${f.period_days_range[1].toFixed(1)} d` : ''}`}
                >
                  {thinnedMembers(f).size}/{f.n_total ?? f.members.length}
                </span>
              </div>
            );
          })}
          {nrho && (
            <div className="row info">
              <span />
              <i style={{ background: NRHO_COLOR }} />
              <span className="name">9:2 NRHO (highlighted)</span>
              <span className="mono muted">{(nrho.period_days ?? 0).toFixed(2)} d</span>
            </div>
          )}
          {highlight && (
            <div className="row clear" onClick={() => setHighlight(null)}>
              <span className="name muted">clear highlight</span>
            </div>
          )}
        </div>
      )}
    </div>
  );
}

/** Collapsed layer toggles (≤ 1500 px): one "Layers ▾" button with a checkbox list. */
function LayersMenu() {
  const layers = useSelene((s) => s.layers);
  const toggleLayer = useSelene((s) => s.toggleLayer);
  const [open, setOpen] = useState(false);
  const ref = useDismiss(open, () => setOpen(false));
  const n = LAYER_LABELS.filter((l) => layers[l.key]).length;
  return (
    <div className="layers-menu" ref={ref}>
      <button className={`toggle${open ? ' active' : ''}`} onClick={() => setOpen((v) => !v)} title="Scene layers">
        Layers ({n}) ▾
      </button>
      {open && (
        <div className="layers-pop">
          {LAYER_LABELS.map(({ key, label, title }) => (
            <label key={key} title={title}>
              <input type="checkbox" checked={layers[key]} onChange={() => toggleLayer(key)} />
              {label}
              <span className="muted" style={{ marginLeft: 'auto', fontSize: 12 }}>
                {title}
              </span>
            </label>
          ))}
        </div>
      )}
    </div>
  );
}

/**
 * Quiet data-provenance chip. The dot is green when everything on screen is live, amber when any endpoint is served
 * by the browser mock, red when the backend is unreachable; the popover lists LIVE / AVAILABLE / MOCK endpoints
 * verbatim (the honesty strip that used to live in the banner).
 */
export function DataSourcesChip() {
  const backend = useBackendStatus();
  const { live, available, mock } = useLiveMockLists();
  const [open, setOpen] = useState(false);
  const ref = useDismiss(open, () => setOpen(false));
  const offline = backend === 'offline';
  const dot = offline ? 'off' : mock.length > 0 ? 'mock' : 'live';
  const title = offline ? 'Backend unreachable — all data from the browser mock (CR3BP in the browser)' : mock.length ? `Browser mock serving: ${mock.join(', ')}` : 'All data on screen fetched from the backend';
  return (
    <div className="src-chip" ref={ref}>
      <button className={offline ? 'offline' : ''} onClick={() => setOpen((v) => !v)} title={title} aria-haspopup="true" aria-expanded={open}>
        <span className={`dot ${dot}`} />
        {offline ? 'OFFLINE · MOCK ▾' : 'DATA SOURCES ▾'}
      </button>
      {open && (
        <div className="src-pop">
          {offline && <div className="row mock"><span className="k">OFFLINE</span><span className="v">backend unreachable — every endpoint served by the browser mock (CR3BP in the browser)</span></div>}
          {live.length > 0 && (
            <div className="row live">
              <span className="k">LIVE</span>
              <span className="v">{live.join(', ')}</span>
            </div>
          )}
          {available.length > 0 && (
            <div className="row avail">
              <span className="k">AVAILABLE</span>
              <span className="v">{available.join(', ')} <span className="muted">(route exists; not consumed by this screen yet)</span></span>
            </div>
          )}
          {mock.length > 0 && !offline && (
            <div className="row mock">
              <span className="k">MOCK</span>
              <span className="v">{mock.join(', ')} <span className="muted">(browser fallback)</span></span>
            </div>
          )}
          {live.length === 0 && available.length === 0 && mock.length === 0 && !offline && <div className="note">Probing the backend…</div>}
          <div className="note">LIVE = fetched from the backend in this session · MOCK = computed in the browser and labelled as such wherever it appears.</div>
        </div>
      )}
    </div>
  );
}

interface Props {
  /** Show frame/layer/demo controls (Ops page only). */
  controls?: boolean;
}

export function TopBar({ controls = true }: Props) {
  const frame = useSelene((s) => s.frame);
  const setFrame = useSelene((s) => s.setFrame);
  const layers = useSelene((s) => s.layers);
  const toggleLayer = useSelene((s) => s.toggleLayer);
  const scenario = useSelene((s) => s.scenario);
  const basisReady = useSelene((s) => s.ephemeris !== null);
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState<string | null>(null);

  async function runDemo() {
    setBusy(true);
    setErr(null);
    try {
      await startDemo();
    } catch (e) {
      setErr(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(false);
    }
  }

  return (
    <header className="topbar">
      <Wordmark />
      <NavLinks />
      {controls && (
        <>
          <div className="group">
            <span className="label">Frame</span>
            <button className={`toggle${frame === 'rotating' ? ' active' : ''}`} onClick={() => setFrame('rotating')} title="Earth–Moon rotating (synodic) frame: Earth–Moon line fixed on +x, Moon pinned at (1−μ, 0, 0)">
              Rotating
            </button>
            <button className={`toggle${frame === 'inertial' ? ' active' : ''}`} onClick={() => setFrame('inertial')} title="Inertial view: Earth-centred, axes = rotating frame at t0; the Earth–Moon line sweeps along the DE440s ephemeris">
              Inertial
            </button>
          </div>
          <div className="group">
            <span className="label">Layers</span>
            <span className="layers-inline">
              {LAYER_LABELS.map(({ key, label, title }) => (
                <button key={key} className={`toggle${layers[key] ? ' active' : ''}`} onClick={() => toggleLayer(key)} title={title}>
                  {label}
                </button>
              ))}
            </span>
            <LayersMenu />
            <FamilyLegend />
          </div>
        </>
      )}
      <div className="right">
        {err && (
          <span className="tag alert" title={err}>
            DEMO ERROR
          </span>
        )}
        <DataSourcesChip />
        {controls && (
          <>
            {scenario && (
              <button className="toggle" onClick={() => resetDemo()} title="Unload the scenario and return to the idle timeline">
                <Icon name="reset" />
                Reset
              </button>
            )}
            <button className="primary" onClick={runDemo} disabled={busy || !basisReady} title={basisReady ? 'Play the scripted 2-minute story' : 'Loading the ephemeris basis…'}>
              {busy || !basisReady ? (
                'Loading…'
              ) : (
                <>
                  <Icon name="play" />
                  {scenario ? 'Replay Demo' : 'Demo Scenario'}
                </>
              )}
            </button>
          </>
        )}
      </div>
    </header>
  );
}
