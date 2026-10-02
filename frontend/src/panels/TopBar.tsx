/** Top navigation bar: wordmark, routes, frame toggle, layer toggles, family legend, Demo Scenario / Reset. */
import { useState } from 'react';
import { NavLink } from 'react-router-dom';
import { useBackendStatus } from '../api/client';
import { resetDemo, startDemo } from '../demo/useDemoDriver';
import { familyColor } from '../scene/constants';
import { useSelene, type Layers } from '../store/useSelene';

const LAYER_LABELS: { key: keyof Layers; label: string; title: string }[] = [
  { key: 'families', label: 'Families', title: 'Periodic-orbit families' },
  { key: 'lagrange', label: 'L-pts', title: 'Lagrange points' },
  { key: 'trails', label: 'Trails', title: 'Object trails' },
  { key: 'clouds', label: 'Clouds', title: 'Uncertainty particle clouds' },
  { key: 'reach', label: 'Reach', title: 'Reachable set' },
  { key: 'fov', label: 'FOV', title: 'Sensor fields of view' },
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

function FamilyLegend() {
  const families = useSelene((s) => s.families);
  const highlight = useSelene((s) => s.highlightFamily);
  const setHighlight = useSelene((s) => s.setHighlightFamily);
  const [open, setOpen] = useState(false);
  if (!families || families.families.length === 0) return null;
  return (
    <div className="legend-chip" onMouseLeave={() => setOpen(false)}>
      <button className={`toggle${open || highlight ? ' active' : ''}`} onClick={() => setOpen((v) => !v)} title="Orbit-family legend (click a family to highlight)">
        <span className="swatches">
          {families.families.slice(0, 6).map((f, i) => (
            <i key={f.name} style={{ background: familyColor(f.name, i) }} />
          ))}
        </span>
        {highlight ?? 'Legend'}
      </button>
      {open && (
        <div className="legend-pop">
          {families.families.map((f, i) => (
            <div key={f.name} className={`row${highlight === f.name ? ' active' : ''}`} onClick={() => setHighlight(highlight === f.name ? null : f.name)}>
              <i style={{ background: familyColor(f.name, i) }} />
              <span className="name">
                {f.name}
                {f.members.some((m) => m.approximate) && (
                  <span className="tag warn" title="Members corrected from an approximate (non-literature) seed — shapes are right, parameters are not reference values" style={{ marginLeft: 6 }}>
                    approx.
                  </span>
                )}
              </span>
              <span className="mono muted">{f.members.length}</span>
            </div>
          ))}
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
  const backend = useBackendStatus();
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
            <button className={`toggle${frame === 'rotating' ? ' active' : ''}`} onClick={() => setFrame('rotating')} title="Earth–Moon rotating (synodic) frame: Earth–Moon line fixed on +x">
              Rotating
            </button>
            <button className={`toggle${frame === 'inertial' ? ' active' : ''}`} onClick={() => setFrame('inertial')} title="Inertial view: the synodic frame turns about +z by the Earth–Moon line angle (ephemeris when loaded)">
              Inertial
            </button>
          </div>
          <div className="group">
            <span className="label">Layers</span>
            {LAYER_LABELS.map(({ key, label, title }) => (
              <button key={key} className={`toggle${layers[key] ? ' active' : ''}`} onClick={() => toggleLayer(key)} title={title}>
                {label}
              </button>
            ))}
            <FamilyLegend />
          </div>
        </>
      )}
      <div className="spacer" />
      {err && (
        <span className="tag alert" title={err}>
          DEMO ERROR
        </span>
      )}
      <span
        className={`tag ${backend === 'online' ? 'ok' : backend === 'offline' ? 'alert' : backend === 'partial' ? 'warn' : 'muted'}`}
        title={backend === 'offline' ? 'Backend unreachable — browser mock data (CR3BP in the browser)' : backend === 'partial' ? 'Backend up; endpoints returning 404 are served from the browser mock' : undefined}
      >
        API {backend.toUpperCase()}
      </span>
      {controls && (
        <>
          {scenario && (
            <button className="toggle" onClick={() => resetDemo()} title="Unload the scenario and return to the idle timeline">
              ⟲ Reset
            </button>
          )}
          <button className="primary" onClick={runDemo} disabled={busy} title="Play the scripted 2-minute story">
            {busy ? 'Loading…' : scenario ? '▶ Replay Demo' : '▶ Demo Scenario'}
          </button>
        </>
      )}
    </header>
  );
}
