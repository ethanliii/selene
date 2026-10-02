/** Top navigation bar: wordmark, routes, frame toggle, layer toggles, Demo Scenario button. */
import { useState } from 'react';
import { NavLink } from 'react-router-dom';
import { api, useBackendStatus } from '../api/client';
import { useSelene, type Layers } from '../store/useSelene';

const LAYER_LABELS: { key: keyof Layers; label: string }[] = [
  { key: 'families', label: 'Families' },
  { key: 'lagrange', label: 'L-pts' },
  { key: 'trails', label: 'Trails' },
  { key: 'clouds', label: 'Clouds' },
  { key: 'fov', label: 'FOV' },
  { key: 'exclusion', label: 'Excl.' },
  { key: 'grid', label: 'Grid' },
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

interface Props {
  /** Show frame/layer/demo controls (Ops page only). */
  controls?: boolean;
}

export function TopBar({ controls = true }: Props) {
  const frame = useSelene((s) => s.frame);
  const setFrame = useSelene((s) => s.setFrame);
  const layers = useSelene((s) => s.layers);
  const toggleLayer = useSelene((s) => s.toggleLayer);
  const loadScenario = useSelene((s) => s.loadScenario);
  const setPlaying = useSelene((s) => s.setPlaying);
  const backend = useBackendStatus();
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState<string | null>(null);

  async function runDemo() {
    setBusy(true);
    setErr(null);
    try {
      const s = await api.demoScenario();
      loadScenario(s);
      setPlaying(true);
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
            <button className={`toggle${frame === 'rotating' ? ' active' : ''}`} onClick={() => setFrame('rotating')} title="Earth–Moon rotating (synodic) frame">
              Rotating
            </button>
            <button className={`toggle${frame === 'inertial' ? ' active' : ''}`} onClick={() => setFrame('inertial')} title="Inertial view (placeholder rotation until ephemeris wired)">
              Inertial
            </button>
          </div>
          <div className="group">
            <span className="label">Layers</span>
            {LAYER_LABELS.map(({ key, label }) => (
              <button key={key} className={`toggle${layers[key] ? ' active' : ''}`} onClick={() => toggleLayer(key)}>
                {label}
              </button>
            ))}
          </div>
        </>
      )}
      <div className="spacer" />
      {err && <span className="tag alert" title={err}>DEMO ERROR</span>}
      <span className={`tag ${backend === 'online' ? 'ok' : backend === 'offline' ? 'alert' : 'muted'}`}>
        API {backend.toUpperCase()}
      </span>
      {controls && (
        <button className="primary" onClick={runDemo} disabled={busy}>
          {busy ? 'Loading…' : '▶ Demo Scenario'}
        </button>
      )}
    </header>
  );
}
