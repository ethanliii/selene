/**
 * 2D rotating-frame plot (SVG) of the Earth–Moon system for the Architecture Studio: the selected
 * architecture's sensor host orbits (schematic outlines from ./model.ts), the sensors' positions at
 * the reference epoch, and the notional target population (SIMULATED objects in DRO / 9:2 NRHO /
 * L1 halo / L2 halo). Identity is carried by direct labels, not colour: sensors are accent cyan,
 * targets are the SIMULATED amber token.
 */
import { useMemo } from 'react';
import { LAGRANGE_ND, L_STAR_KM, MU, ORBITS, orbitOutlineND, R_EARTH_KM, R_GEO_KM, R_MOON_KM, sunDirAt } from './model';
import { MOCK_T0_MS } from './mockArchitecture';
import { CANDIDATE_ORBITS, type ArchitectureDef, type CandidateOrbit } from './types';

// y range reaches past L4/L5 (y = ±0.866) so their markers sit inside the plot, clear of the caption and the x-axis.
const VB = { x0: -0.42, x1: 1.5, y0: -0.95, y1: 0.95 };
const W = 960;
const H = Math.round((W * (VB.y1 - VB.y0)) / (VB.x1 - VB.x0));
const sx = (x: number) => ((x - VB.x0) / (VB.x1 - VB.x0)) * W;
const sy = (y: number) => (1 - (y - VB.y0) / (VB.y1 - VB.y0)) * H;
const path = (pts: [number, number][]) => pts.map(([x, y], i) => `${i ? 'L' : 'M'}${sx(x).toFixed(1)},${sy(y).toFixed(1)}`).join(' ');

const TARGETS: { key: 'DRO' | 'NRHO_9_2' | 'L1_halo' | 'L2_halo'; label: string; phase: number; lx: number; ly: number }[] = [
  { key: 'DRO', label: 'SIM DRO objects', phase: 0.15, lx: 0.56, ly: -0.4 },
  { key: 'NRHO_9_2', label: 'SIM 9:2 NRHO (xy proj.)', phase: 0.5, lx: 1.0, ly: -0.07 },
  { key: 'L1_halo', label: 'SIM L1 halo', phase: 0.3, lx: 0.6, ly: 0.22 },
  { key: 'L2_halo', label: 'SIM L2 halo', phase: 0.7, lx: 1.22, ly: 0.28 },
];

const SENSOR_LABEL_POS: Record<CandidateOrbit, [number, number]> = {
  GEO: [-0.1, 0.14],
  L1_halo: [0.78, -0.24],
  L2_halo: [1.19, -0.14],
  DRO: [0.98, 0.39],
  resonant_3_1: [-0.3, 0.5],
};

export function SystemPlot({ arch }: { arch: ArchitectureDef | null }) {
  const outlines = useMemo(() => {
    const o: Record<string, [number, number][]> = {};
    for (const c of CANDIDATE_ORBITS) o[c.id] = orbitOutlineND(ORBITS[c.id], c.id === 'resonant_3_1' ? 720 : 240);
    for (const t of TARGETS) o[t.key] = orbitOutlineND(ORBITS[t.key], 400);
    return o;
  }, []);
  const sun = sunDirAt(MOCK_T0_MS);
  const usedOrbits = new Set(arch?.sensors.map((s) => s.orbit) ?? []);
  const scaleKm = 100000;
  const scaleW = sx(scaleKm / L_STAR_KM) - sx(0);

  return (
    <svg className="sysplot" viewBox={`0 0 ${W} ${H}`} role="img" aria-label="Rotating-frame plan view of the Earth–Moon system with sensor orbits and target population">
      <defs>
        <marker id="sun-arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto">
          <path d="M0,0 L10,5 L0,10 z" fill="#f5b700" />
        </marker>
      </defs>
      <rect x={0} y={0} width={W} height={H} fill="var(--bg)" />
      {/* grid */}
      {[-0.4, -0.2, 0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4].map((v) => (
        <g key={`gx${v}`}>
          <line x1={sx(v)} x2={sx(v)} y1={0} y2={H} stroke="var(--border)" strokeWidth={1} />
          <text x={sx(v) + 3} y={H - 6} className="sp-tick">
            {v.toFixed(1)}
          </text>
        </g>
      ))}
      {[-0.8, -0.6, -0.4, -0.2, 0, 0.2, 0.4, 0.6, 0.8].map((v) => (
        <g key={`gy${v}`}>
          <line y1={sy(v)} y2={sy(v)} x1={0} x2={W} stroke="var(--border)" strokeWidth={1} />
          <text x={4} y={sy(v) - 3} className="sp-tick">
            {v.toFixed(1)}
          </text>
        </g>
      ))}

      {/* target population */}
      {TARGETS.map((t) => (
        <g key={t.key}>
          <path d={path(outlines[t.key])} fill="none" stroke="var(--sim)" strokeWidth={t.key === 'NRHO_9_2' ? 3 : 1.5} strokeDasharray={t.key === 'NRHO_9_2' ? undefined : '6 5'} opacity={0.8} />
          {[0, 0.33, 0.66].map((ph) => {
            const p = ORBITS[t.key].pos(0, (t.phase + ph) % 1);
            return <circle key={ph} cx={sx(p[0] / L_STAR_KM)} cy={sy(p[1] / L_STAR_KM)} r={4} fill="var(--sim)" stroke="var(--bg)" strokeWidth={2} />;
          })}
          <text x={sx(t.lx)} y={sy(t.ly)} className="sp-label sim">
            {t.label}
          </text>
        </g>
      ))}

      {/* sensor host orbits */}
      {CANDIDATE_ORBITS.filter((c) => usedOrbits.has(c.id)).map((c) => (
        <g key={c.id}>
          <path d={path(outlines[c.id])} fill="none" stroke="var(--accent)" strokeWidth={c.id === 'GEO' ? 2.5 : 2} opacity={0.95} />
          <text x={sx(SENSOR_LABEL_POS[c.id][0])} y={sy(SENSOR_LABEL_POS[c.id][1])} className="sp-label accent">
            {c.id === 'L2_halo' ? 'L2 halo sensor' : `${c.label} sensor orbit`}
          </text>
        </g>
      ))}
      {arch?.sensors.map((s, i) => {
        const p = ORBITS[s.orbit].pos(0, s.phase);
        const px = sx(p[0] / L_STAR_KM);
        const py = sy(p[1] / L_STAR_KM);
        return (
          <g key={s.id}>
            <rect x={px - 6} y={py - 6} width={12} height={12} transform={`rotate(45 ${px} ${py})`} fill="var(--accent)" stroke="var(--bg)" strokeWidth={2} />
            <text x={px + 10} y={py - 8} className="sp-label accent">
              S{i + 1} · m
              <tspan baselineShift="sub" fontSize="11px">
                lim
              </tspan>{' '}
              {s.limiting_mag.toFixed(1)}
            </text>
          </g>
        );
      })}

      {/* Earth, Moon, ground network, L-points */}
      {arch?.ground_network && (
        <>
          <circle cx={sx(-MU)} cy={sy(0)} r={Math.max(10, (R_GEO_KM / L_STAR_KM / (VB.x1 - VB.x0)) * W * 0.55)} fill="none" stroke="var(--accent)" strokeWidth={1} strokeDasharray="2 3" opacity={0.6} />
          <text x={sx(-MU) - 60} y={sy(0) + 42} className="sp-label accent">
            3 notional ground sites
          </text>
        </>
      )}
      <circle cx={sx(-MU)} cy={sy(0)} r={Math.max(6, (R_EARTH_KM / L_STAR_KM / (VB.x1 - VB.x0)) * W)} fill="#0e3a5a" stroke="#2a9fd6" strokeWidth={1.5} />
      <text x={sx(-MU) + 12} y={sy(0) - 12} className="sp-label">
        Earth
      </text>
      <circle cx={sx(1 - MU)} cy={sy(0)} r={Math.max(4, (R_MOON_KM / L_STAR_KM / (VB.x1 - VB.x0)) * W)} fill="#8a8f99" stroke="#d6e2f0" strokeWidth={1} />
      <text x={sx(1 - MU) + 10} y={sy(0) - 10} className="sp-label">
        Moon
      </text>
      {LAGRANGE_ND.filter((l) => l.name !== 'L3').map((l) => (
        <g key={l.name}>
          <path d={`M${sx(l.pos[0])},${sy(l.pos[1]) - 5} l5,5 l-5,5 l-5,-5 z`} fill="none" stroke="var(--accent)" strokeWidth={1.2} />
          <text x={sx(l.pos[0]) + 8} y={sy(l.pos[1]) + 14} className="sp-label accent">
            {l.name}
          </text>
        </g>
      ))}

      {/* Sun direction + scale bar */}
      <g transform={`translate(${W - 70}, 46)`}>
        <circle r={26} fill="none" stroke="var(--muted)" strokeWidth={1} opacity={0.6} />
        <line x1={0} y1={0} x2={sun[0] * 24} y2={-sun[1] * 24} stroke="#f5b700" strokeWidth={2} markerEnd="url(#sun-arrow)" />
        <text x={-30} y={44} className="sp-label warn">
          Sun @ t₀
        </text>
      </g>
      <g transform={`translate(${W - 40 - scaleW}, ${H - 28})`}>
        <line x1={0} x2={scaleW} y1={0} y2={0} stroke="var(--text)" strokeWidth={2} />
        <line x1={0} x2={0} y1={-4} y2={4} stroke="var(--text)" strokeWidth={2} />
        <line x1={scaleW} x2={scaleW} y1={-4} y2={4} stroke="var(--text)" strokeWidth={2} />
        <text x={0} y={-8} className="sp-label">
          100 000 km
        </text>
      </g>
      <text x={12} y={18} className="sp-label muted">
        EARTH–MOON ROTATING FRAME · xy plane · schematic orbit outlines · SIMULATED population, notional actor
      </text>
    </svg>
  );
}
