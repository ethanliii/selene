/**
 * Architecture Trade Studio — placeholder.
 * TODO(later agent, M10): sensor palette (GEO, L1 halo, L2 halo, DRO, resonant) draggable onto orbits
 *   in a small 3D/2D canvas; architecture list; "Run Monte Carlo" -> api.architectureEvaluate(...);
 *   side-by-side Recharts bar/radar comparison of coverage %, custody %, revisit h, detection latency.
 */
import { TopBar } from '../panels/TopBar';

export function ArchitecturePage() {
  return (
    <div className="page">
      <TopBar controls={false} />
      <div className="page-body">
        <div className="card span">
          <h2>Architecture Trade Studio</h2>
          <p>
            Place hypothetical space-based sensors in candidate orbits (GEO, L1 halo, L2 halo, DRO, resonant tours) and score each
            architecture by Monte Carlo on coverage %, custody %, revisit time, and maneuver-detection latency. Built for government
            architecture studies and prime-contractor proposal support.
          </p>
          <p className="todo">TODO(M10): sensor palette · orbit drop targets · POST /api/architecture/evaluate · side-by-side comparison charts</p>
        </div>
        <div className="card">
          <h3>Candidate orbits</h3>
          <p>GEO belt · L1 halo · L2 halo · DRO · 2:1 / 3:1 resonant. Each candidate rides a member of the periodic-orbit library.</p>
        </div>
        <div className="card">
          <h3>Metrics</h3>
          <p>Coverage % of the cislunar volume · custody % over the object set · mean revisit (h) · detection latency mean / p95 (h).</p>
        </div>
        <div className="card">
          <h3>Monte Carlo</h3>
          <p>Seeded runs over object sets and maneuver epochs; results are reproducible and reported honestly, including poor custody.</p>
        </div>
      </div>
    </div>
  );
}
