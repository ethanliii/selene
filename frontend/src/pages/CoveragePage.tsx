/**
 * Coverage page — placeholder.
 * TODO(later agent, M10): POST /api/coverage with a rotating-frame xy grid over [-0.2, 1.4] × [-0.8, 0.8],
 *   render values[t][i] as a heatmap (canvas or SVG rects) with a time slider bound to store.tSec,
 *   overlay Moon/L-points, and highlight blind spots (0 sensors) — lunar glare and daylight cones.
 */
import { TopBar } from '../panels/TopBar';

export function CoveragePage() {
  return (
    <div className="page">
      <TopBar controls={false} />
      <div className="page-body">
        <div className="card span">
          <h2>Coverage &amp; Blind Spots</h2>
          <p>
            Time-resolved heatmap of how many sensors can detect a reference object at each point of the cislunar volume, accounting for
            photometric detectability (size, albedo, phase angle, range vs. limiting magnitude), Sun/Moon/Earth exclusion angles, Earth
            shadow and eclipse, field of view, and ground-site night constraints. Blind spots are the market problem, not something to hide.
          </p>
          <p className="todo">TODO(M10): POST /api/coverage · heatmap renderer · time slider · sensor subset selector</p>
        </div>
        <div className="card">
          <h3>Grid</h3>
          <p>Rotating-frame xy slice through the Earth–Moon plane (optionally z stacks), nondimensional units, 1 L* = 384 400 km.</p>
        </div>
        <div className="card">
          <h3>Expected structure</h3>
          <p>Daylight cone around the anti-ground-site direction, lunar-glare disc around the Moon, and Earth-shadow cylinder — moving with time.</p>
        </div>
      </div>
    </div>
  );
}
