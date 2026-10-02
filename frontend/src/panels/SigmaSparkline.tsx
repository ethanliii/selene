/**
 * Covariance sparkline: σ_pos = sqrt(trace P_pos) [km] vs time (hours), log scale when the range spans decades,
 * with the time cursor as a reference line. Data: per-frame `sigma_pos_km` of the selected object.
 */
import { Area, AreaChart, ReferenceLine, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts';

export interface SigmaPoint {
  h: number;
  sigma: number;
}

export function SigmaSparkline({ data, cursorH }: { data: SigmaPoint[]; cursorH: number }) {
  if (data.length < 2) return <p className="muted small">No covariance history for this object.</p>;
  const vals = data.map((d) => d.sigma).filter((v) => v > 0);
  const min = Math.min(...vals), max = Math.max(...vals);
  const log = max / Math.max(min, 1e-6) > 30;
  return (
    <div className="spark">
      <ResponsiveContainer width="100%" height={84}>
        <AreaChart data={data} margin={{ top: 6, right: 6, bottom: 0, left: 0 }}>
          <defs>
            <linearGradient id="sigFill" x1="0" y1="0" x2="0" y2="1">
              <stop offset="0%" stopColor="#ffb86b" stopOpacity={0.45} />
              <stop offset="100%" stopColor="#ffb86b" stopOpacity={0.02} />
            </linearGradient>
          </defs>
          <XAxis dataKey="h" type="number" domain={['dataMin', 'dataMax']} tick={{ fontSize: 9, fill: '#7c8da3' }} tickFormatter={(v: number) => `${v}h`} stroke="#1c2733" tickLine={false} />
          <YAxis
            scale={log ? 'log' : 'linear'}
            domain={log ? [Math.max(0.1, min / 2), max * 1.5] : [0, 'auto']}
            allowDataOverflow
            width={38}
            tick={{ fontSize: 9, fill: '#7c8da3' }}
            tickFormatter={(v: number) => (v >= 1000 ? `${(v / 1000).toFixed(0)}k` : v >= 10 ? v.toFixed(0) : v.toFixed(1))}
            stroke="#1c2733"
            tickLine={false}
          />
          <Tooltip
            contentStyle={{ background: '#0b1016', border: '1px solid #1c2733', fontSize: 11, fontFamily: 'var(--font-mono)' }}
            labelFormatter={(v) => `T+${Number(v).toFixed(1)} h`}
            formatter={(v: number) => [`${v >= 100 ? v.toFixed(0) : v.toFixed(1)} km`, 'σ_pos']}
          />
          <Area type="monotone" dataKey="sigma" stroke="#ffb86b" strokeWidth={1.4} fill="url(#sigFill)" isAnimationActive={false} dot={false} />
          <ReferenceLine x={cursorH} stroke="#4cc9f0" strokeDasharray="3 3" />
        </AreaChart>
      </ResponsiveContainer>
      <div className="spark-cap mono muted">σ_pos (√tr P) km vs T+h{log ? ' · log scale' : ''}</div>
    </div>
  );
}
