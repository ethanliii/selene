/**
 * Canvas heatmap of coverage over the rotating-frame xy slice, with overlays (Earth, Moon, L1–L5,
 * schematic DRO / NRHO / halo outlines, Sun direction) and a per-cell hover tooltip.
 * Cell index convention: i = ix * ny + iy (x-major), see types.ts.
 */
import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { LAGRANGE_ND, L_STAR_KM, MU, ORBITS, orbitOutlineND, R_EARTH_KM, R_MOON_KM } from './model';
import { BLIND_COLOR, RAMP_LUT, rampGradientCss } from './ramp';
import { REASON_CODES, REASON_LABELS, type BlindReason, type Vec3 } from './types';

export interface HeatmapProps {
  x: number[];
  y: number[];
  /** One value per cell, already reduced for the current mode. */
  values: number[] | Float32Array;
  /** Dominant blind reason code per cell (optional). */
  reasons?: number[] | Uint8Array;
  /** Value mapped to the top of the ramp. */
  vmax: number;
  /** Colour-bar caption and formatter. */
  caption: string;
  format: (v: number) => string;
  /** Explicit colour-bar tick values (defaults to quarter points of vmax) and their short formatter. */
  ticks?: number[];
  tickFormat?: (v: number) => string;
  /** Barycentric Sun unit vector for the displayed epoch (undefined in averaged mode). */
  sunDir?: Vec3;
  overlayFamilies: boolean;
}

interface Hover {
  px: number;
  py: number;
  xnd: number;
  ynd: number;
  value: number;
  reason: BlindReason | null;
}

export function CoverageHeatmap({ x, y, values, reasons, vmax, caption, format, ticks, tickFormat, sunDir, overlayFamilies }: HeatmapProps) {
  const wrapRef = useRef<HTMLDivElement>(null);
  const barRef = useRef<HTMLDivElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [size, setSize] = useState({ w: 600, h: 600 });
  const [hover, setHover] = useState<Hover | null>(null);

  const nx = x.length;
  const ny = y.length;
  const xmin = x[0];
  const xmax = x[nx - 1];
  const ymin = y[0];
  const ymax = y[ny - 1];
  // half-cell padding so cell centres sit on the sample coordinates
  const dx = nx > 1 ? (xmax - xmin) / (nx - 1) : 1;
  const dy = ny > 1 ? (ymax - ymin) / (ny - 1) : 1;
  const bx0 = xmin - dx / 2;
  const bx1 = xmax + dx / 2;
  const by0 = ymin - dy / 2;
  const by1 = ymax + dy / 2;

  // Fit a box with the grid's aspect ratio inside the wrapper.
  useEffect(() => {
    const el = wrapRef.current;
    if (!el) return;
    const ro = new ResizeObserver(() => {
      const r = el.getBoundingClientRect();
      const aspect = (bx1 - bx0) / (by1 - by0);
      const barH = (barRef.current?.offsetHeight ?? 44) + 10; // colour bar + gap
      let w = Math.max(200, r.width);
      let h = w / aspect;
      const maxH = Math.max(200, r.height - barH);
      if (h > maxH) {
        h = maxH;
        w = h * aspect;
      }
      setSize({ w: Math.floor(w), h: Math.floor(h) });
    });
    ro.observe(el);
    return () => ro.disconnect();
  }, [bx0, bx1, by0, by1]);

  const toPx = useCallback(
    (xnd: number, ynd: number): [number, number] => [((xnd - bx0) / (bx1 - bx0)) * size.w, (1 - (ynd - by0) / (by1 - by0)) * size.h],
    [bx0, bx1, by0, by1, size.w, size.h],
  );

  const outlines = useMemo(
    () => ({
      dro: orbitOutlineND(ORBITS.DRO),
      nrho: orbitOutlineND(ORBITS.NRHO_9_2, 400),
      l1: orbitOutlineND(ORBITS.L1_halo),
      l2: orbitOutlineND(ORBITS.L2_halo),
    }),
    [],
  );

  // Offscreen image of the field at grid resolution.
  const field = useMemo(() => {
    const img = new ImageData(nx, ny);
    const d = img.data;
    const blind = [7, 10, 16];
    for (let ix = 0; ix < nx; ix++) {
      for (let iy = 0; iy < ny; iy++) {
        const v = values[ix * ny + iy] ?? 0;
        const row = ny - 1 - iy;
        const o = (row * nx + ix) * 4;
        if (v <= 0) {
          d[o] = blind[0];
          d[o + 1] = blind[1];
          d[o + 2] = blind[2];
        } else {
          const u = Math.max(0, Math.min(1, v / Math.max(vmax, 1e-9)));
          const k = Math.round(u * 255) * 3;
          d[o] = RAMP_LUT[k];
          d[o + 1] = RAMP_LUT[k + 1];
          d[o + 2] = RAMP_LUT[k + 2];
        }
        d[o + 3] = 255;
      }
    }
    return img;
  }, [values, nx, ny, vmax]);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const dpr = window.devicePixelRatio || 1;
    canvas.width = Math.round(size.w * dpr);
    canvas.height = Math.round(size.h * dpr);
    const ctx = canvas.getContext('2d');
    if (!ctx) return;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.fillStyle = BLIND_COLOR;
    ctx.fillRect(0, 0, size.w, size.h);

    // field (nearest-neighbour so cells stay crisp)
    const off = document.createElement('canvas');
    off.width = nx;
    off.height = ny;
    off.getContext('2d')?.putImageData(field, 0, 0);
    ctx.imageSmoothingEnabled = false;
    ctx.drawImage(off, 0, 0, size.w, size.h);
    ctx.imageSmoothingEnabled = true;

    const mono = '10px "IBM Plex Mono", ui-monospace, Menlo, monospace';
    ctx.font = mono;
    ctx.textBaseline = 'middle';
    /** Text on a dark backing box so labels stay legible over bright cells. */
    const text = (t: string, px: number, py: number, color: string) => {
      const w = ctx.measureText(t).width;
      const top = ctx.textBaseline === 'bottom' ? py - 11 : py - 6;
      ctx.fillStyle = 'rgba(5, 7, 11, 0.72)';
      ctx.fillRect(px - 2, top, w + 4, 12);
      ctx.fillStyle = color;
      ctx.fillText(t, px, py);
    };

    const poly = (pts: [number, number][], stroke: string, dash: number[] = [], width = 1) => {
      ctx.beginPath();
      pts.forEach(([px, py], i) => {
        const [cx, cy] = toPx(px, py);
        if (i === 0) ctx.moveTo(cx, cy);
        else ctx.lineTo(cx, cy);
      });
      ctx.setLineDash(dash);
      // dark under-stroke keeps the outline visible over bright cells
      ctx.lineWidth = width + 2;
      ctx.strokeStyle = 'rgba(5, 7, 11, 0.6)';
      ctx.stroke();
      ctx.lineWidth = width;
      ctx.strokeStyle = stroke;
      ctx.stroke();
      ctx.setLineDash([]);
    };

    if (overlayFamilies) {
      poly(outlines.dro, 'rgba(214, 226, 240, 0.75)', [4, 3]);
      poly(outlines.l1, 'rgba(214, 226, 240, 0.45)', [2, 3]);
      poly(outlines.l2, 'rgba(214, 226, 240, 0.45)', [2, 3]);
      poly(outlines.nrho, 'rgba(255, 184, 107, 0.9)', [], 2);
      const label = (xnd: number, ynd: number, t: string, color = 'rgba(214,226,240,0.85)', dxp = 6, dyp = 0) => {
        const [px, py] = toPx(xnd, ynd);
        text(t, px + dxp, py + dyp, color);
      };
      label(1 - MU + 70000 / L_STAR_KM, -135000 / L_STAR_KM, 'DRO', 'rgba(214,226,240,0.85)', 4, 8);
      label(1 - MU + 0.01, -0.03, 'NRHO (xy proj.)', 'rgba(255,184,107,0.95)', 6, 10);
      label(0.836915 - 0.06, 0.2, 'L1 halo', 'rgba(214,226,240,0.6)', -14, -8);
      label(1.155682 + 0.08, 0.25, 'L2 halo (S)', 'rgba(214,226,240,0.6)', 4, -8);
    }

    // Earth & Moon (to scale, min 3 px radius)
    const disc = (xnd: number, ynd: number, rnd: number, fill: string, stroke: string) => {
      const [px, py] = toPx(xnd, ynd);
      const rp = Math.max(3, (rnd / (bx1 - bx0)) * size.w);
      ctx.beginPath();
      ctx.arc(px, py, rp, 0, Math.PI * 2);
      ctx.fillStyle = fill;
      ctx.fill();
      ctx.lineWidth = 1;
      ctx.strokeStyle = stroke;
      ctx.stroke();
    };
    disc(-MU, 0, R_EARTH_KM / L_STAR_KM, '#0e3a5a', '#2a9fd6');
    disc(1 - MU, 0, R_MOON_KM / L_STAR_KM, '#8a8f99', '#d6e2f0');
    const [ex, ey] = toPx(-MU, 0);
    text('Earth', ex + 9, ey - 9, '#d6e2f0');
    const [mx, my] = toPx(1 - MU, 0);
    text('Moon', mx + 8, my - 9, '#d6e2f0');

    // Lagrange points
    for (const { name, pos } of LAGRANGE_ND) {
      if (pos[0] < bx0 || pos[0] > bx1 || pos[1] < by0 || pos[1] > by1) continue;
      const [px, py] = toPx(pos[0], pos[1]);
      ctx.beginPath();
      ctx.moveTo(px, py - 4);
      ctx.lineTo(px + 4, py);
      ctx.lineTo(px, py + 4);
      ctx.lineTo(px - 4, py);
      ctx.closePath();
      ctx.strokeStyle = '#4cc9f0';
      ctx.lineWidth = 1.2;
      ctx.stroke();
      text(name, px + 7, py + (name === 'L1' ? 10 : -9), '#4cc9f0');
    }

    // Sun direction compass (top-left)
    if (sunDir) {
      const cx = 34;
      const cy = 34;
      const r = 22;
      ctx.beginPath();
      ctx.arc(cx, cy, r, 0, Math.PI * 2);
      ctx.strokeStyle = 'rgba(124,141,163,0.6)';
      ctx.lineWidth = 1;
      ctx.stroke();
      const ax = cx + sunDir[0] * r;
      const ay = cy - sunDir[1] * r;
      ctx.beginPath();
      ctx.moveTo(cx, cy);
      ctx.lineTo(ax, ay);
      ctx.strokeStyle = '#f5b700';
      ctx.lineWidth = 2;
      ctx.stroke();
      ctx.beginPath();
      ctx.arc(ax, ay, 3, 0, Math.PI * 2);
      ctx.fillStyle = '#f5b700';
      ctx.fill();
      // mean-synodic-phase Sun direction (±7°, see model.ts sunDirAt)
      text('SUN ±7°', cx + r + 6, cy, '#f5b700');
    }

    // axes ticks (nondimensional L*). Step 0.2 for narrow grids, 0.4 for the wide (±1.6) slice so
    // labels never overlap; x labels are clamped inside the canvas, and y labels skip the bottom
    // 16 px where the x-label row sits.
    const span = bx1 - bx0;
    const step = span > 2.4 ? 0.4 : 0.2;
    const tickColor = 'rgba(214,226,240,0.9)';
    const fmtTick = (v: number) => (Math.abs(v) < 1e-9 ? '0' : v.toFixed(1));
    ctx.textBaseline = 'bottom';
    for (let v = Math.ceil(bx0 / step - 1e-9) * step; v <= bx1 + 1e-9; v += step) {
      const [px] = toPx(v, 0);
      const label = fmtTick(v);
      const tw = ctx.measureText(label).width;
      ctx.fillStyle = tickColor;
      ctx.fillRect(px, size.h - 6, 1, 6);
      const lx = Math.max(2, Math.min(size.w - tw - 2, px - tw / 2));
      text(label, lx, size.h - 7, tickColor);
    }
    ctx.textBaseline = 'middle';
    for (let v = Math.ceil(by0 / step - 1e-9) * step; v <= by1 + 1e-9; v += step) {
      const [, py] = toPx(0, v);
      ctx.fillStyle = tickColor;
      ctx.fillRect(0, py, 6, 1);
      if (py > size.h - 18 || py < 8) continue; // would collide with the x-label row / top edge
      text(fmtTick(v), 8, py, tickColor);
    }
  }, [field, size, nx, ny, toPx, outlines, sunDir, overlayFamilies, bx0, bx1, by0, by1]);

  function onMove(e: React.MouseEvent<HTMLCanvasElement>) {
    const r = e.currentTarget.getBoundingClientRect();
    const px = e.clientX - r.left;
    const py = e.clientY - r.top;
    const xnd = bx0 + (px / size.w) * (bx1 - bx0);
    const ynd = by1 - (py / size.h) * (by1 - by0);
    const ix = Math.round((xnd - xmin) / dx);
    const iy = Math.round((ynd - ymin) / dy);
    if (ix < 0 || ix >= nx || iy < 0 || iy >= ny) {
      setHover(null);
      return;
    }
    const i = ix * ny + iy;
    const code = reasons ? reasons[i] : undefined;
    setHover({ px, py, xnd: x[ix], ynd: y[iy], value: values[i] ?? 0, reason: code === undefined ? null : REASON_CODES[code] ?? null });
  }

  const tipRight = hover ? hover.px > size.w * 0.6 : false;
  return (
    <div className="heat-wrap" ref={wrapRef}>
      <div className="heat-stage" style={{ width: size.w, height: size.h }}>
        <canvas ref={canvasRef} style={{ width: size.w, height: size.h }} onMouseMove={onMove} onMouseLeave={() => setHover(null)} aria-label="Coverage heatmap" role="img" />
        {hover && (
          <div className="heat-tip mono" style={{ left: tipRight ? undefined : hover.px + 14, right: tipRight ? size.w - hover.px + 14 : undefined, top: Math.max(0, hover.py - 10) }}>
            <div>
              x {hover.xnd.toFixed(3)} y {hover.ynd.toFixed(3)} L*
            </div>
            <div className="muted">
              ({(hover.xnd * L_STAR_KM).toFixed(0)} km, {(hover.ynd * L_STAR_KM).toFixed(0)} km)
            </div>
            <div>
              <b>{format(hover.value)}</b>
            </div>
            {hover.value <= 0 && hover.reason && <div className="tip-reason">{REASON_LABELS[hover.reason]}</div>}
          </div>
        )}
      </div>
      <div className="colorbar" ref={barRef}>
        <span className="cb-min mono">blind</span>
        <span className="cb-swatch" style={{ background: BLIND_COLOR }} aria-hidden />
        <span className="cb-ramp" style={{ background: rampGradientCss() }} aria-hidden />
        <span className="cb-ticks mono" aria-hidden>
          {(ticks ?? [0.25, 0.5, 0.75, 1].map((f) => f * vmax)).map((v, i, arr) => (
            <span key={i} style={{ position: 'absolute', left: `${(100 * v) / Math.max(vmax, 1e-9)}%`, transform: i === arr.length - 1 ? 'translateX(-100%)' : 'translateX(-50%)' }}>
              {(tickFormat ?? format)(v)}
            </span>
          ))}
        </span>
        <span className="cb-caption">
          {caption} · axes x, y in L* (1 L* = 384 400 km)
        </span>
      </div>
    </div>
  );
}
