/**
 * Sequential colour ramp for coverage magnitude, interpolated in OKLab so lightness is perceptually
 * monotone (single hue family = the app accent cyan, dark → light on the dark surface).
 * Exactly-zero ("blind") cells use a dedicated near-surface colour so that "no sensor" is visually
 * distinct from "low coverage".
 *
 * OKLab conversions: Björn Ottosson, "A perceptual color space for image processing" (2020).
 */

/** Blind ("no sensor") cells: dark red base + hatch stroke, so blind reads as a finding, not as empty background. */
export const BLIND_COLOR = '#2a0e12';
export const BLIND_HATCH = 'rgba(120, 40, 50, 0.55)';
/** Stops dark → light; OKLab L ≈ 0.33, 0.57, 0.78, 0.96 (monotone). */
const STOPS = ['#133b52', '#1f7ea8', '#4cc9f0', '#dff7ff'];

type Lab = [number, number, number];

function hexToRgb(hex: string): [number, number, number] {
  const h = hex.replace('#', '');
  return [parseInt(h.slice(0, 2), 16) / 255, parseInt(h.slice(2, 4), 16) / 255, parseInt(h.slice(4, 6), 16) / 255];
}
const toLinear = (c: number) => (c <= 0.04045 ? c / 12.92 : Math.pow((c + 0.055) / 1.055, 2.4));
const toSrgb = (c: number) => (c <= 0.0031308 ? 12.92 * c : 1.055 * Math.pow(c, 1 / 2.4) - 0.055);

function rgbToOklab([r, g, b]: [number, number, number]): Lab {
  const lr = toLinear(r);
  const lg = toLinear(g);
  const lb = toLinear(b);
  const l = Math.cbrt(0.4122214708 * lr + 0.5363325363 * lg + 0.0514459929 * lb);
  const m = Math.cbrt(0.2119034982 * lr + 0.6806995451 * lg + 0.1073969566 * lb);
  const s = Math.cbrt(0.0883024619 * lr + 0.2817188376 * lg + 0.6299787005 * lb);
  return [
    0.2104542553 * l + 0.793617785 * m - 0.0040720468 * s,
    1.9779984951 * l - 2.428592205 * m + 0.4505937099 * s,
    0.0259040371 * l + 0.7827717662 * m - 0.808675766 * s,
  ];
}

function oklabToRgb([L, a, b]: Lab): [number, number, number] {
  const l = Math.pow(L + 0.3963377774 * a + 0.2158037573 * b, 3);
  const m = Math.pow(L - 0.1055613458 * a - 0.0638541728 * b, 3);
  const s = Math.pow(L - 0.0894841775 * a - 1.291485548 * b, 3);
  const lr = 4.0767416621 * l - 3.3077115913 * m + 0.2309699292 * s;
  const lg = -1.2684380046 * l + 2.6097574011 * m - 0.3413193965 * s;
  const lb = -0.0041960863 * l - 0.7034186147 * m + 1.707614701 * s;
  const clamp = (v: number) => Math.max(0, Math.min(1, v));
  return [clamp(toSrgb(lr)), clamp(toSrgb(lg)), clamp(toSrgb(lb))];
}

const LAB_STOPS: Lab[] = STOPS.map((h) => rgbToOklab(hexToRgb(h)));

/** Lookup table of 256 RGB triplets (0..255) for u ∈ [0,1]. */
export const RAMP_LUT: Uint8ClampedArray = (() => {
  const lut = new Uint8ClampedArray(256 * 3);
  for (let i = 0; i < 256; i++) {
    const u = i / 255;
    const seg = Math.min(LAB_STOPS.length - 2, Math.floor(u * (LAB_STOPS.length - 1)));
    const f = u * (LAB_STOPS.length - 1) - seg;
    const A = LAB_STOPS[seg];
    const B = LAB_STOPS[seg + 1];
    const rgb = oklabToRgb([A[0] + (B[0] - A[0]) * f, A[1] + (B[1] - A[1]) * f, A[2] + (B[2] - A[2]) * f]);
    lut[i * 3] = Math.round(rgb[0] * 255);
    lut[i * 3 + 1] = Math.round(rgb[1] * 255);
    lut[i * 3 + 2] = Math.round(rgb[2] * 255);
  }
  return lut;
})();

/** CSS colour for a normalised value u ∈ [0,1]. */
export function rampCss(u: number): string {
  const i = Math.max(0, Math.min(255, Math.round(u * 255)));
  return `rgb(${RAMP_LUT[i * 3]},${RAMP_LUT[i * 3 + 1]},${RAMP_LUT[i * 3 + 2]})`;
}

/** CSS linear-gradient string for the colour bar (left = 0, right = 1). */
export function rampGradientCss(): string {
  const parts: string[] = [];
  for (let k = 0; k <= 10; k++) parts.push(`${rampCss(k / 10)} ${k * 10}%`);
  return `linear-gradient(90deg, ${parts.join(', ')})`;
}

/**
 * Fixed categorical palette for up to 4 architectures (dark surface #0b1016), validated with the
 * dataviz palette checker: lightness band PASS, chroma PASS, normal-vision floor 18.4 PASS,
 * contrast ≥ 3:1 PASS, CVD worst adjacent ΔE 7.6 (floor band → always paired with direct labels,
 * 2 px gaps and a legend). Indexed by the architecture's `slot`, which is assigned at creation and
 * kept for its lifetime, so deleting one architecture never recolours the others.
 */
export const ARCH_COLORS = ['#1b96c4', '#9b5de5', '#bd8c00', '#d66080'];
