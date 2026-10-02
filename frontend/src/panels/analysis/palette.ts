/**
 * Categorical series colours for the analysis charts and scene overlays: the dark-surface palette validated with
 * the dataviz validator against the panel surface #0b1016 (lightness band, chroma floor, CVD separation, normal-
 * vision floor, contrast all PASS in this fixed order). Assigned by entity in fixed order, never cycled: a 9th
 * entity folds into OTHER (grey). Text never wears these colours.
 */
export const CATEGORICAL = ['#3987e5', '#d95926', '#199e70', '#c98500', '#d55181', '#008300', '#9085e9', '#e66767'] as const;
export const OTHER = '#6b7684';

export function categorical(i: number): string {
  return i >= 0 && i < CATEGORICAL.length ? CATEGORICAL[i] : OTHER;
}

/** Reachability regions: a fixed slot per region key (the backend's order), so a region keeps its colour across runs. */
export const REGION_ORDER = ['l1_gateway', 'l2_gateway', 'nrho_corridor', 'south_pole_approach', 'llo_shell', 'llo_inner', 'geo_belt_return', 'earth_return_escape', 'lunar_impact'];
export function regionColor(key: string | null | undefined): string {
  if (!key) return OTHER;
  const i = REGION_ORDER.indexOf(key);
  return categorical(i);
}
/** Single-series / status colours (theme tokens; not categorical slots). */
export const SERIES = { sigma: '#ffb86b', truth: '#4cc9f0', nis: '#4cc9f0', nees: '#9b5de5', threshold: '#ff4d4f', familywise: '#f5b700', cursor: '#2dd4bf', burn: '#8d9db3' } as const;
