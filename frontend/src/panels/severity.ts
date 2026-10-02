import type { EventSeverity } from '../api/types';

/** Map event severity to a theme color (CSS custom property reference). */
export function severityColor(s: EventSeverity): string {
  switch (s) {
    case 'alert':
      return 'var(--alert)';
    case 'warn':
      return 'var(--warn)';
    case 'ok':
      return 'var(--ok)';
    default:
      return 'var(--accent)';
  }
}
