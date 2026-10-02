/** Inline 16 px SVG icons for the transport / action buttons (no reliance on glyph fallbacks in system fonts). */
import type { SVGProps } from 'react';

type Name = 'play' | 'pause' | 'rewind' | 'reset' | 'close' | 'copy' | 'download';

const PATHS: Record<Name, JSX.Element> = {
  play: <path d="M4 2.5v11l9-5.5z" fill="currentColor" />,
  pause: <path d="M4 3h3v10H4zM9 3h3v10H9z" fill="currentColor" />,
  rewind: <path d="M3 3h2v10H3zM13 3v10L6 8z" fill="currentColor" />,
  reset: <path d="M8 3a5 5 0 1 1-4.6 3H1.9A6.1 6.1 0 1 0 8 1.9V0L4.5 2.5 8 5z" fill="currentColor" />,
  close: <path d="M3.5 3.5l9 9M12.5 3.5l-9 9" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" />,
  copy: <path d="M5 5h8v8H5zM3 3h8v1.5H4.5V11H3z" fill="none" stroke="currentColor" strokeWidth="1.3" />,
  download: <path d="M8 2v8M4.5 6.5L8 10l3.5-3.5M3 12.5h10" fill="none" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" strokeLinejoin="round" />,
};

export function Icon({ name, size = 14, ...rest }: { name: Name; size?: number } & SVGProps<SVGSVGElement>) {
  return (
    <svg width={size} height={size} viewBox="0 0 16 16" aria-hidden="true" focusable="false" style={{ verticalAlign: '-2px', marginRight: 5 }} {...rest}>
      {PATHS[name]}
    </svg>
  );
}
