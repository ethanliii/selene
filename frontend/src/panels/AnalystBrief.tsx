/**
 * Analyst brief: renders a markdown-ish string (headings '#', bullets '-'/'*', blank-line paragraphs,
 * **bold**) without a markdown dependency. Only the trusted backend produces this string.
 */
import { useMemo, type ReactNode } from 'react';
import { useSelene } from '../store/useSelene';

type Block = { type: 'h'; text: string } | { type: 'p'; text: string } | { type: 'ul'; items: string[] };

export function parseBrief(src: string): Block[] {
  const blocks: Block[] = [];
  let para: string[] = [];
  let list: string[] = [];
  const flush = () => {
    if (para.length) blocks.push({ type: 'p', text: para.join(' ') });
    if (list.length) blocks.push({ type: 'ul', items: list });
    para = [];
    list = [];
  };
  for (const raw of src.split(/\r?\n/)) {
    const line = raw.trim();
    if (!line) {
      flush();
      continue;
    }
    const h = /^#{1,6}\s+(.*)$/.exec(line);
    if (h) {
      flush();
      blocks.push({ type: 'h', text: h[1] });
      continue;
    }
    const li = /^[-*•]\s+(.*)$/.exec(line);
    if (li) {
      if (para.length) {
        blocks.push({ type: 'p', text: para.join(' ') });
        para = [];
      }
      list.push(li[1]);
      continue;
    }
    if (list.length) {
      blocks.push({ type: 'ul', items: list });
      list = [];
    }
    para.push(line);
  }
  flush();
  return blocks;
}

/** Inline **bold** and `code`. */
function inline(text: string): ReactNode[] {
  const parts = text.split(/(\*\*[^*]+\*\*|`[^`]+`)/g);
  return parts.map((p, i) => {
    if (p.startsWith('**') && p.endsWith('**')) return <strong key={i}>{p.slice(2, -2)}</strong>;
    if (p.startsWith('`') && p.endsWith('`')) return <code key={i} className="mono">{p.slice(1, -1)}</code>;
    return <span key={i}>{p}</span>;
  });
}

export function AnalystBrief({ text }: { text?: string }) {
  const storeBrief = useSelene((s) => s.brief);
  const src = text ?? storeBrief;
  const blocks = useMemo(() => parseBrief(src), [src]);
  return (
    <div className="brief">
      <h3 className="panel-title">Analyst brief</h3>
      {blocks.length === 0 ? (
        <p className="empty">The auto-generated brief appears here at the end of the demo scenario.</p>
      ) : (
        blocks.map((b, i) =>
          b.type === 'h' ? (
            <h4 key={i}>{b.text}</h4>
          ) : b.type === 'ul' ? (
            <ul key={i}>
              {b.items.map((it, j) => (
                <li key={j}>{inline(it)}</li>
              ))}
            </ul>
          ) : (
            <p key={i}>{inline(b.text)}</p>
          ),
        )
      )}
    </div>
  );
}
