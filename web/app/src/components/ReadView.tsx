// The read view (SPEC §Section view, simple server-side variant): the
// server's readable blocks rendered as prose. The never-drop contract
// means every block is content — math and raw blocks show their source
// in monospace, visibly, because unrendered beats dropped. This is the
// paper, not a proposal: DiffCard's "diffs are never rendered" stance
// is untouched.
import type { Block, DocSection } from "../types";

function BlockView({ b }: { b: Block }) {
  switch (b.kind) {
    case "heading": {
      const level = Math.min(3, Math.max(1, b.level));
      const Tag = `h${level}` as "h1" | "h2" | "h3";
      return <Tag>{b.text}</Tag>;
    }
    case "paragraph":
      return <p>{b.text}</p>;
    case "list": {
      const Tag = b.ordered ? "ol" : "ul";
      return (
        <Tag>
          {b.items.map((it, i) => (
            <li key={i}>{it}</li>
          ))}
        </Tag>
      );
    }
    case "math":
      return <pre className="math">{b.text}</pre>;
    case "caption":
      return (
        <figure>
          <figcaption>{b.text}</figcaption>
        </figure>
      );
    case "raw":
      return <pre className="raw">{b.text}</pre>;
  }
}

export function ReadView({ sections }: { sections: DocSection[] }) {
  if (sections.length === 0)
    return <p className="read empty">Nothing to read yet.</p>;
  return (
    <article className="read" aria-label="Read view">
      {sections.map((s) => (
        <section key={s.path} data-path={s.path}>
          {s.blocks.map((b, i) => (
            <BlockView key={i} b={b} />
          ))}
        </section>
      ))}
    </article>
  );
}
