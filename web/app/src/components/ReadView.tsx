// The read view (SPEC §Section view, simple server-side variant): the
// server's readable blocks rendered as prose. The never-drop contract
// means every block is content — math renders through KaTeX with its
// raw source kept in the DOM behind the view (CSS hides it only when a
// view precedes it, and unparseable math has no view: unrendered beats
// dropped), and raw blocks show their source in monospace, visibly.
// This is the paper, not a proposal: DiffCard's "diffs are never
// rendered" stance is untouched.
//
// Editing: a block the kernel marked `editable` (pure-prose source) is
// click-to-edit. The open draft lives here, keyed by (path, start),
// with the span and base captured the moment it opened — so a 3 s tick
// that shifts the block list can never pair the draft with a different
// block's seam: the save posts the captured span, and an agent write
// that landed since the fetch bounces 409 instead of clobbering.
// Everything else edits through the source editor: per-block and
// per-section affordances both open it.
import { useState } from "react";
import type { KeyboardEvent, ReactNode } from "react";
import katex from "katex";
import "katex/dist/katex.min.css";
import type { Block, DocSection, FilePatch } from "../types";

interface Props {
  sections: DocSection[];
  onEditSource: (path: string) => void;
  onPatch: (patch: FilePatch) => Promise<void>;
}

// The kernel only ever sets `editable` on these; the `in` check is the
// type-level echo of that rule.
type TextBlock = Extract<Block, { text: string }>;

// The open draft: the block it opened on, seam frozen at open time.
// `orig` is the block's text at open — Escape only closes a draft
// that still equals it, so a key never silently eats typed work.
interface Draft {
  path: string;
  kind: string;
  start: number;
  end: number;
  base: string;
  text: string;
  orig: string;
}

function InlineDraft({
  draft,
  saving,
  onChange,
  onSave,
  onDiscard,
}: {
  draft: Draft;
  saving: boolean;
  onChange: (text: string) => void;
  onSave: () => void;
  onDiscard: () => void;
}) {
  return (
    <div className="edit-inline">
      <textarea
        autoFocus
        value={draft.text}
        rows={Math.min(12, draft.text.split("\n").length + 1)}
        onChange={(e) => onChange(e.target.value)}
        onKeyDown={(e) => {
          if (e.key === "Escape" && draft.text === draft.orig) onDiscard();
        }}
        aria-label={`Edit ${draft.kind}`}
      />
      <span className="edit-acts">
        <button className="btn btn-primary" onClick={onSave} disabled={saving}>
          Save
        </button>
        <button className="btn" onClick={onDiscard} disabled={saving}>
          Discard
        </button>
      </span>
    </div>
  );
}

function SourceLink({ onEditSource }: { onEditSource: () => void }) {
  return (
    <button className="edit-src" onClick={onEditSource}>
      edit source
    </button>
  );
}

// One KaTeX door for both surfaces: parse-or-null, never a throw.
// A null is the never-drop signal — the caller keeps the raw source.
function renderMath(tex: string, display: boolean): string | null {
  try {
    return katex.renderToString(tex, {
      displayMode: display,
      throwOnError: true,
    });
  } catch {
    return null;
  }
}

// The kernel keeps \$ literal in EVERY inline text (readable.py), so
// surfaces that do no math splitting still show it as a plain dollar.
const lit = (text: string) => text.replace(/\\\$/g, "$");

// Display math: \[..\] and $$..$$ arrive WITH their delimiters (the
// kernel keeps them as written); equation/align arrive as inner text.
// KaTeX wants the body alone. On a parse error there is no view at
// all — the raw pre stays visible, which is the never-drop shape.
function MathView({ text }: { text: string }) {
  let tex = text;
  if (tex.startsWith("\\[") && tex.endsWith("\\]")) tex = tex.slice(2, -2);
  else if (tex.startsWith("$$") && tex.endsWith("$$")) tex = tex.slice(2, -2);
  const html = renderMath(tex, true);
  return (
    <>
      {html !== null && (
        <div className="math-view" dangerouslySetInnerHTML={{ __html: html }} />
      )}
      <pre className="math">{text}</pre>
    </>
  );
}

// Unescaped $…$ is inline math; a \$ is money (see lit). A pair KaTeX
// cannot parse renders as written — never-drop again, locally.
const INLINE_MATH = /(?<!\\)\$([^$]+?)(?<!\\)\$/g;

function prose(text: string): ReactNode[] {
  const out: ReactNode[] = [];
  let last = 0;
  for (const m of text.matchAll(INLINE_MATH)) {
    const at = m.index ?? 0;
    if (at > last) out.push(lit(text.slice(last, at)));
    const html = renderMath(m[1], false);
    out.push(
      html === null ? (
        <span key={out.length}>{m[0]}</span>
      ) : (
        <span key={out.length} dangerouslySetInnerHTML={{ __html: html }} />
      ),
    );
    last = at + m[0].length;
  }
  if (out.length === 0) return [lit(text)];
  if (last < text.length) out.push(lit(text.slice(last)));
  return out;
}

function BlockView({
  b,
  path,
  draft,
  saving,
  onOpen,
  onChange,
  onSave,
  onDiscard,
  onEditSource,
}: {
  b: Block;
  path: string;
  draft: Draft | null;
  saving: boolean;
  onOpen: (b: TextBlock) => void;
  onChange: (text: string) => void;
  onSave: () => void;
  onDiscard: () => void;
  onEditSource: (path: string) => void;
}) {
  if (draft)
    return (
      <InlineDraft
        draft={draft}
        saving={saving}
        onChange={onChange}
        onSave={onSave}
        onDiscard={onDiscard}
      />
    );

  const open =
    b.editable && "text" in b ? () => onOpen(b) : undefined;
  // Keyboard parity with the click. No role override: an editable
  // heading must stay a heading (the tests, and screen readers, read
  // the paper's structure through that role).
  const keys = open
    ? {
        tabIndex: 0,
        onKeyDown: (e: KeyboardEvent) => {
          if (e.key === "Enter" || e.key === " ") {
            e.preventDefault(); // Space would scroll the prose
            open();
          }
        },
      }
    : {};
  const cls = b.editable ? "editable" : undefined;
  // The kernel's `editable` is the whole story: list/math/raw never
  // carry it, so everything not prose-editable offers the source.
  const src = !b.editable && (
    <SourceLink onEditSource={() => onEditSource(path)} />
  );

  switch (b.kind) {
    case "heading": {
      const level = Math.min(3, Math.max(1, b.level));
      const Tag = `h${level}` as "h1" | "h2" | "h3";
      return (
        <Tag onClick={open} {...keys} className={cls}>
          {lit(b.text)}
          {src}
        </Tag>
      );
    }
    case "paragraph":
      return (
        <p onClick={open} {...keys} className={cls}>
          {prose(b.text)}
          {src}
        </p>
      );
    case "list": {
      const Tag = b.ordered ? "ol" : "ul";
      return (
        <>
          <Tag>
            {b.items.map((it, i) => (
              <li key={i}>{lit(it)}</li>
            ))}
          </Tag>
          {src}
        </>
      );
    }
    case "math":
      return (
        <>
          <MathView text={b.text} />
          {src}
        </>
      );
    case "caption":
      return (
        <figure>
          <figcaption onClick={open} {...keys} className={cls}>
            {lit(b.text)}
          </figcaption>
          {src}
        </figure>
      );
    case "raw":
      return (
        <>
          <pre className="raw">{b.text}</pre>
          {src}
        </>
      );
  }
}

export function ReadView({ sections, onEditSource, onPatch }: Props) {
  const [draft, setDraft] = useState<Draft | null>(null);
  const [saving, setSaving] = useState(false);

  const save = async () => {
    if (!draft) return;
    setSaving(true);
    try {
      await onPatch({
        path: draft.path,
        start: draft.start,
        end: draft.end,
        base: draft.base,
        text: draft.text,
      });
      setDraft(null); // only a landed save closes the draft
    } catch {
      /* the 409 rode the transcript as a notice; the draft stays open */
    } finally {
      setSaving(false);
    }
  };

  if (sections.length === 0)
    return <p className="read empty">Nothing to read yet.</p>;
  return (
    <article className="read" aria-label="Read view">
      {sections.map((s) => {
        // A draft whose block vanished from the list (an agent write
        // shifted the file) still renders, at the section's end: the
        // user's text stays visible, and its save 409s honestly.
        const orphan =
          draft !== null &&
          draft.path === s.path &&
          !s.blocks.some((b) => b.start === draft.start);
        return (
          <section key={s.path} data-path={s.path}>
            <header className="sec-head">
              <button
                className="btn btn-ghost btn-sm"
                onClick={() => onEditSource(s.path)}
              >
                Edit source
              </button>
            </header>
            {s.blocks.map((b, i) => (
              <BlockView
                key={i}
                b={b}
                path={s.path}
                draft={
                  draft && draft.path === s.path && draft.start === b.start
                    ? draft
                    : null
                }
                saving={saving}
                onOpen={(blk) =>
                  setDraft({
                    path: s.path,
                    kind: blk.kind,
                    start: blk.start,
                    end: blk.end,
                    base: blk.base,
                    text: blk.text,
                    orig: blk.text,
                  })
                }
                onChange={(text) =>
                  setDraft((d) => (d ? { ...d, text } : d))
                }
                onSave={() => void save()}
                onDiscard={() => setDraft(null)}
                onEditSource={onEditSource}
              />
            ))}
            {orphan && draft && (
              <InlineDraft
                draft={draft}
                saving={saving}
                onChange={(text) =>
                  setDraft((d) => (d ? { ...d, text } : d))
                }
                onSave={() => void save()}
                onDiscard={() => setDraft(null)}
              />
            )}
          </section>
        );
      })}
    </article>
  );
}
