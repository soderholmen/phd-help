// The related-work surface (SPEC §6, issue #29): cards for the papers the
// agent's searches turned up, newest first. Presentational — App owns
// the polling and the doors. The panel is a surface, not context: these
// entries never enter per-turn assembly (§4's "never auto-stuffed"
// stays literally true). Cite is a typed turn through the existing
// approval flow; the panel itself never writes to the paper. The steer
// box seeds the agent's pass (keywords + a focus hint) — it is not a
// search box: the user never searches, the agent does.
import { useState } from "react";
import type { RelatedEntry, RelatedPass } from "../types";

interface Props {
  entries: RelatedEntry[];
  searching?: boolean;
  onPin: (docId: string) => void;
  onUnpin: (docId: string) => void;
  onCite: (entry: RelatedEntry) => void;
  // The librarian pass (issue #29): App always passes it now the Find
  // door exists; optional by the new-props rule, and without it the
  // empty state points at the agent's searches instead (and the steer
  // box has nothing to steer).
  onFind?: (keywords: string[], focus: string) => void;
  // What the most recent pass was asked for — the panel says so, so
  // the list is never "papers from somewhere, once".
  lastPass?: RelatedPass | null;
}

// The abstract clamps to two lines (CSS) until opened; past this many
// characters a preview is worth a more/less, below it the clamp never
// bites and the button would only be noise.
const ABS_PREVIEW = 220;

const openUrl = (e: RelatedEntry) =>
  e.arxiv
    ? `https://arxiv.org/abs/${e.arxiv}`
    : e.doi
      ? `https://doi.org/${e.doi}`
      : "";

interface CardProps {
  entry: RelatedEntry;
  onPin: (docId: string) => void;
  onUnpin: (docId: string) => void;
  onCite: (entry: RelatedEntry) => void;
}

function Card({ entry: e, onPin, onUnpin, onCite }: CardProps) {
  const [open, setOpen] = useState(false);
  const url = openUrl(e);
  return (
    <li className="paper">
      <div className="row">
        {e.in_corpus && (
          <span className={`st ${e.in_corpus.status}`}>
            {e.in_corpus.status}
          </span>
        )}
        {e.cited && <span className="st cited">cited</span>}
        <span className="ttl">
          {e.title}
          {e.year ? ` (${e.year})` : ""}
        </span>
        <span className="acts">
          {url && (
            // where to find it (issue #29): the paper itself, in a new
            // tab — a link out, not a door, so it needs no handler
            <a
              className="btn btn-sm"
              href={url}
              target="_blank"
              rel="noopener noreferrer"
            >
              Open
            </a>
          )}
          {e.in_corpus &&
            (e.in_corpus.pinned_here ? (
              <button
                className="btn btn-sm"
                onClick={() => onUnpin(e.in_corpus!.doc_id)}
              >
                Unpin
              </button>
            ) : (
              <button
                className="btn btn-sm"
                onClick={() => onPin(e.in_corpus!.doc_id)}
              >
                Pin
              </button>
            ))}
          {!e.cited && (
            <button className="btn btn-sm" onClick={() => onCite(e)}>
              Cite
            </button>
          )}
        </span>
      </div>
      {e.authors.length > 0 && (
        <div className="meta">{e.authors.join(", ")}</div>
      )}
      {e.why && <div className="why">{e.why}</div>}
      {e.query && <div className="meta">from “{e.query}”</div>}
      {e.abstract && (
        <>
          <div className={`abs${open ? " open" : ""}`}>{e.abstract}</div>
          {e.abstract.length > ABS_PREVIEW && (
            <button className="more" onClick={() => setOpen(!open)}>
              {open ? "less" : "more"}
            </button>
          )}
        </>
      )}
    </li>
  );
}

export function RelatedPanel({
  entries,
  searching = false,
  onPin,
  onUnpin,
  onCite,
  onFind,
  lastPass = null,
}: Props) {
  // The steer survives the 3 s ticks (the panel never unmounts) so a
  // half-typed hint is not eaten by a poll; it seeds the next pass,
  // and lastPass is what the last one actually ran with.
  const [kws, setKws] = useState("");
  const [focus, setFocus] = useState("");
  const run = () =>
    onFind?.(
      kws
        .split(",")
        .map((s) => s.trim())
        .filter(Boolean),
      focus.trim(),
    );
  return (
    <section className="related" aria-label="Related work">
      <div className="row head">
        <h2>Related work</h2>
        {onFind && (
          <button className="btn btn-sm" onClick={run} disabled={searching}>
            {searching ? "Searching…" : "Find papers"}
          </button>
        )}
      </div>
      {onFind && (
        <div className="steer">
          <input
            className="input"
            aria-label="Keywords"
            placeholder="keywords, comma separated"
            value={kws}
            onChange={(ev) => setKws(ev.target.value)}
          />
          <input
            className="input"
            aria-label="Where to search"
            placeholder="where to look (optional)"
            value={focus}
            onChange={(ev) => setFocus(ev.target.value)}
          />
        </div>
      )}
      {lastPass && (
        <div className="lastpass">
          last pass: {lastPass.keywords.join(", ") || "no keywords"}
          {lastPass.focus ? ` · ${lastPass.focus}` : ""} · {lastPass.at}
        </div>
      )}
      <ul>
        {entries.length === 0 && (
          <li className="empty">
            {onFind
              ? "no papers yet — Find papers runs a search pass"
              : "no papers yet — the agent's searches land here"}
          </li>
        )}
        {entries.map((e) => (
          // dedupe's identity is arXiv/DOI/title+first-author — the key
          // mirrors it so two id-less same-title twins can't collide
          <Card
            key={`${e.arxiv}|${e.doi}|${e.title}|${e.authors[0] ?? ""}`}
            entry={e}
            onPin={onPin}
            onUnpin={onUnpin}
            onCite={onCite}
          />
        ))}
      </ul>
    </section>
  );
}
