// The related-work surface (SPEC §6, issue #29): cards for the papers the
// agent's searches turned up, newest first. Presentational — App owns
// the polling and the doors. The panel is a surface, not context: these
// entries never enter per-turn assembly (§4's "never auto-stuffed"
// stays literally true). Cite is a typed turn through the existing
// approval flow; the panel itself never writes to the paper.
import type { RelatedEntry } from "../types";

interface Props {
  entries: RelatedEntry[];
  searching?: boolean;
  onPin: (docId: string) => void;
  onUnpin: (docId: string) => void;
  onCite: (entry: RelatedEntry) => void;
  // The librarian pass. Optional by the new-props rule: until the Find
  // door exists the caller passes nothing, and the empty state points
  // at the agent's searches instead.
  onFind?: () => void;
}

export function RelatedPanel({
  entries,
  searching = false,
  onPin,
  onUnpin,
  onCite,
  onFind,
}: Props) {
  return (
    <section className="related" aria-label="Related work">
      <div className="row head">
        <h2>Related work</h2>
        {onFind && (
          <button className="btn btn-sm" onClick={onFind} disabled={searching}>
            {searching ? "Searching…" : "Find papers"}
          </button>
        )}
      </div>
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
          <li
            key={`${e.arxiv}|${e.doi}|${e.title}|${e.authors[0] ?? ""}`}
            className="paper"
          >
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
            {e.abstract && <div className="abs">{e.abstract}</div>}
          </li>
        ))}
      </ul>
    </section>
  );
}
