# Related work (issue #29): the panel, the recording pipeline, and what is still planned

The right sidebar's third surface: cards for the papers the agent's
searches turned up, newest first. The panel is a **surface, not
context** — related entries never enter per-turn assembly, so SPEC §4's
"never auto-stuffed" stays literally true. Cite from a card is a typed
turn through the existing §5 approval flow; the panel itself never
writes to the paper.

## The store

`.phd-helper/related.json` — `{"entries": [...]}`, capped at 30, atomic
tmp-replace (the `save_gists` rule). `src/phd_helper/related.py` holds
the whole pipeline as pure functions: `to_entry` (a hit, remembered),
`merge` (newest-first, deduped through `search.dedupe` — the search
loop's own identity rules, one copy — with the librarian's `why`, the
cited mark and provenance grafted forward from the entry being
replaced), `mark_cited` (against `refs.bib` via `bibtex.same_paper`),
`corpus_join` (against the registry via `corpus.owned_doc`), and
`parse_json_array` (the librarian's JSON intake, consumed when the pass
lands).

## The recording pipeline (live today)

- `tools.execute_async` grew an `on_results` hook: the `web_search`
  branch hands the **whole** hits — abstracts included — to the caller,
  best-effort (a store fault never sinks the tool result). The result
  dict to the model stays abstract-free: §4's budget is untouched.
- `app.record_related(project, hits, found_by)` is injected at the same
  call site as the corpus autojoin, with the **turn's own project**
  captured at the call site — a search belongs to the paper it was run
  for. It is sync on purpose: the whole read-modify-write runs without
  an await, so no other coroutine can interleave.
- `search.PaperHit` carries `abstract` (last field; arXiv `<summary>`,
  OpenAlex rebuilt from `abstract_inverted_index` — a sparse index
  reads as no abstract, never a half-sentence).
- `GET /project/related` → `{entries, searching}`: the store joined
  against `refs.bib` (cited) and the corpus registry (the Pin
  affordance) — the client joins nothing. Never an HTTPException.
- The shell polls it on the same 3 s tick as everything else. Pin and
  Unpin ride the corpus doors; Cite sends a typed turn naming the paper
  and its id.

## Planned (issue #29, ratified shape)

- **The librarian pass (slice 2).** One "Find papers" button →
  `POST /project/related/find` → a background task: the model reads the
  skeleton + selected section + what is already cited/found, plans 3–6
  keyword queries (JSON array, `parse_json_array`'s intake), searches
  them through the same academic-first stack, ranks its own top ≤10
  with a one-line `why` each, and auto-joins its arXiv-bearing picks
  (the corpus autojoin, unchanged). A plan that will not parse fails
  with a notice; a rank fault falls back to search order. One pass at a
  time: `AppState.related_task` is the guard the door's `searching`
  reads — today nothing starts a pass, so it is always `None` and the
  door reads false. The completion notice fans only to the sitting that
  started it.
- **The citation graph (slice 3).** OpenAlex `cited_by` over the
  references' DOIs (arXiv ids mapped to their DataCite DOIs), folded
  into the pass behind the query candidates, cards marked
  `found_by="graph"` with "cited by N of your references".

## Honest deviations (documented, not hidden)

- **SPEC §6's "No user-facing search UI"** is being amended through
  issue #29: the panel is read-only in slice 1, and the one Find button
  triggers the agent's own pass — the user never types a query, and
  there is no search box.
- **`searching` is a seam, not a feature yet.** The door's field and the
  `related_task` guard exist so slice 2 lands without a wire-shape
  change; today the field is constant false.
- **OpenAlex abstracts are spotty** — most works carry no inverted
  index; those cards fall back to title + why.
- **Cited-marking only sees ids.** `same_paper` matches arXiv id or
  DOI; a hand-written bib entry with neither keeps its card's Cite
  button — a false Cite is worse than a redundant one.
- **Concurrent writers are safe only via the await-free invariant** —
  commented at `record_related`; any future async rewrite must bring a
  lock.
- **The store caps at 30** (~45 KB): the panel is a shortlist, not an
  archive.
