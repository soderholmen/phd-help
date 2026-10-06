# Related work (issue #29): the panel, the two pipelines, and what is still planned

The right sidebar's third surface: cards for the papers the agent's
searches turned up, newest first. The panel is a **surface, not
context** — related entries never enter per-turn assembly, so SPEC §4's
"never auto-stuffed" stays literally true. Cite from a card is a typed
turn through the existing §5 approval flow; the panel itself never
writes to the paper.

## The store

`.phd-helper/related.json` — `{"entries": [...], "last_pass": {...}}`,
entries capped at 30, atomic tmp-replace (the `save_gists` rule).
`last_pass` is the most recent librarian pass's steer — `{keywords,
focus, at}` — kept across entry-only saves, so a later plain search
does not erase what the librarian was asked for. Each entry carries
`query`: the search that surfaced it. `src/phd_helper/related.py` holds
the whole pipeline as pure functions: `to_entry` (a hit, remembered),
`merge` (newest-first, deduped through `search.dedupe` — the search
loop's own identity rules, one copy — with the librarian's `why`, the
cited mark and provenance grafted forward from the entry being
replaced), `mark_cited` (against `refs.bib` via `bibtex.same_paper`),
`corpus_join` (against the registry via `corpus.owned_doc`),
`drop_cited` (the librarian's pre-filter, sharing `mark_cited`'s
`_in_bib` test — one copy), and `parse_json_array` (the librarian's
JSON intake).

## The recording pipeline (live today)

- `tools.execute_async` grew an `on_results` hook: the `web_search`
  branch hands the **whole** hits — abstracts included — and the query
  that found them to the caller, best-effort (a store fault never sinks
  the tool result). The result dict to the model stays abstract-free:
  §4's budget is untouched.
- `app.record_related(project, hits, found_by, query)` is injected at
  the same call site as the corpus autojoin, with the **turn's own
  project** captured at the call site — a search belongs to the paper
  it was run for. It writes through `app.record_entries`, the store's
  one sync read-modify-write: no await inside, so no other coroutine
  can interleave.
- `search.PaperHit` carries `abstract` (last field; arXiv `<summary>`,
  OpenAlex rebuilt from `abstract_inverted_index` — a sparse index
  reads as no abstract, never a half-sentence).
- `GET /project/related` → `{entries, searching, last_pass}`: the store
  joined against `refs.bib` (cited) and the corpus registry (the Pin
  affordance) — the client joins nothing. Never an HTTPException.
- The shell polls it on the same 3 s tick as everything else. Pin and
  Unpin ride the corpus doors; Cite sends a typed turn naming the paper
  and its id. Each card opens its abstract (two-line clamp until
  "more"), links **Open** to the arXiv page (or the DOI when there is
  no arXiv id — no id, no link), and names the query that found it;
  the header says what the last pass searched for.

## The librarian pass (live today)

- One "Find papers" button → `POST /project/related/find` → a
  background task on `AppState.related_task`. One pass at a time: a
  second click while one runs is a no-op, not a queue, and the GET
  door's `searching` reads this guard rather than a stored flag.
- The optional steer body `{keywords, focus}` (`parse_steer`, defensive:
  a torn body is simply no steer) seeds the pass — the keywords and the
  focus hint reach the plan prompt, and the pass records them as
  `last_pass`. They steer the agent's searches; they do not replace
  them.
- `app.run_librarian(state, project, section, seeds, focus)` — the
  project and the sitting's anchor are captured at the click. The model
  reads `librarian_context` (skeleton + anchored section capped
  `LIBRARIAN_SECTION_CAP` + the steer if any + already-cited/found
  titles, so a re-run does not rediscover), plans 3–6 keyword queries
  (JSON array, `parse_json_array`'s intake), searches them through the
  same academic-first stack and the shared `ArxivRateLimited`, drops
  papers already in `refs.bib` (`related.drop_cited`), and ranks its
  own top ≤10 with a one-line `why` each.
- The ladder for untrusted replies: a plan that will not parse fails
  with a notice; a fit-rank fault or a junk answer falls back to
  search order with `why = "top of query '<q>'"`; a fault outside the
  ladder still notices (§8 — a silent dead button is the worse
  failure). Picks land `found_by="librarian"`, and the
  arXiv-bearing ones auto-join (the corpus autojoin, unchanged).
- The completion `notice` (one new ServerEvent, the exact recap fold)
  fans only while `state.project` is still the project the pass
  started on — a switch mid-pass silences it; a new sitting on the
  same paper still hears it, because it is that paper's news.

## Planned (issue #29, ratified shape)

- **The citation graph (slice 3).** OpenAlex `cited_by` over the
  references' DOIs (arXiv ids mapped to their DataCite DOIs), folded
  into the pass behind the query candidates, cards marked
  `found_by="graph"` with "cited by N of your references".

## Honest deviations (documented, not hidden)

- **SPEC §6's "No user-facing search UI"** is amended through issue
  #29 (the SPEC text now says so): the panel surfaces what the agent's
  searches found, and the one Find button triggers the agent's own
  pass. The keywords/focus box is user-facing text, but it is a seed
  for the agent's planning, not a query — there is still no search box:
  the user never searches, the agent does.
- **Model JSON is untrusted end to end.** `parse_json_array` is
  defensive; the fit-rank fault degrades to search order, never a
  dead panel (the ladder above).
- **One Find click can enqueue ≤10 PDFs into MinerU** — the auto-join
  of the pass's picks (user-ratified; more restrained than one live
  `web_search`, which joins every open hit).
- **The librarian's ≤6 queries share `ArxivRateLimited`** — they
  serialize behind live searches; politeness wins over latency.
- **The notice guard is project identity, not sitting identity** — a
  new sitting on the same paper still hears the old pass's completion.
- **OpenAlex abstracts are spotty** — most works carry no inverted
  index; those cards fall back to title + why.
- **Cited-marking only sees ids.** `same_paper` matches arXiv id or
  DOI; a hand-written bib entry with neither keeps its card's Cite
  button — a false Cite is worse than a redundant one.
- **Concurrent writers are safe only via the await-free invariant** —
  commented at `record_entries` (the store's one writer); any future
  async rewrite must bring a lock.
- **The store caps at 30** (~45 KB): the panel is a shortlist, not an
  archive.
