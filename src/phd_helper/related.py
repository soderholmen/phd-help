"""The related-work store (docs/related-work.md).

The panel's whole pipeline — record, merge, cited-mark, corpus-join, and
the librarian's JSON intake — lives here as pure functions over plain
dataclasses. The store is per-project (.phd-helper/related.json) while
the corpus registry is global, so this is its own module rather than a
corpus.py bolt-on. No FastAPI, no LLM, no network: the seams that need
them (project.py, app.py) stay thin and the rules stay testable.
"""

import json
import re
from dataclasses import asdict, dataclass, replace

from phd_helper.bibtex import BibEntry, parse_bib, same_paper
from phd_helper.corpus import owned_doc
from phd_helper.search import dedupe


@dataclass(frozen=True)
class RelatedEntry:
    title: str
    authors: tuple[str, ...] = ()
    year: str = ""
    arxiv: str = ""
    doi: str = ""
    venue: str = ""
    abstract: str = ""
    why: str = ""                # the librarian's one-line fit note
    found_by: str = "search"     # search | librarian | graph
    cited: bool = False          # already in refs.bib (mark_cited)
    query: str = ""              # the search that surfaced it (issue #29)


def to_entry(hit, why: str = "", found_by: str = "search",
             query: str = "") -> RelatedEntry:
    """A search hit, remembered: the panel's entry is the whole PaperHit
    plus how it was found, (later) why it fits and which query surfaced
    it — "so we know what we have searched for"."""
    return RelatedEntry(title=hit.title, authors=tuple(hit.authors),
                        year=hit.year, arxiv=hit.arxiv, doi=hit.doi,
                        venue=hit.venue, abstract=hit.abstract,
                        why=why, found_by=found_by, query=query)


def to_dict(e: RelatedEntry) -> dict:
    d = asdict(e)
    d["authors"] = list(e.authors)   # plain JSON on disk, tuples in RAM
    return d


def from_dict(d) -> RelatedEntry | None:
    """Defensive: a torn or hand-edited row without a title is not an
    entry — it reads as nothing and the next save drops it."""
    if not isinstance(d, dict) or not d.get("title"):
        return None
    return RelatedEntry(
        title=str(d["title"]),
        authors=tuple(str(a) for a in d.get("authors") or ()),
        year=str(d.get("year") or ""),
        arxiv=str(d.get("arxiv") or ""),
        doi=str(d.get("doi") or ""),
        venue=str(d.get("venue") or ""),
        abstract=str(d.get("abstract") or ""),
        why=str(d.get("why") or ""),
        found_by=str(d.get("found_by") or "search"),
        cited=bool(d.get("cited")),
        query=str(d.get("query") or ""))


def merge(existing: list[RelatedEntry], new: list[RelatedEntry],
          cap: int = 30) -> list[RelatedEntry]:
    """Newest first, deduped by the search loop's own identity rules
    (search.dedupe — one copy, no drift). A rediscovered entry keeps the
    panel's memory: the why and cited mark the librarian wrote ride
    forward, and a rediscovery that brings no new why keeps the old
    provenance too — the badge must match the words under it."""
    kept = dedupe([*new, *existing], cap=cap)
    out = []
    for e in kept:
        old = next((o for o in existing if _matches(o, e)), None)
        if old is not None and old is not e:
            e = replace(e, why=e.why or old.why,
                        found_by=e.found_by if e.why else old.found_by,
                        cited=e.cited or old.cited,
                        abstract=e.abstract or old.abstract,
                        query=e.query or old.query)
        out.append(e)
    return out


def _matches(a: RelatedEntry, b: RelatedEntry) -> bool:
    """The identity rules of search.dedupe, mirrored for the graft
    lookup: dedupe decides who survives, this finds who to graft from —
    the two must agree exactly or a collapsed pair loses its why."""
    if a.arxiv and b.arxiv and a.arxiv.lower() == b.arxiv.lower():
        return True
    if a.doi and b.doi and a.doi.lower() == b.doi.lower():
        return True
    return bool(a.title and b.title
                and a.title.lower() == b.title.lower()
                and (a.authors[0] if a.authors else "")
                == (b.authors[0] if b.authors else ""))


def _in_bib(arxiv: str, doi: str, bib) -> bool:
    """The one cited test: same arXiv id or DOI by bibtex.same_paper,
    the cite loop's own identity rule. mark_cited's badge and the
    librarian's drop-list share it so the panel and the pass can never
    disagree about what the paper already cites."""
    if not (arxiv or doi) or not bib:
        return False
    probe = BibEntry(key="", type="", fields={"arxiv": arxiv, "doi": doi})
    return any(same_paper(probe, b) for b in bib)


def mark_cited(entries: list[RelatedEntry], bib_text: str) -> list[RelatedEntry]:
    """cited = the paper is already in refs.bib — same arXiv id or DOI by
    bibtex.same_paper, the cite loop's own identity rule. Where either
    side carries no id (a hand-written bib entry, an id-less hit) the
    match cannot see it and the card keeps its Cite button — a false
    Cite is worse than a redundant one."""
    bib = parse_bib(bib_text)
    return [e if e.cited or not _in_bib(e.arxiv, e.doi, bib)
            else replace(e, cited=True) for e in entries]


def drop_cited(hits, bib_text: str):
    """Candidates the librarian must not re-find: papers already in
    refs.bib. A hit with no id cannot be proven cited and stays — the
    same posture as mark_cited's honest Cite button."""
    bib = parse_bib(bib_text)
    return [h for h in hits if not _in_bib(h.arxiv, h.doi, bib)]


def entries_from_rows(rows) -> list[RelatedEntry]:
    """Rows off the store, torn ones dropped (from_dict's rule) — the
    door and the recorder both read the file this way."""
    return [e for e in (from_dict(r) for r in rows) if e is not None]


def corpus_join(entries: list[RelatedEntry], docs, project: str) -> list[dict]:
    """Per-entry in_corpus view: the registry doc that owns this paper —
    corpus.owned_doc, the registry's own lookup rule, one copy — or None.
    The panel's Pin button only rides a doc, so this join is what
    decides whether a card offers it."""
    out = []
    for e in entries:
        doc = owned_doc(docs, e.arxiv, e.doi)
        row = to_dict(e)
        row["in_corpus"] = None if doc is None else {
            "doc_id": doc.doc_id, "status": doc.status,
            "pinned_here": project in doc.pinned_in}
        out.append(row)
    return out


_FENCE = re.compile(r"```(?:json)?\s*(.*?)```", re.S)


def parse_json_array(text: str) -> list:
    """The model's JSON, defensively: fence content wins when a fence
    exists, else the outermost [...] span; anything that is not a JSON
    array reads as []. An untrusted reply degrades to "no plan", never
    to a crash mid-librarian (the ladder in docs/related-work.md)."""
    if not text:
        return []
    m = _FENCE.search(text)
    body = m.group(1) if m else text
    start, end = body.find("["), body.rfind("]")
    if start == -1 or end <= start:
        return []
    try:
        data = json.loads(body[start:end + 1])
    except json.JSONDecodeError:
        return []
    return data if isinstance(data, list) else []
