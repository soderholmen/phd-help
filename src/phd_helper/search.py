"""Academic-first discovery (SPEC §6): both sources always run and merge —
arXiv ``ti:"..."`` for title precision, OpenAlex ``search=`` for topic
coverage (live probe 2026-09: OpenAlex ranks parasitic clones of famous
papers #1, so single-source OpenAlex cannot be trusted for the cite loop).
All HTTP goes through the injected fetcher — the same seam the cascade is
tested behind; arXiv politeness rides the ArxivRateLimited wrapper.
"""

import json
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from urllib.parse import quote_plus

from phd_helper.bibtex import invert_name
from phd_helper.cascade import Response

# OpenAlex mirrors arXiv papers under this DataCite prefix; the bare id
# behind it lets cite_add take the fast arxiv bibtex path.
ARXIV_DOI_PREFIX = "10.48550/arxiv."


@dataclass(frozen=True)
class PaperHit:
    title: str
    authors: tuple[str, ...]
    year: str
    arxiv: str
    doi: str
    venue: str
    source: str
    # Last and defaulted: the related-work panel's preview rides the hit
    # (docs/related-work.md), but the tool result to the model is built
    # field-by-field and stays abstract-free (§4's context budget).
    abstract: str = ""


async def search_papers(query: str, fetch, mailto: str = "") -> list[PaperHit]:
    core = " ".join(query.split()).replace('"', "")
    if not core:
        return []
    url = (f"https://api.openalex.org/works"
           f"?search={quote_plus(core)}&per-page=10")
    if mailto:
        url += f"&mailto={quote_plus(mailto)}"  # polite pool (SPEC §6)
    resp = await fetch(url)
    oa = _from_openalex(resp) if resp.status == 200 else []
    resp = await fetch("http://export.arxiv.org/api/query?search_query="
                       f"{quote_plus(f'ti:\"{core}\"')}&start=0&max_results=10")
    ax = _from_arxiv(resp) if resp.status == 200 else []
    return _merge(ax, oa)  # precise title hits lead; topics ride OpenAlex


def dedupe(items: list, cap: int = 10) -> list:
    """De-dup by arXiv id, DOI, or title+first-author; first wins.

    Duck-typed on the four identity fields, so the related-work store
    merges its curated entries through the very rules the search loop
    uses — one copy, no drift (docs/related-work.md)."""
    out: list = []
    ids: set[str] = set()
    pairs: set[tuple[str, str]] = set()
    for h in items:
        if h.arxiv and f"a:{h.arxiv.lower()}" in ids:
            continue
        if h.doi and f"d:{h.doi.lower()}" in ids:
            continue
        pair = (h.title.lower(), h.authors[0] if h.authors else "")
        if pair in pairs:
            continue
        if h.arxiv:
            ids.add(f"a:{h.arxiv.lower()}")
        if h.doi:
            ids.add(f"d:{h.doi.lower()}")
        pairs.add(pair)
        out.append(h)
    return out[:cap]


def _merge(primary: list[PaperHit], secondary: list[PaperHit],
           cap: int = 10) -> list[PaperHit]:
    """Precise hits first, so dedupe's first-wins makes primary win."""
    return dedupe([*primary, *secondary], cap)


ATOM = "{http://www.w3.org/2005/Atom}"
ARXIV_NS = "{http://arxiv.org/schemas/atom}"


def _from_arxiv(resp: Response) -> list[PaperHit]:
    try:
        root = ET.fromstring(resp.body)
    except ET.ParseError:
        return []
    hits: list[PaperHit] = []
    for entry in root.findall(f"{ATOM}entry"):
        try:
            raw_id = entry.findtext(f"{ATOM}id") or ""
            arxiv = re.sub(r"v\d+$", "", raw_id.rsplit("/abs/", 1)[-1])
            title = " ".join((entry.findtext(f"{ATOM}title") or "").split())
            if not title:
                continue
            authors = tuple(
                invert_name(" ".join(name.split()))
                for name in (a.findtext(f"{ATOM}name") or ""
                             for a in entry.findall(f"{ATOM}author"))
                if name.strip())
            published = entry.findtext(f"{ATOM}published") or ""
            doi = (entry.findtext(f"{ARXIV_NS}doi") or "").strip()
            summary = " ".join(
                (entry.findtext(f"{ATOM}summary") or "").split())
            hits.append(PaperHit(title=title, authors=authors,
                                 year=published[:4], arxiv=arxiv, doi=doi,
                                 venue="arXiv", source="arxiv",
                                 abstract=summary))
        except (AttributeError, TypeError):  # one malformed entry, not all
            continue
    return hits


def _from_openalex(resp: Response) -> list[PaperHit]:
    try:
        works = json.loads(resp.body).get("results") or []
    except (AttributeError, TypeError, ValueError):
        return []  # valid JSON of the wrong shape is a miss, not a crash
    if not isinstance(works, list):
        return []
    hits: list[PaperHit] = []
    for w in works:
        try:
            title = w.get("display_name") or ""
            if not title:
                continue
            doi = str(w.get("doi") or "").removeprefix("https://doi.org/")
            arxiv = (doi[len(ARXIV_DOI_PREFIX):]
                     if doi.lower().startswith(ARXIV_DOI_PREFIX) else "")
            authors = tuple(
                invert_name(a["author"]["display_name"])
                for a in w.get("authorships", [])
                if (a.get("author") or {}).get("display_name"))
            loc = w.get("primary_location") or {}
            venue = (loc.get("source") or {}).get("display_name") or ""
            hits.append(PaperHit(
                title=title, authors=authors,
                year=str(w.get("publication_year") or ""),
                arxiv=arxiv, doi=doi, venue=venue, source="openalex",
                abstract=_from_inverted_index(
                    w.get("abstract_inverted_index"))))
        except (AttributeError, TypeError):  # one malformed work, not all
            continue
    return hits


def _from_inverted_index(index) -> str:
    """OpenAlex ships no abstract text — only {word: [positions]} — so the
    preview has to be rebuilt by placing each word at its positions.

    Absent or of the wrong shape reads as no abstract, never a partial
    sentence: a card that shows half a preview is worse than one that
    shows none (the fallback is title + why). A real index covers every
    position 0..n-1, so a gap in the placed words is a lost word — the
    whole rebuild reads as nothing rather than a sentence with holes.
    The position cap is the guard against a malformed body allocating a
    list the size of the number in it."""
    if not isinstance(index, dict) or not index:
        return ""
    placed: list[str] = []
    try:
        for word, positions in index.items():
            for p in positions:
                if not isinstance(p, int) or not 0 <= p < 10_000:
                    return ""
                while len(placed) <= p:
                    placed.append("")
                placed[p] = word
    except TypeError:  # positions not iterable of ints — no abstract
        return ""
    if any(not w for w in placed):
        return ""  # a gap is a lost word — no preview, not half one
    return " ".join(placed)
