"""BibTeX resolution cascade (SPEC §6): stop at first success.

arXiv id -> arxiv.org/bibtex/<id>; CS venue -> DBLP; DOI -> Crossref
x-bibtex; last resort -> build from OpenAlex metadata. All HTTP goes
through the injected fetcher — politeness rules (arXiv 3 s single
connection, DBLP browser UA, Crossref mailto) live here, not in callers.
"""

import asyncio
import json
import time
from dataclasses import dataclass
from urllib.parse import quote_plus

from phd_helper.bibtex import BibEntry, invert_name, make_key, parse_entry

# Anubis anti-bot on dblp.org wants a browser-like UA (SPEC §6).
BROWSER_UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
              "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0 Safari/537.36")


@dataclass(frozen=True)
class Lookup:
    arxiv: str = ""
    doi: str = ""
    title: str = ""


@dataclass(frozen=True)
class Response:
    status: int
    body: str


@dataclass(frozen=True)
class ResolveResult:
    entry: BibEntry | None
    source: str | None
    tried: tuple[str, ...] = ()


class ArxivRateLimited:
    """arXiv politeness (SPEC §6): one request per 3 s, single connection.

    Wraps any fetcher; only arxiv.org URLs are spaced, everything else
    passes straight through.
    """

    def __init__(self, fetch, sleep=asyncio.sleep, clock=time.monotonic,
                 min_gap: float = 3.0):
        self._fetch = fetch
        self._sleep = sleep
        self._clock = clock
        self._min_gap = min_gap
        self._last: float | None = None  # when the last arxiv request started
        self._lock = asyncio.Lock()

    async def __call__(self, url, headers=None):
        if "arxiv.org" not in url:
            return await self._fetch(url, headers)
        async with self._lock:  # single connection: serialize arxiv requests
            if self._last is not None:
                wait = self._min_gap - (self._clock() - self._last)
                if wait > 0:
                    await self._sleep(wait)  # only the remaining gap
            self._last = self._clock()
            return await self._fetch(url, headers)


async def resolve_bibtex(lookup: Lookup, fetch,
                         mailto: str = "") -> ResolveResult:
    tried: list[str] = []
    if lookup.arxiv:
        tried.append("arxiv")
        resp = await fetch(f"https://arxiv.org/bibtex/{lookup.arxiv}")
        entry = _try_parse(resp) if resp.status == 200 else None
        if entry is not None:
            return ResolveResult(entry, "arxiv", tuple(tried))
    if lookup.title:
        tried.append("dblp")
        search = await fetch(
            "https://dblp.org/search/publ/api?q="
            f"{quote_plus(lookup.title)}&format=json")
        key = _first_hit_key(search) if search.status == 200 else None
        if key:
            bib = await fetch(f"https://dblp.org/rec/{key}.bib",
                              {"User-Agent": BROWSER_UA})
            entry = _try_parse(bib) if bib.status == 200 else None
            if entry is not None:
                return ResolveResult(entry, "dblp", tuple(tried))
    if lookup.doi:
        tried.append("crossref")
        url = f"https://api.crossref.org/works/{quote_plus(lookup.doi)}/transform"
        if mailto:
            url += f"?mailto={quote_plus(mailto)}"  # polite pool (SPEC §6)
        resp = await fetch(url)
        entry = _try_parse(resp) if resp.status == 200 else None
        if entry is not None:
            return ResolveResult(entry, "crossref", tuple(tried))
    if lookup.doi or lookup.title:
        tried.append("openalex")
        if lookup.doi:
            url = (f"https://api.openalex.org/works/doi:{quote_plus(lookup.doi)}"
                   "?per-page=100")  # no '?' yet, so start the query here
        else:
            url = (f"https://api.openalex.org/works"
                   f"?search={quote_plus(lookup.title)}&per-page=100")
        resp = await fetch(url)  # per-page politeness cap (SPEC §6)
        entry = _from_openalex(resp) if resp.status == 200 else None
        if entry is not None:
            return ResolveResult(entry, "openalex", tuple(tried))
    return ResolveResult(None, None, tuple(tried))


def _try_parse(resp: Response) -> BibEntry | None:
    """Parse a 200 body as bibtex; a non-bibtex body (Anubis/captive-portal
    HTML) is a cascade miss, not a crash."""
    try:
        return parse_entry(resp.body)
    except ValueError:
        return None


def _from_openalex(resp: Response) -> BibEntry | None:
    """Last resort: build an entry from OpenAlex metadata (CC0, SPEC §6)."""
    try:
        data = json.loads(resp.body)
        work = data.get("results", [data])[0] if (
            data.get("results") or data.get("title")) else None
        if work is None:
            return None
        authors = " and ".join(
            invert_name(a["author"]["display_name"])
            for a in work.get("authorships", []))
        fields = {"title": work.get("title", ""),
                  "author": authors,
                  "year": str(work.get("publication_year") or "")}
    except (ValueError, KeyError, IndexError):
        return None
    if not fields["title"]:
        return None
    entry = BibEntry(key="", type="article", fields=fields)
    return BibEntry(key=make_key(entry), type=entry.type, fields=fields)


def _first_hit_key(search: Response) -> str | None:
    try:
        hits = json.loads(search.body)["result"]["hits"].get("hit", [])
    except (ValueError, KeyError):
        return None
    return hits[0]["info"].get("key") if hits else None
