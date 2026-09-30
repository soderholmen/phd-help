"""Academic-first discovery (SPEC §6): arXiv ti:"..." title search and
OpenAlex search= always run and merge, arXiv leading. Fake-HTTP seam like
the cascade — tests assert the recorded fetch calls and the parsed hits,
no real network.
"""

import json

import pytest

from phd_helper.cascade import Response  # the fetcher seam's type
from phd_helper.search import search_papers


@pytest.fixture
def anyio_backend():
    return "asyncio"


def recorder(*responses):
    """Fake fetcher: returns the queued responses in order, records calls."""
    calls = []
    queue = list(responses)

    async def fetch(url, headers=None):
        calls.append((url, headers or {}))
        return queue.pop(0) if queue else Response(404, "")

    return fetch, calls


# Known-good literal shaped from a real OpenAlex /works?search= response.
OPENALEX_SEARCH = json.dumps({"results": [
    {"display_name": "Attention Is All You Need",
     "publication_year": 2017,
     "doi": "https://doi.org/10.48550/arXiv.1706.03762",
     "authorships": [{"author": {"display_name": "Ashish Vaswani"}},
                     {"author": {"display_name": "Noam Shazeer"}}],
     "primary_location": {"source": {"display_name": "NeurIPS"}}},
    {"display_name": "A Second Paper", "publication_year": 2019,
     "doi": "https://doi.org/10.1000/second",
     "authorships": [{"author": {"display_name": "Jane Doe"}}],
     "primary_location": None},
]})


@pytest.mark.anyio
async def test_openalex_search_returns_parsed_hits():
    fetch, calls = recorder(Response(200, OPENALEX_SEARCH))
    hits = await search_papers("attention is all you need", fetch,
                               mailto="me@example.org")
    url = calls[0][0]
    assert url.startswith("https://api.openalex.org/works?search=")
    assert "per-page=" in url and "mailto=me%40example.org" in url
    assert hits[0].title == "Attention Is All You Need"
    assert hits[0].year == "2017"
    assert hits[0].authors == ("Vaswani, Ashish", "Shazeer, Noam")
    assert hits[0].venue == "NeurIPS"
    assert hits[0].source == "openalex"
    # the arXiv DataCite DOI yields the bare id: cite_add takes the fast
    # arxiv bibtex path instead of the DOI detour
    assert hits[0].arxiv == "1706.03762"
    assert hits[0].doi == "10.48550/arXiv.1706.03762"
    assert hits[1].arxiv == "" and hits[1].doi == "10.1000/second"
    assert hits[1].venue == ""


# Known-good literal shaped from a real arXiv Atom feed (wrapped title,
# versioned id — both real quirks the parser must eat).
ARXIV_ATOM = """<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom"
      xmlns:arxiv="http://arxiv.org/schemas/atom">
  <entry>
    <id>http://arxiv.org/abs/1706.03762v7</id>
    <title>Attention Is All
      You Need</title>
    <published>2017-06-12T00:00:00Z</published>
    <author><name>Ashish Vaswani</name></author>
    <author><name>Noam Shazeer</name></author>
  </entry>
</feed>
"""


@pytest.mark.anyio
async def test_arxiv_title_search_is_a_quoted_phrase():
    fetch, calls = recorder(Response(200, '{"results": []}'),
                            Response(200, ARXIV_ATOM))
    hits = await search_papers("attention is all you need", fetch)
    assert calls[1][0].startswith("http://export.arxiv.org/api/query?")
    assert "ti%3A%22attention+is+all+you+need%22" in calls[1][0]
    assert hits[0].arxiv == "1706.03762"  # abs-url prefix + version stripped
    assert hits[0].title == "Attention Is All You Need"  # whitespace folded
    assert hits[0].year == "2017"
    assert hits[0].authors == ("Vaswani, Ashish", "Shazeer, Noam")
    assert hits[0].source == "arxiv"


# Live-probe reality (2026-09): OpenAlex merged the real Attention paper's
# title, authors and citations into a parasitic clone record — wrong year,
# squatted DOI, no arXiv link — and ranks it #1. The arXiv ti: hit must
# survive and lead; the clone must dedupe away behind it.
CLONE_SEARCH = json.dumps({"results": [
    {"display_name": "Attention Is All You Need",
     "publication_year": 2025,
     "doi": "https://doi.org/10.65215/2q58a426",
     "authorships": [{"author": {"display_name": "Ashish Vaswani"}},
                     {"author": {"display_name": "Noam Shazeer"}}],
     "primary_location": None},
]})


@pytest.mark.anyio
async def test_both_sources_run_and_the_clean_arxiv_hit_wins():
    fetch, calls = recorder(Response(200, CLONE_SEARCH),
                            Response(200, ARXIV_ATOM))
    hits = await search_papers("attention is all you need", fetch)
    assert len(calls) == 2  # both sources always: clones poison single-source
    atts = [h for h in hits if h.title == "Attention Is All You Need"]
    assert len(atts) == 1  # title + first author dedupe
    assert atts[0].year == "2017" and atts[0].arxiv == "1706.03762"
    assert hits[0] is atts[0]  # the precise title hit leads


@pytest.mark.anyio
async def test_topic_queries_survive_an_empty_title_search():
    fetch, calls = recorder(Response(200, OPENALEX_SEARCH),
                            Response(200, "<feed/>"))  # ti: finds nothing
    hits = await search_papers("proteins", fetch)
    assert [h.source for h in hits] == ["openalex", "openalex"]


@pytest.mark.anyio
async def test_unparseable_and_failed_bodies_yield_no_hits_not_a_crash():
    # 200-with-HTML (Anubis/captive portal) then a status-0 network miss
    # (what HttpFetcher returns on httpx errors) — both are plain no-hits.
    fetch, calls = recorder(Response(200, "<html>a challenge</html>"),
                            Response(0, "ConnectError: no route"))
    assert await search_papers("anything", fetch) == []
    assert len(calls) == 2  # both sources tried, neither raised


@pytest.mark.anyio
@pytest.mark.parametrize("body", ["[]", '{"results": null}',
                                  '{"results": 5}'])
async def test_valid_json_of_the_wrong_shape_is_a_miss_not_a_crash(body):
    # A proxy returning 200-with-JSON-but-not-the-contract must not kill
    # the turn any more than HTML would (review 97ea4d0 finding 2).
    fetch, calls = recorder(Response(200, body), Response(200, body))
    assert await search_papers("anything", fetch) == []


@pytest.mark.anyio
async def test_old_format_ids_dedupe_case_insensitively():
    # OpenAlex lowercases the id inside its arXiv DOI; arXiv's Atom keeps
    # the original case — the same paper must not take two candidate
    # slots when the titles also differ (revised versions).
    oa = json.dumps({"results": [{
        "display_name": "A Revised Title", "publication_year": 2004,
        "doi": "https://doi.org/10.48550/arxiv.math.gt/0309136",
        "authorships": [{"author": {"display_name": "Some Author"}}],
        "primary_location": None}]})
    atom = ('<?xml version="1.0"?><feed xmlns="http://www.w3.org/2005/Atom">'
            '<entry><id>http://arxiv.org/abs/math.GT/0309136v1</id>'
            '<title>Original Title</title>'
            '<published>2003-09-01T00:00:00Z</published>'
            '<author><name>Some Author</name></author></entry></feed>')
    fetch, calls = recorder(Response(200, oa), Response(200, atom))
    hits = await search_papers("original title", fetch)
    assert len(hits) == 1
    assert hits[0].arxiv == "math.GT/0309136"  # arXiv's own casing wins


@pytest.mark.anyio
async def test_arxiv_doi_element_is_carried_into_the_hit():
    atom = ('<?xml version="1.0"?><feed xmlns="http://www.w3.org/2005/Atom" '
            'xmlns:arxiv="http://arxiv.org/schemas/atom">'
            '<entry><id>http://arxiv.org/abs/1706.03762v7</id>'
            '<title>Attention Is All You Need</title>'
            '<published>2017-06-12T00:00:00Z</published>'
            '<arxiv:doi>10.48550/arXiv.1706.03762</arxiv:doi>'
            '<author><name>Ashish Vaswani</name></author></entry></feed>')
    fetch, calls = recorder(Response(200, '{"results": []}'),
                            Response(200, atom))
    hits = await search_papers("attention", fetch)
    assert hits[0].doi == "10.48550/arXiv.1706.03762"
