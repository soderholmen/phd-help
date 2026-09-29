"""BibTeX resolution cascade (SPEC §6): arXiv -> DBLP -> Crossref ->
OpenAlex-build, stopping at first success. Fake-HTTP seam: tests assert the
recorded fetch calls (URLs, headers, order) — no real network.
"""

import asyncio

import pytest

from phd_helper.cascade import (ArxivRateLimited, Lookup, Response,
                                resolve_bibtex)

# arXiv's own bibtex format (First Last author order), known-good literal.
ARXIV_BIB = """@misc{vaswani2023attention,
  title={Attention Is All You Need},
  author={Ashish Vaswani and Noam Shazeer},
  year={2023},
  eprint={1706.03762},
  archivePrefix={arXiv},
}
"""


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


@pytest.mark.anyio
async def test_arxiv_id_resolves_via_arxiv_bibtex():
    fetch, calls = recorder(Response(200, ARXIV_BIB))
    result = await resolve_bibtex(Lookup(arxiv="1706.03762"), fetch)
    assert result.entry is not None
    assert result.entry.key == "vaswani2023attention"
    assert result.source == "arxiv"
    assert calls[0][0] == "https://arxiv.org/bibtex/1706.03762"


DBLP_SEARCH = ('{"result": {"hits": {"hit": [{"info": '
               '{"key": "conf/nips/VaswaniSPGJLSK17"}}]}}}')
DBLP_BIB = """@inproceedings{VaswaniSPGJLSK17,
  title={Attention Is All You Need},
  author={Ashish Vaswani and Noam Shazeer},
  booktitle={NeurIPS},
  year={2017}
}
"""


@pytest.mark.anyio
async def test_arxiv_miss_falls_to_dblp_search_then_rec_bib():
    fetch, calls = recorder(Response(404, ""), Response(200, DBLP_SEARCH),
                            Response(200, DBLP_BIB))
    result = await resolve_bibtex(
        Lookup(arxiv="1706.03762", title="Attention Is All You Need"), fetch)
    assert result.source == "dblp"
    assert result.entry.key == "VaswaniSPGJLSK17"
    assert "dblp.org/search/publ/api" in calls[1][0]
    assert calls[2][0] == \
        "https://dblp.org/rec/conf/nips/VaswaniSPGJLSK17.bib"
    # Anubis anti-bot => browser-like UA (SPEC §6).
    assert "Mozilla" in calls[2][1].get("User-Agent", "")


CROSSREF_BIB = """@article{doe2020cascade,
  title={The Cascade},
  author={Doe, Jane},
  year={2020},
  doi={10.1234/cascade}
}
"""


@pytest.mark.anyio
async def test_doi_resolves_via_crossref_transform_with_mailto():
    fetch, calls = recorder(Response(200, CROSSREF_BIB))
    result = await resolve_bibtex(
        Lookup(doi="10.1234/cascade"), fetch, mailto="me@example.org")
    assert result.source == "crossref"
    assert result.entry.key == "doe2020cascade"
    url = calls[0][0]
    assert "api.crossref.org/works/10.1234%2Fcascade/transform" in url
    assert "mailto=me%40example.org" in url  # polite pool (SPEC §6)


OPENALEX_WORK = ('{"title": "Attention Is All You Need", '
                 '"publication_year": 2017, "authorships": '
                 '[{"author": {"display_name": "Ashish Vaswani"}}]}')


@pytest.mark.anyio
async def test_openalex_builds_entry_when_all_else_fails():
    fetch, calls = recorder(Response(404, ""),        # dblp search misses
                            Response(200, OPENALEX_WORK))
    result = await resolve_bibtex(
        Lookup(title="Attention Is All You Need"), fetch)
    assert result.source == "openalex"
    assert result.entry.fields["title"] == "Attention Is All You Need"
    assert result.entry.fields["author"] == "Vaswani, Ashish"
    assert result.entry.key == "vaswani2017attention"
    assert "api.openalex.org/works" in calls[1][0]
    assert "per-page=100" in calls[1][0]  # politeness default (SPEC §6)


@pytest.mark.anyio
async def test_all_sources_failing_returns_none_with_tried_trace():
    fetch, calls = recorder(Response(404, ""), Response(404, ""))
    result = await resolve_bibtex(Lookup(title="Nowhere Found"), fetch)
    assert result.entry is None
    assert result.tried == ("dblp", "openalex")


@pytest.mark.anyio
async def test_arxiv_requests_spaced_by_three_seconds():
    fetch, calls = recorder(Response(200, ARXIV_BIB),
                            Response(200, ARXIV_BIB))
    slept = []

    async def sleep(seconds):
        slept.append(seconds)

    now = [0.0]
    polite = ArxivRateLimited(fetch, sleep=sleep, clock=lambda: now[0])
    await resolve_bibtex(Lookup(arxiv="1706.03762"), polite)
    now[0] = 1.0  # only a second since the last request
    await resolve_bibtex(Lookup(arxiv="1706.03762"), polite)
    assert slept == [2.0]  # remaining gap, not the full 3 s (SPEC §6)
    assert len(calls) == 2  # single connection: serial, never concurrent


@pytest.mark.anyio
async def test_arxiv_no_sleep_once_the_gap_already_elapsed():
    fetch, calls = recorder(Response(200, ARXIV_BIB),
                            Response(200, ARXIV_BIB))
    slept = []

    async def sleep(seconds):
        slept.append(seconds)

    now = [0.0]
    polite = ArxivRateLimited(fetch, sleep=sleep, clock=lambda: now[0])
    await resolve_bibtex(Lookup(arxiv="1706.03762"), polite)
    now[0] = 10.0  # well past the 3 s gap
    await resolve_bibtex(Lookup(arxiv="1706.03762"), polite)
    assert slept == []  # a stale boolean must not force a needless wait
    assert len(calls) == 2


@pytest.mark.anyio
async def test_concurrent_arxiv_requests_never_overlap():
    in_flight: list[str] = []
    peak: list[int] = []

    async def fetch(url, headers=None):
        in_flight.append(url)
        peak.append(len(in_flight))
        await asyncio.sleep(0)  # let a second request try to start
        in_flight.pop()
        return Response(200, ARXIV_BIB)

    async def sleep(seconds):
        pass

    polite = ArxivRateLimited(fetch, sleep=sleep, clock=lambda: 0.0)
    await asyncio.gather(
        resolve_bibtex(Lookup(arxiv="1706.03762"), polite),
        resolve_bibtex(Lookup(arxiv="1801.00001"), polite))
    assert max(peak) == 1  # single connection, even under concurrency (§6)


@pytest.mark.anyio
async def test_openalex_doi_branch_puts_per_page_on_a_query_string():
    # The DOI URL has no '?' yet: appending '&per-page=100' to it 404s.
    fetch, calls = recorder(Response(404, ""),        # crossref misses
                            Response(200, OPENALEX_WORK))
    result = await resolve_bibtex(Lookup(doi="10.1234/cascade"), fetch)
    assert result.source == "openalex"
    assert calls[1][0] == ("https://api.openalex.org/works"
                           "/doi:10.1234%2Fcascade?per-page=100")


@pytest.mark.anyio
async def test_non_bibtex_200_body_is_a_miss_not_a_crash():
    # Anubis challenges and captive portals answer 200 with HTML; the
    # cascade must fall through to the next source, not raise.
    fetch, calls = recorder(Response(200, "<html>a challenge page</html>"),
                            Response(404, ""),        # dblp search misses
                            Response(200, OPENALEX_WORK))
    result = await resolve_bibtex(
        Lookup(arxiv="1706.03762", title="Attention Is All You Need"), fetch)
    assert result.source == "openalex"


@pytest.mark.anyio
async def test_openalex_null_publication_year_yields_empty_year():
    work = ('{"title": "Undated Thing", "publication_year": null, '
            '"authorships": [{"author": {"display_name": "Ashish Vaswani"}}]}')
    fetch, calls = recorder(Response(404, ""),        # crossref misses
                            Response(200, work))
    result = await resolve_bibtex(Lookup(doi="10.1/undated"), fetch)
    assert result.source == "openalex"
    assert result.entry.fields["year"] == ""  # not the string 'None'
