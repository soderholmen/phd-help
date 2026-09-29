"""BibTeX resolution cascade (SPEC §6): arXiv -> DBLP -> Crossref ->
OpenAlex-build, stopping at first success. Fake-HTTP seam: tests assert the
recorded fetch calls (URLs, headers, order) — no real network.
"""

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

    polite = ArxivRateLimited(fetch, sleep=sleep)
    await resolve_bibtex(Lookup(arxiv="1706.03762"), polite)
    await resolve_bibtex(Lookup(arxiv="1706.03762"), polite)
    assert slept == [3.0]  # first call free, second waits (SPEC §6)
    assert len(calls) == 2  # single connection: serial, never concurrent
