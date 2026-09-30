"""Agent tools (SPEC §2/§4): strict schemas, the find-anchor validator that
feeds client-side validation, and dispatch against a Project.
"""

import pytest

from phd_helper.bibtex import BibEntry
from phd_helper.cascade import Lookup, ResolveResult, Response
from phd_helper.chunking import Chunk
from phd_helper.corpus import Corpus
from phd_helper.project import Project
from phd_helper.search import PaperHit
from phd_helper.store import ChunkHit, DocInfo
from phd_helper.toolcall import ValidCall
from phd_helper.tools import (OFFERED, TOOL_SCHEMAS, execute,
                              execute_async, make_validators)

MAIN = ("\\documentclass{article}\n\\begin{document}\n"
        "\\input{sections/intro}\n\\end{document}\n")
INTRO = "We use a transformer. It works well.\n"


@pytest.fixture
def paper(tmp_path):
    (tmp_path / "sections").mkdir()
    (tmp_path / "main.tex").write_text(MAIN, encoding="utf-8")
    (tmp_path / "sections" / "intro.tex").write_text(INTRO, encoding="utf-8")
    return Project(tmp_path)


def call(name, args_json, cid="c1"):
    return ValidCall(id=cid, name=name, args=args_json)


def test_schemas_are_openai_strict():
    for schema in TOOL_SCHEMAS:
        params = schema["function"]["parameters"]
        assert params["additionalProperties"] is False
        assert set(params["required"]) == set(params["properties"])


def test_offered_matches_schemas():
    assert set(OFFERED) == {s["function"]["name"] for s in TOOL_SCHEMAS}
    for s in TOOL_SCHEMAS:
        params = s["function"]["parameters"]
        offered = OFFERED[s["function"]["name"]]
        assert set(offered) == set(params["required"])
        assert all(offered[p] == params["properties"][p]["type"]
                   for p in params["required"])


def test_validator_passes_present_anchor(paper):
    v = make_validators(paper)["section_write"]
    assert v({"section": "sections/intro.tex",
              "find": "It works well.", "replace": "x"}) is None


def test_validator_flags_missing_anchor(paper):
    v = make_validators(paper)["section_write"]
    msg = v({"section": "sections/intro.tex",
             "find": "not in there", "replace": "x"})
    assert msg and "not" in msg


def test_validator_flags_unknown_section(paper):
    v = make_validators(paper)["section_write"]
    assert v({"section": "sections/nope.tex", "find": "x", "replace": "y"})


def test_execute_section_read(paper):
    result = execute(call("section_read", {"section": "sections/intro.tex"}),
                     paper)
    assert result["content"] == INTRO


def test_execute_section_write_proposes_pending(paper):
    result = execute(call("section_write", {
        "section": "sections/intro.tex",
        "find": "It works well.", "replace": "It achieves SOTA."}), paper)
    assert result["status"] == "pending"
    assert result["diff_id"]
    assert paper.read_section("sections/intro.tex") == INTRO  # not applied


def test_execute_section_write_lint_failure_bounces(paper):
    result = execute(call("section_write", {
        "section": "sections/intro.tex",
        "find": "It works well.", "replace": "bad \\textbf{brace"}), paper)
    assert "error" in result


def test_execute_unknown_tool_is_error_not_crash(paper):
    result = execute(call("launch_missiles", {}), paper)
    assert "error" in result


# -- cite_add (SPEC §6): async, cascade behind an injected resolver --------

def fake_resolve(entry=None, source="arxiv", tried=("arxiv",)):
    async def resolve(lookup, fetch, mailto=""):
        return ResolveResult(entry, source if entry else None, tried)
    return resolve


RESOLVED = BibEntry(key="shazeer2024mesh", type="article", fields={
    "title": "Mesh Anything", "author": "Shazeer, Noam", "year": "2024",
    "arxiv": "2401.00002"})


@pytest.mark.anyio
async def test_cite_add_proposes_one_pending_diff(paper):
    result = await execute_async(
        call("cite_add", {"section": "sections/intro.tex",
                          "find": "We use a transformer.",
                          "replace": "We use a transformer "
                                     "\\cite{2401.00002}.",
                          "arxiv": "2401.00002", "doi": "", "title": ""}),
        paper, resolve=fake_resolve(RESOLVED))
    assert result["status"] == "pending"
    assert result["key"] == "shazeer2024mesh"
    assert "\\cite{shazeer2024mesh}" in result["replace"]  # rewritten
    assert paper.read_section("sections/intro.tex") == INTRO  # not applied


@pytest.mark.anyio
async def test_cite_add_unresolved_paper_bounces(paper):
    result = await execute_async(
        call("cite_add", {"section": "sections/intro.tex",
                          "find": "We use a transformer.",
                          "replace": "x \\cite{2401.00002}.",
                          "arxiv": "2401.00002", "doi": "", "title": ""}),
        paper, resolve=fake_resolve(None, tried=("arxiv", "dblp",
                                                 "crossref", "openalex")))
    assert "error" in result
    assert "arxiv" in result["error"]  # tried trace helps the model retry


@pytest.mark.anyio
async def test_cite_add_without_a_fetcher_bounces_not_typeerror(paper):
    # fetch defaults to None: a mis-wired caller must get a tool-error
    # bounce, not a TypeError raised from inside the cascade.
    result = await execute_async(
        call("cite_add", {"section": "sections/intro.tex",
                          "find": "We use a transformer.",
                          "replace": "x \\cite{2401.00002}.",
                          "arxiv": "2401.00002", "doi": "", "title": ""}),
        paper)
    assert "error" in result


@pytest.mark.anyio
async def test_cite_add_bad_anchor_bounces(paper):
    result = await execute_async(
        call("cite_add", {"section": "sections/intro.tex",
                          "find": "not in there",
                          "replace": "x \\cite{2401.00002}.",
                          "arxiv": "2401.00002", "doi": "", "title": ""}),
        paper, resolve=fake_resolve(RESOLVED))
    assert "error" in result


# -- web_search (SPEC §6): discovery feeding cite_add -----------------------

def fake_search(hits):
    async def search(query, fetch, mailto=""):
        return hits
    return search


HITS = [PaperHit(title="Mesh Anything", authors=("Shazeer, Noam",),
                 year="2024", arxiv="2401.00002", doi="", venue="arXiv",
                 source="arxiv")]


@pytest.mark.anyio
async def test_web_search_returns_candidates_for_the_model(paper):
    result = await execute_async(
        call("web_search", {"query": "mesh anything"}), paper,
        search=fake_search(HITS))
    hit = result["results"][0]
    assert hit["n"] == 1
    assert hit["title"] == "Mesh Anything" and hit["arxiv"] == "2401.00002"
    assert hit["authors"] == "Shazeer, Noam"  # compact for the model
    assert "cite_add" in result["note"]  # the chain the model should take


@pytest.mark.anyio
async def test_web_search_empty_is_a_result_not_an_error(paper):
    result = await execute_async(
        call("web_search", {"query": "zzz"}), paper, search=fake_search([]))
    assert result["results"] == []
    assert "error" not in result


@pytest.mark.anyio
async def test_web_search_without_a_fetcher_bounces(paper):
    result = await execute_async(call("web_search", {"query": "x"}), paper)
    assert "search unavailable" in result["error"]


# -- auto-join (SPEC §6): openly downloadable PDFs join the corpus ---------

@pytest.mark.anyio
async def test_web_search_auto_joins_arxiv_hits(paper):
    joined = []

    async def autojoin(arxiv_ids):
        joined.append(arxiv_ids)

    hits = [PaperHit(title="Mesh Anything", authors=(), year="2024",
                     arxiv="2401.00002", doi="", venue="arXiv",
                     source="arxiv"),
            PaperHit(title="Paywalled", authors=(), year="2020",
                     arxiv="", doi="10.1000/p", venue="ACM",
                     source="openalex")]
    result = await execute_async(
        call("web_search", {"query": "mesh anything"}), paper,
        search=fake_search(hits), autojoin=autojoin)
    assert joined == [["2401.00002"]]  # arXiv PDFs only; paywalled stays bib-only
    assert len(result["results"]) == 2  # the search result stands regardless


@pytest.mark.anyio
async def test_autojoin_failure_never_sinks_the_search_result(paper):
    async def autojoin(arxiv_ids):
        raise RuntimeError("server offline")
    result = await execute_async(
        call("web_search", {"query": "mesh anything"}), paper,
        search=fake_search(HITS), autojoin=autojoin)
    assert len(result["results"]) == 1 and "error" not in result


def test_cite_add_validator_checks_anchor(paper):
    v = make_validators(paper)["cite_add"]
    assert v({"section": "sections/intro.tex", "find": "It works well.",
              "replace": "x", "arxiv": "1", "doi": "", "title": ""}) is None
    assert v({"section": "sections/intro.tex", "find": "nope",
              "replace": "x", "arxiv": "1", "doi": "", "title": ""})


# -- corpus_search / corpus_doc (SPEC §6): the store behind a fake ---------


class FakeStore:
    """Keyword-overlap stand-in for LanceDB; boost_ids doubles the score."""

    def __init__(self, fail=False):
        self.fail = fail
        self.rows: list[tuple[str, Chunk]] = []
        self.infos: dict[str, DocInfo] = {}

    async def search(self, query, k, boost_ids):
        if self.fail:
            raise RuntimeError("LanceDB down")
        terms = query.lower().split()
        scored = []
        for doc_id, c in self.rows:
            hay = f"{c.text} {c.section_path}".lower()
            s = sum(hay.count(t) for t in terms)
            if s:
                scored.append((s * (2 if doc_id in boost_ids else 1),
                               doc_id, c))
        scored.sort(key=lambda x: -x[0])
        return [ChunkHit(doc_id=d, text=c.text, section_path=c.section_path,
                         page_start=c.page_start, page_end=c.page_end,
                         block_start=c.block_start, block_end=c.block_end,
                         kind=c.kind, score=float(s))
                for s, d, c in scored[:k]]

    async def doc(self, doc_id):
        return self.infos.get(doc_id)

    async def index(self, doc_id, chunks):
        self.rows.extend((doc_id, c) for c in chunks)

    async def remove(self, doc_id):
        self.rows = [(d, c) for d, c in self.rows if d != doc_id]


def chunk(text, block=0, page=1, section="", kind="text", abstract=False):
    return Chunk(text=text, embed_text=text, section_path=section,
                 page_start=page, page_end=page, block_start=block,
                 block_end=block, kind=kind, is_abstract=abstract)


@pytest.fixture
def corpus(tmp_path):
    return Corpus(tmp_path / "corpus")


def own(corpus, store, *, title, arxiv="", doi="", body, hits=()):
    rec, _ = corpus.add_pdf(body, title=title, arxiv=arxiv, doi=doi)
    for c in hits:
        store.rows.append((rec.doc_id, c))
    return rec


@pytest.mark.anyio
async def test_corpus_search_returns_ranked_chunks_with_locators(paper, corpus):
    store = FakeStore()
    rec = own(corpus, store, title="Parakeet", arxiv="2401.00001",
              body=b"a", hits=[chunk("the parakeet model scales audio",
                                      block=12, page=3, section="Method")])
    result = await execute_async(
        call("corpus_search", {"query": "parakeet audio", "k": 5,
                              "boost_pinned": False}),
        paper, corpus=corpus, store=store)
    hit = result["results"][0]
    assert hit["doc_id"] == rec.doc_id
    assert hit["title"] == "Parakeet"  # registry join, not stored per chunk
    assert hit["section"] == "Method"
    assert hit["locator"] == "p.3, blocks 12-12"
    assert "corpus_doc" in result["note"]  # the chain the model should take


@pytest.mark.anyio
async def test_corpus_search_skips_chunks_without_a_record(paper, corpus):
    # A failed supersede can leave chunks whose registry record is gone;
    # the registry is the source of truth — an orphan must not surface
    # with a blank title the agent would read as a real paper.
    store = FakeStore()
    rec = own(corpus, store, title="Real", body=b"a",
              hits=[chunk("attention scores attention", block=3, page=2)])
    store.rows.append(("ghost", chunk("attention scores attention")))
    result = await execute_async(
        call("corpus_search", {"query": "attention scores", "k": 5,
                              "boost_pinned": False}),
        paper, corpus=corpus, store=store)
    assert [r["doc_id"] for r in result["results"]] == [rec.doc_id]


@pytest.mark.anyio
async def test_corpus_search_boosts_pinned_but_never_filters(paper, corpus):
    store = FakeStore()
    pinned = own(corpus, store, title="Pinned", body=b"p",
                 hits=[chunk("shared topic word", section="A")])
    other = own(corpus, store, title="Other", body=b"o",
                hits=[chunk("shared topic word word", section="B")])
    corpus.pin(pinned.doc_id, paper.root.name)
    result = await execute_async(
        call("corpus_search", {"query": "shared topic word", "k": 5,
                              "boost_pinned": True}),
        paper, corpus=corpus, store=store)
    ids = [h["doc_id"] for h in result["results"]]
    assert ids == [pinned.doc_id, other.doc_id]  # boosted to the top, both kept
    result = await execute_async(
        call("corpus_search", {"query": "shared topic word", "k": 5,
                              "boost_pinned": False}),
        paper, corpus=corpus, store=store)
    assert [h["doc_id"] for h in result["results"]] == [other.doc_id,
                                                        pinned.doc_id]


@pytest.mark.anyio
async def test_corpus_search_k_limits_results(paper, corpus):
    store = FakeStore()
    own(corpus, store, title="T", body=b"x",
        hits=[chunk(f"topic {i}") for i in range(6)])
    result = await execute_async(
        call("corpus_search", {"query": "topic", "k": 2,
                              "boost_pinned": False}),
        paper, corpus=corpus, store=store)
    assert len(result["results"]) == 2


@pytest.mark.anyio
async def test_corpus_search_empty_is_a_result_not_an_error(paper, corpus):
    store = FakeStore()
    result = await execute_async(
        call("corpus_search", {"query": "zzz", "k": 5,
                              "boost_pinned": False}),
        paper, corpus=corpus, store=store)
    assert result["results"] == [] and "error" not in result


@pytest.mark.anyio
async def test_corpus_search_down_degrades_to_web_search_advice(paper, corpus):
    # §8: embeddings/LanceDB down -> tool error, agent continues via web
    # search. Both the missing store and a faulting one degrade the same.
    result = await execute_async(
        call("corpus_search", {"query": "x", "k": 5, "boost_pinned": False}),
        paper, corpus=corpus, store=None)
    assert "web_search" in result["error"]
    result = await execute_async(
        call("corpus_search", {"query": "x", "k": 5, "boost_pinned": False}),
        paper, corpus=corpus, store=FakeStore(fail=True))
    assert "web_search" in result["error"]


@pytest.mark.anyio
async def test_corpus_doc_returns_abstract_headings_bib_status(paper, corpus):
    store = FakeStore()
    rec = own(corpus, store, title="Mesh Anything", arxiv="2401.00002",
              body=b"m")
    store.infos[rec.doc_id] = DocInfo(
        abstract="We segment anything.", headings=("Intro", "Method"))
    (paper.root / "refs.bib").write_text(
        "@article{shazeer2024mesh, title={Mesh Anything}, "
        "arxiv={2401.00002}}\n", encoding="utf-8")
    result = await execute_async(call("corpus_doc", {"doc_id": rec.doc_id}),
                                 paper, corpus=corpus, store=store)
    assert result["title"] == "Mesh Anything"
    assert result["abstract"] == "We segment anything."
    assert result["headings"] == ["Intro", "Method"]
    assert result["status"] == "queued"  # per-PDF status is visible to the agent
    assert "shazeer2024mesh" in result["bib"]  # already cited -> reuse the key


@pytest.mark.anyio
async def test_corpus_doc_not_in_bib_and_unknown_id(paper, corpus):
    store = FakeStore()
    rec = own(corpus, store, title="Uncited", doi="10.1000/u", body=b"u")
    store.infos[rec.doc_id] = DocInfo(abstract="", headings=())
    result = await execute_async(call("corpus_doc", {"doc_id": rec.doc_id}),
                                 paper, corpus=corpus, store=store)
    assert result["bib"] == "not in refs.bib"
    result = await execute_async(call("corpus_doc", {"doc_id": "nope"}),
                                 paper, corpus=corpus, store=store)
    assert "error" in result


@pytest.mark.anyio
async def test_corpus_doc_down_degrades(paper, corpus):
    result = await execute_async(call("corpus_doc", {"doc_id": "x"}),
                                 paper, corpus=corpus, store=None)
    assert "web_search" in result["error"]
