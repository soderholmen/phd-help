"""Agent tools (SPEC §2/§4): strict schemas, the find-anchor validator that
feeds client-side validation, and dispatch against a Project.
"""

import json

import pytest

from phd_helper.bibtex import BibEntry
from phd_helper.patches import section_hash
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


def test_section_create_validator_rejects_an_existing_target(paper):
    v = make_validators(paper)["section_create"]
    assert v({"section": "sections/intro.tex", "content": "x\n"})


def test_section_create_validator_accepts_a_new_path(paper):
    v = make_validators(paper)["section_create"]
    assert v({"section": "sections/method.tex", "content": "x\n"}) is None


# -- section_write from_draft (stepwise writing) ---------------------------


def test_section_write_from_draft_is_a_required_boolean():
    assert OFFERED["section_write"] == {"section": "string",
                                        "find": "string",
                                        "replace": "string",
                                        "from_draft": "boolean"}


def test_validator_accepts_from_draft_with_a_captured_draft(paper):
    v = make_validators(paper, draft="We propose X.")["section_write"]
    assert v({"section": "sections/intro.tex", "find": "It works well.",
              "replace": "", "from_draft": True}) is None


def test_validator_flags_from_draft_without_a_draft(paper):
    v = make_validators(paper)["section_write"]  # nothing captured
    msg = v({"section": "sections/intro.tex", "find": "It works well.",
             "replace": "", "from_draft": True})
    assert msg and "draft" in msg


def test_validator_flags_from_draft_with_replace_too(paper):
    # the XOR: a draft to apply OR text to type, never both
    v = make_validators(paper, draft="x")["section_write"]
    msg = v({"section": "sections/intro.tex", "find": "It works well.",
             "replace": "typed text", "from_draft": True})
    assert msg and "replace" in msg


def test_validator_still_checks_the_anchor_under_from_draft(paper):
    v = make_validators(paper, draft="x")["section_write"]
    assert v({"section": "sections/intro.tex", "find": "not there",
              "replace": "", "from_draft": True})


def test_execute_section_create_returns_pending_with_created(paper):
    result = execute(call("section_create", {"section": "sections/method.tex",
                                             "content": "M.\n"}), paper)

    assert result["status"] == "pending"
    assert result["section"] == "main.tex"
    assert result["created"] == {"path": "sections/method.tex",
                                 "content": "M.\n"}
    # nothing on disk until the approval lands
    assert not (paper.root / "sections" / "method.tex").exists()


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

    async def autojoin(open_hits):
        joined.append(open_hits)

    hits = [PaperHit(title="Mesh Anything", authors=(), year="2024",
                     arxiv="2401.00002", doi="", venue="arXiv",
                     source="arxiv"),
            PaperHit(title="Paywalled", authors=(), year="2020",
                     arxiv="", doi="10.1000/p", venue="ACM",
                     source="openalex")]
    result = await execute_async(
        call("web_search", {"query": "mesh anything"}), paper,
        search=fake_search(hits), autojoin=autojoin)
    # arXiv PDFs only; paywalled stays bib-only. Whole hits ride along so
    # the corpus gets the real title, not "Untitled" (§6 embed prefix).
    assert [[h.arxiv for h in batch] for batch in joined] == [["2401.00002"]]
    assert joined[0][0].title == "Mesh Anything"
    assert len(result["results"]) == 2  # the search result stands regardless


@pytest.mark.anyio
async def test_autojoin_failure_never_sinks_the_search_result(paper):
    async def autojoin(open_hits):
        raise RuntimeError("server offline")
    result = await execute_async(
        call("web_search", {"query": "mesh anything"}), paper,
        search=fake_search(HITS), autojoin=autojoin)
    assert len(result["results"]) == 1 and "error" not in result


# -- related-work recording (docs/related-work.md): the panel sees searches

@pytest.mark.anyio
async def test_web_search_reports_the_whole_hits_and_query_to_the_panel(paper):
    seen = []
    result = await execute_async(
        call("web_search", {"query": "mesh anything"}), paper,
        search=fake_search(HITS),
        on_results=lambda hits, query: seen.append((hits, query)))
    assert [h.arxiv for h in seen[0][0]] == ["2401.00002"]
    assert seen[0][1] == "mesh anything"  # the card names its search (#29)
    # the model's result is built field-by-field and stays abstract-free
    # (§4 budget); the panel gets the whole hits through the callback.
    assert "abstract" not in result["results"][0]


@pytest.mark.anyio
async def test_a_store_fault_never_sinks_the_search_result(paper):
    def on_results(hits, query):
        raise RuntimeError("disk full")
    result = await execute_async(
        call("web_search", {"query": "mesh anything"}), paper,
        search=fake_search(HITS), on_results=on_results)
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
    assert hit["locator"] == "p.3, block 12"  # single block, no degenerate range
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


# -- memory_write (SPEC §4 voice door) ---------------------------------------


def test_memory_write_replaces_the_memory_file(paper):
    result = execute(call("memory_write", {"content":
                                           "# Decisions\n- harrier"}), paper)
    assert result == {"status": "written", "chars": len("# Decisions\n- harrier")}
    assert paper.load_memory() == "# Decisions\n- harrier"


def test_memory_write_is_offered_with_content_required():
    assert OFFERED["memory_write"] == {"content": "string"}


# -- pending_decide (SPEC §3 approval window: voice resolves diffs) ---------


def test_pending_decide_apply_lands_the_diff(paper):
    diff = paper.propose_patch("sections/intro.tex", "It works well.",
                               "It works.")

    result = execute(call("pending_decide",
                          {"diff_id": diff.id, "decision": "apply"}), paper)

    assert result["status"] == "resolved"
    r = result["resolutions"][0]
    assert r["diff_id"] == diff.id and r["applied"] is True
    assert r["section"] == "sections/intro.tex"
    assert "It works.\n" in paper.read_section("sections/intro.tex")
    assert paper.list_pending("sections/intro.tex") == []


def test_pending_decide_discard_clears_without_writing(paper):
    diff = paper.propose_patch("sections/intro.tex", "It works well.",
                               "Nope.")

    result = execute(call("pending_decide",
                          {"diff_id": diff.id, "decision": "discard"}), paper)

    r = result["resolutions"][0]
    assert r["applied"] is False and r["reason"] == "discarded"
    assert paper.read_section("sections/intro.tex") == INTRO
    assert paper.list_pending("sections/intro.tex") == []


def test_pending_decide_all_resolves_every_pending_diff(paper):
    (paper.root / "sections" / "methods.tex").write_text(
        "Methods body.\n", encoding="utf-8")
    d1 = paper.propose_patch("sections/intro.tex", "It works well.",
                             "It works.")
    d2 = paper.propose_patch("sections/methods.tex", "Methods body.",
                             "Methods text.")

    result = execute(call("pending_decide",
                          {"diff_id": "all", "decision": "apply"}), paper)

    assert {r["diff_id"] for r in result["resolutions"]} == {d1.id, d2.id}
    assert all(r["applied"] for r in result["resolutions"])
    assert paper.pending.list_all() == []


def test_pending_decide_apply_bounce_keeps_the_diff_pending(paper):
    diff = paper.propose_patch("sections/intro.tex", "It works well.",
                               "It works.")
    paper.write_section("sections/intro.tex", "User rewrote the file.\n")

    result = execute(call("pending_decide",
                          {"diff_id": diff.id, "decision": "apply"}), paper)

    r = result["resolutions"][0]
    assert r["applied"] is False and r["reason"]
    assert paper.list_pending("sections/intro.tex")  # still pending (§5)


def test_pending_decide_unknown_id_bounces(paper):
    result = execute(call("pending_decide",
                          {"diff_id": "0042", "decision": "apply"}), paper)
    assert "error" in result


def test_pending_decide_rejects_an_unknown_decision(paper):
    diff = paper.propose_patch("sections/intro.tex", "It works well.", "x")
    result = execute(call("pending_decide",
                          {"diff_id": diff.id, "decision": "maybe"}), paper)
    assert "error" in result


def test_pending_decide_validator_checks_id_and_decision(paper):
    v = make_validators(paper)["pending_decide"]
    assert v({"diff_id": "all", "decision": "apply"}) is None
    diff = paper.propose_patch("sections/intro.tex", "It works well.", "x")
    assert v({"diff_id": diff.id, "decision": "discard"}) is None
    assert v({"diff_id": "0099", "decision": "apply"}) is not None
    assert v({"diff_id": diff.id, "decision": "keep"}) is not None


def test_pending_decide_is_offered_with_both_params():
    assert OFFERED["pending_decide"] == {"diff_id": "string",
                                         "decision": "string"}


def test_pending_decide_all_stays_inside_the_session_window(paper):
    # A diff from an earlier sitting is on disk but was never shown in
    # this window — "apply all" must not write it without approval (§5).
    (paper.root / "sections" / "methods.tex").write_text(
        "Methods body.\n", encoding="utf-8")
    seen = paper.propose_patch("sections/intro.tex", "It works well.",
                               "It works.")
    unseen = paper.propose_patch("sections/methods.tex", "Methods body.",
                                 "Methods text.")

    result = execute(call("pending_decide",
                          {"diff_id": "all", "decision": "apply"}),
                     paper, window={seen.id})

    assert [r["diff_id"] for r in result["resolutions"]] == [seen.id]
    assert paper.list_pending("sections/methods.tex")
    assert "Methods text." not in \
        paper.read_section("sections/methods.tex")
    assert unseen.id not in {r["diff_id"] for r in result["resolutions"]}


def test_pending_decide_window_excludes_foreign_ids(paper):
    diff = paper.propose_patch("sections/intro.tex", "It works well.",
                               "It works.")

    result = execute(call("pending_decide",
                          {"diff_id": diff.id, "decision": "apply"}),
                     paper, window=set())

    assert "error" in result
    assert paper.list_pending("sections/intro.tex")


def test_pending_decide_ambiguous_legacy_id_bounces(paper):
    # Pre-global-ids, two sections could hold the same id; id-only
    # addressing must refuse to guess, not apply both.
    diff = paper.propose_patch("sections/intro.tex", "It works well.",
                               "It works.")
    (paper.root / "sections" / "methods.tex").write_text(
        "Methods body.\n", encoding="utf-8")
    # A pre-global-ids, pre-section-field file: same stem in another dir,
    # section decoded from the dir name.
    (paper.pending._dir("sections/methods.tex")).mkdir(parents=True,
                                                       exist_ok=True)
    (paper.pending._dir("sections/methods.tex") / f"{diff.id}.json").write_text(
        json.dumps({"find": "Methods body.", "replace": "Methods text.",
                    "base_hash": section_hash("Methods body.\n"),
                    "proposed_text": "Methods text.\n",
                    "bib_append": None}), encoding="utf-8")

    result = execute(call("pending_decide",
                          {"diff_id": diff.id, "decision": "discard"}),
                     paper)

    assert "error" in result and "ambiguous" in result["error"]
    assert paper.list_pending("sections/intro.tex")
    assert paper.list_pending("sections/methods.tex")
    assert "It works." not in paper.read_section("sections/intro.tex")


def test_pending_decide_apply_survives_a_deleted_section_file(paper):
    (paper.root / "sections" / "methods.tex").write_text(
        "Methods body.\n", encoding="utf-8")
    d1 = paper.propose_patch("sections/methods.tex", "Methods body.",
                             "Methods text.")
    d2 = paper.propose_patch("sections/intro.tex", "It works well.",
                             "It works.")
    (paper.root / "sections" / "methods.tex").unlink()

    result = execute(call("pending_decide",
                          {"diff_id": "all", "decision": "apply"}), paper)

    by = {r["diff_id"]: r for r in result["resolutions"]}
    assert by[d1.id]["applied"] is False and by[d1.id]["reason"]
    assert by[d2.id]["applied"] is True  # the pass continues past the loss


# -- git_commit (issue: every project a repo) ------------------------------


def git_run(script):
    from phd_helper import gitrepo
    calls = []

    async def run(argv, cwd):
        calls.append(list(argv))
        step = script.get(gitrepo._label(argv), (0, "", ""))
        return step(argv) if callable(step) else step

    return run, calls


def test_git_commit_is_offered_with_a_required_message():
    assert OFFERED["git_commit"] == {"message": "string"}


@pytest.mark.anyio
async def test_git_commit_lazy_inits_then_commits_the_message(paper):
    run, calls = git_run({"status": (0, "M  main.tex\n", ""),
                          "commit": (0, "", ""),
                          "rev-parse": (0, "1a2b3c4\n", "")})
    result = await execute_async(call("git_commit", {"message": "intro done"}),
                                 paper, git_run=run, git_name="N",
                                 git_email="e@x")
    assert result["commit"] == "1a2b3c4"
    assert calls[0] == ["init", "-b", "main"]
    assert any("intro done" in arg for c in calls for arg in c)


@pytest.mark.anyio
async def test_git_commit_empty_message_is_auto_dated(paper):
    run, calls = git_run({"status": (0, "M  main.tex\n", ""),
                          "commit": (0, "", ""),
                          "rev-parse": (0, "abc1234\n", "")})
    await execute_async(call("git_commit", {"message": ""}), paper,
                        git_run=run)
    assert any("Voice commit (" in arg for c in calls for arg in c)


@pytest.mark.anyio
async def test_git_commit_clean_tree_says_so(paper):
    run, _ = git_run({"status": (0, "", "")})
    result = await execute_async(call("git_commit", {"message": "m"}), paper,
                                 git_run=run)
    assert result["commit"] is None
    assert "clean" in result["note"]


@pytest.mark.anyio
async def test_git_commit_fault_bounces_as_a_tool_error(paper):
    run, _ = git_run({"status": (0, "M x\n", ""),
                      "commit": (128, "", "fatal: index.lock")})
    result = await execute_async(call("git_commit", {"message": "m"}), paper,
                                 git_run=run)
    assert "git commit failed" in result["error"]


# -- undo_last (SPEC:137: undo lands directly, no approval window) --------

def test_undo_last_is_offered_with_no_arguments():
    assert OFFERED["undo_last"] == {}


def test_undo_last_reverts_the_last_apply_and_names_it(paper):
    paper.propose_patch("sections/intro.tex", "It works well.", "It works.")
    diff = paper.pending.list_all()[0]
    paper.apply_pending("sections/intro.tex", diff.id)

    result = execute(call("undo_last", {}), paper)

    assert result["status"] == "undone"
    assert result["section"] == "sections/intro.tex"
    assert "It works well." in result["undid"]
    assert paper.read_section("sections/intro.tex") == INTRO


def test_undo_last_says_so_when_history_is_empty(paper):
    result = execute(call("undo_last", {}), paper)
    assert "nothing to undo" in result["error"]


def test_undo_last_twice_bounces_honestly(paper):
    # The second undo targets the same (still-newest) entry; its inverse
    # patch cannot re-anchor what is already undone. No redo, no crash.
    paper.propose_patch("sections/intro.tex", "It works well.", "It works.")
    diff = paper.pending.list_all()[0]
    paper.apply_pending("sections/intro.tex", diff.id)
    assert execute(call("undo_last", {}), paper)["status"] == "undone"

    second = execute(call("undo_last", {}), paper)
    assert "error" in second
    assert paper.read_section("sections/intro.tex") == INTRO
