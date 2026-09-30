"""Agent tools (SPEC §2/§4): strict schemas, the find-anchor validator that
feeds client-side validation, and dispatch against a Project.
"""

import pytest

from phd_helper.bibtex import BibEntry
from phd_helper.cascade import Lookup, ResolveResult, Response
from phd_helper.project import Project
from phd_helper.search import PaperHit
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


def test_cite_add_validator_checks_anchor(paper):
    v = make_validators(paper)["cite_add"]
    assert v({"section": "sections/intro.tex", "find": "It works well.",
              "replace": "x", "arxiv": "1", "doi": "", "title": ""}) is None
    assert v({"section": "sections/intro.tex", "find": "nope",
              "replace": "x", "arxiv": "1", "doi": "", "title": ""})
