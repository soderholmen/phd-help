"""Agent tools (SPEC §2/§4): strict schemas, the find-anchor validator that
feeds client-side validation, and dispatch against a Project.
"""

import pytest

from phd_helper.project import Project
from phd_helper.toolcall import ValidCall
from phd_helper.tools import OFFERED, TOOL_SCHEMAS, execute, make_validators

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
        assert OFFERED[s["function"]["name"]] == \
            s["function"]["parameters"]["required"]


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
