"""Project layer (SPEC §5/§7): the write path the agent goes through —
propose (lint before the diff is shown) -> pending -> approve (hash check /
re-anchor, snapshot, write) -> undo. Filesystem-backed, tested on tmp_path.
"""

from pathlib import Path

import pytest

from phd_helper.project import Project, ProposeError

MAIN = (
    "\\documentclass{article}\n\\begin{document}\n"
    "\\input{sections/intro}\n\\input{sections/related}\n"
    "\\end{document}\n")
INTRO = "We use a transformer. It works well.\n"
RELATED = "Prior work studied RNNs.\n"


@pytest.fixture
def paper(tmp_path: Path) -> Project:
    (tmp_path / "sections").mkdir()
    (tmp_path / "main.tex").write_text(MAIN, encoding="utf-8")
    (tmp_path / "sections" / "intro.tex").write_text(INTRO, encoding="utf-8")
    (tmp_path / "sections" / "related.tex").write_text(RELATED, encoding="utf-8")
    return Project(tmp_path)


def test_section_tree_from_root_graph(paper):
    paths = [n.path for n in paper.section_tree()]
    assert paths == ["sections/intro.tex", "sections/related.tex"]


def test_read_section(paper):
    assert paper.read_section("sections/intro.tex") == INTRO


def test_propose_creates_pending_without_writing(paper):
    diff = paper.propose_patch("sections/intro.tex",
                               "It works well.", "It achieves SOTA.")
    assert diff.proposed_text == "We use a transformer. It achieves SOTA.\n"
    assert paper.read_section("sections/intro.tex") == INTRO  # untouched
    assert [d.id for d in paper.list_pending("sections/intro.tex")] == [diff.id]


def test_propose_rejects_missing_anchor(paper):
    with pytest.raises(ProposeError, match="not present"):
        paper.propose_patch("sections/intro.tex", "not there", "x")
    assert paper.list_pending("sections/intro.tex") == []


def test_propose_lints_before_showing_diff(paper):
    with pytest.raises(ProposeError, match="[Ll]int"):
        paper.propose_patch("sections/intro.tex",
                            "It works well.", "broken \\textbf{brace")
    assert paper.list_pending("sections/intro.tex") == []


def test_approve_writes_snapshots_and_clears_pending(paper):
    diff = paper.propose_patch("sections/intro.tex",
                               "It works well.", "It achieves SOTA.")
    result = paper.apply_pending("sections/intro.tex", diff.id)
    assert result.applied
    assert paper.read_section("sections/intro.tex") == \
        "We use a transformer. It achieves SOTA.\n"
    assert paper.list_pending("sections/intro.tex") == []
    assert paper.history.entries("sections/intro.tex")  # snapshot recorded


def test_approve_after_external_edit_reanchors(paper):
    diff = paper.propose_patch("sections/intro.tex",
                               "It works well.", "It achieves SOTA.")
    # User edits elsewhere in the file (Syncthing/VS Code arriving edit).
    paper.write_section("sections/intro.tex",
                        "We use a transformer. It works well. Also new.\n")
    result = paper.apply_pending("sections/intro.tex", diff.id)
    assert result.applied
    assert "SOTA" in paper.read_section("sections/intro.tex")
    assert "Also new" in paper.read_section("sections/intro.tex")


def test_approve_ambiguous_reanchor_rejects(paper):
    diff = paper.propose_patch("sections/intro.tex",
                               "It works well.", "It achieves SOTA.")
    # Duplicate the anchor so the close-match re-anchor is ambiguous.
    paper.write_section("sections/intro.tex",
                        "It works well. It works well.\n")
    result = paper.apply_pending("sections/intro.tex", diff.id)
    assert not result.applied
    assert result.reason  # reason shown, never silently dropped


def test_undo_reverts_last_apply(paper):
    diff = paper.propose_patch("sections/intro.tex",
                               "It works well.", "It achieves SOTA.")
    paper.apply_pending("sections/intro.tex", diff.id)
    assert paper.undo_last("sections/intro.tex").applied
    assert paper.read_section("sections/intro.tex") == INTRO


def test_pending_survives_reopen(tmp_path, paper):
    paper.propose_patch("sections/intro.tex",
                        "It works well.", "It achieves SOTA.")
    reopened = Project(tmp_path)
    pend = reopened.list_pending("sections/intro.tex")
    assert len(pend) == 1
    assert pend[0].patch.replace == "It achieves SOTA."
