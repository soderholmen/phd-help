"""Section history (SPEC §5): snapshots of the file's saved state before
each agent apply, persistent per project in .phd-helper/."""

import os

from phd_helper.history import SectionHistory
from phd_helper.patches import AnchoredPatch, section_hash


def test_snapshots_are_listed_newest_first(tmp_path):
    history = SectionHistory(tmp_path / ".phd-helper")

    first = history.snapshot("sections/intro.tex", "state before apply one\n")
    second = history.snapshot("sections/intro.tex", "state before apply two\n")

    entries = history.entries("sections/intro.tex")

    assert [e.id for e in entries] == [second, first]


def test_history_keeps_only_the_last_n_snapshots_per_file(tmp_path):
    history = SectionHistory(tmp_path / ".phd-helper", keep=3)

    evicted = history.snapshot("sections/intro.tex", "one\n")
    history.snapshot("sections/intro.tex", "two\n")
    history.snapshot("sections/intro.tex", "three\n")
    newest = history.snapshot("sections/intro.tex", "four\n")

    assert [e.id for e in history.entries("sections/intro.tex")] == [
        newest,
        "0002",
        "0001",
    ]
    assert evicted == "0000"


def test_revert_returns_the_snapshots_bytes_wholesale(tmp_path):
    history = SectionHistory(tmp_path / ".phd-helper")
    sid = history.snapshot("sections/intro.tex", "the file as it was\n")

    assert history.revert_text("sections/intro.tex", sid) == "the file as it was\n"


def test_history_survives_a_restart(tmp_path):
    project = tmp_path / ".phd-helper"
    sid = SectionHistory(project).snapshot("sections/intro.tex", "before the crash\n")

    reopened = SectionHistory(project)

    assert [e.id for e in reopened.entries("sections/intro.tex")] == [sid]
    assert reopened.revert_text("sections/intro.tex", sid) == "before the crash\n"


def test_snapshots_are_per_section(tmp_path):
    history = SectionHistory(tmp_path / ".phd-helper")

    history.snapshot("sections/intro.tex", "intro state\n")
    history.snapshot("sections/method.tex", "method state\n")

    assert [e.id for e in history.entries("sections/intro.tex")] == ["0000"]
    assert [e.id for e in history.entries("sections/method.tex")] == ["0000"]


def test_the_newest_entry_across_sections_is_picked_by_mtime(tmp_path):
    # sid is a per-section counter, so 0000 exists in every section and
    # meta carries no timestamp: mtime is the only cross-section order.
    history = SectionHistory(tmp_path / ".phd-helper")
    intro = history.record_apply(
        "sections/intro.tex", "old intro\n",
        AnchoredPatch("old intro", "new intro", section_hash("old intro\n")),
        "new intro\n",
    )
    method = history.record_apply(
        "sections/method.tex", "old method\n",
        AnchoredPatch("old method", "new method", section_hash("old method\n")),
        "new method\n",
    )
    # method's entry is older; intro's apply is the last change made.
    old = 1_000_000.0
    os.utime(history._dir("sections/method.tex") / f"{method}.tex", (old, old))

    assert history.latest_entry_across(
        ["sections/intro.tex", "sections/method.tex"]
    ) == ("sections/intro.tex", intro)


def test_the_newest_entry_across_sections_reports_nothing_when_empty(tmp_path):
    history = SectionHistory(tmp_path / ".phd-helper")

    assert history.latest_entry_across(["sections/intro.tex", "sections/x.tex"]) is None


def test_undo_of_a_specific_past_apply_reverses_just_that_apply(tmp_path):
    history = SectionHistory(tmp_path / ".phd-helper")
    s0 = "alpha\nbeta\ngamma\n"
    s1 = "alpha\nBETA\ngamma\n"
    s2 = "alpha\nBETA\nGAMMA\n"
    first = history.record_apply(
        "sections/s.tex", s0, AnchoredPatch("beta", "BETA", section_hash(s0)), s1
    )
    history.record_apply(
        "sections/s.tex", s1, AnchoredPatch("gamma", "GAMMA", section_hash(s1)), s2
    )

    result = history.undo_entry("sections/s.tex", first, s2)

    assert result.applied is True
    assert result.text == "alpha\nbeta\nGAMMA\n"
