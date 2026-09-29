"""Pending diffs (SPEC §7): persistent per project in .phd-helper/,
re-presented on reopen through the apply-time hash check / re-anchor path."""

from phd_helper.patches import AnchoredPatch, section_hash
from phd_helper.pending import PendingDiffs


def test_a_proposed_diff_survives_a_restart(tmp_path):
    store = PendingDiffs(tmp_path / ".phd-helper")
    patch = AnchoredPatch("alpha\n", "ALPHA\n", section_hash("alpha\n"))
    pid = store.propose("sections/s.tex", patch, "ALPHA\n")

    reopened = PendingDiffs(tmp_path / ".phd-helper")
    pending = reopened.list_pending("sections/s.tex")

    assert [p.id for p in pending] == [pid]
    assert pending[0].patch == patch


def test_reconcile_represents_an_unchanged_pending_diff_as_is(tmp_path):
    store = PendingDiffs(tmp_path / ".phd-helper")
    patch = AnchoredPatch("alpha\n", "ALPHA\n", section_hash("alpha\n"))
    store.propose("sections/s.tex", patch, "ALPHA\n")

    outcomes = store.reconcile("sections/s.tex", "alpha\n")

    assert len(outcomes) == 1
    assert outcomes[0].bounce_reason is None
    assert outcomes[0].diff.proposed_text == "ALPHA\n"


def test_reconcile_reanchors_a_stale_pending_diff_to_the_current_text(tmp_path):
    store = PendingDiffs(tmp_path / ".phd-helper")
    stale_base = "alpha\nbeta\n"
    patch = AnchoredPatch("alpha\n", "ALPHA\n", section_hash(stale_base))
    store.propose("sections/s.tex", patch, "ALPHA\nbeta\n")

    current = "alpha\nbeta\n\nUser note added while the diff was pending.\n"
    outcomes = store.reconcile("sections/s.tex", current)

    assert outcomes[0].bounce_reason is None
    assert outcomes[0].diff.proposed_text == (
        "ALPHA\nbeta\n\nUser note added while the diff was pending.\n"
    )


def test_reconcile_bounces_an_unanchorable_diff_with_a_reason(tmp_path):
    store = PendingDiffs(tmp_path / ".phd-helper")
    patch = AnchoredPatch("alpha\n", "ALPHA\n", section_hash("alpha\n"))
    store.propose("sections/s.tex", patch, "ALPHA\n")

    outcomes = store.reconcile("sections/s.tex", "beta\ngamma\n")

    assert outcomes[0].bounce_reason
    assert store.list_pending("sections/s.tex") == []


def test_resolving_a_pending_diff_removes_it(tmp_path):
    store = PendingDiffs(tmp_path / ".phd-helper")
    patch = AnchoredPatch("alpha\n", "ALPHA\n", section_hash("alpha\n"))
    pid = store.propose("sections/s.tex", patch, "ALPHA\n")

    store.resolve("sections/s.tex", pid)

    assert store.list_pending("sections/s.tex") == []
