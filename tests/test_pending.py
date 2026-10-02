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


# -- globally unique ids + list_all (SPEC §3 approval window) ----------------
# Voice addresses a diff by its id ("apply 0002"), and the shell's reducer
# keys cards by diff_id alone — ids must be unique across sections.


def test_diff_ids_are_unique_across_sections(tmp_path):
    store = PendingDiffs(tmp_path / ".phd-helper")
    a = store.propose("sections/a.tex",
                      AnchoredPatch("x\n", "X\n", section_hash("x\n")), "X\n")
    b = store.propose("sections/b.tex",
                      AnchoredPatch("y\n", "Y\n", section_hash("y\n")), "Y\n")

    assert a != b


def test_list_all_returns_every_pending_diff_with_its_section(tmp_path):
    store = PendingDiffs(tmp_path / ".phd-helper")
    a = store.propose("sections/a.tex",
                      AnchoredPatch("x\n", "X\n", section_hash("x\n")), "X\n")
    b = store.propose("sections/deep/b.tex",
                      AnchoredPatch("y\n", "Y\n", section_hash("y\n")), "Y\n")

    found = {(d.id, d.section_path) for d in store.list_all()}

    assert found == {(a, "sections/a.tex"), (b, "sections/deep/b.tex")}


def test_propose_survives_a_stray_nonnumeric_file(tmp_path):
    store = PendingDiffs(tmp_path / ".phd-helper")
    d = store._dir("sections/s.tex")
    d.mkdir(parents=True)
    (d / "notes.json").write_text("{}", encoding="utf-8")

    pid = store.propose("sections/s.tex",
                        AnchoredPatch("x\n", "X\n", section_hash("x\n")),
                        "X\n")

    assert pid == "0000"


def test_list_all_skips_a_torn_file(tmp_path):
    store = PendingDiffs(tmp_path / ".phd-helper")
    store.propose("sections/s.tex",
                  AnchoredPatch("x\n", "X\n", section_hash("x\n")), "X\n")
    (store._dir("sections/s.tex") / "0001.json").write_text(
        '{"find": "x', encoding="utf-8")  # power loss mid-write

    assert len(store.list_all()) == 1


# -- file creation riding the approval (section_create) ---------------------
# The bib_append precedent generalized: a new file's bytes ride the same
# pending JSON as the main.tex patch that wires it in — one approval, both
# halves land together or neither does.


def test_a_create_diff_round_trips_through_disk(tmp_path):
    store = PendingDiffs(tmp_path / ".phd-helper")
    main = "a\n\\end{document}\n"
    patch = AnchoredPatch("\\end{document}",
                          "\\input{sections/new}\n\\end{document}",
                          section_hash(main))
    store.propose("main.tex", patch,
                  "a\n\\input{sections/new}\n\\end{document}\n",
                  create=("sections/new.tex", "New part\n"))

    d = PendingDiffs(tmp_path / ".phd-helper").list_pending("main.tex")[0]

    assert d.create_path == "sections/new.tex"
    assert d.create_content == "New part\n"


def test_a_plain_patch_carries_no_create(tmp_path):
    # Old pending files (and plain section_write diffs) have no create
    # field at all — they must read as plain patches, not half-creates.
    store = PendingDiffs(tmp_path / ".phd-helper")
    store.propose("sections/s.tex",
                  AnchoredPatch("x\n", "X\n", section_hash("x\n")), "X\n")

    d = PendingDiffs(tmp_path / ".phd-helper").list_pending("sections/s.tex")[0]

    assert d.create_path is None
    assert d.create_content is None
