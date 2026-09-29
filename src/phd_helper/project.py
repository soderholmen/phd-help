"""A paper project on the server filesystem (SPEC §5/§7).

The agent's write path: propose (deterministic lint runs on the result
BEFORE the diff is shown — failures bounce to the agent) -> pending diff ->
approve (apply-time hash check / re-anchor, snapshot, write) -> undo. All
persistence lives in ``.phd-helper/`` inside the project folder.
"""

from pathlib import Path

from phd_helper.bibtex import (BibEntry, dedupe_key, format_entry, make_key,
                               parse_bib, parse_entry, same_paper, with_key)
from phd_helper.history import SectionHistory
from phd_helper.lint import lint_latex
from phd_helper.pending import PendingDiff, PendingDiffs
from phd_helper.patches import AnchoredPatch, ApplyResult, apply_patch, section_hash
from phd_helper.sections import parse_section_tree


class ProposeError(Exception):
    """A proposed patch that never becomes a diff: missing anchor or lint
    failure. The message is the bounce-back for the model."""


class Project:
    def __init__(self, root_dir: Path, root_file: str = "main.tex"):
        self.root = Path(root_dir)
        self.root_file = root_file
        state = self.root / ".phd-helper"
        self.pending = PendingDiffs(state)
        self.history = SectionHistory(state)

    # -- reading ----------------------------------------------------------

    def _files(self) -> dict[str, str]:
        out = {}
        for p in self.root.rglob("*.tex"):
            out[str(p.relative_to(self.root)).replace("\\", "/")] = \
                p.read_text(encoding="utf-8")
        return out

    def section_tree(self):
        return parse_section_tree(self._files(), self.root_file)

    def read_section(self, path: str) -> str:
        return (self.root / path).read_text(encoding="utf-8")

    def read_bib(self) -> str:
        bib = self.root / "refs.bib"
        return bib.read_text(encoding="utf-8") if bib.exists() else ""

    def _bib_keys(self) -> set[str]:
        return {e.key for e in parse_bib(self.read_bib())}

    # -- the write path -----------------------------------------------------

    def propose_patch(self, path: str, find: str, replace: str) -> PendingDiff:
        """Lint-checked proposal; becomes a pending diff, never a write."""
        current = self.read_section(path)
        if find not in current:
            raise ProposeError(
                f"find text is not present in the section: {find[:80]!r}")
        patch = AnchoredPatch(find=find, replace=replace,
                              base_hash=section_hash(current))
        trial = apply_patch(current, patch)
        if not trial.applied:
            raise ProposeError(trial.reason)
        problems = lint_latex(trial.text, self._bib_keys())
        if problems:
            raise ProposeError("lint failed: " + "; ".join(
                p.message for p in problems))
        diff_id = self.pending.propose(path, patch, trial.text)
        return PendingDiff(id=diff_id, section_path=path,
                           patch=patch, proposed_text=trial.text)

    def propose_cite(self, path: str, find: str, replace: str,
                     lookup, entry: BibEntry) -> PendingDiff:
        """The cite loop (§6): bib entry + prose + \\cite in one approval.

        The agent cites by the lookup id it used; the final key is decided
        here — reused for an owned paper, renumbered on collision — and the
        \\cite command is rewritten to it.
        """
        existing = parse_bib(self.read_bib())
        match = next((e for e in existing if same_paper(e, entry)), None)
        if match is not None:
            final_key, bib_append = match.key, None
        else:
            final_key = dedupe_key(entry.key or make_key(entry),
                                   {e.key for e in existing})
            bib_append = format_entry(with_key(entry, final_key))
        lookup_id = lookup.arxiv or lookup.doi or lookup.title
        cited = f"\\cite{{{lookup_id}}}"
        if cited not in replace:
            raise ProposeError(
                f"replace must cite the paper as {cited}")
        replace = replace.replace(cited, f"\\cite{{{final_key}}}")
        current = self.read_section(path)
        if find not in current:
            raise ProposeError(
                f"find text is not present in the section: {find[:80]!r}")
        patch = AnchoredPatch(find=find, replace=replace,
                              base_hash=section_hash(current))
        trial = apply_patch(current, patch)
        if not trial.applied:
            raise ProposeError(trial.reason)
        keys = {e.key for e in existing} | {final_key}
        problems = lint_latex(trial.text, keys)
        if problems:
            raise ProposeError("lint failed: " + "; ".join(
                p.message for p in problems))
        diff_id = self.pending.propose(path, patch, trial.text,
                                       bib_append=bib_append)
        return PendingDiff(id=diff_id, section_path=path, patch=patch,
                           proposed_text=trial.text, bib_append=bib_append,
                           cite_key=final_key)

    def list_pending(self, path: str) -> list[PendingDiff]:
        return self.pending.list_pending(path)

    def apply_pending(self, path: str, diff_id: str) -> ApplyResult:
        """Approve: bib re-check, clobber check / re-anchor, snapshot, write."""
        pend = [d for d in self.pending.list_pending(path) if d.id == diff_id]
        if not pend:
            return ApplyResult(applied=False, reason="no such pending diff")
        diff = pend[0]
        append = diff.bib_append
        if append is not None:
            # Propose-time checks go stale: re-check the key at approval (§6).
            incoming = parse_entry(append)
            held = {e.key: e for e in parse_bib(self.read_bib())}.get(
                incoming.key)
            if held is not None:
                if not same_paper(held, incoming):
                    return ApplyResult(
                        applied=False,
                        reason=f"key '{held.key}' was taken by another paper "
                               "since this diff was proposed")
                append = None  # identical entry already landed: skip append
        current = self.read_section(path)
        result = apply_patch(current, diff.patch)
        if not result.applied:
            return result  # ambiguous re-anchor: reason shown, stays pending
        self.write_section(path, result.text)
        if append:
            # One approval covers all three (§6): the entry rides the diff.
            bib = self.read_bib()
            if bib and not bib.endswith("\n"):
                bib += "\n"  # never glue the entry onto the last line
            (self.root / "refs.bib").write_text(bib + append, encoding="utf-8")
        self.history.record_apply(path, current, diff.patch, result.text)
        self.pending.resolve(path, diff_id)
        return result

    def reject_pending(self, path: str, diff_id: str) -> None:
        self.pending.resolve(path, diff_id)

    def write_section(self, path: str, text: str) -> None:
        (self.root / path).write_text(text, encoding="utf-8")

    # -- undo ---------------------------------------------------------------

    def undo_last(self, path: str) -> ApplyResult:
        entries = self.history.entries(path)  # newest first
        if not entries:
            return ApplyResult(applied=False, reason="nothing to undo")
        return self.undo_entry(path, entries[0].id)

    def undo_entry(self, path: str, entry_id: str) -> ApplyResult:
        current = self.read_section(path)
        result = self.history.undo_entry(path, entry_id, current)
        if result.applied:
            self.write_section(path, result.text)
        return result
