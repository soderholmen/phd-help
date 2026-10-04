"""A paper project on the server filesystem (SPEC §5/§7).

The agent's write path: propose (deterministic lint runs on the result
BEFORE the diff is shown — failures bounce to the agent) -> pending diff ->
approve (apply-time hash check / re-anchor, snapshot, write) -> undo. All
persistence lives in ``.phd-helper/`` inside the project folder.
"""

import json
import re
from pathlib import Path, PurePosixPath

from phd_helper import sessionlog
from phd_helper.bibtex import (BibEntry, dedupe_key, format_entry, make_key,
                               parse_bib, parse_entry, same_paper, with_key)
from phd_helper.history import SectionHistory
from phd_helper.lint import lint_latex
from phd_helper.pending import PendingDiff, PendingDiffs
from phd_helper.patches import AnchoredPatch, ApplyResult, apply_patch, section_hash
from phd_helper.sections import parse_section_tree
from phd_helper.skeleton import build_skeleton


class ProposeError(Exception):
    """A proposed patch that never becomes a diff: missing anchor or lint
    failure. The message is the bounce-back for the model."""


class Project:
    def __init__(self, root_dir: Path, root_file: str = "main.tex"):
        self.root = Path(root_dir)
        self.root_file = root_file
        state = self.root / ".phd-helper"
        self.state_dir = state
        self.pending = PendingDiffs(state)
        self.history = SectionHistory(state)

    # -- reading ----------------------------------------------------------

    def files(self, degrade: bool = False) -> dict[str, str]:
        """The project's .tex files, for callers that walk the tree and
        the bytes together (gist staleness) without re-rglobbing.
        degrade=True: an unreadable file reads as absent instead of
        failing the call — the read view shows the rest of the paper."""
        return self._files(degrade)

    def _files(self, degrade: bool = False) -> dict[str, str]:
        out = {}
        for p in self.root.rglob("*.tex"):
            rel = p.relative_to(self.root)
            if rel.parts[0] == ".phd-helper":
                continue  # state dir: history snapshots are bytes, not files
            try:
                out[str(rel).replace("\\", "/")] = p.read_text(encoding="utf-8")
            except OSError:
                if not degrade:
                    raise
        return out

    def section_tree(self):
        return parse_section_tree(self._files(), self.root_file)

    def skeleton(self) -> str:
        """The §4 never-drop paper map: title, abstract, headings."""
        return build_skeleton(self._files(), self.root_file)

    def read_section(self, path: str) -> str:
        return (self.root / path).read_text(encoding="utf-8")

    # -- rolling summaries (SPEC §7) ------------------------------------------

    def load_summaries(self) -> dict:
        """{path: [{"date":…, "text":…}, …]}; dated session dividers."""
        try:
            return json.loads(
                (self.state_dir / "summaries.json").read_text(
                    encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {}

    def save_summaries(self, summaries: dict) -> None:
        self.state_dir.mkdir(parents=True, exist_ok=True)
        tmp = self.state_dir / "summaries.json.tmp"
        tmp.write_text(json.dumps(summaries, ensure_ascii=False),
                       encoding="utf-8")
        tmp.replace(self.state_dir / "summaries.json")

    # -- verbatim conversation history (SPEC §7) ----------------------------

    def append_chat(self, records) -> None:
        sessionlog.append(self.state_dir / "chat.jsonl", records)

    def chat_tail(self) -> list[dict]:
        """The verbatim messages after the last divider — what a reopened
        sitting resumes with (older talk rides on as rolling summaries)."""
        return sessionlog.read_tail(self.state_dir / "chat.jsonl")

    def chat_divider(self) -> dict | None:
        return sessionlog.last_divider(self.state_dir / "chat.jsonl")

    def write_chat_divider(self, record: dict) -> None:
        sessionlog.append(self.state_dir / "chat.jsonl", [record])

    # -- paper memory (SPEC §4) ---------------------------------------------

    def load_memory(self) -> str:
        """The persistent decisions/claims/terminology/TODOs file; a
        project that never distilled one reads as empty."""
        try:
            return (self.state_dir / "memory.md").read_text(
                encoding="utf-8")
        except OSError:
            return ""

    def save_memory(self, text: str) -> None:
        # Not a paper file: memory is agent state under .phd-helper/,
        # so it skips the §5 pending-diff approval path (the §4 side
        # panel is the user's edit surface).
        self.state_dir.mkdir(parents=True, exist_ok=True)
        (self.state_dir / "memory.md").write_text(text, encoding="utf-8")

    # -- gist cache (SPEC §4) -----------------------------------------------

    def load_gists(self) -> dict:
        """{path: {"sha":…, "gist":…}}; a missing or torn file reads as
        no gists yet — staleness regenerates them, nothing else (§8)."""
        try:
            return json.loads(
                (self.state_dir / "gists.json").read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {}

    def save_gists(self, cache: dict) -> None:
        # Atomic replace: a background refresh and a turn's read must
        # never meet a half-written cache.
        self.state_dir.mkdir(parents=True, exist_ok=True)
        tmp = self.state_dir / "gists.json.tmp"
        tmp.write_text(json.dumps(cache, ensure_ascii=False),
                       encoding="utf-8")
        tmp.replace(self.state_dir / "gists.json")

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

    def propose_create(self, path: str, content: str,
                       after: str | None = None) -> PendingDiff:
        """section_create: a new part + its wiring into the root file, one
        approval. The patch is the ``\\input`` insertion; the new file's
        bytes ride the diff the way a bib entry rides a cite diff — both
        halves land together on apply, or neither does."""
        p = PurePosixPath(path.replace("\\", "/"))
        if (not path.endswith(".tex") or p.is_absolute()
                or ".." in p.parts or path == self.root_file):
            raise ProposeError(
                f"cannot create {path!r}: must be a relative .tex path "
                "inside the project")
        if (self.root / p).exists():
            raise ProposeError(f"file already exists: {path}")
        problems = lint_latex(content, self._bib_keys())
        if problems:
            raise ProposeError("lint failed: " + "; ".join(
                pr.message for pr in problems))
        main = self.read_section(self.root_file)
        stem = str(p)[: -len(".tex")]
        if after is not None:
            line = self._input_line_for(main, after)
            if line is None:
                raise ProposeError(
                    f"no \\input line for '{after}' in {self.root_file}")
            find, replace = line, f"{line}\n\\input{{{stem}}}"
        else:
            # default: the part joins the end of the document
            find = "\\end{document}"
            replace = f"\\input{{{stem}}}\n{find}"
        patch = AnchoredPatch(find=find, replace=replace,
                              base_hash=section_hash(main))
        trial = apply_patch(main, patch)
        if not trial.applied:
            raise ProposeError(trial.reason)
        diff_id = self.pending.propose(self.root_file, patch, trial.text,
                                       create=(str(p), content))
        return PendingDiff(id=diff_id, section_path=self.root_file,
                           patch=patch, proposed_text=trial.text,
                           create_path=str(p), create_content=content)

    def _input_line_for(self, main: str, after: str) -> str | None:
        """The exact \\input line in main that resolves to `after` — the
        anchor for an insert-after, quoted from disk truth, not guessed."""
        want = after if after.endswith(".tex") else f"{after}.tex"
        for line in main.splitlines():
            m = re.search(r"\\(?:input|include)\{([^}]*)\}", line)
            if m:
                arg = m.group(1)
                if (arg if arg.endswith(".tex") else f"{arg}.tex") == want:
                    return line
        return None

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
        if diff.create_path is not None:
            # Propose-time guards go stale: a file that appeared since
            # the proposal bounces the whole diff — bytes the user wrote
            # by hand outrank the agent's proposal, wiring included.
            if (self.root / diff.create_path).exists():
                return ApplyResult(
                    applied=False,
                    reason=f"file already exists: {diff.create_path}")
        current = self.read_section(path)
        result = apply_patch(current, diff.patch)
        if not result.applied:
            return result  # ambiguous re-anchor: reason shown, stays pending
        if diff.create_path is not None:
            # The wiring trial passed: land the file first, then the
            # line that pulls it in — a half-create without the wiring
            # would be an invisible orphan.
            target = self.root / diff.create_path
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(diff.create_content or "", encoding="utf-8")
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

    def undo_last_change(self) -> dict:
        """Undo the last change to the paper, whichever section it
        touched, and name what went back — the door's and the voice
        tool's shared kernel, so both adapters stay thin. Exactly one
        of "undone"/"reason": {"undone": {"section", "find",
        "replace"}} (display-truncated) or {"reason": …, "empty": bool}
        — `empty` separates "nothing to undo" (a normal answer, the
        door's 200) from a conflict: the inverse patch cannot
        re-anchor, or the file or its meta moved under us (the door's
        409). (files() reads the bytes we don't need; the paper is
        small and this is a user-triggered door, not a per-turn path.)"""
        try:
            found = self.history.latest_entry_across(
                list(self._files(degrade=True)))
            if found is None:
                return {"reason": "nothing to undo", "empty": True}
            path, sid = found
            result = self.undo_entry(path, sid)
            if not result.applied:
                return {"reason": result.reason}
            meta = self.history.entry_meta(path, sid)
        except OSError as e:
            return {"reason": f"undo failed: {e}"}
        return {"undone": {"section": path, "find": meta["find"][:80],
                           "replace": meta["replace"][:80]}}
