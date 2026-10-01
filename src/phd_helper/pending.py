"""Pending diffs (SPEC §7): a proposed diff stays pending across restarts,
re-presented on reopen through the apply-time hash check / re-anchor path.
"""

import json
from dataclasses import dataclass, replace
from pathlib import Path

from phd_helper.patches import AnchoredPatch, apply_patch


@dataclass(frozen=True)
class PendingDiff:
    id: str
    section_path: str
    patch: AnchoredPatch
    proposed_text: str
    # The cite loop (§6): a bib entry rides the same approval as the patch.
    bib_append: str | None = None
    cite_key: str | None = None


@dataclass(frozen=True)
class ReconciledDiff:
    """A pending diff as it should be re-presented on reopen, or the reason
    it bounced (stale and not re-anchorable)."""

    diff: PendingDiff
    bounce_reason: str | None


class PendingDiffs:
    def __init__(self, project_dir: Path):
        self._root = project_dir / "pending"

    def _dir(self, section_path: str) -> Path:
        slug = section_path.replace("/", "__").replace("\\", "__")
        return self._root / slug

    def propose(self, section_path: str, patch: AnchoredPatch,
                proposed_text: str, bib_append: str | None = None) -> str:
        d = self._dir(section_path)
        d.mkdir(parents=True, exist_ok=True)
        # Ids are globally unique, not per-section: voice addresses a diff
        # by id alone (§3), and the shell's reducer keys cards by diff_id.
        # Non-numeric stems are strays, not diffs — they must not poison
        # every proposal in the project.
        existing = [int(p.stem) for p in self._root.rglob("*.json")
                    if p.stem.isdigit()]
        pid = f"{max(existing, default=-1) + 1:04d}"
        (d / f"{pid}.json").write_text(
            json.dumps(
                {
                    "find": patch.find,
                    "replace": patch.replace,
                    "base_hash": patch.base_hash,
                    "proposed_text": proposed_text,
                    "bib_append": bib_append,
                    "section": section_path,
                }
            ),
            encoding="utf-8",
        )
        return pid

    def list_all(self) -> list[PendingDiff]:
        """Every pending diff across all sections, addressed by its globally
        unique id — the approval window's view (§3)."""
        return [diff for p in sorted(self._root.rglob("*.json"))
                if p.stem.isdigit()
                and (diff := self._read(p)) is not None]

    def _read(self, p: Path,
              section: str | None = None) -> PendingDiff | None:
        """One meta->PendingDiff reader for both listings. A torn file
        (propose's write is not atomic) reads as absent: invisible to
        decisions, recoverable by hand — never a crash that takes every
        turn with it (§8)."""
        try:
            meta = json.loads(p.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        if section is None:
            section = meta.get("section") or \
                p.parent.name.replace("__", "/")  # pre-section-field files
        return PendingDiff(
            id=p.stem,
            section_path=section,
            patch=AnchoredPatch(
                meta["find"], meta["replace"], meta["base_hash"]
            ),
            proposed_text=meta["proposed_text"],
            bib_append=meta.get("bib_append"),
        )

    def list_pending(self, section_path: str) -> list[PendingDiff]:
        d = self._dir(section_path)
        if not d.is_dir():
            return []
        return [diff for p in sorted(d.glob("*.json"))
                if (diff := self._read(p, section_path)) is not None]

    def resolve(self, section_path: str, diff_id: str) -> None:
        """The approval window closed — applied or discarded, either way the
        diff stops being pending."""
        (self._dir(section_path) / f"{diff_id}.json").unlink()

    def reconcile(self, section_path: str, current_text: str) -> list[ReconciledDiff]:
        """Re-present each pending diff through the apply-time hash check /
        re-anchor path: stale-but-re-anchorable re-anchors (the re-presented
        diff shows the re-anchored result); ambiguous bounces with reason."""
        outcomes: list[ReconciledDiff] = []
        for diff in self.list_pending(section_path):
            result = apply_patch(current_text, diff.patch)
            if result.applied:
                outcomes.append(
                    ReconciledDiff(replace(diff, proposed_text=result.text), None)
                )
            else:
                (self._dir(section_path) / f"{diff.id}.json").unlink()
                outcomes.append(ReconciledDiff(diff, result.reason))
        return outcomes
