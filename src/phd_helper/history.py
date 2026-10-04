"""Section history (SPEC §5): the per-section list of snapshots — the file's
saved state before each agent apply or full-section rewrite — persistent per
project in .phd-helper/ (rides Syncthing, user-inspectable, no git).
"""

import json
from dataclasses import dataclass
from pathlib import Path

from phd_helper.patches import AnchoredPatch, ApplyResult, apply_patch, inverse_patch


@dataclass(frozen=True)
class SnapshotEntry:
    id: str


class SectionHistory:
    def __init__(self, project_dir: Path, keep: int = 20):
        self._root = project_dir / "history"
        self._keep = keep

    def _dir(self, section_path: str) -> Path:
        slug = section_path.replace("/", "__").replace("\\", "__")
        return self._root / slug

    def snapshot(self, section_path: str, prior_text: str) -> str:
        d = self._dir(section_path)
        d.mkdir(parents=True, exist_ok=True)
        existing = [int(p.stem) for p in d.glob("*.tex")]
        sid = f"{max(existing, default=-1) + 1:04d}"
        (d / f"{sid}.tex").write_text(prior_text, encoding="utf-8")
        for stale in sorted(d.glob("*.tex"))[: -self._keep]:
            stale.unlink()
            stale.with_suffix(".json").unlink(missing_ok=True)
        return sid

    def record_apply(
        self,
        section_path: str,
        prior_text: str,
        patch: AnchoredPatch,
        applied_text: str,
    ) -> str:
        """Snapshot the prior bytes and remember the apply, so the history
        list can undo this specific apply later (any past apply is
        undoable, not just the latest)."""
        sid = self.snapshot(section_path, prior_text)
        meta = {
            "find": patch.find,
            "replace": patch.replace,
            "base_hash": patch.base_hash,
            "applied_text": applied_text,
        }
        (self._dir(section_path) / f"{sid}.json").write_text(
            json.dumps(meta), encoding="utf-8"
        )
        return sid

    def entry_meta(self, section_path: str, entry_id: str) -> dict:
        """What the apply did (find/replace/base_hash/applied_text) —
        the undo door and the voice tool name the change with it."""
        return json.loads(
            (self._dir(section_path) / f"{entry_id}.json").read_text(
                encoding="utf-8"
            )
        )

    def undo_entry(
        self, section_path: str, entry_id: str, current_text: str
    ) -> ApplyResult:
        """Undo this entry's apply against the file as it stands now, riding
        the same re-anchor path as any patch."""
        meta = json.loads(
            (self._dir(section_path) / f"{entry_id}.json").read_text(encoding="utf-8")
        )
        patch = AnchoredPatch(meta["find"], meta["replace"], meta["base_hash"])
        return apply_patch(current_text, inverse_patch(meta["applied_text"], patch))

    def revert_text(self, section_path: str, snapshot_id: str) -> str:
        """The file's bytes as of that snapshot — revert discards everything
        since, including the user's own edits (on-screen confirmation is the
        caller's gate, SPEC §5)."""
        return (self._dir(section_path) / f"{snapshot_id}.tex").read_text(
            encoding="utf-8"
        )

    def entries(self, section_path: str) -> list[SnapshotEntry]:
        d = self._dir(section_path)
        if not d.is_dir():
            return []
        return [
            SnapshotEntry(id=p.stem)
            for p in sorted(d.glob("*.tex"), reverse=True)
        ]

    def latest_entry_across(
        self, section_paths: list[str]
    ) -> tuple[str, str] | None:
        """The newest entry over the given sections — the undo door's
        \"last change to the paper\". sid is a per-section counter and
        meta has no timestamp, so mtime is the only cross-section order
        (single-machine reality; Syncthing mtime drift is the known
        caveat). Returns (section_path, sid), or None when no section
        has history."""
        best: tuple[float, str, str] | None = None
        for path in section_paths:
            entries = self.entries(path)
            if not entries:
                continue
            sid = entries[0].id  # newest first
            mtime = (self._dir(path) / f"{sid}.tex").stat().st_mtime
            if best is None or mtime > best[0]:
                best = (mtime, path, sid)
        return None if best is None else (best[1], best[2])
