"""Anchored patches: the default write into a section (SPEC §5).

An anchored patch is an exact find/replace whose untouched prose stays
byte-identical. The base hash is the section's state when the patch was
computed — the apply-time clobber check compares against it: if the file
changed since, the patch deterministically fuzzy-reanchors to a unique
close match, or is rejected on ambiguity. The user's own edits are never
clobbered.
"""

import difflib
import hashlib
from dataclasses import dataclass


@dataclass(frozen=True)
class AnchoredPatch:
    find: str
    replace: str
    base_hash: str


@dataclass(frozen=True)
class ApplyResult:
    applied: bool
    text: str | None = None
    reason: str | None = None


def section_hash(section: str) -> str:
    return hashlib.sha256(section.encode("utf-8")).hexdigest()


def inverse_patch(applied_text: str, patch: AnchoredPatch) -> AnchoredPatch:
    """The undo of an apply: the same machinery, find and replace swapped.

    Stamped against the section as it stands right after the apply, so the
    undo rides the same clobber check and re-anchor path as any patch.
    """
    return AnchoredPatch(
        find=patch.replace,
        replace=patch.find,
        base_hash=section_hash(applied_text),
    )


CLOSE_MATCH_CUTOFF = 0.75


def apply_patch(section: str, patch: AnchoredPatch) -> ApplyResult:
    matches = section.count(patch.find)
    if matches == 1:
        return ApplyResult(
            applied=True,
            text=section.replace(patch.find, patch.replace, 1),
        )
    if matches > 1:
        return ApplyResult(
            applied=False,
            reason="find text matches multiple places in the section",
        )
    # find is absent: only a stale patch (file changed since it was computed)
    # gets the fuzzy re-anchor path; a fresh patch with an absent find is bogus.
    if patch.base_hash == section_hash(section):
        return ApplyResult(
            applied=False,
            reason="find text is not present in the section",
        )
    span = _unique_close_match(section, patch.find)
    if span is None:
        return ApplyResult(
            applied=False,
            reason="find text is not present in the section",
        )
    merged = _replay_patch_onto_span(patch, span)
    if merged is None:
        return ApplyResult(
            applied=False,
            reason="the user's edit conflicts with this patch",
        )
    return ApplyResult(applied=True, text=section.replace(span, merged, 1))


def _unique_close_match(section: str, find: str) -> str | None:
    """The one span in section that closely matches find, or None if not unique.

    Deterministic window scan: every window (sized 0.8x..1.25x of find) whose
    similarity clears the cutoff is a hit; overlapping hits cluster to one
    location. Exactly one cluster with hits above the cutoff is a unique
    close match; zero or many is ambiguity.
    """
    n = len(find)
    if n == 0 or len(section) < n:
        return None
    hits: list[tuple[int, float, str]] = []
    matcher = difflib.SequenceMatcher()
    matcher.set_seq2(find)
    for width in range(max(1, int(n * 0.8)), int(n * 1.25) + 1):
        for start in range(len(section) - width + 1):
            window = section[start : start + width]
            matcher.set_seq1(window)
            ratio = matcher.ratio()
            if ratio >= CLOSE_MATCH_CUTOFF:
                hits.append((start, ratio, window))
    if not hits:
        return None
    hits.sort(key=lambda h: (h[0], len(h[2])))
    clusters: list[list[tuple[int, float, str]]] = []
    for hit in hits:
        if clusters and hit[0] < clusters[-1][-1][0] + n:
            clusters[-1].append(hit)
        else:
            clusters.append([hit])
    if len(clusters) != 1:
        return None
    best = max(clusters[0], key=lambda h: h[1])
    return best[2]


MERGE_GAP = 4


def _merged_edits(find: str, replace: str) -> list[tuple[int, int, int, int]]:
    """difflib opcodes coalesced into whole-word-ish edits.

    Raw opcodes are character-grained ("u"->"i", "l"->"tered"), and a
    one-character target is ambiguous inside a span. Edits separated by an
    equal run shorter than MERGE_GAP are merged into one edit.
    """
    edits: list[tuple[int, int, int, int]] = []
    cur: tuple[int, int, int, int] | None = None
    for tag, i1, i2, j1, j2 in difflib.SequenceMatcher(None, find, replace).get_opcodes():
        if tag == "equal":
            if cur is not None:
                if i2 - i1 >= MERGE_GAP:
                    edits.append(cur)
                    cur = None
                else:
                    cur = (cur[0], i2, cur[2], j2)
            continue
        cur = (cur[0], i2, cur[2], j2) if cur is not None else (i1, i2, j1, j2)
    if cur is not None:
        edits.append(cur)
    return edits


def _replay_patch_onto_span(patch: AnchoredPatch, span: str) -> str | None:
    """Apply the patch's own edits onto the close-match span, right to left.

    Each edit is located by its exact target text inside the span, so prose
    the user changed elsewhere in the span survives. If an edit's target is
    missing (the user edited exactly what the patch changes) or ambiguous,
    the merge is a conflict.
    """
    result = span
    for i1, i2, j1, j2 in reversed(_merged_edits(patch.find, patch.replace)):
        target = patch.find[i1:i2]
        new = patch.replace[j1:j2]
        if target:
            if result.count(target) != 1:
                return None
            result = result.replace(target, new, 1)
        else:
            # pure insertion: anchor on surrounding context
            before = patch.find[max(0, i1 - 16) : i1]
            after = patch.find[i1 : i1 + 16]
            anchor = before + after
            if result.count(anchor) != 1:
                return None
            result = result.replace(anchor, before + new + after, 1)
    return result
