"""Per-turn context assembly (SPEC §4).

Assembly is a priority list, window-agnostic: the selected section, paper
memory and skeleton never drop; over budget, pinned-paper abstracts/headings
go first, then conversation beyond the rolling summary, then far-away
section gists.
"""

from collections.abc import Callable, Sequence
from dataclasses import dataclass


@dataclass(frozen=True)
class PinnedSource:
    doc_id: str
    abstract: str
    headings: str


@dataclass(frozen=True)
class Turn:
    speaker: str
    text: str


@dataclass(frozen=True)
class ContextInputs:
    section: str
    skeleton: str
    memory: str
    distant_gists: str
    rolling_summary: str
    turns: tuple[Turn, ...] = ()
    pinned: tuple[PinnedSource, ...] = ()


@dataclass(frozen=True)
class Context:
    parts: list[str]
    # How many conversation turns survived the drop tiers. Conversation
    # only ever drops oldest-first, so the wiring rebuilds the sent
    # message list as the last N exchanges.
    conversation_kept: int = 0


def approx_tokens(text: str) -> int:
    """Server-side token estimate: ~4 chars/token, no tokenizer dep. The
    budget is an estimate by nature (window-agnostic priority, §4); this
    only has to be monotone."""
    return max(1, len(text) // 4) if text else 0


# Drop tiers: 0 never drops; over budget, tiers drop in the order 2, 1, 3.
_NEVER = 0
_CONVERSATION = 1
_PINNED = 2
_FAR_GISTS = 3
_DROP_ORDER = (_PINNED, _CONVERSATION, _FAR_GISTS)


def assemble_context(
    inputs: ContextInputs,
    budget: int,
    count_tokens: Callable[[str], int],
) -> Context:
    parts: list[tuple[int, str]] = [
        (_NEVER, inputs.skeleton),
        (_FAR_GISTS, inputs.distant_gists),
        (_NEVER, inputs.memory),
        (_NEVER, inputs.section),
    ]
    parts += [(_PINNED, f"{s.abstract}\n{s.headings}") for s in inputs.pinned]
    parts.append((_NEVER, inputs.rolling_summary))
    parts += [(_CONVERSATION, t.text) for t in inputs.turns]

    def total() -> int:
        return sum(count_tokens(text) for _, text in parts)

    for tier in _DROP_ORDER:
        while total() > budget:
            candidates = [i for i, (t, _) in enumerate(parts) if t == tier]
            if not candidates:
                break
            # pinned drops most-recently-attached first; conversation drops
            # oldest first, down to the rolling summary (tier 0)
            parts.pop(candidates[-1] if tier == _PINNED else candidates[0])

    return Context(
        parts=[text for _, text in parts],
        conversation_kept=sum(1 for tier, _ in parts if tier == _CONVERSATION))
