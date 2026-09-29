"""Per-turn context assembly (SPEC §4): a priority list, window-agnostic.
Never drop the selected section, paper memory, or skeleton; drop in order —
pinned-paper abstracts/headings, then conversation beyond the rolling
summary, then far-away section gists."""

from phd_helper.context import ContextInputs, PinnedSource, Turn, assemble_context

INPUTS = ContextInputs(
    section="SECTION BODY",
    skeleton="PAPER SKELETON",
    memory="PAPER MEMORY",
    distant_gists="FAR GISTS",
    rolling_summary="OLD CONVERSATION SUMMARY",
    turns=(Turn("user", "TURN ONE"), Turn("agent", "TURN TWO")),
    pinned=(PinnedSource(doc_id="d1", abstract="PINNED ABSTRACT", headings="PINNED HEADINGS"),),
)


def count_words(text: str) -> int:
    return len(text.split())


def test_context_that_fits_the_budget_holds_every_part_in_order():
    context = assemble_context(INPUTS, budget=1000, count_tokens=count_words)

    joined = "\n".join(context.parts)
    for part in (
        "PAPER SKELETON",
        "FAR GISTS",
        "PAPER MEMORY",
        "SECTION BODY",
        "PINNED ABSTRACT",
        "OLD CONVERSATION SUMMARY",
        "TURN ONE",
        "TURN TWO",
    ):
        assert part in joined

    positions = [joined.index(p) for p in ("PAPER SKELETON", "SECTION BODY", "PINNED ABSTRACT", "TURN TWO")]
    assert positions == sorted(positions)


def test_over_budget_drops_pinned_source_abstracts_and_headings_first():
    # total is 19 words; 15 fits only once the 4-word pinned part is gone
    context = assemble_context(INPUTS, budget=15, count_tokens=count_words)

    joined = "\n".join(context.parts)
    assert "PINNED ABSTRACT" not in joined
    for kept in (
        "PAPER SKELETON",
        "FAR GISTS",
        "PAPER MEMORY",
        "SECTION BODY",
        "OLD CONVERSATION SUMMARY",
        "TURN ONE",
        "TURN TWO",
    ):
        assert kept in joined


def test_still_over_budget_drops_oldest_conversation_turns_first():
    # 19 total; pinned (4) drops to 15, then the oldest turn (2) to 13
    context = assemble_context(INPUTS, budget=13, count_tokens=count_words)

    joined = "\n".join(context.parts)
    assert "TURN ONE" not in joined
    assert "TURN TWO" in joined
    assert "OLD CONVERSATION SUMMARY" in joined
    assert "PINNED ABSTRACT" not in joined


def test_still_over_budget_drops_far_away_section_gists_last():
    # 19 total; pinned (4) and both turns (4) drop to 11, then gists (2) to 9
    context = assemble_context(INPUTS, budget=9, count_tokens=count_words)

    joined = "\n".join(context.parts)
    assert "FAR GISTS" not in joined
    for kept in ("PAPER SKELETON", "PAPER MEMORY", "SECTION BODY", "OLD CONVERSATION SUMMARY"):
        assert kept in joined


def test_never_drops_the_section_memory_or_skeleton_even_when_budget_is_hopeless():
    context = assemble_context(INPUTS, budget=1, count_tokens=count_words)

    joined = "\n".join(context.parts)
    for kept in ("PAPER SKELETON", "PAPER MEMORY", "SECTION BODY"):
        assert kept in joined
