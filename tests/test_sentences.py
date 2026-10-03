# The sentence gate: pure, deterministic, no clock/IO (segmenting.py's
# idiom). Tests feed deltas in arbitrary splits — the gate must see the
# same sentences however the token stream chunks them.
from phd_helper.sentences import SentenceGate


def collect(gate, deltas):
    out = []
    for d in deltas:
        out.extend(gate.push(d))
    tail = gate.flush()
    if tail:
        out.append(tail)
    return out


def test_splits_on_terminal_punctuation():
    g = SentenceGate()
    assert collect(g, ["Let me look at the intro. ",
                       "It needs a citation! ", "Done?"]) == [
        "Let me look at the intro.", "It needs a citation!", "Done?"]


def test_abbreviation_period_does_not_split():
    g = SentenceGate()
    assert collect(g, ["See e.g. the paper, etc. More text."]) == [
        "See e.g. the paper, etc. More text."]
    g = SentenceGate()
    assert collect(g, ["As shown in Fig. 3, i.e. the loss drops."]) == [
        "As shown in Fig. 3, i.e. the loss drops."]


def test_single_letter_initial_does_not_split():
    g = SentenceGate()
    assert collect(g, ["Cite J. Smith here. Thanks."]) == [
        "Cite J. Smith here.", "Thanks."]


def test_decimal_and_version_do_not_split():
    g = SentenceGate()
    assert collect(g, ["Section 3.14 explains v2. Next."]) == [
        "Section 3.14 explains v2.", "Next."]


def test_never_splits_inside_braces():
    g = SentenceGate()
    assert collect(g, ["Patch \\section{Intro. Give} now. Done."]) == [
        "Patch \\section{Intro. Give} now.", "Done."]


def test_never_splits_inside_inline_math():
    g = SentenceGate()
    assert collect(g, ["The loss $a. b$ converges. Good."]) == [
        "The loss $a. b$ converges.", "Good."]


def test_escaped_dollar_does_not_toggle_math():
    g = SentenceGate()
    assert collect(g, ["Cost \\$5. Pay now."]) == ["Cost \\$5.", "Pay now."]


def test_paragraph_break_splits():
    g = SentenceGate()
    assert collect(g, ["First para\n\nSecond para."]) == [
        "First para", "Second para."]


def test_paragraph_break_respects_min_chars():
    g = SentenceGate(min_chars=4)
    # a stub paragraph merges forward like a stub sentence — the
    # sidecar never speaks "ab" alone
    assert collect(g, ["ab\n\ncd ef gh."]) == ["ab\n\ncd ef gh."]


def test_flush_returns_tail():
    g = SentenceGate()
    assert g.push("No punctuation at the end") == []
    assert g.flush() == "No punctuation at the end"


def test_flush_returns_armed_tail():
    g = SentenceGate()
    # a period + space only ARMS the break; the next non-space char
    # confirms it — end of message confirms it too
    assert g.push("One. ") == []
    assert g.flush() == "One."


def test_flush_empty_after_clean_split():
    g = SentenceGate()
    assert g.push("One.\n\n") == ["One."]  # the paragraph break confirms
    assert g.flush() == ""


def test_drop_tail_discards_partial():
    g = SentenceGate()
    # a tool-call fragment appearing mid-sentence: the partial is gone
    assert g.push("Let me check. Now the too") == ["Let me check."]
    assert g.drop_tail() == "Now the too"
    assert g.flush() == ""


def test_short_fragment_merges_forward():
    g = SentenceGate(min_chars=4)
    assert collect(g, ["No. Yes. A longer one."]) == [
        "No. Yes.", "A longer one."]


def test_list_marker_merges_forward_at_the_default():
    # The live model streams "1. Do this" lists (probe-pinned); a bare
    # "1." pushed alone would be the sidecar speaking a dot.
    g = SentenceGate()
    assert collect(g, ["1. First item. Second item."]) == [
        "1. First item.", "Second item."]


def test_unbalanced_brace_flushes_as_tail():
    g = SentenceGate()
    assert collect(g, ["Broken \\textbf{unclosed. Still"]) == [
        "Broken \\textbf{unclosed. Still"]


def test_split_is_independent_of_delta_boundaries():
    text = ("See e.g. the paper. It cites J. Smith on $a. b$ loss. "
            "Patch \\section{Intro. Now} ends.\n\nNext para here.")
    expected = collect(SentenceGate(), [text])
    assert len(expected) == 4  # three sentences + the paragraph tail
    for i in range(1, len(text)):
        assert collect(SentenceGate(), [text[:i], text[i:]]) == expected, \
            f"split at {i} changed the sentences"
