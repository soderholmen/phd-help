# saytext: what the voice says from the text (stepwise writing).
# Pure kernel — draft capture from the final text, fence-free speech,
# LaTeX-to-words for the TTS. No clock, no I/O (sentences.py's idiom):
# the streaming parts must see the same output however the token
# stream chunks the input.
from phd_helper.saytext import (Unfence, extract_draft, for_speech,
                                strip_fences)


def collect(u, deltas):
    out = ""
    for d in deltas:
        out += u.push(d)
    return out + u.flush()


# --- extract_draft ----------------------------------------------------

def test_extract_draft_returns_block_content():
    text = ("Here you go:\n\n```latex\nWe propose X.\nMore prose.\n"
            "```\nSay add when you want it.")
    assert extract_draft(text) == "We propose X.\nMore prose."


def test_extract_draft_returns_the_last_complete_block():
    text = ("```latex\nfirst\n```\nchatter\n```latex\nsecond\n```")
    assert extract_draft(text) == "second"


def test_extract_draft_needs_a_closing_fence():
    # an unclosed block is a truncated turn, not a draft
    assert extract_draft("```latex\nunclosed") is None
    assert extract_draft("no fences here") is None


def test_extract_draft_tolerates_indented_and_tagged_fences():
    assert extract_draft("  ```\ncontent\n  ```  ") == "content"
    assert extract_draft("```latex\nx\n```") == "x"


# --- Unfence ----------------------------------------------------------

def test_unfence_drops_delimiter_lines_keeps_content():
    u = Unfence()
    out = collect(u, ["Draft:\n```latex\nWe propose X.\n```\nOK."])
    assert out == "Draft:\n\nWe propose X.\n\nOK."


def test_unfence_survives_fence_split_across_deltas():
    u = Unfence()
    assert collect(u, ["```", "`latex\nWe propose.\n``", "`"]) == \
        "\nWe propose.\n"


def test_unfence_streams_prose_without_waiting_for_a_newline():
    # The overlap must survive the filter: a line is released by its
    # first ordinary character, never held to a newline — a reply
    # without newlines must still reach the gate as it streams.
    u = Unfence()
    assert u.push("We propose X. ") == "We propose X. "
    assert u.flush() == ""


def test_unfence_flush_emits_backtick_stub_remainder():
    u = Unfence()
    assert u.push("Body.\n``") == "Body.\n"
    assert u.flush() == "``"  # two backticks never make a fence


def test_unfence_flush_dies_on_fence_remainder():
    u = Unfence()
    # the fence line's newline never arrived — held, then killed
    assert u.push("Body.\n```") == "Body.\n"
    assert u.flush() == ""


def test_unfence_drop_tail_discards_held_line():
    # a tool-call fragment appearing mid-line: the held line goes
    # quiet, exactly SentenceGate.drop_tail's intent
    u = Unfence()
    assert u.push("We propose X.\n``") == "We propose X.\n"
    u.drop_tail()
    assert u.flush() == ""


def test_strip_fences_matches_the_streaming_filter():
    text = "A\n```latex\nB\n```\nC"
    assert strip_fences(text) == collect(Unfence(), [text])


# --- for_speech -------------------------------------------------------

def test_for_speech_cite_and_ref_families():
    assert for_speech(
        "As shown \\cite{a} and \\citep[see][p.~2]{b,c}, see "
        "\\autoref{fig:x} and \\cref{eq:y}.") == \
        "As shown citation and citation, see reference and reference."


def test_for_speech_wrappers_unwrap_nested():
    assert for_speech("A \\textbf{bold \\emph{deep}} word.") == \
        "A bold deep word."


def test_for_speech_generic_command_keeps_argument():
    assert for_speech("Use \\mycommand[opt]{the thing} now.") == \
        "Use the thing now."


def test_for_speech_bare_command_dropped():
    assert for_speech("More here. \\newpage Still here.") == \
        "More here. Still here."


def test_for_speech_environment_names_dropped():
    assert for_speech("\\begin{itemize} one \\end{itemize}") == "one"


def test_for_speech_latex_word_survives_the_bare_rule():
    assert for_speech("\\LaTeX{} rules.") == "LaTeX rules."


def test_for_speech_math_becomes_formula():
    assert for_speech("with $E=mc^2$ inside, and $$x_1$$ display.") == \
        "with formula inside, and formula display."


def test_for_speech_keeps_escaped_dollar():
    assert for_speech("Cost \\$5. Pay now.") == "Cost \\$5. Pay now."


def test_for_speech_collapses_whitespace():
    assert for_speech("line one\nline two   here.") == \
        "line one line two here."
