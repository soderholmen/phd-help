"""What the voice says from the text (stepwise-writing slice).

Pure and deterministic, like sentences.py. Three jobs on the speech
feed: ``extract_draft`` captures the fenced draft from a turn's final
assistant text (the apply turn then references it instead of
re-decoding the paragraph into tool arguments); ``Unfence`` keeps the
``` delimiter lines out of the spoken stream while the draft content
itself is spoken; ``for_speech`` turns LaTeX prose into words, because
the TTS engine hallucinates on dense symbol strings — the note the
SYSTEM_PROMPT gives the model cannot be honored inside a real draft,
which IS LaTeX. The screen and the history keep the raw source; only
the voice hears the paraphrase.
"""
import re

FENCE = "```"


def extract_draft(text: str) -> str | None:
    """The content of the last COMPLETE fenced block, or None.
    Complete = an opening fence line and a later closing fence line;
    an unclosed block is a truncated turn, not a draft."""
    blocks, current = [], None
    for line in text.splitlines():
        if line.lstrip().startswith(FENCE):
            if current is None:
                current = []
            else:
                blocks.append("\n".join(current).strip())
                current = None
        elif current is not None:
            current.append(line)
    return blocks[-1] if blocks else None


class Unfence:
    """Streaming line filter: fence delimiter lines are dropped (their
    newline kept, so the gate's blank-line rule still separates the
    draft from the prose around it); every other byte passes through.

    A line is held only while it could still BE a fence — whitespace
    and fewer than three backticks so far. The first ordinary character
    releases the line to stream, so ordinary prose is never delayed to
    a newline (the overlap must survive the filter). A fence split
    across deltas ("``" + "`latex") is still one fence: the backticks
    keep the line held."""

    def __init__(self):
        self._held = ""      # chars of a not-yet-classified line
        self._open = False   # a delimiter line is proven; skip to \n

    def push(self, delta: str) -> str:
        out = []
        for ch in delta:
            if self._open:
                if ch == "\n":
                    out.append("\n")
                    self._open = False
                continue  # the rest of a delimiter line dies
            if self._held is None:
                out.append(ch)   # this line is proven prose: stream
                if ch == "\n":
                    self._held = ""  # the next line gets classified again
                continue
            if ch == "\n":
                # a blank or backtick-stub line ("``") is prose after all
                out.append(self._held + "\n")
                self._held = ""
                continue
            self._held += ch
            stripped = self._held.lstrip()
            if stripped and not stripped.startswith("`"):
                out.append(self._held)  # first ordinary char: release
                self._held = None
            elif stripped.count("`") >= 3:
                self._open = True       # a delimiter line
                self._held = ""
        return "".join(out)

    def flush(self) -> str:
        """The held remainder: a fence fragment dies, a backtick stub
        lives (the message ended without a trailing newline)."""
        rest = self._held if isinstance(self._held, str) else ""
        self._held, self._open = "", False
        return rest

    def drop_tail(self) -> None:
        """The partial line a tool-call fragment or a bounce
        invalidated goes quiet (SentenceGate's seam, same intent)."""
        self._held, self._open = "", False


def strip_fences(text: str) -> str:
    """One-shot Unfence — the kill-switch leg pushes the whole reply
    at once and needs the same fence-free speech."""
    u = Unfence()
    return u.push(text) + u.flush()


_CITES = (r"(?:cite|citep|citet|citealp|autocite|parencite|textcite"
          r"|nocite)\*?")
_REFS = r"(?:ref|autoref|eqref|cref|Cref|nameref|pageref)\*?"
_OPTS = r"(?:\[[^\]]*\])*"
_WRAPPER = (r"(?:textbf|emph|textit|texttt|textsf|underline|textsc"
            r"|text)\*?\{([^{}]*)\}")
_GENERIC = re.compile(r"\\[a-zA-Z]+\*?" + _OPTS + r"\{([^{}]*)\}")
_WRAPPER_RE = re.compile(r"\\" + _WRAPPER)


def _sub_loop(pattern: re.Pattern, text: str) -> str:
    """Substitute until stable: innermost braces first, so nesting
    (\\textbf{\\emph{x}}) unwraps layer by layer."""
    while True:
        text, n = pattern.subn(r"\1", text)
        if not n:
            return text


def for_speech(text: str) -> str:
    """LaTeX prose -> words the TTS can say. Order is load-bearing:
    \\LaTeX before the generic bare-command rule, display math before
    inline, environments before the generic brace rule (else
    \\begin{itemize} speaks "itemize")."""
    t = re.sub(r"\\LaTeX\b", "LaTeX", text)
    t = re.sub(r"\\TeX\b", "TeX", t)
    t = re.sub(r"(?<!\\)\$\$.+?(?<!\\)\$\$", " formula ", t)
    t = re.sub(r"(?<!\\)\$[^$\n]+?(?<!\\)\$", " formula ", t)
    t = re.sub(r"\\" + _CITES + _OPTS + r"\s*\{[^{}]*\}", " citation ", t)
    t = re.sub(r"\\" + _REFS + _OPTS + r"\s*\{[^{}]*\}", " reference ", t)
    t = re.sub(r"\\(?:begin|end)\*?\{[^{}]*\}", " ", t)
    t = _sub_loop(_WRAPPER_RE, t)
    t = _sub_loop(_GENERIC, t)
    t = re.sub(r"\\\\", " ", t)          # LaTeX line break
    t = re.sub(r"\\[a-zA-Z]+\*?", " ", t)  # bare commands: \newpage
    t = re.sub(r"\{\s*\}", " ", t)       # leftover empty braces
    t = re.sub(r"\s+([.,;:!?])", r"\1", t)  # substitution residue
    return re.sub(r"\s+", " ", t).strip()
