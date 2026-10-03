"""Sentence gate for streaming prose (sentence-level TTS slice).

Pure and deterministic — no clock, no I/O, no knowledge of who streams
what: feed token deltas, get completed sentences out. The boundary
rules bend around LaTeX prose, because the assistant talks about
patches: a period inside ``\\section{Intro. Give}`` or ``$a. b$`` is
not a sentence end, and neither is the one in "e.g." or "J. Smith".

The gate is a char-wise state machine (deltas are small; no
whole-buffer regex rescans): a ``.!?`` opens a candidate break that
whitespace arms and the next non-space character confirms — a
lowercase or digit continues the sentence ("vs. the", "3. 14"),
anything else ends it. A blank line is a boundary on its own.
"""

ABBREVIATIONS = {
    "i.e", "e.g", "etc", "cf", "vs", "v", "no", "sec", "fig", "eq",
    "dr", "mr", "mrs", "ms", "prof", "st", "approx", "al",  # "et al."
}


class SentenceGate:
    """push() deltas, get completed sentences; flush()/drop_tail() the
    partial at end-of-message. Sentences shorter than ``min_chars``
    merge forward rather than being emitted alone."""

    def __init__(self, min_chars: int = 3):
        # 3, not 2: the live model streams numbered lists, and a list
        # marker ("1. Break it down") must merge forward rather than be
        # pushed alone — the sidecar would speak a bare "dot".
        self.min_chars = min_chars
        self._buf = ""
        self._depth = 0      # brace nesting: never break inside {…}
        self._math = False   # $-toggle: never break inside $…$
        self._esc = False    # previous char was a backslash
        self._pending = False   # last char was .!? and boundary-eligible
        self._awaiting = False  # whitespace armed the candidate
        self._last = ""

    def push(self, delta: str) -> list[str]:
        out: list[str] = []
        for ch in delta:
            self._step(ch, out)
        return out

    def flush(self) -> str:
        """The tail as-is (a reply that ended without punctuation)."""
        tail = self._buf.strip()
        self._reset()
        return tail

    def drop_tail(self) -> str:
        """flush() under the caller's intent: the partial is thrown
        away, not delivered (a tool-call fragment or a bounce
        invalidated it). Returns what was dropped."""
        return self.flush()

    def _reset(self) -> None:
        self._buf = ""
        self._depth = 0
        self._math = False
        self._esc = False
        self._pending = False
        self._awaiting = False
        self._last = ""

    def _step(self, ch: str, out: list[str]) -> None:
        # 1. the armed candidate resolves on the first non-space char
        if self._awaiting and not ch.isspace():
            self._awaiting = False
            if (not ch.islower() and not ch.isdigit()
                    and len(self._buf.strip()) >= self.min_chars):
                out.append(self._buf.strip())
                self._reset()
            else:
                self._pending = False
        # 2. escapes: the next char is literal (affects $ and braces)
        if self._esc:
            self._buf += ch
            self._esc = False
            self._last = ch
            return
        if ch == "\\":
            self._buf += ch
            self._esc = True
            self._last = ch
            return
        # 3. a blank line ends the current sentence, outside math/braces
        # (min_chars applies here too: a stub paragraph merges forward
        # rather than being pushed alone)
        if (ch == "\n" and self._last == "\n"
                and len(self._buf.strip()) >= self.min_chars
                and not self._depth and not self._math):
            out.append(self._buf.strip())
            self._reset()
            self._last = ch
            return
        # 4. region state
        if ch == "$":
            self._math = not self._math  # "$$" nets out by toggling twice
        elif ch == "{":
            self._depth += 1
        elif ch == "}" and self._depth > 0:
            self._depth -= 1
        # 5. boundary bookkeeping
        self._buf += ch
        if ch in ".!?":
            self._pending = not (self._depth or self._math
                                 or (ch == "." and self._abbreviation()))
        elif ch.isspace():
            if self._pending:
                self._awaiting = True
                self._pending = False
        else:
            self._pending = False
        self._last = ch

    def _abbreviation(self) -> bool:
        """True when the token before the period reads as an
        abbreviation or a single-letter initial, not a sentence end."""
        tokens = self._buf.split()
        if not tokens:
            return False
        token = tokens[-1].rstrip(".").lstrip("(\"'[{").lower()
        return token in ABBREVIATIONS or (
            len(token) == 1 and token.isalpha())
