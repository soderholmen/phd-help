"""Deterministic LaTeX lint (SPEC §5).

Runs on the resulting file before the diff is shown; failures bounce back to
the agent and never reach the user's approval attention.
"""

import re
from collections import Counter
from dataclasses import dataclass


@dataclass(frozen=True)
class LintError:
    rule: str
    message: str


def lint_latex(text: str, bib_keys: set[str] | None = None) -> list[LintError]:
    errors: list[LintError] = []

    # \{ and \} are literal characters, not grouping
    grouping = re.sub(r"\\([{}])", "", text)
    if grouping.count("{") != grouping.count("}"):
        errors.append(LintError("brace-balance", "unbalanced braces"))

    begins = Counter(re.findall(r"\\begin\{([^}]*)\}", text))
    ends = Counter(re.findall(r"\\end\{([^}]*)\}", text))
    if begins != ends:
        errors.append(LintError("begin-end", "unmatched \\begin/\\end"))

    if bib_keys is not None:
        cited = {
            key.strip()
            for keys in re.findall(
                r"\\cite\w*(?:\[[^\]]*\])*\{([^}]*)\}", text
            )
            for key in keys.split(",")
            if key.strip()
        }
        unknown = cited - bib_keys
        if unknown:
            errors.append(
                LintError("unknown-citation", f"unknown bib keys: {sorted(unknown)}")
            )

    return errors
