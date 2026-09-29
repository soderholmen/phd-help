"""BibTeX entry handling (SPEC §6): the cite loop's key discipline.

Keys like vaswani2023attention — first author surname + year + first
significant title word — deduped; re-citing an owned paper reuses its key.
"""

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class BibEntry:
    key: str
    type: str
    fields: dict


TITLE_STOPWORDS = {"a", "an", "the", "of", "on", "for", "in", "with", "to",
                   "is", "are", "and"}


def parse_bib(text: str) -> list[BibEntry]:
    """Every entry in a .bib file, in order."""
    out = []
    for m in re.finditer(r"@(\w+)\s*\{", text):
        try:
            close = _matching_brace(text, m.end() - 1)
        except ValueError:
            continue
        out.append(parse_entry(text[m.start():close + 1]))
    return out


def parse_entry(text: str) -> BibEntry:
    """Parse one @type{key, field = value, ...} entry.

    Values may be braced ({...}, nested), quoted ("..."), or bare
    (year=2017, pages=1--10). A malformed field never aborts the entry:
    an unterminated quote takes the remainder, and parsing always
    terminates (i strictly increases each iteration).
    """
    at = text.index("@")
    brace = text.index("{", at)
    key, _, body = text[brace + 1:].partition(",")
    fields: dict[str, str] = {}
    i = 0
    while True:
        name_m = re.search(r"(\w+)\s*=\s*", body[i:])
        if name_m is None:
            break
        name = name_m.group(1).lower()
        j = i + name_m.end()
        value, j = _read_value(body, j)
        fields[name] = value.strip()
        comma = body.find(",", j)
        if comma == -1:
            break
        i = comma + 1
    return BibEntry(key=key.strip(), type=text[at + 1:brace].strip().lower(),
                    fields=fields)


def _read_value(body: str, j: int) -> tuple[str, int]:
    """Read the field value at body[j]; return (value, index after it)."""
    if j < len(body) and body[j] == "{":
        try:
            close = _matching_brace(body, j)
        except ValueError:
            return body[j + 1:], len(body)  # unbalanced: take the remainder
        return body[j + 1:close], close + 1
    if j < len(body) and body[j] in "\"'":
        quote = body[j]
        close = body.find(quote, j + 1)
        if close == -1:
            return body[j + 1:], len(body)  # unterminated quote: no crash
        return body[j + 1:close], close + 1
    bare = re.match(r"[^,\s}]+", body[j:])  # bare token: 2017, 1--10, jan
    if bare:
        return bare.group(0), j + bare.end()
    return "", j


def _matching_brace(text: str, open_at: int) -> int:
    depth = 0
    for i in range(open_at, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                return i
    raise ValueError("unbalanced braces in bib entry")


def format_entry(entry: BibEntry) -> str:
    lines = [f"@{entry.type}{{{entry.key},"]
    for name, value in entry.fields.items():
        lines.append(f"  {name}={{{value}}},")
    lines.append("}")
    return "\n".join(lines) + "\n"


def with_key(entry: BibEntry, key: str) -> BibEntry:
    from dataclasses import replace
    return replace(entry, key=key)


def dedupe_key(key: str, existing: set[str]) -> str:
    """Collision renumbering: vaswani2023attention -> ...b -> ...c."""
    if key not in existing:
        return key
    for letter in "bcdefghijklmnopqrstuvwxyz":
        if key + letter not in existing:
            return key + letter
    raise ValueError(f"no free variant of key '{key}'")


def same_paper(a: BibEntry, b: BibEntry) -> bool:
    """Identifier match: same arXiv id (version-insensitive) or same DOI.

    The arXiv id lives in 'arxiv' for entries we build, but in 'eprint'
    for arXiv's own bibtex — check both, or re-citing an owned arXiv paper
    would duplicate it instead of reusing its key (SPEC §6).
    """
    aid, bid = _arxiv_id(a), _arxiv_id(b)
    if aid and bid and aid == bid:
        return True
    adoi = a.fields.get("doi", "").strip().lower()
    bdoi = b.fields.get("doi", "").strip().lower()
    return bool(adoi and bdoi and adoi == bdoi)


def _arxiv_id(entry: BibEntry) -> str:
    raw = (entry.fields.get("arxiv") or entry.fields.get("eprint") or "")
    return re.sub(r"v\d+$", "", raw.strip().lower().removeprefix("arxiv:"))


def invert_name(display: str) -> str:
    """'Ashish Vaswani' -> 'Vaswani, Ashish' — the bibtex surname-first form."""
    parts = display.split()
    return f"{parts[-1]}, {' '.join(parts[:-1])}" if len(parts) > 1 else display


def make_key(entry: BibEntry) -> str:
    surname = entry.fields.get("author", "").split(",")[0].strip().lower()
    year = entry.fields.get("year", "")
    words = re.findall(r"[a-z]+", entry.fields.get("title", "").lower())
    word = next((w for w in words if w not in TITLE_STOPWORDS), "")
    return f"{surname}{year}{word}"
