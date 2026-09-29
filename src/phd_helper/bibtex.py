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
    """Parse one @type{key, field = {value}, ...} entry."""
    at = text.index("@")
    brace = text.index("{", at)
    head, rest = text[brace + 1:], text[brace + 1:]
    key, _, body = head.partition(",")
    fields: dict[str, str] = {}
    i = 0
    while True:
        name_m = re.search(r"(\w+)\s*=\s*", body[i:])
        if name_m is None:
            break
        name = name_m.group(1).lower()
        j = i + name_m.end()
        if j >= len(body) or body[j] not in "{\":":
            break
        if body[j] in "{\":":
            close = _matching_brace(body, j) if body[j] == "{" else \
                body.index(body[j], j + 1)
            fields[name] = body[j + 1:close].strip()
            j = close + 1
        i = j + body[j:].find(",") + 1 if "," in body[j:] else len(body)
    return BibEntry(key=key.strip(), type=text[at + 1:brace].strip().lower(),
                    fields=fields)


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
    """Identifier match: same arXiv id (version-insensitive) or same DOI."""
    for field in ("arxiv", "doi"):
        va = _norm_id(field, a.fields.get(field, ""))
        vb = _norm_id(field, b.fields.get(field, ""))
        if va and vb and va == vb:
            return True
    return False


def _norm_id(field: str, value: str) -> str:
    value = value.strip().lower()
    if field == "arxiv":
        return re.sub(r"v\d+$", "", value.removeprefix("arxiv:"))
    return value


def make_key(entry: BibEntry) -> str:
    surname = entry.fields.get("author", "").split(",")[0].strip().lower()
    year = entry.fields.get("year", "")
    words = re.findall(r"[a-z]+", entry.fields.get("title", "").lower())
    word = next((w for w in words if w not in TITLE_STOPWORDS), "")
    return f"{surname}{year}{word}"
