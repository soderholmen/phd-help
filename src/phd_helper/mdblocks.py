"""MinerU 4.0 markdown -> Blocks (pure; feeds the chunker, SPEC §6).

MinerU 4.0's public content format is markdown with structure markers:
``<!-- page N of M -->`` comments and ``![alt](doc:.../page:N/block:M)``
figure locators (verified against the real CLI). Tables arrive as pipe
tables or one-line raw HTML; display math as ``$$`` fences. This maps
that onto the chunker's Block vocabulary — block ids are assigned
sequentially in document order, stable for a given parse.
"""

import re

from phd_helper.chunking import Block

_PAGE = re.compile(r"<!--\s*page\s+(\d+)\s+of\s+\d+\s*-->")
_HEADING = re.compile(r"^(#{1,6})\s+(.+?)\s*#*\s*$")
_FIGURE = re.compile(r"^!\[([^\]]*)\]\([^)]*\)\s*$")
_SEP = re.compile(r"^\|?[\s:|-]+\|[\s:|-]*$")  # pipe-table separator row


def _starts_new_block(line: str) -> bool:
    return bool(_PAGE.match(line) or _HEADING.match(line)
                or line.startswith("$$") or line.startswith("|")
                or "<table" in line or _FIGURE.match(line))


def markdown_to_blocks(md: str) -> list[Block]:
    lines = md.splitlines()
    page = 1
    blocks: list[Block] = []

    def emit(kind: str, text: str, level: int = 0) -> None:
        text = text.strip()
        if text:
            blocks.append(Block(kind, text, page, len(blocks), level))

    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if not line:
            i += 1
            continue
        if (pm := _PAGE.match(line)):
            page = int(pm.group(1))
            i += 1
            continue
        if (hm := _HEADING.match(line)):
            emit("heading", hm.group(2), level=len(hm.group(1)))
            i += 1
            continue
        if (fm := _FIGURE.match(line)):
            emit("figure", fm.group(1) or "figure")  # locator is noise
            i += 1
            continue
        if line.startswith("$$"):
            buf = [line]
            closed = line.count("$$") >= 2  # single-line $$x$$
            while not closed and i + 1 < len(lines):
                i += 1
                buf.append(lines[i].strip())
                closed = "$$" in lines[i]
            emit("equation", "\n".join(buf))
            i += 1
            continue
        if "<table" in line:  # MinerU emits HTML tables, usually one line
            buf = [line]
            while "</table>" not in " ".join(buf) and i + 1 < len(lines):
                nxt = lines[i + 1].strip()
                if not nxt:
                    break
                i += 1
                buf.append(nxt)
            emit("table", "\n".join(buf))
            i += 1
            continue
        if line.startswith("|") and i + 1 < len(lines) \
                and _SEP.match(lines[i + 1].strip()):
            buf = []
            while i < len(lines) and lines[i].strip().startswith("|"):
                buf.append(lines[i].strip())
                i += 1
            emit("table", "\n".join(buf))
            continue
        buf = [line]  # plain paragraph: run to the blank line or marker
        i += 1
        while i < len(lines) and (nxt := lines[i].strip()) \
                and not _starts_new_block(nxt):
            buf.append(nxt)
            i += 1
        emit("text", " ".join(buf))
    return blocks
