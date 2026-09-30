"""MinerU 4.0 markdown -> Blocks (pure; feeds the chunker, SPEC §6).

MinerU 4.0's public content format is markdown with structure markers:
``<!-- page N of M -->`` comments and ``![alt](doc:.../page:N/block:M)``
figure locators (verified against the real CLI). Tables arrive as pipe
tables or one-line raw HTML; display math as ``$$`` fences. This maps
that onto the chunker's Block vocabulary. Block ids are **page-local**
(1-based, reset at each page marker — MinerU's own convention): a
document-wide counter would shift every id after it whenever a reparse
segments one page differently, which is exactly what §6's "stable
page/block citation locators" exists to prevent. Figures carry
MinerU's real ``page:N/block:M`` from their asset URL; the rest take
the next free id on their page.
"""

import re

from phd_helper.chunking import Block

_PAGE = re.compile(r"<!--\s*page\s+(\d+)\s+of\s+\d+\s*-->")
_HEADING = re.compile(r"^(#{1,6})\s+(.+?)\s*#*\s*$")
_FIGURE = re.compile(r"^!\[([^\]]*)\]\(([^)]*)\)\s*$")
_FIG_LOC = re.compile(r"page:(\d+)/block:(\d+)")  # MinerU's real locator
_SEP = re.compile(r"^\|?[\s:|-]+\|[\s:|-]*$")  # pipe-table separator row
# The caption paragraph that gives a figure marker its meaning; MinerU
# keeps it as the next text block. The float label + number must be
# followed *immediately* by ":" or "." — prose that merely mentions a
# figure ("Figure 1 shows that…") reads as prose, not as a caption.
_CAPTION = re.compile(
    r"^(?:figure|fig\.|table|listing|algorithm)\s+[a-z0-9ivx]+"
    r"(?:\s*\([a-z0-9ivx]+\))?\s*[:.]", re.IGNORECASE)


def _starts_new_block(line: str) -> bool:
    return bool(_PAGE.match(line) or _HEADING.match(line)
                or line.startswith("$$") or line.startswith("|")
                or "<table" in line or _FIGURE.match(line))


def markdown_to_blocks(md: str) -> list[Block]:
    lines = md.splitlines()
    page = 1
    next_block = 1  # page-local id space; reset at each page marker
    blocks: list[Block] = []

    def emit(kind: str, text: str, level: int = 0,
             block: int | None = None, at_page: int | None = None) -> None:
        nonlocal next_block
        text = text.strip()
        if not text:
            return
        b = block if block is not None else next_block
        # A figure's real id is claimed, not counted: the counter skips
        # past it so no two blocks on a page ever share a locator id.
        next_block = max(next_block, b) + 1
        blocks.append(Block(kind, text,
                            at_page if at_page is not None else page,
                            b, level))

    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if not line:
            i += 1
            continue
        if (pm := _PAGE.match(line)):
            page = int(pm.group(1))
            next_block = 1
            i += 1
            continue
        if (hm := _HEADING.match(line)):
            emit("heading", hm.group(2), level=len(hm.group(1)))
            i += 1
            continue
        if (fm := _FIGURE.match(line)):
            # MinerU's asset URL carries the real page:N/block:M — the
            # PDF-stable locator §6:161 makes load-bearing. Keep it.
            lm = _FIG_LOC.search(fm.group(2))
            emit("figure", fm.group(1) or "figure",
                 block=int(lm.group(2)) if lm else None,
                 at_page=int(lm.group(1)) if lm else None)
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
        text = " ".join(buf)
        prev = blocks[-1] if blocks else None
        if prev is not None and prev.kind == "figure" and prev.page == page \
                and _CAPTION.match(text):
            # A bare "Image block" marker is a contentless chunk (§6
            # wants figures as *useful* standalone chunks); the caption
            # paragraph right below it is the figure's actual text.
            blocks[-1] = Block("figure", f"{prev.text} — {text}",
                               prev.page, prev.block)
        else:
            emit("text", text)
    return blocks
