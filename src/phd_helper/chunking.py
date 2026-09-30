"""Section-aware chunking of an extracted PDF (SPEC §6).

Input is MinerU's block tree normalized to ``Block`` (the server adapter
maps model.json onto this; chunking itself is pure and testable). Rules
from §6: split on section headings, prepend the heading path
(``paper title » section``) to the *embedded* text only, tables/figures
as standalone chunks, ~512-1024 token cap, and stable page/block
locators so the agent can say "§3.2 of the parakeet paper" precisely.

The token count is a deterministic char/4 heuristic — no tokenizer
dependency in the pure core; the cap is a soft ceiling either way.
"""

import re
from dataclasses import dataclass

# MinerU block kinds normalized to this vocabulary by the server adapter.
TEXT_LIKE = ("text", "equation", "other")
STANDALONE = ("table", "figure")


@dataclass(frozen=True)
class Block:
    kind: str  # "heading" | "text" | "table" | "figure" | "equation" | "other"
    text: str
    page: int  # 1-based page number (MinerU page_idx + 1)
    block: int  # stable block index — the citation locator
    level: int = 0  # heading level 1..n; 0 for non-headings


@dataclass(frozen=True)
class Chunk:
    text: str  # raw text for display — never carries the prefix
    embed_text: str  # "title » section path" + text, what gets embedded
    section_path: str  # "Method » Training", "" before the first heading
    page_start: int
    page_end: int
    block_start: int
    block_end: int
    kind: str  # "text" | "table" | "figure"
    is_abstract: bool


def est_tokens(text: str) -> int:
    return max(1, round(len(text) / 4))


def chunk_document(title: str, blocks: list[Block],
                   token_cap: int = 1024) -> list[Chunk]:
    chunks: list[Chunk] = []
    headings: list[tuple[int, str]] = []  # (level, text) stack
    pending: list[Block] = []
    pending_tokens = 0

    def path() -> str:
        return " » ".join(text for _, text in headings)

    def emit(text: str, kind: str, first: Block, last: Block,
             path_at_emit: str) -> None:
        prefix = f"{title} » {path_at_emit}" if path_at_emit else title
        chunks.append(Chunk(
            text=text, embed_text=f"{prefix}\n{text}",
            section_path=path_at_emit,
            page_start=first.page, page_end=last.page,
            block_start=first.block, block_end=last.block,
            kind=kind,
            is_abstract=path_at_emit.strip().lower() == "abstract"))

    def flush() -> None:
        nonlocal pending_tokens
        if pending:
            emit(" ".join(b.text.strip() for b in pending), "text",
                 pending[0], pending[-1], path())
            pending.clear()
            pending_tokens = 0

    for b in blocks:
        if not b.text.strip():
            continue
        if b.kind == "heading":
            flush()
            while headings and headings[-1][0] >= b.level:
                headings.pop()
            headings.append((b.level, b.text.strip()))
        elif b.kind in STANDALONE:
            flush()  # tables/figures never merge with prose (§6)
            emit(b.text.strip(), b.kind, b, b, path())
        else:  # text-like: accumulate under the cap
            n = est_tokens(b.text)
            if n > token_cap:  # oversized paragraph: split on sentences
                flush()
                for part in _split_to_cap(b.text.strip(), token_cap):
                    emit(part, "text", b, b, path())
                continue
            if pending_tokens + n > token_cap:
                flush()
            pending.append(b)
            pending_tokens += n
    flush()
    return chunks


def _split_to_cap(text: str, token_cap: int) -> list[str]:
    """Group sentences into runs that fit the cap; a single oversized
    sentence stays atomic rather than being cut mid-word."""
    sentences = re.split(r"(?<=[.!?])\s+", text)
    parts: list[str] = []
    run: list[str] = []
    run_tokens = 0
    for s in sentences:
        n = est_tokens(s)
        if run and run_tokens + n > token_cap:
            parts.append(" ".join(run))
            run, run_tokens = [], 0
        run.append(s)
        run_tokens += n
    if run:
        parts.append(" ".join(run))
    return parts
