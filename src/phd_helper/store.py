"""The corpus index seam (SPEC §6).

The production adapter is embedded LanceDB (Tantivy BM25 + vectors + RRF
fusion + rerank in one library call) with the harrier embedder behind it —
all server-side, wired when the 3090 stack is up. Until then (and in
tests) the agent tools run against any object with this shape; a missing
or faulting store degrades to tool errors per the §8 matrix.
"""

from dataclasses import dataclass
from typing import Protocol

from phd_helper.chunking import Chunk


@dataclass(frozen=True)
class ChunkHit:
    doc_id: str
    text: str
    section_path: str
    page_start: int
    page_end: int
    block_start: int
    block_end: int
    kind: str
    score: float

    @property
    def locator(self) -> str:
        """The §6 citation locator ("p.4, blocks 43-47")."""
        pages = (f"p.{self.page_start}" if self.page_start == self.page_end
                 else f"pp.{self.page_start}-{self.page_end}")
        return f"{pages}, blocks {self.block_start}-{self.block_end}"


@dataclass(frozen=True)
class DocInfo:
    abstract: str
    headings: tuple[str, ...]


class CorpusStore(Protocol):
    """Hybrid search over the one global chunk index (§6)."""

    async def search(self, query: str, k: int,
                     boost_ids: set[str]) -> list[ChunkHit]:
        """Ranked chunks; docs in boost_ids (pinned to the active project)
        are boosted, never filtered to."""
        ...

    async def doc(self, doc_id: str) -> DocInfo | None:
        """Abstract + heading list for a doc, or None if unknown."""
        ...

    async def index(self, doc_id: str, chunks: list[Chunk]) -> None:
        """(Re)place all chunks of one document in the index."""
        ...

    async def remove(self, doc_id: str) -> None:
        """Drop a superseded document's chunks."""
        ...
