"""Embedded LanceDB adapter for the CorpusStore seam (SPEC §6).

One global table of chunks; hybrid search is one library call — Tantivy
BM25 + vector, RRF-fused (verified against lancedb 0.39: the working
form is ``search(query_type="hybrid").vector(q).text(query)``). When a
reranker is wired (server/rerank.py), the fused top-50 goes through it
and its scores replace the RRF ones; a reranker that faults or returns
a misaligned score list degrades to the RRF order rather than sinking
the search (§8). The pinned-doc boost rides on top of the final scores:
pinned docs get a multiplier and the list re-sorts, so a pinned doc
wins ties and near ties but a bad pinned doc still loses to a much
better match — boosted, never filtered to (§6).

The embedder and reranker are the sync seams (server/embed.py,
server/rerank.py); every method here is async and offloads model work
to a worker thread. The table is created lazily on first index, sized
to the embedder's first vector — an empty store searches to [] and docs
to None (§8).
"""

import asyncio

import lancedb
import pyarrow as pa
from lancedb.index import FTS

from phd_helper.chunking import leaf
from phd_helper.store import ChunkHit, DocInfo

TABLE = "chunks"
BOOST = 1.5
RERANK_CANDIDATES = 50  # §6: rerank over top-50, not over k


def _quote(value: str) -> str:
    """SQL-literal escape for the doc_id filters (ids are hex, but the
    store must not care who calls it with what)."""
    return value.replace("'", "''")


def _schema(dim: int) -> pa.Schema:
    return pa.schema([
        pa.field("doc_id", pa.string()),
        pa.field("text", pa.string()),
        pa.field("section_path", pa.string()),
        pa.field("page_start", pa.int32()),
        pa.field("page_end", pa.int32()),
        pa.field("block_start", pa.int32()),
        pa.field("block_end", pa.int32()),
        pa.field("kind", pa.string()),
        pa.field("is_abstract", pa.bool_()),
        pa.field("vector", pa.list_(pa.float32(), dim)),
    ])


class LanceStore:
    """CorpusStore over an embedded LanceDB directory — no server."""

    def __init__(self, db_path, embedder, reranker=None):
        self._db = lancedb.connect(str(db_path))
        self.embedder = embedder
        self.reranker = reranker
        self._fts_ready = False

    def _table(self):
        if TABLE not in self._db.list_tables().tables:
            return None
        return self._db.open_table(TABLE)

    def _ensure_fts(self, t) -> None:
        if self._fts_ready:
            return
        try:
            t.create_index("text", config=FTS())
        except Exception:
            pass  # already indexed (idempotent across restarts)
        self._fts_ready = True

    async def index(self, doc_id: str, chunks: list) -> None:
        vecs = None
        if chunks:
            texts = [c.embed_text for c in chunks]
            vecs = await asyncio.to_thread(self.embedder.encode, texts)
        t = self._table()
        if t is None:
            if not chunks:
                return  # nothing to store, nothing to replace
            t = self._db.create_table(TABLE,
                                       schema=_schema(len(vecs[0])),
                                       exist_ok=True)
        if t.count_rows():
            t.delete(f"doc_id = '{_quote(doc_id)}'")  # (re)place semantics
        if chunks:
            t.add([{
                "doc_id": doc_id, "text": c.text,
                "section_path": c.section_path,
                "page_start": c.page_start, "page_end": c.page_end,
                "block_start": c.block_start, "block_end": c.block_end,
                "kind": c.kind, "is_abstract": c.is_abstract,
                "vector": v,
            } for c, v in zip(chunks, vecs)])
        self._ensure_fts(t)

    async def remove(self, doc_id: str) -> None:
        t = self._table()
        if t is not None and t.count_rows():
            t.delete(f"doc_id = '{_quote(doc_id)}'")

    async def search(self, query: str, k: int,
                     boost_ids: set[str]) -> list[ChunkHit]:
        t = self._table()
        if t is None or t.count_rows() == 0:
            return []
        q = (await asyncio.to_thread(self.embedder.encode, [query]))[0]
        self._ensure_fts(t)
        # Over-fetch: the boost re-sort (and the rerank stage) must
        # choose from a wide enough field that a good doc outside the
        # raw top-k — or a pinned doc outside it — can still land.
        limit = RERANK_CANDIDATES if self.reranker else max(k * 3, 10)
        rows = (t.search(query_type="hybrid")
                .vector(q).text(query).limit(limit).to_list())
        rows = await self._rerank(query, rows)
        for r in rows:
            if r["doc_id"] in boost_ids:
                r["_relevance_score"] *= BOOST
        rows.sort(key=lambda r: r["_relevance_score"], reverse=True)
        return [ChunkHit(doc_id=r["doc_id"], text=r["text"],
                         section_path=r["section_path"],
                         page_start=int(r["page_start"]),
                         page_end=int(r["page_end"]),
                         block_start=int(r["block_start"]),
                         block_end=int(r["block_end"]),
                         kind=r["kind"],
                         score=float(r["_relevance_score"]))
                for r in rows[:k]]

    async def _rerank(self, query: str, rows: list[dict]) -> list[dict]:
        """Re-score the fused candidates with the cross-encoder, in
        place on the score field (§6). The doc text carries the section
        path — the same heading context the embedder saw, and what lets
        the reranker tell two chunks with shared prose apart.

        Degrades, never sinks (§8): a faulting reranker or one whose
        scores don't line up with the candidates leaves the RRF order
        untouched — a misaligned score list would graft scores onto the
        wrong chunks, which is worse than no rerank at all."""
        if self.reranker is None or not rows:
            return rows
        docs = [f"{r['section_path']}\n{r['text']}" for r in rows]
        try:
            scores = await asyncio.to_thread(self.reranker.rerank,
                                            query, docs)
        except Exception:
            return rows
        if len(scores) != len(rows):
            return rows
        for r, s in zip(rows, scores):
            r["_relevance_score"] = float(s)
        return rows

    async def doc(self, doc_id: str) -> DocInfo | None:
        t = self._table()
        if t is None:
            return None
        rows = (t.search().where(f"doc_id = '{_quote(doc_id)}'")
                .limit(100000).to_list())
        if not rows:
            return None
        abstract = next((r["text"] for r in rows if r["is_abstract"]), "")
        headings: list[str] = []
        for r in rows:  # heading order = document order
            h = leaf(r["section_path"])
            if h and h not in headings:
                headings.append(h)
        return DocInfo(abstract=abstract, headings=tuple(headings))
