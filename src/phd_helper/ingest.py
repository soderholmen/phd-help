"""The ingestion pipeline (SPEC §6): two doors, one pipeline.

Both doors (web-UI upload, agent fetch) register the PDF in the corpus
registry and run the same unit: extract (MinerU behind a Protocol) ->
chunk (pure) -> index (embedder + LanceDB behind a Protocol), with the
registry's status machine making every step visible and failures
retryable. Without the heavy adapters wired (the 3090 stack), the
pipeline is not runnable and docs sit queued — indexing pauses, it does
not rot (§8).
"""

from typing import Protocol

from phd_helper.chunking import Block, chunk_document
from phd_helper.corpus import Corpus, CorpusError, DocRecord


class IngestError(Exception):
    """A PDF that could not be fetched (paywalled, dead link)."""


class Extractor(Protocol):
    async def extract(self, pdf: bytes) -> list[Block]:
        """MinerU standard tier: PDF bytes -> normalized block tree."""
        ...


class PdfFetcher(Protocol):
    async def __call__(self, url: str) -> bytes:
        """Raises IngestError on any non-OK fetch (arXiv-spaced by the
        caller's rate limiter)."""
        ...


class Ingestor:
    def __init__(self, corpus: Corpus, extractor=None, store=None,
                 fetch_pdf: PdfFetcher | None = None):
        self.corpus = corpus
        self.extractor = extractor
        self.store = store
        self.fetch_pdf = fetch_pdf

    def runnable(self) -> bool:
        return self.extractor is not None and self.store is not None

    async def ingest(self, doc_id: str) -> None:
        """One PDF, start to finish; status tracks it either way."""
        rec = self.corpus.get(doc_id)
        if rec is None:
            return  # superseded mid-flight: nothing to write back to
        self.corpus.set_status(doc_id, "extracting")
        try:
            pdf = self.corpus.pdf_path(doc_id).read_bytes()
            blocks = await self.extractor.extract(pdf)
            chunks = chunk_document(rec.title or "Untitled", blocks)
            await self.store.index(doc_id, chunks)
            if rec.supersedes:
                await self.store.remove(rec.supersedes)  # new version (§6)
            self.corpus.mark_indexed(doc_id, chunk_count=len(chunks))
        except Exception as e:
            try:
                self.corpus.set_status(
                    doc_id, "failed", error=f"{type(e).__name__}: {e}")
            except CorpusError:
                pass  # record vanished underneath us; nothing to mark

    async def drain_queued(self) -> None:
        """Startup resume: everything still queued (or recovered from a
        crash) goes through the pipeline in registration order."""
        for rec in self.corpus.list():
            if rec.status == "queued":
                await self.ingest(rec.doc_id)

    async def fetch_arxiv(self, arxiv_id: str) -> DocRecord:
        """Door 2 (agent fetch / search auto-join): fetch the PDF, then the
        shared pipeline. A dead or paywalled link raises — those papers
        get a bib entry only, never a half-registered corpus doc."""
        pdf = await self.fetch_pdf(f"https://arxiv.org/pdf/{arxiv_id}")
        rec, new = self.corpus.add_pdf(pdf, arxiv=arxiv_id, source="agent")
        if new:
            await self.ingest(rec.doc_id)
        return rec
