"""The reference-corpus registry (SPEC §6): one global corpus shared by
all paper projects.

Dedup is by content hash; a new arXiv/DOI version of an owned paper
supersedes the old record on re-fetch (pins carry over). Per-PDF status
is a small state machine so failures are visible, not rot:

    queued -> extracting -> indexed
                 \\-> failed -> queued (retry)
    extracting -> queued (startup reconcile after a crash)

Pins are per-project links into this global store (§6: search boosts
pinned docs, never filters to them). PDFs live on the server filesystem
under ``files/``; Syncthing replicates the folder.
"""

import hashlib
import json
import time
from dataclasses import dataclass, field, replace
from pathlib import Path

STATUSES = ("queued", "extracting", "indexed", "failed")
_LEGAL = {
    "queued": {"extracting"},
    "extracting": {"indexed", "failed", "queued"},  # queued: crash recovery
    "failed": {"queued"},
    "indexed": set(),  # terminal; supersede replaces the record instead
}


class CorpusError(Exception):
    """Illegal status transition or unknown doc id."""


@dataclass(frozen=True)
class DocRecord:
    doc_id: str
    sha256: str
    title: str
    arxiv: str
    doi: str
    year: str
    source: str  # "upload" | "agent"
    status: str
    error: str = ""
    pinned_in: tuple[str, ...] = ()
    added_at: float = 0.0
    indexed_at: float = 0.0
    chunk_count: int = 0
    # doc id this record replaced (new arXiv version re-fetched, §6) — the
    # pipeline evicts the superseded doc's chunks from the index.
    supersedes: str = ""


@dataclass
class _State:
    docs: list[DocRecord] = field(default_factory=list)


class Corpus:
    def __init__(self, root_dir: Path):
        self.root = Path(root_dir)
        self._index = self.root / "corpus.json"

    # -- persistence ------------------------------------------------------

    def _load(self) -> list[DocRecord]:
        try:
            raw = json.loads(self._index.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return []
        fields = set(DocRecord.__dataclass_fields__) - {"pinned_in"}
        return [DocRecord(**{k: v for k, v in d.items() if k in fields},
                          pinned_in=tuple(d.get("pinned_in", ())))
                for d in raw.get("docs", [])]

    def _save(self, docs: list[DocRecord]) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        tmp = self._index.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(
            {"docs": [d.__dict__ for d in docs]}, indent=1),
            encoding="utf-8")
        tmp.replace(self._index)

    # -- the two doors ------------------------------------------------------

    def add_pdf(self, data: bytes, *, title: str = "", arxiv: str = "",
                doi: str = "", year: str = "", source: str = "upload",
                ) -> tuple[DocRecord, bool]:
        """Register a PDF: dedup by content hash, supersede a newer version
        of an owned paper. Returns (record, is_new)."""
        docs = self._load()
        sha = hashlib.sha256(data).hexdigest()
        dup = next((d for d in docs if d.sha256 == sha), None)
        if dup is not None:
            return dup, False
        owned = next((d for d in docs if (arxiv and d.arxiv == arxiv) or
                      (doi and d.doi == doi)), None)
        rec = DocRecord(
            doc_id=sha[:16], sha256=sha, title=title, arxiv=arxiv, doi=doi,
            year=year, source=source, status="queued",
            pinned_in=owned.pinned_in if owned else (),
            supersedes=owned.doc_id if owned else "",
            added_at=time.time())
        if owned is not None:  # new version of an owned paper (§6)
            (self.root / "files" / f"{owned.doc_id}.pdf").unlink(missing_ok=True)
            docs.remove(owned)
        docs.append(rec)
        self.pdf_path(rec.doc_id).parent.mkdir(parents=True, exist_ok=True)
        self.pdf_path(rec.doc_id).write_bytes(data)
        self._save(docs)
        return rec, True

    def pdf_path(self, doc_id: str) -> Path:
        return self.root / "files" / f"{doc_id}.pdf"

    # -- reading ------------------------------------------------------------

    def get(self, doc_id: str) -> DocRecord | None:
        return next((d for d in self._load() if d.doc_id == doc_id), None)

    def list(self) -> list[DocRecord]:
        return self._load()  # insertion order, oldest first

    # -- status machine -------------------------------------------------------

    def set_status(self, doc_id: str, status: str, error: str = "") -> DocRecord:
        docs = self._load()
        rec = self._find(docs, doc_id)
        if status not in _LEGAL.get(rec.status, set()):
            raise CorpusError(
                f"{doc_id}: {rec.status} -> {status} is not a legal "
                "transition")
        updated = replace(rec, status=status, error=error)
        self._write(docs, updated)
        return updated

    def mark_indexed(self, doc_id: str, chunk_count: int) -> DocRecord:
        docs = self._load()
        rec = self._find(docs, doc_id)
        if "indexed" not in _LEGAL[rec.status]:
            raise CorpusError(
                f"{doc_id}: {rec.status} -> indexed is not a legal transition")
        updated = replace(rec, status="indexed", error="",
                          indexed_at=time.time(), chunk_count=chunk_count)
        self._write(docs, updated)
        return updated

    def retry(self, doc_id: str) -> DocRecord:
        return self.set_status(doc_id, "queued")  # only failed -> queued

    def reconcile_startup(self) -> list[str]:
        """A crash mid-extraction leaves 'extracting' records; they go back
        to the queue so indexing resumes on its own after a restart."""
        docs = self._load()
        recovered = []
        for d in docs:
            if d.status == "extracting":
                self._write(docs, replace(d, status="queued"))
                recovered.append(d.doc_id)
        return recovered

    # -- pins (per-project links into the global corpus) --------------------

    def pin(self, doc_id: str, project: str) -> None:
        docs = self._load()
        rec = self._find(docs, doc_id)
        if project not in rec.pinned_in:
            self._write(docs, replace(rec, pinned_in=(*rec.pinned_in, project)))

    def unpin(self, doc_id: str, project: str) -> None:
        docs = self._load()
        rec = next((d for d in docs if d.doc_id == doc_id), None)
        if rec is None or project not in rec.pinned_in:
            return
        self._write(docs, replace(
            rec, pinned_in=tuple(p for p in rec.pinned_in if p != project)))

    def pinned_ids(self, project: str) -> set[str]:
        return {d.doc_id for d in self._load() if project in d.pinned_in}

    # -- internals ----------------------------------------------------------

    @staticmethod
    def _find(docs: list[DocRecord], doc_id: str) -> DocRecord:
        rec = next((d for d in docs if d.doc_id == doc_id), None)
        if rec is None:
            raise CorpusError(f"no such corpus doc: {doc_id}")
        return rec

    def _write(self, docs: list[DocRecord], updated: DocRecord) -> None:
        docs[:] = [updated if d.doc_id == updated.doc_id else d
                    for d in docs]
        self._save(docs)
