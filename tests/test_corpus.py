"""The reference-corpus registry (SPEC §6): one global corpus across
projects, dedup by content hash, visible per-PDF status as a small state
machine, per-project pins, and a new arXiv version superseding the old
record on re-fetch. Filesystem-backed like the other persistence modules.
"""

import pytest

from phd_helper.corpus import Corpus, CorpusError


@pytest.fixture
def corpus(tmp_path):
    return Corpus(tmp_path / "corpus")


def add(corpus, body=b"pdf bytes v1", **kw):
    meta = {"title": "Attention Is All You Need", "arxiv": "1706.03762",
            "source": "agent"}
    meta.update(kw)
    rec, new = corpus.add_pdf(body, **meta)
    return rec


def test_add_pdf_registers_queued_and_stores_the_pdf(corpus):
    rec = add(corpus)
    assert rec.status == "queued"
    assert rec.title == "Attention Is All You Need"
    assert rec.arxiv == "1706.03762"
    assert rec.source == "agent"
    assert corpus.pdf_path(rec.doc_id).read_bytes() == b"pdf bytes v1"
    assert corpus.get(rec.doc_id) == rec


def test_dedup_by_content_hash(corpus):
    first = add(corpus)
    again, new = corpus.add_pdf(b"pdf bytes v1", title="Duplicate Title")
    assert not new
    assert again.doc_id == first.doc_id
    assert again.title == "Attention Is All You Need"  # original wins
    assert len(corpus.list()) == 1


def test_new_arxiv_version_supersedes_the_old_record(corpus):
    old = add(corpus)
    corpus.pin(old.doc_id, "my-paper")
    new = add(corpus, body=b"pdf bytes v2")
    assert new.doc_id != old.doc_id
    assert new.status == "queued"
    assert new.supersedes == old.doc_id  # the pipeline evicts the old chunks
    assert corpus.get(old.doc_id) is None  # superseded, not kept around
    assert corpus.list() == [new]
    assert corpus.pinned_ids("my-paper") == {new.doc_id}  # pins carry over


def test_supersede_matches_on_doi_too(corpus):
    old = add(corpus, arxiv="", doi="10.1000/x")
    new = add(corpus, body=b"other bytes", arxiv="", doi="10.1000/x")
    assert corpus.get(old.doc_id) is None
    assert corpus.get(new.doc_id) == new


def test_status_happy_path(corpus):
    rec = add(corpus)
    corpus.set_status(rec.doc_id, "extracting")
    corpus.mark_indexed(rec.doc_id, chunk_count=42)
    out = corpus.get(rec.doc_id)
    assert out.status == "indexed"
    assert out.chunk_count == 42
    assert out.indexed_at >= out.added_at


def test_failure_then_retry(corpus):
    rec = add(corpus)
    corpus.set_status(rec.doc_id, "extracting")
    corpus.set_status(rec.doc_id, "failed", error="MinerU crashed")
    assert corpus.get(rec.doc_id).error == "MinerU crashed"
    corpus.retry(rec.doc_id)
    assert corpus.get(rec.doc_id).status == "queued"
    assert corpus.get(rec.doc_id).error == ""


def test_illegal_transitions_are_rejected(corpus):
    rec = add(corpus)
    with pytest.raises(CorpusError):
        corpus.set_status(rec.doc_id, "indexed")  # skips extracting
    corpus.set_status(rec.doc_id, "extracting")
    corpus.mark_indexed(rec.doc_id, chunk_count=1)
    with pytest.raises(CorpusError):
        corpus.set_status(rec.doc_id, "queued")  # indexed is terminal
    with pytest.raises(CorpusError):
        corpus.retry(rec.doc_id)  # only failed retries


def test_unknown_doc_id(corpus):
    assert corpus.get("nope") is None
    with pytest.raises(CorpusError):
        corpus.set_status("nope", "extracting")


def test_pins_are_per_project(corpus):
    a = add(corpus, title="A", arxiv="a")
    b = add(corpus, body=b"other", title="B", arxiv="b")
    corpus.pin(a.doc_id, "proj1")
    corpus.pin(b.doc_id, "proj2")
    corpus.pin(a.doc_id, "proj1")  # idempotent
    assert corpus.pinned_ids("proj1") == {a.doc_id}
    assert corpus.pinned_ids("proj2") == {b.doc_id}
    corpus.unpin(a.doc_id, "proj1")
    assert corpus.pinned_ids("proj1") == set()
    corpus.unpin(a.doc_id, "proj1")  # unknown pin is a no-op


def test_startup_reconcile_recovers_crashed_extractions(corpus):
    a = add(corpus)
    b = add(corpus, body=b"two", title="B", arxiv="b")
    c = add(corpus, body=b"three", title="C", arxiv="c")
    corpus.set_status(a.doc_id, "extracting")
    corpus.set_status(b.doc_id, "extracting")
    corpus.mark_indexed(b.doc_id, chunk_count=1)  # b finished before crash
    recovered = corpus.reconcile_startup()
    assert recovered == [a.doc_id]
    assert corpus.get(a.doc_id).status == "queued"
    assert corpus.get(b.doc_id).status == "indexed"
    assert corpus.get(c.doc_id).status == "queued"


def test_registry_survives_restart(corpus):
    rec = add(corpus)
    corpus.set_status(rec.doc_id, "extracting")
    reopened = Corpus(corpus.root)
    assert reopened.get(rec.doc_id).status == "extracting"
    assert reopened.pdf_path(rec.doc_id).read_bytes() == b"pdf bytes v1"


def test_list_is_oldest_first(corpus):
    a = add(corpus)
    b = add(corpus, body=b"two", title="B", arxiv="b")
    assert [r.doc_id for r in corpus.list()] == [a.doc_id, b.doc_id]
