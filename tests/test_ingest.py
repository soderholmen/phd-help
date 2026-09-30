"""The ingestion pipeline (SPEC §6): two doors, one pipeline. The unit is
Ingestor.ingest — extracting -> chunking -> indexing with visible per-PDF
status. MinerU and the embedder/LanceDB store ride Protocol seams (fakes
here; the real adapters wait on the 3090 stack).
"""

import pytest

from phd_helper.chunking import Block
from phd_helper.corpus import Corpus
from phd_helper.ingest import Ingestor, IngestError


class FakeExtractor:
    def __init__(self, blocks=None, fail=False):
        self.blocks = blocks if blocks is not None else [
            Block("text", "the model scales audio", page=2, block=7)]
        self.fail = fail
        self.seen: list[bytes] = []

    async def extract(self, pdf: bytes):
        self.seen.append(pdf)
        if self.fail:
            raise RuntimeError("MinerU died")
        return self.blocks


class FakeStore:
    def __init__(self, fail=False):
        self.fail = fail
        self.indexed: dict[str, list] = {}
        self.removed: list[str] = []

    async def index(self, doc_id, chunks):
        if self.fail:
            raise RuntimeError("LanceDB down")
        self.indexed[doc_id] = chunks

    async def remove(self, doc_id):
        self.removed.append(doc_id)
        self.indexed.pop(doc_id, None)


@pytest.fixture
def corpus(tmp_path):
    return Corpus(tmp_path / "corpus")


def make(corpus, extractor=None, store=None, fetch_pdf=None):
    return Ingestor(corpus, extractor or FakeExtractor(),
                    store if store is not None else FakeStore(),
                    fetch_pdf=fetch_pdf)


@pytest.mark.anyio
async def test_ingest_indexes_chunks_and_marks_indexed(corpus):
    store = FakeStore()
    ing = make(corpus, store=store)
    rec, _ = corpus.add_pdf(b"%PDF-1.7 fake", title="Parakeet",
                            arxiv="2401.00001", source="agent")
    await ing.ingest(rec.doc_id)
    out = corpus.get(rec.doc_id)
    assert out.status == "indexed"
    assert out.chunk_count == 1
    chunks = store.indexed[rec.doc_id]
    assert chunks[0].embed_text == "Parakeet\nthe model scales audio"
    assert ing.runnable()


@pytest.mark.anyio
async def test_extractor_failure_is_visible_and_retryable(corpus):
    extractor = FakeExtractor(fail=True)
    store = FakeStore()
    ing = make(corpus, extractor=extractor, store=store)
    rec, _ = corpus.add_pdf(b"bad", title="T")
    await ing.ingest(rec.doc_id)
    out = corpus.get(rec.doc_id)
    assert out.status == "failed"
    assert "MinerU died" in out.error
    # the user asks for a retry: queued again, then it lands
    corpus.retry(rec.doc_id)
    extractor.fail = False
    await ing.ingest(rec.doc_id)
    assert corpus.get(rec.doc_id).status == "indexed"


@pytest.mark.anyio
async def test_store_failure_marks_failed(corpus):
    ing = make(corpus, store=FakeStore(fail=True))
    rec, _ = corpus.add_pdf(b"x", title="T")
    await ing.ingest(rec.doc_id)
    assert corpus.get(rec.doc_id).status == "failed"


@pytest.mark.anyio
async def test_superseded_doc_chunks_are_evicted(corpus):
    store = FakeStore()
    ing = make(corpus, store=store)
    old, _ = corpus.add_pdf(b"v1", title="Paper", arxiv="1706.03762")
    await ing.ingest(old.doc_id)
    new, _ = corpus.add_pdf(b"v2", title="Paper", arxiv="1706.03762")
    await ing.ingest(new.doc_id)
    assert store.removed == [old.doc_id]
    assert set(store.indexed) == {new.doc_id}  # old chunks evicted


@pytest.mark.anyio
async def test_failed_supersede_leaves_no_orphan_chunks(corpus):
    # add_pdf deletes the old record immediately; the old chunks must go
    # before the new extract, or a failed re-fetch leaves chunks whose
    # registry record is gone.
    extractor = FakeExtractor()
    store = FakeStore()
    ing = make(corpus, extractor=extractor, store=store)
    old, _ = corpus.add_pdf(b"v1", title="Paper", arxiv="1706.03762")
    await ing.ingest(old.doc_id)
    assert store.indexed  # old chunks live
    extractor.fail = True
    new, _ = corpus.add_pdf(b"v2", title="Paper", arxiv="1706.03762")
    await ing.ingest(new.doc_id)
    assert corpus.get(new.doc_id).status == "failed"
    assert store.indexed == {}  # index matches the registry: no orphans


@pytest.mark.anyio
async def test_concurrent_ingest_does_not_poison_the_owner(corpus):
    # upload + autojoin of the same paper race: the loser must not mark
    # the winner's doc failed.
    ing = make(corpus)
    rec, _ = corpus.add_pdf(b"x", title="T")
    corpus.set_status(rec.doc_id, "extracting")  # another ingest owns it
    await ing.ingest(rec.doc_id)  # returns quietly
    assert corpus.get(rec.doc_id).status == "extracting"


@pytest.mark.anyio
async def test_ingest_of_a_vanished_doc_is_quiet(corpus):
    # superseded mid-extraction: the record is gone; no crash, no write
    ing = make(corpus)
    rec, _ = corpus.add_pdf(b"v1", title="P", arxiv="a")
    corpus.add_pdf(b"v2", title="P", arxiv="a")  # supersedes rec
    await ing.ingest(rec.doc_id)  # must not raise


@pytest.mark.anyio
async def test_paused_pipeline_is_not_runnable(corpus):
    ing = Ingestor(corpus, extractor=None, store=None)
    assert not ing.runnable()


@pytest.mark.anyio
async def test_drain_queued_indexes_everything_queued(corpus):
    store = FakeStore()
    ing = make(corpus, store=store)
    a, _ = corpus.add_pdf(b"a", title="A")
    b, _ = corpus.add_pdf(b"b", title="B")
    corpus.set_status(b.doc_id, "extracting")  # crashed mid-flight
    corpus.reconcile_startup()  # -> queued
    await ing.drain_queued()
    assert corpus.get(a.doc_id).status == "indexed"
    assert corpus.get(b.doc_id).status == "indexed"


# -- door 2: the agent fetch (arXiv) + auto-join ----------------------------

@pytest.mark.anyio
async def test_fetch_arxiv_registers_and_indexes(corpus):
    fetched = []

    async def fetch_pdf(url):
        fetched.append(url)
        return b"%PDF arxiv bytes"
    store = FakeStore()
    ing = make(corpus, store=store, fetch_pdf=fetch_pdf)
    rec = await ing.fetch_arxiv("2401.00001")
    assert fetched == ["https://arxiv.org/pdf/2401.00001"]
    assert rec.source == "agent" and rec.arxiv == "2401.00001"
    assert corpus.get(rec.doc_id).status == "indexed"
    # auto-join again (same paper, same search twice): the registry is
    # checked before the network — no re-download of an owned paper,
    # no re-index, and the §6 1-req/3-s budget is spent on new papers
    again = await ing.fetch_arxiv("2401.00001")
    assert again.doc_id == rec.doc_id
    assert fetched == ["https://arxiv.org/pdf/2401.00001"]
    assert len(store.indexed[rec.doc_id]) == 1  # indexed once


@pytest.mark.anyio
async def test_fetch_arxiv_carries_the_search_title(corpus):
    # §6's embed prefix is `paper title » section`; the auto-join title
    # must reach the record, or every agent-fetched doc embeds "Untitled".
    async def fetch_pdf(url):
        return b"%PDF arxiv bytes"
    store = FakeStore()
    ing = make(corpus, store=store, fetch_pdf=fetch_pdf)
    rec = await ing.fetch_arxiv("2401.00001", title="Mesh Anything",
                                year="2024")
    assert rec.title == "Mesh Anything" and rec.year == "2024"
    assert store.indexed[rec.doc_id][0].embed_text.startswith("Mesh Anything")


@pytest.mark.anyio
async def test_fetch_arxiv_versioned_rehit_spends_no_fetch(corpus):
    # A later search returns the owned paper under a different version
    # string (or bare id): it is owned, so auto-join spends no fetch and
    # registers nothing new — re-fetching a version is the upload door's
    # job (§6 budget; supersede is covered in test_corpus).
    fetched = []

    async def fetch_pdf(url):
        fetched.append(url)
        return b"%PDF v" + str(len(fetched)).encode()
    store = FakeStore()
    ing = make(corpus, store=store, fetch_pdf=fetch_pdf)
    v1 = await ing.fetch_arxiv("2301.00001v1", title="P")
    for probe in ("2301.00001v2", "2301.00001", "arXiv:2301.00001v3"):
        again = await ing.fetch_arxiv(probe, title="P")
        assert again.doc_id == v1.doc_id
    assert len(fetched) == 1  # one download, ever
    assert [d.doc_id for d in corpus.list()] == [v1.doc_id]  # one record
    assert store.removed == []


@pytest.mark.anyio
async def test_fetch_arxiv_paywalled_or_dead_raises_without_a_record(corpus):
    async def fetch_pdf(url):
        raise IngestError("PDF fetch failed (404)")
    ing = make(corpus, fetch_pdf=fetch_pdf)
    with pytest.raises(IngestError):
        await ing.fetch_arxiv("9999.00001")
    assert corpus.list() == []  # paywalled papers get a bib entry only (§6)
