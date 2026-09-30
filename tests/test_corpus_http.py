"""The corpus HTTP surface (SPEC §6): upload door, per-PDF status list,
retry. Tested at the HTTP seam with the real Corpus + Ingestor and fake
extractor/store — the app's thin pass-throughs and the paused behavior
(§8) are what only the app owns.
"""

import time
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from phd_helper.corpus import Corpus
from phd_helper.ingest import Ingestor
from phd_helper.server.app import create_app


class FakeExtractor:
    def __init__(self, fail=False):
        self.fail = fail

    async def extract(self, pdf):
        if self.fail:
            raise RuntimeError("MinerU died")
        from phd_helper.chunking import Block
        return [Block("text", "body", page=1, block=0)]


class FakeStore:
    def __init__(self):
        self.indexed = {}

    async def index(self, doc_id, chunks):
        self.indexed[doc_id] = chunks

    async def remove(self, doc_id):
        self.indexed.pop(doc_id, None)


class FakeHttp:
    async def aclose(self):
        pass


def make_env(tmp_path, extractor=None, store=None):
    corpus = Corpus(tmp_path / "corpus")
    extractor = extractor if extractor is not None else FakeExtractor()
    store = store if store is not None else FakeStore()
    ingestor = Ingestor(corpus, extractor, store)
    state = SimpleNamespace(http=FakeHttp(), corpus=corpus,
                            corpus_store=store, ingestor=ingestor,
                            ingest_tasks=set(),
                            project=SimpleNamespace(
                                root=tmp_path / "my-paper"))
    return state, TestClient(create_app(state=state))


def settle(client, doc_id, want="indexed", tries=200):
    """Background ingest runs on the app's loop; poll until it settles."""
    doc = {}
    for _ in range(tries):
        doc = next(d for d in client.get("/corpus/docs").json()
                   if d["doc_id"] == doc_id)
        if doc["status"] == want:
            return doc
        time.sleep(0.005)
    raise AssertionError(f"doc stuck in {doc.get('status')}")


def test_upload_indexes_and_shows_status(tmp_path):
    _, client = make_env(tmp_path)
    with client:
        r = client.post("/corpus/upload?title=Parakeet&arxiv=2401.00001",
                        content=b"%PDF-1.7 fake")
        assert r.status_code == 200
        body = r.json()
        assert body["new"] and body["status"] in ("queued", "extracting")
        doc = settle(client, body["doc_id"])
        assert doc["title"] == "Parakeet" and doc["arxiv"] == "2401.00001"
        assert doc["source"] == "upload" and doc["chunk_count"] == 1


def test_upload_dedups_by_content_hash(tmp_path):
    _, client = make_env(tmp_path)
    with client:
        first = client.post("/corpus/upload", content=b"same bytes").json()
        settle(client, first["doc_id"])
        again = client.post("/corpus/upload", content=b"same bytes").json()
        assert again["doc_id"] == first["doc_id"] and not again["new"]
        assert len(client.get("/corpus/docs").json()) == 1


def test_upload_empty_body_is_a_400(tmp_path):
    _, client = make_env(tmp_path)
    with client:
        assert client.post("/corpus/upload", content=b"").status_code == 400


def test_paused_pipeline_leaves_docs_queued(tmp_path):
    # No extractor/store wired (the 3090 stack is down): uploads register
    # and stay visibly queued — indexing pauses, it does not rot (§8).
    corpus = Corpus(tmp_path / "corpus")
    state = SimpleNamespace(http=FakeHttp(), corpus=corpus,
                            corpus_store=None,
                            ingestor=Ingestor(corpus, None, None),
                            ingest_tasks=set())
    with TestClient(create_app(state=state)) as client:
        body = client.post("/corpus/upload", content=b"x").json()
        settle(client, body["doc_id"], want="queued", tries=5)


def test_failure_is_visible_and_retry_lands(tmp_path):
    extractor = FakeExtractor(fail=True)
    _, client = make_env(tmp_path, extractor=extractor)
    with client:
        body = client.post("/corpus/upload", content=b"x").json()
        doc = settle(client, body["doc_id"], want="failed")
        assert "MinerU died" in doc["error"]
        extractor.fail = False
        r = client.post(f"/corpus/{body['doc_id']}/retry")
        assert r.status_code == 200
        settle(client, body["doc_id"], want="indexed")


def test_pin_attaches_doc_to_active_project(tmp_path):
    _, client = make_env(tmp_path)
    with client:
        doc_id = client.post("/corpus/upload", content=b"x").json()["doc_id"]
        r = client.post(f"/corpus/{doc_id}/pin")
        assert r.status_code == 200
        assert r.json()["pinned_in"] == ["my-paper"]  # the active project
        docs = client.get("/corpus/docs").json()
        assert next(d for d in docs if d["doc_id"] == doc_id)["pinned_in"] \
            == ["my-paper"]
        # idempotent
        assert client.post(f"/corpus/{doc_id}/pin").json()["pinned_in"] \
            == ["my-paper"]
        r = client.post(f"/corpus/{doc_id}/unpin")
        assert r.json()["pinned_in"] == []


def test_pin_unknown_doc_is_a_409(tmp_path):
    _, client = make_env(tmp_path)
    with client:
        assert client.post("/corpus/nosuchdoc/pin").status_code == 409


def test_retry_only_for_failed_docs(tmp_path):
    _, client = make_env(tmp_path)
    with client:
        body = client.post("/corpus/upload", content=b"x").json()
        settle(client, body["doc_id"])  # indexed
        r = client.post(f"/corpus/{body['doc_id']}/retry")
        assert r.status_code == 409  # indexed is terminal; supersede instead
        assert client.post("/corpus/nosuchdoc/retry").status_code == 409
