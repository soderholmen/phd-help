"""The corpus sidecar clients (SPEC §6; docs/corpus-stack.md).

Same disease as the audio stack, same cure: torch is SAC-blocked in this
venv, so the harrier embedder and the Qwen reranker live in a py3.12
sidecar and the backend holds thin sync httpx clients. Sync is not a
shortcut — LanceStore calls both seams exclusively through
asyncio.to_thread, so a sync client is a true drop-in, and the last two
tests prove it through the real store.

Failure policy differs from SidecarStt on purpose: the voice client
swallows (honest silence must not break the WebSocket), the corpus
clients RAISE after counting — the callers already own the §8 degraded
paths (RRF order, CORPUS_DOWN), and a swallowed encode failure would
index zero-vector chunks.
"""

import hashlib
import json
import re

import httpx
import pytest

from phd_helper.chunking import Chunk
from phd_helper.server.corpus_sidecar import SidecarEmbedder, SidecarReranker
from phd_helper.server.lancedb_store import LanceStore

@pytest.fixture
def anyio_backend():
    return "asyncio"


DIM = 64  # the store tests' FakeEmbedder width: same corpus, same math


def bow(text):
    """Word-hashed bag-of-words — the exact FakeEmbedder math from
    test_lancedb_store, so the drop-in proof exercises real vector
    ranking through the sidecar wire instead of a canned order."""
    v = [0.0] * DIM
    for w in re.findall(r"\w+", text.lower()):
        v[int(hashlib.md5(w.encode()).hexdigest(), 16) % DIM] += 1.0
    norm = sum(x * x for x in v) ** 0.5 or 1.0
    return [x / norm for x in v]


def chunk(text, *, path="", page=1, kind="text", abstract=False, b0=0, b1=0):
    prefix = f"title » {path}" if path else "title"
    return Chunk(text=text, embed_text=f"{prefix}\n{text}",
                 section_path=path, page_start=page, page_end=page,
                 block_start=b0, block_end=b1, kind=kind,
                 is_abstract=abstract)


ATTN = [
    chunk("We propose the transformer, based on attention mechanisms.",
          path="Abstract", page=1, abstract=True),
    chunk("Attention weights are computed by scaled dot product.",
          path="Model Architecture » Scaled Dot-Product Attention",
          page=3, b0=10, b1=12),
]
OTHER = [
    chunk("Parakeets mimic human speech with their syrinx.",
          path="Abstract", page=1, abstract=True),
    chunk("Vocal learning in songbirds follows a motor pathway.",
          path="Discussion", page=5, b0=20, b1=21),
]


def make_embedder(handler, **kw):
    client = httpx.Client(transport=httpx.MockTransport(handler),
                          base_url="http://corpus.test")
    return SidecarEmbedder("http://corpus.test", http=client, **kw)


def make_reranker(handler, **kw):
    client = httpx.Client(transport=httpx.MockTransport(handler),
                          base_url="http://corpus.test")
    return SidecarReranker("http://corpus.test", http=client, **kw)


def down(request):
    raise httpx.ConnectError("sidecar down")


def ok(request):
    if request.url.path == "/embed":
        return httpx.Response(200, json={
            "vectors": [[0.0] * 4
                        for _ in json.loads(request.content)["texts"]]})
    if request.url.path == "/health":
        return httpx.Response(200, json={"status": "ok"})
    return httpx.Response(200, json={
        "scores": [0.0] * len(json.loads(request.content)["docs"])})


# -- the wire contract -------------------------------------------------------

def test_encode_posts_texts_and_returns_the_vectors():
    seen = {}

    def handler(request):
        seen["path"] = request.url.path
        seen["body"] = json.loads(request.content)
        return httpx.Response(200, json={"vectors": [[0.1, 0.2], [0.3, 0.4]]})

    emb = make_embedder(handler)
    assert emb.encode(["alpha", "beta"]) == [[0.1, 0.2], [0.3, 0.4]]
    assert seen == {"path": "/embed", "body": {"texts": ["alpha", "beta"]}}


def test_rerank_posts_query_and_docs_and_sigmoids_the_margins():
    seen = {}

    def handler(request):
        seen["path"] = request.url.path
        seen["body"] = json.loads(request.content)
        return httpx.Response(200, json={"scores": [9.0, -9.0, 0.0]})

    rr = make_reranker(handler)
    scores = rr.rerank("what is attention", ["doc a", "doc b", "doc c"])
    assert seen == {"path": "/rerank",
                    "body": {"query": "what is attention",
                             "docs": ["doc a", "doc b", "doc c"]}}
    # The wire carries raw margins; the client owns the probability
    # contract — same numbers as the in-process reranker's tests.
    assert scores[0] > 0.99 and scores[1] < 0.01
    assert scores[2] == pytest.approx(0.5)


# -- the failure policy: raise, count, self-heal ------------------------------

CLIENTS = [
    (SidecarEmbedder, lambda c: c.encode(["hello"])),
    (SidecarReranker, lambda c: c.rerank("q", ["a doc"])),
]


@pytest.mark.parametrize("cls,call", CLIENTS)
def test_transport_failures_raise_count_and_self_heal(cls, call):
    state = {"down": True}

    def handler(request):
        if state["down"]:
            raise httpx.ConnectError("sidecar down")
        return ok(request)

    c = cls("http://corpus.test", http=httpx.Client(
        transport=httpx.MockTransport(handler), base_url="http://corpus.test"))
    for _ in range(2):
        with pytest.raises(httpx.ConnectError):
            call(c)
        assert not c.faulted()
    with pytest.raises(httpx.ConnectError):
        call(c)
    assert c.faulted()  # three consecutive: /health flips
    state["down"] = False
    call(c)             # one success against the recovered sidecar...
    assert not c.faulted()  # ...self-heals, like the audio clients


@pytest.mark.parametrize("cls,call", CLIENTS)
def test_http_500_raises_and_counts(cls, call):
    c = cls("http://corpus.test", http=httpx.Client(
        transport=httpx.MockTransport(lambda r: httpx.Response(500)),
        base_url="http://corpus.test"))
    for _ in range(3):
        with pytest.raises(httpx.HTTPStatusError):
            call(c)
    assert c.faulted()


def test_healthy_reads_the_sidecar_health_endpoint():
    up = make_embedder(lambda r: httpx.Response(200, json={"status": "ok"}))
    assert up.healthy() is True
    assert make_embedder(down).healthy() is False
    assert make_embedder(lambda r: httpx.Response(503)).healthy() is False


def test_healthy_never_resets_the_failure_counter():
    # The counter beats a fresh probe (the audio precedent): a faulted
    # chip stays faulted until a real encode succeeds, so /health can
    # never oscillate ok/faulted while calls still fail.
    state = {"down": True}

    def handler(request):
        if state["down"]:
            raise httpx.ConnectError("sidecar down")
        return ok(request)

    emb = make_embedder(handler)
    for _ in range(3):
        with pytest.raises(httpx.ConnectError):
            emb.encode(["x"])
    assert emb.faulted()
    state["down"] = False
    assert emb.healthy() is True
    assert emb.faulted()  # ...probe alone does not clear it


def test_close_only_closes_the_client_it_owns():
    injected = httpx.Client(base_url="http://corpus.test")
    SidecarEmbedder("http://corpus.test", http=injected).close()
    assert not injected.is_closed  # borrowed, not ours to close
    SidecarReranker("http://corpus.test").close()  # owned: must not raise
    injected.close()


# -- the drop-in proof: the real LanceStore over the sidecar wire -------------

def corpus_handler(rerank_status=200):
    def handler(request):
        if request.url.path == "/embed":
            return httpx.Response(200, json={
                "vectors": [bow(t) for t in json.loads(request.content)["texts"]]})
        if request.url.path == "/rerank":
            if rerank_status != 200:
                return httpx.Response(rerank_status)
            docs = json.loads(request.content)["docs"]
            return httpx.Response(200, json={
                "scores": [9.0 if "transformer" in d else -9.0
                           for d in docs]})
        return httpx.Response(404)
    return handler


@pytest.mark.anyio
async def test_sidecar_embedder_is_a_true_drop_in_for_lancestore(tmp_path):
    store = LanceStore(tmp_path / "db", make_embedder(corpus_handler()))
    await store.index("attn", ATTN)
    await store.index("bird", OTHER)
    # Same corpus, same bow math, same assertion as the FakeEmbedder
    # store test — the wire changed, the ranking did not.
    hits = await store.search("vocal learning pathway speech", k=4,
                              boost_ids=set())
    assert [h.doc_id for h in hits][:2] == ["bird", "bird"]


@pytest.mark.anyio
async def test_sidecar_reranker_replaces_rrf_order_through_the_store(
        tmp_path):
    handler = corpus_handler()
    store = LanceStore(tmp_path / "db", make_embedder(handler),
                       reranker=make_reranker(handler))
    await store.index("attn", ATTN)
    await store.index("bird", OTHER)
    hits = await store.search("vocal learning pathway speech", k=4,
                              boost_ids=set())
    # RRF ranks the bird prose first (the query's words are its words),
    # but the cross-encoder margins override it — and the score the
    # agent sees is the client-side sigmoid of the wire margin.
    assert hits[0].doc_id == "attn"
    assert hits[0].score > 0.99


@pytest.mark.anyio
async def test_rerank_fault_degrades_to_rrf_order_through_the_store(
        tmp_path):
    handler = corpus_handler(rerank_status=500)
    store = LanceStore(tmp_path / "db", make_embedder(handler),
                       reranker=make_reranker(handler))
    await store.index("attn", ATTN)
    await store.index("bird", OTHER)
    # Degrades, never sinks (§8): a raising reranker leaves the fused
    # order standing — the search itself must not fault.
    hits = await store.search("vocal learning pathway speech", k=4,
                              boost_ids=set())
    assert [h.doc_id for h in hits][:2] == ["bird", "bird"]
