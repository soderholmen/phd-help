"""The LanceDB adapter (SPEC §6) against the real embedded library with a
fake embedder — the adapter's own logic (schema, replace-on-reindex,
boost re-sort, doc assembly, empty-store degradation) is what's under
test; the harrier model itself is proven by the GPU probe, and faking
the embedder keeps the suite deterministic and GPU-free.

FakeEmbedder is a word-hashed bag-of-words: shared words mean close
vectors, so vector search does real keyword-ish ranking without a model.
"""

import hashlib
import re

import pytest

from phd_helper.chunking import Chunk
from phd_helper.server.lancedb_store import LanceStore


@pytest.fixture
def anyio_backend():
    return "asyncio"


class FakeEmbedder:
    DIM = 64

    def encode(self, texts):
        out = []
        for t in texts:
            v = [0.0] * self.DIM
            for w in re.findall(r"\w+", t.lower()):
                v[int(hashlib.md5(w.encode()).hexdigest(), 16)
                  % self.DIM] += 1.0
            norm = sum(x * x for x in v) ** 0.5 or 1.0
            out.append([x / norm for x in v])
        return out


def chunk(text, *, path="", page=1, kind="text", abstract=False,
          b0=0, b1=0):
    prefix = f"title » {path}" if path else "title"
    return Chunk(text=text, embed_text=f"{prefix}\n{text}",
                 section_path=path, page_start=page, page_end=page,
                 block_start=b0, block_end=b1, kind=kind,
                 is_abstract=abstract)


@pytest.fixture
def store(tmp_path):
    return LanceStore(tmp_path / "db", FakeEmbedder())


ATTN = [
    chunk("We propose the transformer, based on attention mechanisms.",
          path="Abstract", page=1, abstract=True, b0=0, b1=0),
    chunk("Attention weights are computed by scaled dot product.",
          path="Model Architecture » Scaled Dot-Product Attention",
          page=3, b0=10, b1=12),
]
OTHER = [
    chunk("Parakeets mimic human speech with their syrinx.",
          path="Abstract", page=1, abstract=True, b0=0, b1=0),
    chunk("Vocal learning in songbirds follows a motor pathway.",
          path="Discussion", page=5, b0=20, b1=21),
]


@pytest.mark.anyio
async def test_search_ranks_and_carries_locators(store):
    await store.index("attn", ATTN)
    await store.index("bird", OTHER)
    hits = await store.search("scaled dot product attention", k=3,
                              boost_ids=set())
    assert hits and hits[0].doc_id == "attn"
    top = hits[0]
    assert "scaled dot product" in top.text.lower()
    assert (top.page_start, top.block_start, top.block_end) == (3, 10, 12)
    assert top.section_path.endswith("Scaled Dot-Product Attention")


@pytest.mark.anyio
async def test_boost_lifts_pinned_doc_without_filtering(store):
    await store.index("attn", ATTN)
    await store.index("bird", OTHER)
    plain = await store.search("vocal learning pathway speech", k=4,
                               boost_ids=set())
    assert [h.doc_id for h in plain][:2] == ["bird", "bird"]
    boosted = await store.search("vocal learning pathway speech", k=4,
                                 boost_ids={"attn"})
    # Boosted, never filtered to (§6): the off-topic pinned doc is still
    # in the result set, and its score carries the multiplier.
    assert {h.doc_id for h in boosted} == {"bird", "attn"}
    by_doc = {h.doc_id: h.score for h in boosted}
    assert by_doc["attn"] > max(h.score for h in plain
                                if h.doc_id == "attn")
    # ...but a multiplier does not beat a much better raw match.
    assert boosted[0].doc_id == "bird"


@pytest.mark.anyio
async def test_reindex_replaces_chunks(store):
    await store.index("attn", ATTN)
    await store.index("attn", ATTN[:1])  # re-ingest a corrected version
    hits = await store.search("scaled dot product", k=5, boost_ids=set())
    assert all("scaled dot product" not in h.text.lower() for h in hits)


@pytest.mark.anyio
async def test_remove_drops_the_doc(store):
    await store.index("attn", ATTN)
    await store.index("bird", OTHER)
    await store.remove("bird")
    hits = await store.search("parakeet speech syrinx", k=5,
                              boost_ids=set())
    assert all(h.doc_id != "bird" for h in hits)
    assert await store.doc("bird") is None


@pytest.mark.anyio
async def test_doc_assembles_abstract_and_headings(store):
    await store.index("attn", ATTN)
    info = await store.doc("attn")
    assert info is not None
    assert "transformer" in info.abstract
    assert "Scaled Dot-Product Attention" in info.headings
    assert await store.doc("nosuch") is None


@pytest.mark.anyio
async def test_empty_store_degrades_cleanly(store):
    assert await store.search("anything", k=5, boost_ids=set()) == []
    assert await store.doc("x") is None
    await store.remove("x")  # must not raise on a store with no table


# -- the §6 rerank stage: cross-encoder over the fused top-50 --------------

class FakeReranker:
    """Deterministic stand-in for Qwen3-Reranker: scores come from a
    test-supplied function, so the final order is set by the reranker,
    not by RRF — which is exactly what these tests must show."""

    def __init__(self, score=lambda q, d: 0.5, fail=False, wrong_len=False):
        self.score = score
        self.fail = fail
        self.wrong_len = wrong_len
        self.calls: list[tuple[str, list[str]]] = []

    def rerank(self, query, docs):
        self.calls.append((query, list(docs)))
        if self.fail:
            raise RuntimeError("reranker OOM")
        scores = [self.score(query, d) for d in docs]
        return scores[:-1] if self.wrong_len and scores else scores


def rerank_store(tmp_path, reranker):
    return LanceStore(tmp_path / "db", FakeEmbedder(), reranker=reranker)


@pytest.mark.anyio
async def test_reranker_reorders_beyond_rrf(tmp_path):
    rr = FakeReranker(score=lambda q, d: 0.9 if "syrinx" in d else 0.1)
    store = rerank_store(tmp_path, rr)
    await store.index("attn", ATTN)
    await store.index("bird", OTHER)
    hits = await store.search("scaled dot product attention", k=2,
                              boost_ids=set())
    # RRF puts the attention chunk first; the reranker's judgement wins.
    assert hits[0].doc_id == "bird" and hits[0].score == pytest.approx(0.9)


@pytest.mark.anyio
async def test_rerank_docs_carry_the_section_path(tmp_path):
    rr = FakeReranker()
    store = rerank_store(tmp_path, rr)
    await store.index("attn", ATTN)
    await store.search("attention", k=1, boost_ids=set())
    docs = rr.calls[0][1]
    assert any("Scaled Dot-Product Attention" in d for d in docs)


def _odds_boost(p):
    return p * 1.5 / (1.0 + p * 0.5)


@pytest.mark.anyio
async def test_boost_applies_after_the_rerank(tmp_path):
    # attn 0.75 vs bird 0.8: bird leads on merit; the pinned odds
    # boost (0.75 -> 0.82) flips the near tie — boost rides the
    # reranked scale.
    rr = FakeReranker(
        score=lambda q, d: 0.75 if "attention" in d.lower() else 0.8)
    store = rerank_store(tmp_path, rr)
    await store.index("attn", ATTN)
    await store.index("bird", OTHER)
    plain = await store.search("attention speech", k=4, boost_ids=set())
    assert plain[0].doc_id == "bird"
    boosted = await store.search("attention speech", k=4,
                                 boost_ids={"attn"})
    assert boosted[0].doc_id == "attn"
    assert boosted[0].score == pytest.approx(_odds_boost(0.75))


@pytest.mark.anyio
async def test_boost_never_leaps_over_a_much_better_match(tmp_path):
    # The saturation trap: probabilities cap at 1.0, so a raw x1.5
    # would let a pinned 0.7 (-> 1.05) beat a near-certain 0.999. On
    # odds, 0.7 -> 0.78 and the excellent unpinned chunk stays first.
    rr = FakeReranker(
        score=lambda q, d: 0.7 if "attention" in d.lower() else 0.999)
    store = rerank_store(tmp_path, rr)
    await store.index("attn", ATTN)
    await store.index("bird", OTHER)
    boosted = await store.search("attention speech", k=4,
                                 boost_ids={"attn"})
    assert boosted[0].doc_id == "bird"
    assert boosted[0].score == pytest.approx(0.999)


@pytest.mark.anyio
async def test_reranker_nan_scores_fall_back(tmp_path):
    # len() can't catch NaN: it would poison the sort into an
    # undefined order and reach the agent as a score. Non-finite
    # scores are a fault, and the RRF order stands (§8).
    rr = FakeReranker(score=lambda q, d: float("nan"))
    store = rerank_store(tmp_path, rr)
    await store.index("attn", ATTN)
    await store.index("bird", OTHER)
    hits = await store.search("scaled dot product attention", k=2,
                              boost_ids=set())
    assert hits[0].doc_id == "attn"  # RRF order, not NaN chaos
    assert all(h.score == h.score for h in hits)  # no NaN reported


@pytest.mark.anyio
async def test_reranker_uncoercible_scores_fall_back(tmp_path):
    # Coercion must happen inside the guard: a reranker handing back
    # None is a fault to degrade on, not a TypeError after it.
    rr = FakeReranker(score=lambda q, d: None)
    store = rerank_store(tmp_path, rr)
    await store.index("attn", ATTN)
    await store.index("bird", OTHER)
    hits = await store.search("scaled dot product attention", k=2,
                              boost_ids=set())
    assert hits[0].doc_id == "attn"


@pytest.mark.anyio
async def test_reranker_failure_falls_back_to_rrf_order(tmp_path):
    # §8: a down reranker degrades the ranking, it never sinks the
    # search — the RRF-fused order and scores must stand untouched.
    rr = FakeReranker(fail=True)
    store = rerank_store(tmp_path, rr)
    await store.index("attn", ATTN)
    await store.index("bird", OTHER)
    hits = await store.search("scaled dot product attention", k=3,
                              boost_ids=set())
    ref_store = LanceStore(tmp_path / "db-no-reranker", FakeEmbedder())
    await ref_store.index("attn", ATTN)
    await ref_store.index("bird", OTHER)
    ref = await ref_store.search("scaled dot product attention", k=3,
                                boost_ids=set())
    assert [(h.doc_id, h.block_start) for h in hits] == \
           [(h.doc_id, h.block_start) for h in ref]


@pytest.mark.anyio
async def test_reranker_wrong_length_falls_back(tmp_path):
    # A reranker returning a misaligned score list must not silently
    # graft scores onto the wrong chunks.
    rr = FakeReranker(score=lambda q, d: 0.9, wrong_len=True)
    store = rerank_store(tmp_path, rr)
    await store.index("attn", ATTN)
    await store.index("bird", OTHER)
    hits = await store.search("scaled dot product attention", k=2,
                              boost_ids=set())
    assert hits[0].doc_id == "attn"  # RRF order, not the bogus 0.9s


# -- the §8 health probe: open + count, never the models --------------------


@pytest.mark.anyio
async def test_healthy_true_for_empty_and_populated_store(store):
    assert await store.healthy() is True  # no table yet: empty, not broken
    await store.index("attn", ATTN)
    assert await store.healthy() is True


@pytest.mark.anyio
async def test_healthy_never_loads_the_embedder(store):
    # /health rides every heartbeat: the probe is open+count, not a
    # search — a probe that encoded would drag the GPU model in.
    class CountingEmbedder(FakeEmbedder):
        def __init__(self):
            self.calls = 0

        def encode(self, texts):
            self.calls += 1
            return super().encode(texts)
    store.embedder = CountingEmbedder()
    assert await store.healthy() is True
    assert store.embedder.calls == 0


@pytest.mark.anyio
async def test_healthy_false_when_the_store_faults(tmp_path):
    # A configured store that raises on open is a fault, not "ok" (§8):
    # the locked-file / corrupt-manifest case the issue names.
    store = LanceStore(tmp_path / "db", FakeEmbedder())

    class Locked:
        def list_tables(self):
            raise OSError("lance manifest locked")
    store._db = Locked()
    assert await store.healthy() is False


@pytest.mark.anyio
async def test_reranker_sees_the_wide_candidate_field(tmp_path):
    # §6 reranks over top-50, not over k: with k=1 the reranker must
    # still judge every indexed chunk (12 here), so a chunk RRF buried
    # can still be lifted into the answer.
    rr = FakeReranker()
    store = rerank_store(tmp_path, rr)
    await store.index("many", [chunk(f"topic word {i} filler text here",
                                      page=i + 1, b0=i, b1=i)
                               for i in range(12)])
    await store.search("topic word filler", k=1, boost_ids=set())
    assert len(rr.calls[0][1]) == 12
