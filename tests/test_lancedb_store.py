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
