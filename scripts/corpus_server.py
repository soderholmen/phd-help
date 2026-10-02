"""harrier embed + Qwen rerank sidecar (docs/corpus-stack.md).

Standalone on purpose: it runs in .venv-embed (py3.12 + sentence-
transformers + torch), which does not have phd_helper installed — and
must never import it, so the backend venv (where Smart App Control
blocks torch) can never be tempted to load a model in-process. Pinned
by an AST test in tests/test_audio.py.

One process hosts both models because they share the corpus seams, but
each loads independently and lazily: a search that only reranks must
not pay for the embedder, and vice versa. The wire carries raw rerank
margins — the sigmoid is the client's (the tested adapter owns the
probability contract), while normalize_embeddings stays server-side:
LanceDB's L2-as-cosine depends on it, and no client may opt out.

Run:  .venv-embed/Scripts/python.exe scripts/corpus_server.py --port 8091
"""

import argparse
import asyncio
import os
import threading

from fastapi import FastAPI
from pydantic import BaseModel

EMBED_MODEL = os.environ.get("PHD_EMBED_MODEL",
                             "microsoft/harrier-oss-v1-0.6b")
RERANK_MODEL = os.environ.get("PHD_RERANK_MODEL", "Qwen/Qwen3-Reranker-0.6B")
# fp16 halves the VRAM bill against the voice sidecars on one 16 GB
# card; on CPU fp16 inference is slow and numerically shaky, so CPU
# forces fp32 regardless. PHD_CORPUS_DTYPE=fp32 is the escape hatch if
# an embedding starts to drift.
DTYPE = os.environ.get("PHD_CORPUS_DTYPE", "fp16")

app = FastAPI(title="phd-helper corpus sidecar")

_embedder = None
_reranker = None
_embed_load_lock = threading.Lock()   # double-checked lazy loads, per model
_rerank_load_lock = threading.Lock()
_embed_infer_lock = threading.Lock()   # one GPU: one forward pass per model
_rerank_infer_lock = threading.Lock()


def _load_kwargs():
    import torch
    if torch.cuda.is_available():
        dtype = torch.float16 if DTYPE == "fp16" else torch.float32
        return {"device": "cuda", "model_kwargs": {"torch_dtype": dtype}}
    return {"device": "cpu"}


def get_embedder():
    global _embedder
    if _embedder is None:
        with _embed_load_lock:
            if _embedder is None:
                from sentence_transformers import SentenceTransformer
                _embedder = SentenceTransformer(EMBED_MODEL,
                                                **_load_kwargs())
                # a raise above leaves None: next call retries the load
    return _embedder


def get_reranker():
    global _reranker
    if _reranker is None:
        with _rerank_load_lock:
            if _reranker is None:
                from sentence_transformers import CrossEncoder
                _reranker = CrossEncoder(RERANK_MODEL, max_length=1024,
                                         **_load_kwargs())
    return _reranker


@app.get("/health")
def health():
    return {"status": "ok", "embed_model": EMBED_MODEL,
            "rerank_model": RERANK_MODEL, "dtype": DTYPE,
            "loaded": {"embed": _embedder is not None,
                       "rerank": _reranker is not None}}


class EmbedBody(BaseModel):
    texts: list[str]


class RerankBody(BaseModel):
    query: str
    docs: list[str]


def _encode(texts: list[str]):
    model = get_embedder()
    with _embed_infer_lock:
        vecs = model.encode(texts, normalize_embeddings=True)
    return [v.tolist() for v in vecs]


@app.post("/embed")
async def embed(body: EmbedBody):
    if not body.texts:
        return {"vectors": []}  # never load a 1.2 GB model for nothing
    # to_thread, not inline: the lazy load and the infer lock on the
    # event loop would stall this sidecar's own /health — a merely busy
    # sidecar must not read faulted to the backend.
    return {"vectors": await asyncio.to_thread(_encode, body.texts)}


def _rerank(query: str, docs: list[str]):
    model = get_reranker()
    with _rerank_infer_lock:
        margins = model.predict([(query, d) for d in docs])
    return [float(m) for m in margins]


@app.post("/rerank")
async def rerank(body: RerankBody):
    if not body.docs:
        return {"scores": []}
    return {"scores": await asyncio.to_thread(_rerank,
                                              body.query, body.docs)}


if __name__ == "__main__":
    import uvicorn
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8091)
    args = parser.parse_args()
    uvicorn.run(app, host="127.0.0.1", port=args.port)
