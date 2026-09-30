"""The §6 rerank stage: Qwen3-Reranker-0.6B as a cross-encoder over the
RRF-fused top-50 (probe-verified on the 5070 under SAC: sentence-
transformers' CrossEncoder speaks this model natively, prompt template
and all).

The model scores a (query, doc) pair by how much it would answer the
query — a real read of the chunk, not bag-of-words proximity, which is
what lifts "3.2.1 Scaled Dot-Product Attention" over a chunk that just
mentions attention in passing.

Its raw scores are margins (signed: +9 relevant, -9 not). They are
squashed through a sigmoid before leaving here, because the pinned-doc
boost is a multiplier — on a signed scale, multiplying by 1.5 would
bury a pinned doc, inverting the §6 boost. Probabilities keep the
scale positive, so boosting can only ever lift.

Lazy like the embedder: the model loads on first search, so startup
never waits on it and the §8 degraded path pays nothing.
"""

import math


class QwenReranker:
    MODEL = "Qwen/Qwen3-Reranker-0.6B"

    def __init__(self, model_name: str | None = None,
                 device: str | None = None, max_length: int = 1024):
        self.model_name = model_name or self.MODEL
        self._device = device
        self._max_length = max_length
        self._model = None

    def _load(self):
        if self._model is None:
            import torch  # heavy: only when the local stack is wired
            from sentence_transformers import CrossEncoder
            device = self._device or (
                "cuda" if torch.cuda.is_available() else "cpu")
            self._model = CrossEncoder(self.model_name, device=device,
                                       max_length=self._max_length)
        return self._model

    def rerank(self, query: str, docs: list[str]) -> list[float]:
        """Sync — LanceStore offloads this to a worker thread. One
        P(relevant) per (query, doc) pair, in input order."""
        margins = self._load().predict([(query, d) for d in docs])
        return [1.0 / (1.0 + math.exp(-float(m))) for m in margins]
