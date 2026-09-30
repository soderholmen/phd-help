"""The harrier embedder (SPEC §6): one sentence-transformers model behind
the tiny sync ``encode`` seam LanceStore talks to.

Lazy on purpose: the ~1.2 GB model loads on first use, so app startup
never waits on it and the §8 degraded path pays nothing. Device picks
CUDA when a local GPU is up (the 5070 dev box, later the 3090), CPU
otherwise — indexing is background work and a search embeds one query,
so this is quality-critical, not latency-critical.
"""


class HarrierEmbedder:
    MODEL = "microsoft/harrier-oss-v1-0.6b"

    def __init__(self, model_name: str | None = None,
                 device: str | None = None):
        self.model_name = model_name or self.MODEL
        self._device = device
        self._model = None

    def _load(self):
        if self._model is None:
            import torch  # heavy: only when the local stack is wired
            from sentence_transformers import SentenceTransformer
            device = self._device or (
                "cuda" if torch.cuda.is_available() else "cpu")
            self._model = SentenceTransformer(self.model_name,
                                              device=device)
        return self._model

    def encode(self, texts: list[str]) -> list[list[float]]:
        """Sync — LanceStore offloads this to a worker thread.
        Normalized so LanceDB's L2 distance behaves like cosine."""
        vecs = self._load().encode(list(texts), normalize_embeddings=True)
        return [v.tolist() for v in vecs]
