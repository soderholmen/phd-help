"""Corpus sidecar clients (SPEC §6; docs/corpus-stack.md).

The harrier embedder and the Qwen reranker live in one py3.12 sidecar
process (scripts/corpus_server.py) — torch is SAC-blocked in this venv,
the same disease the audio slice cured, the same cure. Sync is not a
shortcut: LanceStore calls both seams exclusively through
asyncio.to_thread, so a sync httpx.Client is a true drop-in for
HarrierEmbedder/QwenReranker — no store changes, no async migration.

Failure policy is deliberately the opposite of SidecarStt's: the voice
client swallows so the WebSocket never breaks, these RAISE after
counting. The callers already own the §8 degraded paths (_rerank falls
back to RRF order, tools.py maps failures to CORPUS_DOWN), and a
swallowed encode failure would silently index zero-vector chunks —
worse than no search at all. Three consecutive failures flip faulted()
for /health; one success self-heals the counter.
"""

import math

import httpx


class _Sidecar:
    """Client ownership + the fault counter, shared by the corpus
    family only. The audio clients keep their own copies on purpose:
    they differ in plane (async) and policy (swallow), and a
    cross-family base would abstract over one real difference."""

    def __init__(self, url: str, *, http: httpx.Client | None,
                 timeout: float, fault_threshold: int):
        self._http = http or httpx.Client(base_url=url, timeout=timeout)
        self._owns_http = http is None
        self._failures = 0
        self._fault_threshold = fault_threshold

    def faulted(self) -> bool:
        return self._failures >= self._fault_threshold

    def healthy(self) -> bool:
        """Sync probe with its own short timeout (the client's minutes-
        long inference timeout must never apply to a health check).
        Deliberately never touches the failure counter: the counter
        beats a fresh probe, so /health can't oscillate ok/faulted
        while real calls still fail. app.py offloads this via
        asyncio.to_thread — a wedged sidecar must not stall the loop
        the shell's 3 s health poll runs on."""
        try:
            return self._http.get("/health", timeout=2.0).status_code == 200
        except httpx.HTTPError:
            return False

    def close(self) -> None:
        if self._owns_http:
            self._http.close()  # borrowed clients (tests) are not ours


class SidecarEmbedder(_Sidecar):
    def __init__(self, url: str, http: httpx.Client | None = None,
                 timeout: float = 120.0, fault_threshold: int = 3):
        # 120 s: the first call after startup pays the sidecar's lazy
        # model load; later calls are milliseconds.
        super().__init__(url, http=http, timeout=timeout,
                         fault_threshold=fault_threshold)

    def encode(self, texts: list[str]) -> list[list[float]]:
        """Sync — LanceStore offloads this to a worker thread. The
        server normalizes (L2 ≈ cosine depends on it); the wire carries
        the vectors as-is."""
        try:
            r = self._http.post("/embed", json={"texts": list(texts)})
            r.raise_for_status()
            vectors = r.json()["vectors"]
        except httpx.HTTPError:
            self._failures += 1
            raise
        self._failures = 0
        return vectors


class SidecarReranker(_Sidecar):
    def __init__(self, url: str, http: httpx.Client | None = None,
                 timeout: float = 60.0, fault_threshold: int = 3):
        super().__init__(url, http=http, timeout=timeout,
                         fault_threshold=fault_threshold)

    def rerank(self, query: str, docs: list[str]) -> list[float]:
        """Sync — offloaded like encode. The wire carries raw margins;
        the sigmoid lives here, client-side, so the probability
        contract stays with the tested adapter (and the §6 boost's
        positive-scale requirement, per rerank.py, keeps holding)."""
        try:
            r = self._http.post("/rerank",
                                json={"query": query, "docs": list(docs)})
            r.raise_for_status()
            margins = r.json()["scores"]
        except httpx.HTTPError:
            self._failures += 1
            raise
        self._failures = 0
        return [1.0 / (1.0 + math.exp(-float(m))) for m in margins]
