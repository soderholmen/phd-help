"""Sidecar STT client (SPEC §3; docs/audio-stack.md).

The model (parakeet-tdt-0.6b-v3) lives in a py3.12 sidecar process —
torch is blocked in this venv by Smart App Control — so this client owns
everything that does *not* need torch: the endpointing kernel (segmenting)
runs here, and the sidecar only turns a finished utterance's raw PCM16
into a final. One POST per utterance, ~0.46 s measured, awaited so the
event loop keeps serving everyone else.

Failure policy (§8, honest silence): a dead sidecar must never break the
WebSocket — feed() swallows transport errors and returns no finals, so
typed chat continues untouched. Three consecutive failures flip faulted()
for /health; one success self-heals the counter when the sidecar returns.
"""

import json

import httpx

from ..segmenting import EnergyVad, UtteranceGate


class SidecarStt:
    def __init__(self, url: str, http: httpx.AsyncClient | None = None,
                 vad=None, hangover_s: float = 0.6,
                 min_utterance_s: float = 0.25, timeout_s: float = 15.0,
                 fault_threshold: int = 3):
        self._http = http or httpx.AsyncClient(base_url=url,
                                               timeout=timeout_s)
        self._owns_http = http is None
        self._gate = UtteranceGate(vad or EnergyVad(),
                                   hangover_s=hangover_s,
                                   min_utterance_s=min_utterance_s)
        self._failures = 0
        self._fault_threshold = fault_threshold

    async def feed(self, pcm16_bytes: bytes) -> list[str]:
        self._gate.push(pcm16_bytes)
        if not self._gate.ready():
            return []
        utterance = self._gate.pop()
        try:
            r = await self._http.post("/transcribe", content=utterance)
            r.raise_for_status()
            text = (r.json().get("text") or "").strip()
        except (httpx.HTTPError, json.JSONDecodeError):
            # Honest silence covers transport + malformed body — but not
            # every ValueError, or a genuine bug would hide as "no finals".
            self._failures += 1
            return []
        self._failures = 0
        return [text] if text else []

    def faulted(self) -> bool:
        return self._failures >= self._fault_threshold

    async def healthy(self) -> bool:
        try:
            return (await self._http.get("/health")).status_code == 200
        except httpx.HTTPError:
            return False

    async def aclose(self) -> None:
        if self._owns_http:
            await self._http.aclose()
