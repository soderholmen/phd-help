"""MOSS-TTS-Realtime sidecar client (SPEC §3; docs/audio-stack.md).

The exact request sequence the probe measured (scripts/tts_server.py
hosts the vendored server): start the session with the voice-prompt path,
push the text with is_final, then stream the session's audio back as
pcm_s16le chunks. Every turn re-POSTs start with the same session_id —
the server caches the prompt tokens, which is what makes turn 2 cheap.

Failure policy (§8): TTS down means screen-only replies, so errors end
the generator quietly; three consecutive failures flip faulted().
"""

import httpx


class MossTts:
    def __init__(self, url: str, prompt_wav: str,
                 http: httpx.AsyncClient | None = None,
                 session_id: str = "phd-backend", timeout_s: float = 180.0,
                 fault_threshold: int = 3):
        self._http = http or httpx.AsyncClient(base_url=url,
                                               timeout=timeout_s)
        self._owns_http = http is None
        self._prompt_wav = prompt_wav
        self._session_id = session_id
        self._failures = 0
        self._fault_threshold = fault_threshold
        # The rate the sidecar actually produces (its TARGET_SR); the
        # browser player schedules at this, so a relaunched sidecar at
        # another rate never plays chipmunks. Updated when a stream opens.
        self.sample_rate = 24000

    async def synthesize(self, text: str):
        if not text.strip():
            return                      # silence costs no network
        try:
            r = await self._http.post("/tts/session/start", json={
                "session_id": self._session_id,
                "prompt_audio": self._prompt_wav})
            r.raise_for_status()
            r = await self._http.post("/tts/session/push", json={
                "session_id": self._session_id, "text": text,
                "is_final": True})
            r.raise_for_status()
            async with self._http.stream(
                    "GET", f"/tts/session/{self._session_id}/audio") as resp:
                resp.raise_for_status()
                self.sample_rate = int(resp.headers.get(
                    "X-Audio-Sample-Rate", 24000))
                async for chunk in resp.aiter_bytes():
                    yield chunk
            self._failures = 0
        except httpx.HTTPError:
            self._failures += 1         # generator ends; reply stays on-screen

    def faulted(self) -> bool:
        return self._failures >= self._fault_threshold

    async def healthy(self) -> bool:
        try:
            return (await self._http.get("/health")).status_code == 200
        except httpx.HTTPError:
            return False

    async def aclose(self) -> None:
        try:                            # best effort — the sidecar may be gone
            await self._http.post("/tts/session/close",
                                  json={"session_id": self._session_id},
                                  timeout=5.0)  # never let a hung sidecar
        except httpx.HTTPError:         # hold lifespan shutdown for 180 s
            pass
        if self._owns_http:
            await self._http.aclose()
