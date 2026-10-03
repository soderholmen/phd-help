"""MOSS-TTS-Realtime sidecar client (SPEC §3; docs/audio-stack.md).

The session protocol is incremental (fast_api.py): start the turn with
the voice-prompt path, open the audio GET, push text pieces as they
complete (is_final=False), then a final push closes the turn and the
audio stream ends on the sidecar's sentinel. The GET must be issued
AFTER the start POST returns — the generator captures the turn's queue
at generator start, so an early GET would attach to the previous
turn's queue and hang. Every turn re-POSTs start with the same
session_id, which keeps the session's worker alive (the model itself is
cached process-wide, and the prompt is re-encoded every turn regardless
— fast_api.py re-encodes it in _handle_start_turn). There is no cancel
endpoint, so abort() closes the session outright: the worker is
single-threaded, so an abandoned turn would otherwise keep synthesizing
behind the next turn's start on the same queue (and force-finish into a
garbled tail) — deleting the session gives the next turn a fresh worker
and skips the force-finish entirely.

Failure policy (§8): TTS down means screen-only replies, so a failed
stream ends the chunk iterator quietly and three consecutive failed
streams flip faulted(); a clean stream resets the counter.
"""

import asyncio
from uuid import uuid4

import httpx


class MossTtsStream:
    """One turn's incremental synthesis. push() opens the session on the
    first non-empty text; finish() closes the text (audio drains);
    abort() abandons. chunks() yields PCM16 until end-of-stream — it
    ends on every path, including failure, so the caller's episode
    never hangs open."""

    def __init__(self, tts: "MossTts"):
        self._tts = tts
        self._queue: asyncio.Queue = asyncio.Queue()
        self._reader: asyncio.Task | None = None
        self._opened = False
        self._ended = False
        self._counted = False

    async def push(self, text: str) -> None:
        if self._ended or not text.strip():
            return                      # silence costs no network
        if not self._opened:
            self._opened = True
            try:
                await self._tts._start()
            except httpx.HTTPError:
                await self._fail()
                return
            # GET after start (module docstring); the reader owns the
            # response, the socket side never sees it. The GET may lose
            # the race with the first push — safe by construction: the
            # sidecar creates the turn's UNBOUNDED audio_queue at start
            # and the GET generator attaches to that same queue, so
            # audio decoded before the GET lands is already queued.
            self._reader = asyncio.create_task(self._read())
        try:
            await self._tts._push(text, is_final=False)
        except httpx.HTTPError:
            await self._fail()

    async def _read(self) -> None:
        # The stream, not the provider, scores its own outcome: one
        # failure per stream (a failed start/push and the reader dying
        # with it are one episode, not three strikes).
        try:
            await self._tts._read_audio(self._queue)
            self._tts._failures = 0     # a clean stream self-heals
        except httpx.HTTPError:
            self._count()               # chunks end; reply stays on-screen
        finally:
            self._queue.put_nowait(None)

    def _count(self) -> None:
        if not self._counted:
            self._counted = True
            self._tts._failures += 1

    async def finish(self) -> None:
        if self._ended:
            return
        if not self._opened:
            self._end()                 # never touched the network
            return
        try:
            await self._tts._push("", is_final=True)  # sidecar: text
        except httpx.HTTPError:         # optional, is_final closes
            await self._fail()

    async def abort(self) -> None:
        if self._ended:
            return
        if self._reader is not None:
            self._reader.cancel()       # stop reading
        self._end()                     # chunks end before any I/O: a
                                        # cancel landing on the close below
                                        # must not strand the stream
        if self._opened:
            # Stop the sidecar, not just the read: the worker is
            # single-threaded, so an abandoned turn keeps synthesizing
            # (and force-finishes into garbage on the next start) until
            # it is closed out of. Best effort — a dead sidecar must
            # never hold the barge.
            await self._tts._close_session()

    async def chunks(self):
        while True:
            item = await self._queue.get()
            if item is None:
                return
            yield item

    async def _fail(self) -> None:
        self._count()
        if self._reader is not None:
            self._reader.cancel()
        self._end()

    def _end(self) -> None:
        if not self._ended:
            self._ended = True
            self._queue.put_nowait(None)


class MossTts:
    def __init__(self, url: str, prompt_wav: str,
                 http: httpx.AsyncClient | None = None,
                 session_id: str | None = None, timeout_s: float = 180.0,
                 fault_threshold: int = 3):
        self._http = http or httpx.AsyncClient(base_url=url,
                                               timeout=timeout_s)
        self._owns_http = http is None
        self._prompt_wav = prompt_wav
        # One session per instance, not one per process: a fixed id makes
        # every MossTts (a second backend, a restart, a diagnostic probe)
        # share the sidecar's single worker session and audio queue, so
        # their turns interleave into each other's audio. Reused across
        # THIS instance's turns, which keeps its worker (and the
        # process-wide cached model) warm — not the prompt, which the
        # sidecar re-encodes every turn anyway.
        self._session_id = session_id or uuid4().hex
        self._failures = 0
        self._fault_threshold = fault_threshold
        # The rate the sidecar actually produces (its TARGET_SR); the
        # browser player schedules at this, so a relaunched sidecar at
        # another rate never plays chipmunks. Updated when a stream opens.
        self.sample_rate = 24000

    def stream(self) -> MossTtsStream:
        return MossTtsStream(self)

    async def _start(self) -> None:
        r = await self._http.post("/tts/session/start", json={
            "session_id": self._session_id,
            "prompt_audio": self._prompt_wav})
        r.raise_for_status()

    async def _push(self, text: str, is_final: bool) -> None:
        r = await self._http.post("/tts/session/push", json={
            "session_id": self._session_id, "text": text,
            "is_final": is_final})
        r.raise_for_status()

    async def _read_audio(self, queue: asyncio.Queue) -> None:
        # Pump only: the owning stream scores the outcome (one failure
        # per stream) and always closes the queue.
        async with self._http.stream(
                "GET", f"/tts/session/{self._session_id}/audio") as resp:
            resp.raise_for_status()
            self.sample_rate = int(resp.headers.get(
                "X-Audio-Sample-Rate", 24000))
            async for chunk in resp.aiter_bytes():
                await queue.put(chunk)

    def faulted(self) -> bool:
        return self._failures >= self._fault_threshold

    async def healthy(self) -> bool:
        try:
            return (await self._http.get("/health")).status_code == 200
        except httpx.HTTPError:
            return False

    async def _close_session(self) -> None:
        # Best effort — the sidecar may be gone, and a hung one must
        # never hold a barge (or lifespan shutdown) for the full 180 s.
        try:
            await self._http.post("/tts/session/close",
                                  json={"session_id": self._session_id},
                                  timeout=5.0)
        except httpx.HTTPError:
            pass

    async def aclose(self) -> None:
        await self._close_session()
        if self._owns_http:
            await self._http.aclose()
