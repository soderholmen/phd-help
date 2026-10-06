"""TTS providers for the audio episode (SPEC §3; docs/audio-stack.md).

MossTts — the MOSS-TTS-Realtime sidecar client. The session protocol is
incremental (fast_api.py): start the turn with
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

KokoroTts — client for kokoro-fastapi, the OpenAI-compatible
/v1/audio/speech endpoint (the shared instance on :8880, which already
serves voicemode). One bounded request per completed sentence, whole
audio per response; chunks() drains the sentences in order, so the
episode's PCM stays one continuous buffer even though the requests
race. It became the default voice after the MOSS worker was caught
dropping pushes ("push_text ignored: no active turn") on backend-driven
sessions while a direct probe of the same sidecar synthesized fine —
the session interleaving is MOSS-side, and the swap sidesteps it.
PHD_TTS=moss selects the incremental path again.

Failure policy (§8) is shared: TTS down means screen-only replies, so a
failed stream ends the chunk iterator quietly and three consecutive
failed streams flip faulted(); a clean stream resets the counter.
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


class KokoroTtsStream:
    """One turn's synthesis over the OpenAI speech endpoint: every
    completed sentence becomes one bounded request, fired the moment it
    is pushed (the overlap with generation is the point). chunks()
    awaits the requests in sentence order, so out-of-order completions
    never scramble the PCM. It ends on every path — clean, failed,
    aborted — so the caller's episode never hangs open."""

    def __init__(self, tts: "KokoroTts"):
        self._tts = tts
        self._tasks: list[asyncio.Task] = []
        self._more = asyncio.Event()      # a push or finish happened
        self._ended = False               # no more text is coming
        self._aborted = False
        self._counted = False

    async def push(self, text: str) -> None:
        if self._ended or self._aborted or not text.strip():
            return                      # silence costs no network
        self._tasks.append(asyncio.create_task(self._tts._synthesize(text)))
        self._more.set()

    async def finish(self) -> None:
        if not self._ended:
            self._ended = True
            self._more.set()            # let chunks() drain and close

    async def abort(self) -> None:
        if self._aborted:
            return
        self._aborted = True
        self._ended = True
        for task in self._tasks:
            task.cancel()               # stop paying for unplayed audio
        self._more.set()

    async def chunks(self):
        i = 0
        while True:
            if i >= len(self._tasks):
                if self._ended:
                    if not self._aborted:
                        self._tts._failures = 0  # a clean stream self-heals
                    return
                self._more.clear()      # no await between check and wait:
                await self._more.wait()  # a push can never be missed
                continue
            task = self._tasks[i]
            i += 1
            try:
                data = await task
            except asyncio.CancelledError:
                if self._aborted:
                    return              # our own cancel, not the host's
                raise
            except httpx.HTTPError:
                if not self._counted:   # one failure per stream, as Moss
                    self._counted = True
                    self._tts._failures += 1
                for t in self._tasks[i:]:
                    t.cancel()
                return                  # §8: screen-only; chunks END
            if data:
                yield data


class KokoroTts:
    """The kokoro-fastapi voice (module docstring). No session to keep
    or close: each sentence is its own request, so there is no worker to
    strand and abort() is purely local."""

    def __init__(self, url: str, voice: str = "af_sky",
                 model: str = "tts-1",
                 http: httpx.AsyncClient | None = None,
                 timeout_s: float = 60.0, fault_threshold: int = 3):
        self._http = http or httpx.AsyncClient(base_url=url,
                                               timeout=timeout_s)
        self._owns_http = http is None
        self._voice = voice
        self._model = model
        self._failures = 0
        self._fault_threshold = fault_threshold
        # Kokoro-82M records at 24 kHz and the raw PCM response carries
        # no header, so the adapter declares the rate the browser player
        # schedules at (the MossTts shape, but static here).
        self.sample_rate = 24000

    def stream(self) -> KokoroTtsStream:
        return KokoroTtsStream(self)

    async def _synthesize(self, text: str) -> bytes:
        r = await self._http.post("/v1/audio/speech", json={
            "model": self._model, "input": text, "voice": self._voice,
            "response_format": "pcm"})
        r.raise_for_status()
        return r.content

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
