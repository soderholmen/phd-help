"""Sidecar STT client (SPEC §3; docs/audio-stack.md).

The models (parakeet-tdt-0.6b-v3, silero-vad) live in a py3.12 sidecar
process — torch is blocked in this venv by Smart App Control — so this
client owns everything that does *not* need torch: the endpointing kernel
(segmenting) runs here, and the sidecar answers per-frame speech verdicts
(/vad) and turns a finished utterance's raw PCM16 into a final
(/transcribe). One POST per utterance, ~0.46 s measured, awaited so the
event loop keeps serving everyone else.

Endpointing has two legs (docs/audio-stack.md): "silero" asks the
sidecar's VAD per mic blob — the energy gate cut real speech at
micro-pauses (the live listen-test heard "Yeah." where a sentence was
said) — and "energy" (the constructor default; the shipped config
default is silero) runs the pure gate alone. In silero mode EnergyVad
stays the per-frame fallback, so a
/vad fault degrades endpointing quality without ever dropping the
utterance.

Failure policy (§8, honest silence): a dead sidecar must never break the
WebSocket — feed() swallows transport errors and returns no finals, so
typed chat continues untouched. Three consecutive failures on either leg
flip faulted() for /health (the vad leg keeps its own count: a working
/transcribe must not mask a down /vad); one success self-heals that leg
when the sidecar returns.

Live partials (docs/audio-stack.md): with `on_partial` set, every mic
blob while an utterance is in progress kicks a best-effort POST to the
streaming sidecar's /stream (the gate's `pending()` bytes), and the
growing hypothesis rides the callback. Best-effort by construction: one
request in flight at a time (a newer prefix supersedes the wait), a
failed partial is simply no partial, and a generation counter makes
every late answer die at the close — the final transcript is the
record, the ghost text is decoration. The finals path is untouched.
"""

import asyncio
import json
from collections import deque
from uuid import uuid4

import httpx

from ..segmenting import FRAME_BYTES, EnergyVad, UtteranceGate


class RemoteVad:
    """Sidecar verdicts (silero), consumed one per gate frame.

    feed() awaits /vad before pushing, so the deque holds the verdicts
    for exactly the frames the gate is about to process: both sides count
    complete 30 ms frames of the same byte stream, and the byte offset in
    the request keeps them in step across a lost POST. Starved, the
    fallback EnergyVad answers the frame — degradation is per-frame and
    an utterance is never dropped for a missing verdict.
    """

    def __init__(self, fallback=None):
        self._verdicts = deque()
        self._fallback = fallback or EnergyVad()

    def extend(self, verdicts) -> None:
        self._verdicts.extend(verdicts)

    def drain(self) -> None:
        # The gate stops consulting the VAD the frame an utterance closes
        # (segmenting.py early-returns on ready), but the sidecar counted
        # every byte of that blob: the post-close verdicts left in the
        # deque would misanswer every later frame — one stale verdict per
        # close, forever. Dropping them realigns exactly, because the
        # gate's frame grid stays global (post-close frames are consumed,
        # just unanswered) and the next blob starts on a frame boundary.
        self._verdicts.clear()

    def speech(self, pcm16_bytes: bytes) -> bool:
        if self._verdicts:
            return self._verdicts.popleft()
        return self._fallback.speech(pcm16_bytes)


class SidecarStt:
    def __init__(self, url: str, http: httpx.AsyncClient | None = None,
                 vad=None, hangover_s: float = 0.6,
                 min_utterance_s: float = 0.25, timeout_s: float = 15.0,
                 fault_threshold: int = 3, vad_mode: str = "energy",
                 on_partial=None, partials_url: str | None = None,
                 partials_http: httpx.AsyncClient | None = None,
                 partial_timeout_s: float = 5.0):
        self._http = http or httpx.AsyncClient(base_url=url,
                                               timeout=timeout_s)
        self._owns_http = http is None
        # The partials leg is a SECOND client: the streaming sidecar is
        # its own process (port 8092) — a nemotron load must never
        # queue behind parakeet work, and PHD_PARTIALS=0 simply leaves
        # this None (no ghost text, no process, finals untouched).
        self._on_partial = on_partial
        self._partials = partials_http
        self._owns_partials = False
        if on_partial is not None and self._partials is None:
            self._partials = httpx.AsyncClient(base_url=partials_url,
                                               timeout=partial_timeout_s)
            self._owns_partials = True
        self._partial_gen = 0         # bumped at every close: the guard
        self._partial_task = None     # that kills late ghost text
        self._ghost_live = False      # a hypothesis is on screen: a
                                      # clear is owed (see _end_partial)
        self._stream = uuid4().hex      # one VAD stream per session
        self._fed = 0                   # whole-frame bytes sent: the
                                        # sidecar's grid position
        self._vad_partial = b""         # sub-frame tail held for the next POST
        self._remote = None
        if vad_mode == "silero":
            # the vad param is the fallback under this leg
            self._remote = RemoteVad(vad)
            gate_vad = self._remote
        elif vad_mode == "energy":
            gate_vad = vad or EnergyVad()
        else:
            raise ValueError(f"unknown vad_mode {vad_mode!r}")
        self._gate = UtteranceGate(gate_vad, hangover_s=hangover_s,
                                   min_utterance_s=min_utterance_s)
        self._failures = 0
        self._vad_failures = 0
        self._fault_threshold = fault_threshold

    async def feed(self, pcm16_bytes: bytes) -> list[str]:
        if self._remote is not None:
            verdicts = await self._fetch_verdicts(pcm16_bytes)
            if verdicts:
                self._remote.extend(verdicts)
        self._gate.push(pcm16_bytes)
        if not self._gate.ready():
            if self._ghost_live and not self._gate.pending():
                # A blip under min_utterance was discarded in the gate
                # (segmenting.py): it never sets ready, so no pop() close
                # will ever land for it — this feed is the only clear the
                # ghost it earned will ever get.
                await self._end_partial()
            else:
                self._pump_partial()
            return []
        utterance = self._gate.pop()
        await self._end_partial()
        if self._remote is not None:
            self._remote.drain()
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

    def _pump_partial(self) -> None:
        # Best-effort by construction: one request in flight (its answer
        # describes an older prefix than whatever the next blob will
        # carry, so waiting on two buys nothing), and an empty gate
        # means the mic is idle — silence never reaches the sidecar.
        if self._on_partial is None or self._partials is None:
            return
        task = self._partial_task
        if task is not None and not task.done():
            return
        pcm = self._gate.pending()
        if not pcm:
            return
        gen = self._partial_gen
        self._partial_task = asyncio.create_task(self._fetch_partial(pcm,
                                                                      gen))

    async def _fetch_partial(self, pcm: bytes, gen: int) -> None:
        try:
            r = await self._partials.post("/stream", content=pcm)
            r.raise_for_status()
            text = (r.json().get("partial") or "").strip()
        except (httpx.HTTPError, json.JSONDecodeError):
            return                  # a failed partial is no partial
        # The close bumped the generation while this was in flight: the
        # answer describes an utterance that already produced its final
        # — ghost text must not outlive the words it guessed.
        if gen == self._partial_gen and text:
            self._ghost_live = True
            await self._on_partial(text)

    async def _end_partial(self) -> None:
        # Utterance closed (or a blip discarded — it never closes, so
        # feed() routes here too): late answers die (gen), the in-flight
        # one is cancelled, and the ghost clears — with a final coming
        # (the user_text fold clears too, belt) and without one (this is
        # the only clear), the composer never keeps stale ghost text.
        self._ghost_live = False
        self._partial_gen += 1
        task = self._partial_task
        self._partial_task = None
        if task is not None and not task.done():
            task.cancel()
        if self._on_partial is not None and self._partials is not None:
            await self._on_partial("")

    async def _fetch_verdicts(self, blob: bytes) -> list[bool]:
        # /vad only ever sees whole frames. The gate's grid and the
        # sidecar's both count from byte 0, so a whole-frame offset means
        # a resync can never skip a frame the gate will still process —
        # a straddling frame would hand the gate the NEXT frame's verdict
        # and shift the deque by one, forever. Held bytes ride the next
        # POST. The offset advances even when the POST fails: the next
        # request resyncs the sidecar to the gate's real position, so a
        # lost POST costs those frames' verdicts (energy answers them),
        # never a permanent misalignment.
        buf = self._vad_partial + blob
        whole = len(buf) - len(buf) % FRAME_BYTES
        out, self._vad_partial = buf[:whole], buf[whole:]
        off = self._fed
        self._fed += len(out)
        if not out:
            return []
        try:
            r = await self._http.post("/vad", params={"stream": self._stream,
                                                      "off": off},
                                      content=out)
            r.raise_for_status()
            speech = r.json().get("speech") or []
        except (httpx.HTTPError, json.JSONDecodeError):
            self._vad_failures += 1
            return []
        self._vad_failures = 0
        return speech

    def faulted(self) -> bool:
        return (self._failures >= self._fault_threshold
                or self._vad_failures >= self._fault_threshold)

    async def healthy(self) -> bool:
        try:
            return (await self._http.get("/health")).status_code == 200
        except httpx.HTTPError:
            return False

    async def aclose(self) -> None:
        if self._owns_http:
            await self._http.aclose()
        if self._owns_partials:
            await self._partials.aclose()
