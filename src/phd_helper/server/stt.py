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
"""

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
                 fault_threshold: int = 3, vad_mode: str = "energy"):
        self._http = http or httpx.AsyncClient(base_url=url,
                                               timeout=timeout_s)
        self._owns_http = http is None
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
            return []
        utterance = self._gate.pop()
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
