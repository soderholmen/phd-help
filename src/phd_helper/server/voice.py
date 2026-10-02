"""Voice pipeline seams (SPEC §3).

The models never live in this process: Smart App Control blocks torch in
this venv, and NeMo wants py3.12 besides (docs/audio-stack.md). The real
implementations are therefore sidecar HTTP servers in separate venvs —
parakeet in server/stt.py's SidecarStt, MOSS-TTS in server/tts.py's
MossTts — and these Protocols are the client-side contract they satisfy.
The stubs keep the same shape so tests and the PHD_AUDIO_STACK=off path
need no sidecars at all.

Deferred, seam intact: live partials (NeMo 3.0 has no stateful per-chunk
streaming API). Browser audio-out shipped: run_turn streams the chunks
plus an audio_start{sample_rate}/audio_end bookend to the endpoint
holder's socket, and the shell's PcmPlayer plays and barges in.
"""

import json
from typing import AsyncIterator, Protocol


class SttProvider(Protocol):
    async def feed(self, pcm16_bytes: bytes) -> list[str]:
        """Consume 16 kHz mono PCM16 chunks; return authoritative finals.

        Async because a real provider may await a sidecar request at
        utterance-end; the event loop must stay free for everything else.
        """

    def faulted(self) -> bool: ...


class TtsProvider(Protocol):
    sample_rate: int
    """PCM16 rate the chunks are at; audio_start carries it to the
    player. A provider may update it when a stream opens."""

    async def synthesize(self, text: str) -> AsyncIterator[bytes]:
        """Yield audio chunks for one sentence (180 ms TTFB target)."""

    def faulted(self) -> bool: ...


class StubStt:
    """Captures and meters audio but produces no finals — transcripts
    arrive typed while PHD_AUDIO_STACK=off. Honest, not fake."""

    def __init__(self):
        self.bytes_received = 0

    async def feed(self, pcm16_bytes: bytes) -> list[str]:
        self.bytes_received += len(pcm16_bytes)
        return []

    def faulted(self) -> bool:
        return False


class StubTts:
    sample_rate = 24000

    def __init__(self):
        self.requests = 0

    async def synthesize(self, text: str):
        self.requests += 1
        return
        yield b""  # pragma: no cover — makes this an async generator

    def faulted(self) -> bool:
        return False


CONTROL_TYPES = {"heartbeat", "arm", "disarm", "typed", "barge_in",
                 "approve", "reject", "select_section"}


def parse_control(text: str) -> dict:
    """Parse a client control frame. Raises ValueError on anything the
    protocol doesn't define — the session replies with an error event."""
    try:
        msg = json.loads(text)
    except json.JSONDecodeError as e:
        raise ValueError(f"control frame is not JSON: {e}")
    if not isinstance(msg, dict) or msg.get("type") not in CONTROL_TYPES:
        raise ValueError(f"unknown control type: {msg.get('type')!r}"
                         if isinstance(msg, dict) else "control frame "
                         "must be an object")
    return msg
