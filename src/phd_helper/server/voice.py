"""Voice pipeline seams (SPEC §3).

The real STT (silero-vad + nemotron-streaming + parakeet) and TTS
(MOSS-TTS-Realtime) land on the 3090 inside this process later; the
WebSocket protocol and turn-taking are built against these Protocols now,
so swapping stubs for models touches nothing else.
"""

import json
from typing import AsyncIterator, Protocol


class SttProvider(Protocol):
    def feed(self, pcm16_bytes: bytes) -> list[str]:
        """Consume 16 kHz mono PCM16 chunks; return authoritative finals."""

    def faulted(self) -> bool: ...


class TtsProvider(Protocol):
    async def synthesize(self, text: str) -> AsyncIterator[bytes]:
        """Yield audio chunks for one sentence (180 ms TTFB target)."""

    def faulted(self) -> bool: ...


class StubStt:
    """Captures and meters audio but produces no finals — transcripts
    arrive typed until the 3090 stack lands. Honest, not fake."""

    def __init__(self):
        self.bytes_received = 0

    def feed(self, pcm16_bytes: bytes) -> list[str]:
        self.bytes_received += len(pcm16_bytes)
        return []

    def faulted(self) -> bool:
        return False


class StubTts:
    def __init__(self):
        self.requests = 0

    async def synthesize(self, text: str):
        self.requests += 1
        return
        yield b""  # pragma: no cover — makes this an async generator

    def faulted(self) -> bool:
        return False


CONTROL_TYPES = {"heartbeat", "arm", "disarm", "typed", "barge_in"}


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
