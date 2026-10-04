"""Endpointing kernel: decide, from raw PCM16, when an utterance has ended.

Pure and deterministic — no clock, no audio model, no I/O. Time is counted
in frames (30 ms of 16 kHz mono PCM16), so tests drive it with scripted VAD
verdicts and the production path needs nothing more than the mic's bytes.

SPEC §3 names silero-vad for this job. Silero is a torch model, and torch
is blocked in this venv by Smart App Control (docs/audio-stack.md), so the
production VAD here is an energy gate with an adaptive noise floor. The
deviation is contained: UtteranceGate takes any object with
``speech(frame) -> bool``, so a sidecar-hosted silero (a WS verdict stream)
can replace EnergyVad without touching the state machine.
"""

import array
import math

SAMPLE_RATE = 16000
FRAME_S = 0.03
FRAME_SAMPLES = int(SAMPLE_RATE * FRAME_S)          # 480
FRAME_BYTES = FRAME_SAMPLES * 2                     # PCM16 mono: 960


class EnergyVad:
    """RMS gate with an adaptive floor.

    A frame is speech when its RMS clears ``max(threshold,
    floor * floor_factor)``. The floor is an EMA of the signal with
    asymmetric rates — it rises slowly under a sustained hum (a fan, a
    server room) so the hum stops reading as speech, and falls quickly when
    the room goes quiet so a threshold-calibrated start-up is not held
    hostage by a loud room it once measured.
    """

    def __init__(self, threshold: float = 0.012, floor: float = 0.004,
                 floor_factor: float = 4.0):
        self.threshold = threshold
        self._floor = floor
        self.floor_factor = floor_factor

    def speech(self, pcm16_bytes: bytes) -> bool:
        # audioop died with Python 3.13; 480 samples per frame makes the
        # pure-Python RMS a rounding error on the budget.
        samples = array.array("h")
        samples.frombytes(pcm16_bytes)
        if not samples:
            return False
        rms = math.sqrt(sum(x * x for x in samples) / len(samples)) / 32768.0
        gate = max(self.threshold, self._floor * self.floor_factor)
        if rms > gate:
            self._floor += 0.001 * (rms - self._floor)   # slow rise
            return True
        self._floor += 0.05 * (rms - self._floor)        # fast fall
        return False


class UtteranceGate:
    """IDLE → SPEAKING → TAIL → ready, counted in frames.

    ``push`` takes arbitrary-length PCM16 chunks (the mic's 100 ms blobs)
    and runs the VAD over whole 30 ms frames, carrying the partial frame
    across pushes. Once the hangover elapses after the last speech frame,
    ``ready()`` flips; ``pop()`` hands back the utterance *plus* its
    hangover tail (harmless leading/trailing silence for the ASR) and
    resets. A blip shorter than ``min_utterance_s`` is discarded where it
    stands; a runaway utterance is capped at ``max_utterance_s``.
    """

    def __init__(self, vad, hangover_s: float, min_utterance_s: float = 0.25,
                 max_utterance_s: float = 30.0):
        self.vad = vad
        self._hangover = max(1, int(hangover_s / FRAME_S))
        self._min_speech = max(1, int(min_utterance_s / FRAME_S))
        self._max_frames = max(1, int(max_utterance_s / FRAME_S))
        self._partial = b""
        self._buffer: list[bytes] = []
        self._speaking = False
        self._speech_frames = 0
        self._tail = 0
        self._ready = False

    def push(self, pcm16_bytes: bytes) -> None:
        data = self._partial + pcm16_bytes
        view = memoryview(data)
        self._partial = b""
        for off in range(0, len(view) - len(view) % FRAME_BYTES, FRAME_BYTES):
            self._frame(bytes(view[off:off + FRAME_BYTES]))
        self._partial = bytes(view[len(view) - len(view) % FRAME_BYTES:])

    def _frame(self, frame_bytes: bytes) -> None:
        if self._ready:
            return                      # utterance closed; next pop() reopens
        is_speech = self.vad.speech(frame_bytes)
        if not self._speaking:
            if not is_speech:
                return                  # IDLE swallows silence
            self._speaking = True
            self._speech_frames = 0
            self._tail = 0
            self._buffer = []
        if len(self._buffer) < self._max_frames:
            self._buffer.append(frame_bytes)
        if is_speech:
            self._speech_frames += 1
            self._tail = 0
        else:
            self._tail += 1
            if self._tail >= self._hangover:
                if self._speech_frames >= self._min_speech:
                    self._ready = True
                else:
                    self._buffer = []   # a blip; discarded, not leaked
                self._speaking = False

    def pending(self) -> bytes:
        """The utterance in progress — empty when idle. This is what a
        live-partial sidecar gets re-transcribed as the mic keeps
        feeding; `pop()` still owns the authoritative close."""
        return b"".join(self._buffer) if self._speaking else b""

    def ready(self) -> bool:
        return self._ready

    def pop(self) -> bytes:
        if not self._ready:
            return b""
        utterance = b"".join(self._buffer)
        self._ready = False
        self._speaking = False
        self._buffer = []
        return utterance
