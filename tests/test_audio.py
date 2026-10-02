"""The audio-stack seams: the pure endpointing kernel (segmenting.py) and
the sidecar adapters (server/stt.py, server/tts.py). The kernel is tested
with scripted VAD verdicts — no clock, no audio model, no GPU."""

import ast
import json
import struct
from pathlib import Path

import httpx
import pytest

from phd_helper.segmenting import FRAME_BYTES, EnergyVad, UtteranceGate
from phd_helper.server.stt import SidecarStt
from phd_helper.server.tts import MossTts

FRAME_SAMPLES = FRAME_BYTES // 2  # 16 kHz mono PCM16, 30 ms


def frame(amplitude: int) -> bytes:
    return struct.pack("<h", amplitude) * FRAME_SAMPLES


SILENCE = frame(0)
LOUD = frame(16000)   # RMS ≈ 0.49 — unmistakable speech
HUM = frame(655)      # RMS ≈ 0.02 — above the floor, below speech


class FakeVad:
    """Scripted per-frame verdicts; repeats the last entry when drained."""

    def __init__(self, script):
        self.script = list(script)
        self.seen = 0

    def speech(self, pcm16_bytes):
        i = min(self.seen, len(self.script) - 1)
        self.seen += 1
        return self.script[i]


@pytest.fixture
def anyio_backend():
    return "asyncio"


# --- EnergyVad -----------------------------------------------------------

def test_energy_vad_fires_on_loud_and_not_on_silence():
    vad = EnergyVad()
    assert vad.speech(LOUD)
    assert not vad.speech(SILENCE)


def test_energy_vad_floor_absorbs_a_sustained_hum():
    # A hum above the fixed threshold must stop reading as speech once the
    # adaptive floor has risen to cover it (a fan, a server room).
    vad = EnergyVad()
    assert vad.speech(HUM)          # first contact: still "speech"
    for _ in range(300):
        verdict = vad.speech(HUM)
    assert not verdict
    assert vad.speech(LOUD)         # real speech still clears the gate


# --- UtteranceGate -------------------------------------------------------

def test_gate_readies_exactly_after_the_hangover():
    gate = UtteranceGate(FakeVad([True] * 10 + [False] * 50),
                         hangover_s=0.6, min_utterance_s=0.1)
    for _ in range(10):
        gate.push(LOUD)
    for _ in range(19):             # 0.57 s of silence: still hanging
        gate.push(SILENCE)
    assert not gate.ready()
    gate.push(SILENCE)              # the 20th hangover frame (0.6 s)
    assert gate.ready()


def test_pop_returns_utterance_plus_tail_and_resets():
    gate = UtteranceGate(FakeVad([True] * 10 + [False] * 50),
                         hangover_s=0.6, min_utterance_s=0.1)
    for _ in range(10):
        gate.push(LOUD)
    for _ in range(20):
        gate.push(SILENCE)
    utterance = gate.pop()
    assert utterance == LOUD * 10 + SILENCE * 20
    assert not gate.ready()
    assert gate.pop() == b""        # drained until the next utterance


def test_blip_under_the_minimum_never_readies():
    gate = UtteranceGate(FakeVad([True] * 2 + [False] * 50),
                         hangover_s=0.6, min_utterance_s=0.25)
    for _ in range(2):
        gate.push(LOUD)
    for _ in range(30):
        gate.push(SILENCE)
    assert not gate.ready()
    assert gate.pop() == b""        # the buffer was discarded, not leaked


def test_long_utterance_caps_at_max():
    gate = UtteranceGate(FakeVad([True] * 1400 + [False] * 50),
                         hangover_s=0.6, min_utterance_s=0.25,
                         max_utterance_s=30.0)
    for _ in range(1400):           # 42 s of "speech"
        gate.push(LOUD)
    for _ in range(20):
        gate.push(SILENCE)
    assert gate.ready()
    assert len(gate.pop()) == int(30.0 / 0.03) * FRAME_BYTES


def test_gate_buffers_partial_frames_across_pushes():
    # The mic sends 100 ms chunks; a frame boundary can fall mid-chunk.
    gate = UtteranceGate(FakeVad([True] * 10 + [False] * 50),
                         hangover_s=0.6, min_utterance_s=0.1)
    blob = LOUD * 10 + SILENCE * 20
    half = len(blob) // 2 + 1       # deliberately mid-frame
    gate.push(blob[:half])
    gate.push(blob[half:])
    assert gate.ready()
    assert gate.pop() == blob


# --- SidecarStt ----------------------------------------------------------
#
# The gate above is proven; here it just needs to fire, so the scripted VAD
# runs one utterance per 30 frames and the test feeds exactly that blob.

UTTERANCE = LOUD * 10 + SILENCE * 20


def four_utterances():
    return FakeVad(([True] * 10 + [False] * 20) * 4)


def make_stt(handler, **kw):
    client = httpx.AsyncClient(transport=httpx.MockTransport(handler),
                               base_url="http://stt.test")
    kw.setdefault("vad", FakeVad([True] * 10 + [False] * 20))
    return SidecarStt("http://stt.test", http=client,
                      hangover_s=0.6, min_utterance_s=0.1, **kw)


@pytest.mark.anyio
async def test_sidecar_stt_silence_never_calls_the_sidecar():
    calls = []
    stt = make_stt(lambda r: calls.append(r) or httpx.Response(
        200, json={"text": "x"}), vad=FakeVad([False] * 200))
    assert await stt.feed(SILENCE * 100) == []
    assert calls == []


@pytest.mark.anyio
async def test_sidecar_stt_posts_the_buffered_utterance_once():
    bodies = []

    def handler(request):
        bodies.append(request.content)
        return httpx.Response(200, json={"text": "  rear and curse  "})

    stt = make_stt(handler)
    assert await stt.feed(UTTERANCE) == ["rear and curse"]
    assert bodies == [UTTERANCE]     # utterance + hangover tail, raw PCM16


@pytest.mark.anyio
async def test_sidecar_stt_empty_transcript_is_no_final():
    stt = make_stt(lambda r: httpx.Response(200, json={"text": "   "}))
    assert await stt.feed(UTTERANCE) == []


@pytest.mark.anyio
async def test_sidecar_stt_swallows_connection_errors():
    def handler(request):
        raise httpx.ConnectError("sidecar down", request=request)

    stt = make_stt(handler)
    assert await stt.feed(UTTERANCE) == []   # §8: STT down = typed chat


@pytest.mark.anyio
async def test_sidecar_stt_faults_after_three_and_self_heals():
    state = {"n": 0}

    def handler(request):
        state["n"] += 1
        if state["n"] <= 3:
            return httpx.Response(500)
        return httpx.Response(200, json={"text": "back"})

    stt = make_stt(handler, vad=four_utterances())
    for _ in range(3):
        assert await stt.feed(UTTERANCE) == []
    assert stt.faulted()
    assert await stt.feed(UTTERANCE) == ["back"]
    assert not stt.faulted()          # one success resets the counter


@pytest.mark.anyio
async def test_sidecar_stt_healthy_probes_the_sidecar():
    ok = make_stt(lambda r: httpx.Response(200))
    assert await ok.healthy()
    down = make_stt(lambda r: (_ for _ in ()).throw(
        httpx.ConnectError("nope", request=r)))
    assert not await down.healthy()


# --- MossTts -------------------------------------------------------------

def make_tts(handler, **kw):
    client = httpx.AsyncClient(transport=httpx.MockTransport(handler),
                               base_url="http://tts.test")
    return MossTts("http://tts.test", "ref.mp3", http=client,
                   session_id="phd", **kw)


@pytest.mark.anyio
async def test_moss_tts_runs_the_probe_sequence_and_streams_audio():
    seen = []

    def handler(request):
        seen.append((request.method, request.url.path,
                     json.loads(request.content) if request.content else None))
        if request.url.path == "/tts/session/phd/audio":
            return httpx.Response(200, content=b"AB" * 100)
        return httpx.Response(200, json={"status": "ok"})

    tts = make_tts(handler)
    chunks = [c async for c in tts.synthesize("hello")]
    assert b"".join(chunks) == b"AB" * 100
    assert [p for _, p, _ in seen] == ["/tts/session/start",
                                       "/tts/session/push",
                                       "/tts/session/phd/audio"]
    assert seen[0][2] == {"session_id": "phd", "prompt_audio": "ref.mp3"}
    assert seen[1][2] == {"session_id": "phd", "text": "hello",
                          "is_final": True}


@pytest.mark.anyio
async def test_moss_tts_reuses_the_session_across_turns():
    starts = []

    def handler(request):
        if request.url.path == "/tts/session/start":
            starts.append(json.loads(request.content)["session_id"])
        if request.url.path == "/tts/session/phd/audio":
            return httpx.Response(200, content=b"zz")
        return httpx.Response(200, json={})

    tts = make_tts(handler)
    assert [c async for c in tts.synthesize("one")] == [b"zz"]
    assert [c async for c in tts.synthesize("two")] == [b"zz"]
    assert starts == ["phd", "phd"]   # same sid: prompt tokens stay cached


@pytest.mark.anyio
async def test_moss_tts_error_ends_the_stream_clean_and_faults():
    state = {"n": 0}

    def handler(request):
        state["n"] += 1
        if state["n"] <= 3:
            raise httpx.ConnectError("warmup failed", request=request)
        return httpx.Response(200, content=b"ok") if (
            request.url.path == "/tts/session/phd/audio"
        ) else httpx.Response(200, json={})

    tts = make_tts(handler)
    assert [c async for c in tts.synthesize("a")] == []
    assert [c async for c in tts.synthesize("b")] == []
    assert [c async for c in tts.synthesize("c")] == []
    assert tts.faulted()
    assert [c async for c in tts.synthesize("d")] == [b"ok"]
    assert not tts.faulted()


@pytest.mark.anyio
async def test_moss_tts_empty_text_skips_the_network():
    calls = []
    tts = make_tts(lambda r: calls.append(r) or httpx.Response(200))
    assert [c async for c in tts.synthesize("   ")] == []
    assert calls == []


@pytest.mark.anyio
async def test_moss_tts_close_is_best_effort():
    closed = []

    def handler(request):
        closed.append(request.url.path)
        raise httpx.ConnectError("already gone", request=request)

    tts = make_tts(handler)
    await tts.aclose()                # must not raise into lifespan shutdown
    assert closed == ["/tts/session/close"]


@pytest.mark.anyio
async def test_moss_tts_surfaces_the_sample_rate_header():
    # Voice-out schedules the browser's AudioBufferSourceNodes at the
    # rate the sidecar actually produced (fast_api's X-Audio-Sample-Rate,
    # TARGET_SR-configurable) — guessing 24 kHz would play chipmunks if
    # the sidecar is ever relaunched at another rate.
    def handler(request):
        if request.url.path == "/tts/session/phd/audio":
            return httpx.Response(200, content=b"zz",
                                  headers={"X-Audio-Sample-Rate": "16000"})
        return httpx.Response(200, json={})

    tts = make_tts(handler)
    assert tts.sample_rate == 24000          # before any stream: the default
    [c async for c in tts.synthesize("hi")]
    assert tts.sample_rate == 16000          # the sidecar's own truth


@pytest.mark.anyio
async def test_moss_tts_sample_rate_defaults_when_the_header_is_absent():
    def handler(request):
        if request.url.path == "/tts/session/phd/audio":
            return httpx.Response(200, content=b"zz")
        return httpx.Response(200, json={})

    tts = make_tts(handler)
    [c async for c in tts.synthesize("hi")]
    assert tts.sample_rate == 24000


# --- sidecar topology pin ---------------------------------------------------

def test_sidecar_scripts_never_import_the_backend():
    # The sidecar venvs have no phd_helper installed, and the backend
    # venv must never gain a path to loading torch in-process (SAC). The
    # only honest guarantee is a pin — an AST one, so prose may name the
    # sin.
    for script in ("asr_server.py", "tts_server.py", "corpus_server.py"):
        src = (Path(__file__).resolve().parents[1] / "scripts"
               / script).read_text(encoding="utf-8")
        imported = set()
        for node in ast.walk(ast.parse(src)):
            if isinstance(node, ast.Import):
                imported.update(a.name for a in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported.add(node.module)
        assert not [m for m in imported if m.split(".")[0] == "phd_helper"], \
            f"scripts/{script} must stay standalone (sidecar venv)"
