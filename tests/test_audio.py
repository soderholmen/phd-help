"""The audio-stack seams: the pure endpointing kernel (segmenting.py) and
the sidecar adapters (server/stt.py, server/tts.py). The kernel is tested
with scripted VAD verdicts — no clock, no audio model, no GPU."""

import ast
import asyncio
import json
import struct
from pathlib import Path

import httpx
import pytest

from phd_helper.segmenting import FRAME_BYTES, EnergyVad, UtteranceGate
from phd_helper.server.stt import RemoteVad, SidecarStt
from phd_helper.server.tts import KokoroTts, MossTts

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


def test_gate_pending_shares_the_utterance_in_progress():
    # The live-partial pump's feed: what the mic has said so far, while
    # it is still saying it. pop() keeps owning the authoritative bytes.
    gate = UtteranceGate(FakeVad([True] * 4 + [False] * 50),
                         hangover_s=0.6, min_utterance_s=0.1)
    assert gate.pending() == b""           # idle: nothing to guess at
    gate.push(LOUD * 4)
    assert gate.pending() == LOUD * 4      # in progress: the partial feed
    gate.push(SILENCE * 20)
    assert gate.ready()
    assert gate.pending() == b""           # closed: pop() has the bytes


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


# --- SidecarStt live partials ---------------------------------------------
#
# The ghost-text leg: mid-utterance blobs kick a best-effort /stream of
# the gate's pending bytes. It is decoration — the finals path above is
# the record — so the tests below are mostly about the leg NOT leaking
# into the record: no turn starts, no late answer outlives the close.

def make_partial_stt(stream_handler, transcribe_handler=None, **kw):
    heard = []

    async def on_partial(text):
        heard.append(text)

    kw.setdefault("vad", FakeVad([True] * 10 + [False] * 20))
    finals = transcribe_handler or (lambda r: httpx.Response(
        200, json={"text": "the final word"}))
    stt = SidecarStt(
        "http://stt.test",
        http=httpx.AsyncClient(transport=httpx.MockTransport(finals),
                               base_url="http://stt.test"),
        hangover_s=0.6, min_utterance_s=0.1,
        on_partial=on_partial,
        partials_http=httpx.AsyncClient(
            transport=httpx.MockTransport(stream_handler),
            base_url="http://partials.test"),
        **kw)
    return stt, heard


@pytest.mark.anyio
async def test_a_partial_provider_emits_partial_events_without_a_turn():
    streams = []

    def stream_handler(request):
        streams.append(request.content)
        return httpx.Response(200, json={"partial": "the quick"})

    def transcribe(request):
        raise AssertionError("a partial must never reach /transcribe")

    stt, heard = make_partial_stt(stream_handler, transcribe)
    assert await stt.feed(LOUD * 10) == []      # mid-utterance: no close
    await stt._partial_task                     # the pump's own request
    assert heard == ["the quick"]               # ghost text, and only it
    assert streams == [LOUD * 10]               # the gate's pending bytes


@pytest.mark.anyio
async def test_only_one_partial_is_in_flight_and_the_next_carries_the_longer_prefix():
    import asyncio
    hold = asyncio.Event()
    streams = []

    async def stream_handler(request):
        streams.append(request.content)
        if len(streams) == 1:
            await hold.wait()                   # hold request #1
        return httpx.Response(200, json={"partial": "x"})

    stt, heard = make_partial_stt(stream_handler)
    await stt.feed(LOUD * 4)
    await asyncio.sleep(0)                      # let request #1 in
    await stt.feed(LOUD * 4)                    # second blob: skipped
    assert len(streams) == 1
    hold.set()
    await stt._partial_task
    # The next blob sees the finished task and re-posts — with the
    # LONGER pending prefix (the skipped bytes were never lost: the
    # gate keeps buffering).
    await stt.feed(LOUD * 2)
    await stt._partial_task
    assert len(streams) == 2 and len(streams[1]) > len(streams[0])


@pytest.mark.anyio
async def test_a_discarded_blip_clears_the_ghost_it_earned():
    def stream_handler(request):
        return httpx.Response(200, json={"partial": "half a word"})

    # 2 speech frames under a 3-frame minimum: the gate discards the
    # blip WITHOUT a close (segmenting.py), so no pop() will ever clear
    # — the feed that notices the discard is the ghost's only clear.
    stt, heard = make_partial_stt(stream_handler,
                                 vad=FakeVad([True] * 2 + [False] * 30))
    assert await stt.feed(LOUD * 2) == []         # blip forming, ghost earned
    await stt._partial_task
    assert heard == ["half a word"]
    assert await stt.feed(SILENCE * 30) == []     # hangover: blip discarded
    assert heard == ["half a word", ""]           # ... and the ghost cleared


@pytest.mark.anyio
async def test_a_late_partial_dies_at_the_close_and_the_ghost_clears():
    import asyncio
    hold = asyncio.Event()

    async def stream_handler(request):
        await hold.wait()
        return httpx.Response(200, json={"partial": "stale ghost"})

    stt, heard = make_partial_stt(stream_handler)
    await stt.feed(LOUD * 10)
    await asyncio.sleep(0)                        # partial now in flight
    in_flight = stt._partial_task
    finals = await stt.feed(SILENCE * 20)         # the close lands
    assert finals == ["the final word"]
    assert heard == [""]                          # ghost cleared at close
    hold.set()
    await asyncio.gather(in_flight, return_exceptions=True)
    assert heard == [""]                          # the late answer died


@pytest.mark.anyio
async def test_a_failed_partial_is_silence_and_the_final_is_untouched():
    def stream_handler(request):
        return httpx.Response(500)

    stt, heard = make_partial_stt(stream_handler)
    assert await stt.feed(LOUD * 10) == []
    await stt._partial_task
    assert heard == []                            # no ghost, no complaint
    assert await stt.feed(SILENCE * 20) == ["the final word"]
    assert heard == [""]                          # close still clears
    assert not stt.faulted()                      # the finals leg is fine


@pytest.mark.anyio
async def test_close_without_a_final_still_clears_the_ghost():
    def stream_handler(request):
        return httpx.Response(200, json={"partial": "half a word"})

    stt, heard = make_partial_stt(
        stream_handler, lambda r: httpx.Response(200, json={"text": "  "}))
    await stt.feed(LOUD * 10)
    await stt._partial_task
    assert heard == ["half a word"]
    assert await stt.feed(SILENCE * 20) == []     # closed, nothing final
    assert heard == ["half a word", ""]           # the ghost did not stick


@pytest.mark.anyio
async def test_the_stub_never_invents_partials():
    from phd_helper.server.voice import StubStt
    heard = []
    stub = StubStt(on_partial=lambda t: heard.append(t))
    assert await stub.feed(LOUD * 10) == []       # same contract, silence
    assert heard == []


# --- RemoteVad + the silero leg ------------------------------------------
#
# The energy gate cuts real speech at micro-pauses (the live listen-test
# heard "Yeah." where a sentence was said — docs/audio-stack.md). The
# silero leg asks the sidecar per mic blob and consumes its verdicts one
# per gate frame; EnergyVad stays the per-frame fallback, so a /vad fault
# degrades endpointing quality without ever dropping the utterance.

def test_remote_vad_pops_scripted_verdicts_then_falls_back_to_energy():
    rv = RemoteVad(EnergyVad())
    rv.extend([True, False])
    assert rv.speech(SILENCE) is True       # the verdict wins over the bytes
    assert rv.speech(SILENCE) is False
    assert rv.speech(LOUD)                  # starved: energy answers
    assert not rv.speech(SILENCE)


def make_silero_stt(vad_handler, transcribe_handler, **kw):
    def handler(request):
        return (vad_handler(request) if request.url.path == "/vad"
                else transcribe_handler(request))
    client = httpx.AsyncClient(transport=httpx.MockTransport(handler),
                               base_url="http://stt.test")
    return SidecarStt("http://stt.test", http=client, vad_mode="silero",
                      hangover_s=0.6, min_utterance_s=0.1, **kw)


@pytest.mark.anyio
async def test_silero_leg_verdicts_drive_the_gate_to_a_final():
    seen = []

    def vad(request):
        seen.append(request)
        return httpx.Response(200, json={"speech": [True] * 10 + [False] * 20})

    stt = make_silero_stt(
        vad, lambda r: httpx.Response(200, json={"text": "whole sentence"}))
    assert await stt.feed(UTTERANCE) == ["whole sentence"]
    assert seen[0].url.params["stream"]          # per-session stream identity
    assert seen[0].url.params["off"] == "0"


@pytest.mark.anyio
async def test_silero_leg_carries_stream_and_offset_across_blobs():
    # The offset is the sidecar's frame-grid anchor: it lets a stream
    # resync after a lost POST instead of misaligning forever.
    offs = []

    def vad(request):
        off = int(request.url.params["off"])
        offs.append((request.url.params["stream"], off))
        speech = [True] * 5 if off else [True] * 5 + [False] * 20
        return httpx.Response(200, json={"speech": speech})

    stt = make_silero_stt(
        vad, lambda r: httpx.Response(200, json={"text": "x"}))
    assert await stt.feed(LOUD * 5) == []
    assert await stt.feed(LOUD * 5 + SILENCE * 20) == ["x"]
    assert offs[0][0] == offs[1][0]              # one stream per session
    assert [o for _, o in offs] == [0, 5 * FRAME_BYTES]


@pytest.mark.anyio
async def test_silero_leg_asks_the_sidecar_even_for_silence():
    calls = []

    def vad(request):
        calls.append(request)
        return httpx.Response(200, json={"speech": [False] * 10})

    stt = make_silero_stt(
        vad, lambda r: httpx.Response(200, json={"text": "x"}))
    assert await stt.feed(SILENCE * 10) == []
    assert len(calls) == 1        # the bytes go where the verdicts come from


@pytest.mark.anyio
async def test_silero_leg_vad_fault_falls_back_to_energy_and_still_delivers():
    def vad(request):
        raise httpx.ConnectError("vad leg down", request=request)

    stt = make_silero_stt(
        vad, lambda r: httpx.Response(200, json={"text": "x"}))
    assert await stt.feed(UTTERANCE) == ["x"]   # energy heard the LOUD frames
    assert not stt.faulted()                     # one failure is not a fault


@pytest.mark.anyio
async def test_silero_leg_vad_failures_reach_health_while_transcribe_works():
    # Degraded endpointing is visible: /transcribe being fine does not
    # mask /vad being down — the vad leg keeps its own failure count.
    def vad(request):
        return httpx.Response(500)

    stt = make_silero_stt(
        vad, lambda r: httpx.Response(200, json={"text": "x"}))
    for _ in range(3):
        assert await stt.feed(UTTERANCE) == ["x"]   # degraded, still working
    assert stt.faulted()                             # /health says so honestly


@pytest.mark.anyio
async def test_silero_leg_realigns_verdicts_after_an_utterance_close():
    # The gate stops consulting the VAD the frame it closes, so blob 1
    # leaves 5 unconsumed verdicts (frames after the close). Drained at
    # pop, blob 2's speech is judged speech from its first frame; left,
    # every later frame is misanswered by the stale verdicts forever.
    def vad(request):
        off = int(request.url.params["off"])
        if off == 0:
            speech = [True] * 5 + [False] * 25      # closes at frame 24
        elif off == 30 * FRAME_BYTES:
            speech = [True] * 10                    # blob 2: all speech
        else:
            speech = [False] * 20                   # blob 3: the hangover
        return httpx.Response(200, json={"speech": speech})

    heard = []

    def transcribe(request):
        heard.append(len(request.content))
        return httpx.Response(200, json={"text": "x"})

    stt = make_silero_stt(vad, transcribe)
    assert await stt.feed(LOUD * 30) == ["x"]
    assert await stt.feed(LOUD * 10) == []
    assert await stt.feed(SILENCE * 20) == ["x"]
    # utterance 1: 5 speech + 20 tail; utterance 2: blob 2's 10 speech
    # (none swallowed) + blob 3's 20 tail
    assert heard == [25 * FRAME_BYTES, 30 * FRAME_BYTES]


@pytest.mark.anyio
async def test_silero_leg_posts_only_whole_frames():
    # A whole-frame offset is what makes a resync lossless: the sidecar
    # can never skip a frame the gate will still process. The sub-frame
    # tail is held and rides the next POST.
    seen = []

    def vad(request):
        seen.append((request.url.params["off"], len(request.content)))
        n = len(request.content) // FRAME_BYTES
        return httpx.Response(200, json={"speech": [False] * n})

    stt = make_silero_stt(vad, lambda r: httpx.Response(200,
                                                       json={"text": "x"}))
    await stt.feed(SILENCE * 3 + b"\x00\x01" * 100)     # 3 frames + 200 B
    await stt.feed(SILENCE * 2)
    assert seen == [("0", 3 * FRAME_BYTES),
                    (str(3 * FRAME_BYTES), 2 * FRAME_BYTES)]


def test_sidecar_stt_rejects_an_unknown_vad_mode():
    # The config guard catches typos at startup; this is the adapter's
    # own second line — a mode that is neither leg must fail loudly,
    # never silently run the gate the user just complained about.
    with pytest.raises(ValueError, match="vad_mode"):
        SidecarStt("http://stt.test", vad_mode="silro")


# --- sidecar /vad mapping --------------------------------------------------
#
# The mapping (global frame grid, offset resync, buffer trim, chunk/frame
# overlap) is pure Python by design — torch lives behind load_vad — so the
# trickiest logic in the slice is unit-tested with a fake model. The real
# silero is proven by scripts/smoke_audio.py.

def load_asr_script():
    import importlib.util
    path = Path(__file__).resolve().parents[1] / "scripts" / "asr_server.py"
    spec = importlib.util.spec_from_file_location("asr_server", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mod._vad_streams.clear()
    return mod


def pcm(amplitude: int, samples: int) -> bytes:
    return struct.pack(f"<{samples}h", *((amplitude,) * samples))


def test_vad_mapping_emits_one_verdict_per_global_frame():
    mod = load_asr_script()
    mod.load_vad = lambda: (lambda chunk: 0.9 if chunk[0] else 0.0)
    blob = pcm(10000, 1600)                       # a 100 ms mic blob
    assert mod._vad_verdicts("s", 0, blob) == [True] * 3    # 1600 // 480
    assert mod._vad_verdicts("s", len(blob), blob) == [True] * 3
    # the partial frame carried across the POST — no drift, no duplicates


def test_vad_mapping_resyncs_on_offset_mismatch():
    mod = load_asr_script()
    mod.load_vad = lambda: (lambda chunk: 0.9 if chunk[0] else 0.0)
    blob = pcm(10000, 1600)
    mod._vad_verdicts("s", 0, blob)
    # a lost POST: the next off skips ahead. Fresh state, no crash, and
    # verdicts resume on the global grid — the skipped frames are simply
    # absent (the backend's energy fallback answers them).
    v = mod._vad_verdicts("s", 2 * len(blob), blob)
    assert v == [True] * 3       # resync at sample 3200: frames 7, 8, 9


def test_vad_mapping_frame_is_speech_if_any_overlapping_chunk_is():
    mod = load_asr_script()
    mod.load_vad = lambda: (lambda chunk: 0.9 if any(chunk) else 0.0)
    samples = [0] * 1440
    for i in range(400, 512):
        samples[i] = 9000        # loud inside silero chunk 0 only
    body = struct.pack("<1440h", *samples)
    # conservative overlap: chunk 0 touches frames 0 AND 1 — both speech;
    # frame 2 sees only the silent chunk 1
    assert mod._vad_verdicts("s", 0, body) == [True, True, False]


def test_vad_mapping_trim_is_invisible_to_the_stream():
    mod = load_asr_script()
    mod.load_vad = lambda: (lambda chunk: 0.9 if any(chunk) else 0.0)
    data = (pcm(9000, 480) + pcm(0, 480)) * 8
    streamed = []
    blob = 3200
    for off in range(0, len(data), blob):
        streamed += mod._vad_verdicts("t", off, data[off:off + blob])
    single = mod._vad_verdicts("u", 0, data)      # fresh stream, one POST
    assert streamed == single
    assert len(single) == 16                      # 7680 samples, 16 frames


# --- partials sidecar mapping ----------------------------------------------
#
# Same rule as the /vad mapping: torch lives behind load_model, so the
# chunking (the model's own validator's exact sizes — 49 then 56 mel
# frames, tail zero-padded) is unit-tested against a fake adapter. The
# real nemotron is proven by the probe (.probe/probe_rnnt_stream.py).

def load_partials_script():
    import importlib.util
    path = (Path(__file__).resolve().parents[1] / "scripts"
            / "partials_server.py")
    spec = importlib.util.spec_from_file_location("partials_server", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class FakeAdapter:
    first, step = 49, 56                    # the probe's numbers

    def __init__(self, n_frames, text="the quick brown"):
        self.n = n_frames
        self.text = text
        self.slices = []                    # (frames, pad_to) per chunk
        self.generate_calls = 0

    def feats(self, body):
        return type("F", (), {"shape": (1, self.n, 128)})()

    def slice(self, feats, a, b, pad_to=0):
        self.slices.append((b - a, pad_to))
        return (a, b)

    def generate(self, chunks):
        self.generate_calls += 1
        return self.text


def test_stream_mapping_slices_the_probe_chunk_sizes_and_pads_the_tail():
    mod = load_partials_script()
    # The probe's 11.8 s file: 1178 mel frames = 49 + 20x56 + 9, so 22
    # chunks and the 9-frame tail padded to 56 (the validator demands
    # exact sizes).
    ad = FakeAdapter(1178)
    assert mod._hypothesis(ad, b"\x00\x01" * 100) == "the quick brown"
    assert ad.slices[0] == (49, 0)
    assert ad.slices[1:-1] == [(56, 0)] * 20
    assert ad.slices[-1] == (9, 56)
    assert ad.generate_calls == 1


def test_stream_mapping_says_nothing_under_one_first_chunk():
    mod = load_partials_script()
    ad = FakeAdapter(30)
    assert mod._hypothesis(ad, b"\x00\x01" * 8) == ""   # nothing causal
    assert mod._hypothesis(ad, b"") == ""
    assert ad.generate_calls == 0           # the sidecar never wakes


def test_as_text_flattens_the_decode_list_shape():
    # The probe's decode came back as ['The quick ...'] — a sidecar that
    # str()-ed that would speak in brackets.
    mod = load_partials_script()
    assert mod._as_text(["The quick brown"]) == "The quick brown"
    assert mod._as_text("The quick brown") == "The quick brown"


# --- MossTts -------------------------------------------------------------

def make_tts(handler, session_id="phd", **kw):
    client = httpx.AsyncClient(transport=httpx.MockTransport(handler),
                               base_url="http://tts.test")
    return MossTts("http://tts.test", "ref.mp3", http=client,
                   session_id=session_id, **kw)


@pytest.mark.anyio
async def test_moss_tts_stream_runs_the_incremental_wire_sequence():
    seen = []

    def handler(request):
        seen.append((request.method, request.url.path,
                     json.loads(request.content) if request.content else None))
        if request.url.path == "/tts/session/phd/audio":
            return httpx.Response(200, content=b"AB" * 100)
        return httpx.Response(200, json={"status": "ok"})

    tts = make_tts(handler)
    s = tts.stream()
    await s.push("one. ")
    await s.push("two.")
    await s.finish()
    assert b"".join([c async for c in s.chunks()]) == b"AB" * 100
    paths = [p for _, p, _ in seen]
    assert paths[0] == "/tts/session/start"
    assert "/tts/session/phd/audio" in paths          # the reader's GET
    pushes = [b for _, p, b in seen if p == "/tts/session/push"]
    assert [b["text"] for b in pushes] == ["one. ", "two.", ""]
    assert [b["is_final"] for b in pushes] == [False, False, True]
    assert seen[0][2] == {"session_id": "phd", "prompt_audio": "ref.mp3"}


@pytest.mark.anyio
async def test_moss_tts_stream_opens_audio_only_after_start_returns():
    # fast_api's GET generator captures the turn's queue AT generator
    # start: a GET issued before the start POST would attach to the
    # PREVIOUS turn's queue and hang. Pin the order.
    seen = []

    def handler(request):
        seen.append(request.url.path)
        if request.url.path == "/tts/session/phd/audio":
            return httpx.Response(200, content=b"zz")
        return httpx.Response(200, json={})

    tts = make_tts(handler)
    s = tts.stream()
    await s.push("hello")
    await s.finish()
    [c async for c in s.chunks()]
    assert seen.index("/tts/session/phd/audio") > seen.index(
        "/tts/session/start")


@pytest.mark.anyio
async def test_moss_tts_stream_reuses_the_session_across_turns():
    starts = []

    def handler(request):
        if request.url.path == "/tts/session/start":
            starts.append(json.loads(request.content)["session_id"])
        if request.url.path == "/tts/session/phd/audio":
            return httpx.Response(200, content=b"zz")
        return httpx.Response(200, json={})

    tts = make_tts(handler)
    for text in ("one", "two"):
        s = tts.stream()
        await s.push(text)
        await s.finish()
        assert [c async for c in s.chunks()] == [b"zz"]
    assert starts == ["phd", "phd"]   # same sid: prompt tokens stay cached


@pytest.mark.anyio
async def test_moss_tts_stream_abort_closes_without_final_and_next_start_works():
    # No cancel endpoint exists, so abort closes the session outright:
    # the single-threaded worker stops the abandoned turn instead of
    # synthesizing it (and force-finishing into garbage) before the next
    # turn's audio. The client's job is to stop reading AND stop paying.
    seen = []

    def handler(request):
        seen.append((request.url.path,
                     json.loads(request.content) if request.content else None))
        if request.url.path == "/tts/session/phd/audio":
            return httpx.Response(200, content=b"zz")
        return httpx.Response(200, json={})

    tts = make_tts(handler)
    s = tts.stream()
    await s.push("stale sentence")
    await s.abort()
    got = [c async for c in s.chunks()]  # must END, not hang — the contract
    assert got in ([], [b"zz"])          # partial or none, both honest
    finals = [b for p, b in seen
              if p == "/tts/session/push" and b.get("is_final")]
    assert finals == []                  # abandoned, never force-finished
    assert "/tts/session/close" in [p for p, _ in seen]  # worker stopped
    s2 = tts.stream()
    await s2.push("fresh")
    await s2.finish()
    assert [c async for c in s2.chunks()] == [b"zz"]
    assert [p for p, _ in seen].count("/tts/session/start") == 2


@pytest.mark.anyio
async def test_moss_tts_abort_before_open_touches_no_network():
    # A stream that never opened a session has nothing to close — abort
    # must not invent a close POST for a session that was never started.
    calls = []
    tts = make_tts(lambda r: calls.append(r) or httpx.Response(200))
    s = tts.stream()
    await s.push("   ")                  # whitespace: never opens
    await s.abort()
    assert calls == []


@pytest.mark.anyio
async def test_moss_tts_session_id_is_unique_per_instance():
    # A fixed id would make a second MossTts (a probe, a restart, a
    # second backend) share the sidecar's one worker session and audio
    # queue — turns interleave into each other's audio. Each instance
    # owns its own id; reuse is per-instance, across its own turns.
    starts = []

    def handler(request):
        if request.url.path == "/tts/session/start":
            starts.append(json.loads(request.content)["session_id"])
        if request.url.path.endswith("/audio"):
            return httpx.Response(200, content=b"z")
        return httpx.Response(200, json={})

    for _ in range(2):
        s = make_tts(handler, session_id=None).stream()
        await s.push("hello")
        await s.finish()
        [c async for c in s.chunks()]
    assert starts[0] and starts[0] != starts[1]


@pytest.mark.anyio
async def test_moss_tts_stream_error_ends_chunks_clean_and_faults():
    state = {"n": 0}

    def handler(request):
        state["n"] += 1
        if state["n"] <= 3:
            raise httpx.ConnectError("warmup failed", request=request)
        return httpx.Response(200, content=b"ok") if (
            request.url.path == "/tts/session/phd/audio"
        ) else httpx.Response(200, json={})

    tts = make_tts(handler)
    for _ in range(3):
        s = tts.stream()
        await s.push("a")
        await s.finish()
        assert [c async for c in s.chunks()] == []
    assert tts.faulted()
    s = tts.stream()
    await s.push("d")
    await s.finish()
    assert [c async for c in s.chunks()] == [b"ok"]
    assert not tts.faulted()


@pytest.mark.anyio
async def test_moss_tts_empty_push_skips_the_network():
    calls = []
    tts = make_tts(lambda r: calls.append(r) or httpx.Response(200))
    s = tts.stream()
    await s.push("   ")
    await s.finish()          # never opened: the episode touched no network
    assert [c async for c in s.chunks()] == []
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
    s = tts.stream()
    await s.push("hi")
    await s.finish()
    [c async for c in s.chunks()]
    assert tts.sample_rate == 16000          # the sidecar's own truth


@pytest.mark.anyio
async def test_moss_tts_sample_rate_defaults_when_the_header_is_absent():
    def handler(request):
        if request.url.path == "/tts/session/phd/audio":
            return httpx.Response(200, content=b"zz")
        return httpx.Response(200, json={})

    tts = make_tts(handler)
    s = tts.stream()
    await s.push("hi")
    await s.finish()
    [c async for c in s.chunks()]
    assert tts.sample_rate == 24000


# --- KokoroTts -----------------------------------------------------------

def make_kokoro(handler, **kw):
    client = httpx.AsyncClient(transport=httpx.MockTransport(handler),
                               base_url="http://kokoro.test")
    return KokoroTts("http://kokoro.test", http=client, **kw)


@pytest.mark.anyio
async def test_kokoro_stream_speaks_each_pushed_sentence_in_order():
    seen = []

    def handler(request):
        body = json.loads(request.content)
        seen.append(body)
        return httpx.Response(200, content=body["input"].encode() * 4)

    tts = make_kokoro(handler)
    s = tts.stream()
    await s.push("one.")
    await s.push("two.")
    await s.finish()
    audio = b"".join([c async for c in s.chunks()])
    assert audio == b"one." * 4 + b"two." * 4   # one continuous PCM buffer
    assert [b["input"] for b in seen] == ["one.", "two."]
    assert all(b["response_format"] == "pcm" for b in seen)
    assert all(b["voice"] == "af_sky" for b in seen)


@pytest.mark.anyio
async def test_kokoro_keeps_order_when_the_second_sentence_finishes_first():
    # Per-sentence requests race; the wire must still play them in
    # sentence order — the episode's PCM is one continuous buffer.
    async def handler(request):
        body = json.loads(request.content)
        if body["input"] == "slow one.":
            await asyncio.sleep(0.05)
            return httpx.Response(200, content=b"S")
        return httpx.Response(200, content=b"F")

    tts = make_kokoro(handler)
    s = tts.stream()
    await s.push("slow one.")
    await s.push("fast two.")
    await s.finish()
    assert b"".join([c async for c in s.chunks()]) == b"SF"


@pytest.mark.anyio
async def test_kokoro_stream_never_pays_for_silence():
    calls = []
    tts = make_kokoro(
        lambda r: calls.append(r) or httpx.Response(200, content=b""))
    s = tts.stream()
    await s.push("   ")                          # whitespace: no request
    await s.finish()
    assert [c async for c in s.chunks()] == []
    assert calls == []


@pytest.mark.anyio
async def test_kokoro_failed_sentence_ends_chunks_and_faults_after_three():
    # §8: TTS down means screen-only — the iterator ENDS, it never
    # hangs, and three consecutive failed streams flip faulted().
    def handler(request):
        raise httpx.ConnectError("kokoro down", request=request)

    tts = make_kokoro(handler)
    for _ in range(3):
        s = tts.stream()
        await s.push("hello")
        await s.finish()
        assert [c async for c in s.chunks()] == []
    assert tts.faulted()


@pytest.mark.anyio
async def test_kokoro_clean_stream_resets_the_fault_counter():
    state = {"down": True}

    def handler(request):
        if state["down"]:
            raise httpx.ConnectError("kokoro down", request=request)
        return httpx.Response(200, content=b"AA")

    tts = make_kokoro(handler)
    for _ in range(2):                           # two strikes stand
        s = tts.stream()
        await s.push("x")
        await s.finish()
        [c async for c in s.chunks()]
    state["down"] = False
    s = tts.stream()                             # one clean stream heals
    await s.push("x")
    await s.finish()
    assert [c async for c in s.chunks()] == [b"AA"]
    state["down"] = True
    for _ in range(2):                           # back to two, not five
        s = tts.stream()
        await s.push("x")
        await s.finish()
        [c async for c in s.chunks()]
    assert not tts.faulted()


@pytest.mark.anyio
async def test_kokoro_a_silent_turn_does_not_heal_the_fault_counter():
    # Only a stream that actually spoke heals (Moss heals after a real
    # audio read): a whitespace turn over a down kokoro made no request
    # and must not reset the strikes.
    def handler(request):
        raise httpx.ConnectError("kokoro down", request=request)

    tts = make_kokoro(handler)
    s = tts.stream()                             # strike 1
    await s.push("x")
    await s.finish()
    [c async for c in s.chunks()]
    s = tts.stream()                             # silent turn, no network
    await s.push("   ")
    await s.finish()
    assert [c async for c in s.chunks()] == []
    for _ in range(2):                           # strikes 2 and 3 stand
        s = tts.stream()
        await s.push("x")
        await s.finish()
        [c async for c in s.chunks()]
    assert tts.faulted()


@pytest.mark.anyio
async def test_kokoro_abort_ends_chunks_and_stops_paying_for_unplayed_audio():
    # A barge abandons the turn: chunks must END (the episode closes),
    # and a sentence still on the wire is cancelled, not finished.
    in_flight = {"cancelled": 0}

    async def handler(request):
        body = json.loads(request.content)
        if body["input"] == "second.":
            try:
                await asyncio.sleep(30)
            except asyncio.CancelledError:
                in_flight["cancelled"] += 1
                raise
        return httpx.Response(200, content=b"A")

    tts = make_kokoro(handler)
    s = tts.stream()
    await s.push("first.")
    await s.push("second.")
    got = []
    async for chunk in s.chunks():
        got.append(chunk)
        await s.abort()                          # barge mid-second-sentence
    assert got == [b"A"]
    await asyncio.sleep(0)                       # let the cancel land
    assert in_flight["cancelled"] == 1


@pytest.mark.anyio
async def test_kokoro_abort_before_any_push_touches_no_network():
    calls = []
    tts = make_kokoro(
        lambda r: calls.append(r) or httpx.Response(200, content=b""))
    s = tts.stream()
    await s.abort()
    assert [c async for c in s.chunks()] == []
    assert calls == []


def test_kokoro_reports_the_rate_it_records_at():
    # The raw PCM response carries no header: the adapter declares the
    # rate, and the browser player schedules at it (Kokoro-82M is 24 kHz).
    assert KokoroTts("http://kokoro.test").sample_rate == 24000


@pytest.mark.anyio
async def test_kokoro_health_reads_the_sidecar_health_door():
    ok = make_kokoro(
        lambda r: httpx.Response(200, json={"status": "healthy"}))
    assert await ok.healthy()
    down = make_kokoro(lambda r: httpx.Response(503))
    assert not await down.healthy()


# --- sidecar topology pin ---------------------------------------------------

def test_sidecar_scripts_never_import_the_backend():
    # The sidecar venvs have no phd_helper installed, and the backend
    # venv must never gain a path to loading torch in-process (SAC). The
    # only honest guarantee is a pin — an AST one, so prose may name the
    # sin.
    for script in ("asr_server.py", "tts_server.py", "corpus_server.py",
                   "partials_server.py"):
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
