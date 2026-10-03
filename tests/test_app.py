"""App assembly at the HTTP seam (SPEC §1): startup/shutdown wiring and
turn-error containment. The voice loop itself is proven end-to-end against
live vLLM (§2); this file covers what only the app owns.
"""

import asyncio
import json
import time

import pytest
from fastapi.testclient import TestClient
from types import SimpleNamespace

from phd_helper.corpus import Corpus
from phd_helper.gists import body_sha
from phd_helper.toolcall import ValidCall
from phd_helper.endpoint import VoiceEndpoint
from phd_helper.project import Project
from phd_helper import sessionlog
from phd_helper.saytext import for_speech, strip_fences
from phd_helper.server.app import (Session, create_app, end_sitting,
                                   ensure_sitting, idle_expired)
from phd_helper.server.config import Config
from phd_helper.server.llm import LlmError
from phd_helper.server.voice import StubStt, StubTts
from phd_helper.store import DocInfo


class FakeHttp:
    async def aclose(self):
        pass


@pytest.fixture
def anyio_backend():
    return "asyncio"


def test_app_shutdown_closes_the_http_client(tmp_path):
    closed = []

    class Http:
        async def aclose(self):
            closed.append(True)

    state = SimpleNamespace(http=Http(), corpus=Corpus(tmp_path / "c"),
                            ingestor=None, ingest_tasks=set())
    with TestClient(create_app(state=state)):
        pass
    assert closed == [True]  # shared httpx client must not leak


class BoomLlm:
    async def chat(self, *args, **kwargs):
        raise RuntimeError("boom")  # anything outside the known envelope


@pytest.mark.anyio
async def test_unexpected_error_becomes_an_error_event_not_silence(tmp_path):
    (tmp_path / "sections").mkdir()
    (tmp_path / "main.tex").write_text(
        "\\documentclass{article}\n\\begin{document}\n"
        "\\input{sections/intro}\n\\end{document}\n", encoding="utf-8")
    (tmp_path / "sections" / "intro.tex").write_text("Hi.\n", encoding="utf-8")
    state = SimpleNamespace(project=Project(tmp_path), llm=BoomLlm(),
                            tts=None, config=Config(stream_tts=False),
                            fetch=None,
                            corpus=Corpus(tmp_path / "c"),
                            corpus_store=None, crossref_mailto="",
                            openalex_mailto="",
                            ingest_tasks=set(), gist_task=None)
    session = Session(state)
    events = []

    async def capture(event):
        events.append(event)

    session.send = capture
    await session.run_turn("hello")  # must not raise out of the task
    assert any(e.get("type") == "error" and e.get("where") == "turn"
               for e in events)


# -- per-turn context assembly wiring (SPEC §4, issue #20) -----------------
# The §4 priority list must reach the LLM: skeleton + selected section +
# pinned abstracts/headings, with the drop tiers actually dropping.


class RecordingLlm:
    def __init__(self):
        self.calls: list[list[dict]] = []

    async def chat(self, messages, **kwargs):
        self.calls.append([dict(m) for m in messages])
        return {"role": "assistant", "content": "ok"}, [], "ok"


class DocStore:
    """Store half of the §4 pinned contribution: doc() -> DocInfo."""

    def __init__(self, fail=False):
        self.fail = fail

    async def doc(self, doc_id):
        if self.fail:
            raise RuntimeError("lancedb locked")
        return DocInfo("TRANSFORMER ABSTRACT",
                       ("1 Introduction", "2 Architecture"))


def make_env(tmp_path, store=None, budget=8000):
    paper = tmp_path / "my-paper"
    (paper / "sections").mkdir(parents=True, exist_ok=True)
    (paper / "main.tex").write_text(
        "\\title{My Paper}\n\\begin{document}\n"
        "\\input{sections/intro}\n\\end{document}\n", encoding="utf-8")
    (paper / "sections" / "intro.tex").write_text(
        "\\section{Introduction}\nIntro body prose.\n", encoding="utf-8")
    corpus = Corpus(tmp_path / "c")
    rec, _ = corpus.add_pdf(b"%PDF-1", title="Attention Is All You Need",
                            arxiv="1706.03762")
    corpus.pin(rec.doc_id, "my-paper")
    state = SimpleNamespace(project=Project(paper), llm=RecordingLlm(),
                            tts=StubTts(), config=Config(
                                context_budget_tokens=budget,
                                # the one-shot leg: the chat-only fakes
                                # here never need chat_stream; the voice
                                # block turns the streaming leg on
                                stream_tts=False),
                            fetch=None, corpus=corpus, corpus_store=store,
                            crossref_mailto="", openalex_mailto="",
                            ingest_tasks=set(), gist_task=None,
                            sitting=None, endpoint=VoiceEndpoint())
    return state, Session(state)


def sent_text(session) -> str:
    return "\n".join(str(m.get("content") or "")
                     for m in session.state.llm.calls[0])


@pytest.mark.anyio
async def test_pinned_source_abstract_and_headings_reach_the_llm(tmp_path):
    state, session = make_env(tmp_path, store=DocStore())
    await session.run_turn("hello")
    joined = sent_text(session)
    assert "TRANSFORMER ABSTRACT" in joined  # §4: always contribute
    assert "2 Architecture" in joined
    assert "My Paper" in joined  # the skeleton rides every turn


@pytest.mark.anyio
async def test_pinned_contribution_degrades_to_the_title_when_store_is_gone(
        tmp_path):
    for store in (None, DocStore(fail=True)):  # down, or faulting mid-call
        state, session = make_env(tmp_path, store=store)
        await session.run_turn("hello")
        joined = sent_text(session)
        assert "Attention Is All You Need" in joined  # never vanishes
        assert "TRANSFORMER ABSTRACT" not in joined


@pytest.mark.anyio
async def test_over_budget_drops_pinned_but_never_the_map_or_the_request(
        tmp_path):
    # A pinned abstract big enough to blow the budget: the §4 tier order
    # must drop it, keep skeleton + section, and clamp the conversation
    # so the current request itself never drops.
    class BigDocStore(DocStore):
        async def doc(self, doc_id):
            return DocInfo("X" * 4000, ("1 Introduction",))

    state, session = make_env(tmp_path, store=BigDocStore(), budget=500)
    assert session.select_section("sections/intro.tex")
    await session.run_turn("hello")
    joined = sent_text(session)
    assert "X" * 4000 not in joined
    assert "My Paper" in joined
    assert "Intro body prose" in joined
    assert "hello" in joined


@pytest.mark.anyio
async def test_hopeless_budget_still_sends_the_current_request(tmp_path):
    state, session = make_env(tmp_path, store=DocStore(), budget=1)
    await session.run_turn("the current ask")
    assert session.state.llm.calls[0][-1]["content"] == "the current ask"


@pytest.mark.anyio
async def test_conversation_drops_keep_exchanges_whole(tmp_path):
    # An old exchange with a tool round-trip: dropping conversation must
    # never orphan a tool result from its assistant tool_calls message.
    state, session = make_env(tmp_path, store=DocStore(), budget=1)
    session.history += [
        {"role": "user", "content": "old question"},
        {"role": "assistant", "content": None,
         "tool_calls": [{"id": "t1", "type": "function",
                         "function": {"name": "section_read",
                                      "arguments": "{}"}}]},
        {"role": "tool", "tool_call_id": "t1", "content": "{}"},
        {"role": "assistant", "content": "old answer"},
    ]
    await session.run_turn("new question")
    msgs = session.state.llm.calls[0]
    assert not any(m.get("content") == "old question" for m in msgs)
    assert not any(m.get("tool_call_id") == "t1" for m in msgs)
    assert msgs[-1]["content"] == "new question"


def test_select_section_anchors_only_known_tree_nodes(tmp_path):
    _, session = make_env(tmp_path)
    assert session.select_section("sections/intro.tex")
    assert session.selected == "sections/intro.tex"
    assert not session.select_section("sections/nonexistent.tex")
    assert session.selected == "sections/intro.tex"  # unchanged


def test_empty_path_clears_the_anchor(tmp_path):
    # An anchored section rides every later turn (§4); without a deselect
    # the tab is stuck with it until reconnect.
    _, session = make_env(tmp_path)
    assert session.select_section("sections/intro.tex")
    assert session.select_section("")
    assert session.selected is None


@pytest.mark.anyio
async def test_pinned_docs_are_fetched_concurrently(tmp_path):
    # store.doc is a blocking LanceDB query per doc; sequential awaits
    # would add N round-trips of latency to every voice turn.
    import asyncio

    inflight = {"now": 0, "max": 0}

    class SlowStore(DocStore):
        async def doc(self, doc_id):
            inflight["now"] += 1
            inflight["max"] = max(inflight["max"], inflight["now"])
            await asyncio.sleep(0.01)
            inflight["now"] -= 1
            return await super().doc(doc_id)

    state, session = make_env(tmp_path, store=SlowStore())
    rec2, _ = state.corpus.add_pdf(b"%PDF-2", title="Second Paper")
    state.corpus.pin(rec2.doc_id, "my-paper")
    await session.run_turn("hello")
    assert inflight["max"] == 2


@pytest.mark.anyio
async def test_selected_section_body_rides_the_context(tmp_path):
    state, session = make_env(tmp_path, store=DocStore())
    assert session.select_section("sections/intro.tex")
    await session.run_turn("tighten it")
    assert "Intro body prose" in sent_text(session)


def test_select_section_control_frame_is_acknowledged(tmp_path):
    state, _ = make_env(tmp_path)
    state.endpoint = VoiceEndpoint(ping_interval=2.0, lease_timeout=60.0)
    state.stt = StubStt()
    state.http = FakeHttp()
    state.ingestor = None
    state.ingest_tasks = set()
    with TestClient(create_app(state=state)) as client:
        with client.websocket_connect("/ws/voice?client=c1") as ws:
            assert ws.receive_json()["type"] == "hello"
            ws.send_json({"type": "select_section",
                          "section": "sections/intro.tex"})
            assert ws.receive_json() == {"type": "section_selected",
                                         "section": "sections/intro.tex"}
            ws.send_json({"type": "select_section", "section": "nope.tex"})
            err = ws.receive_json()
            assert err["type"] == "error" and err["where"] == "control"


class VoiceStt:
    """An STT that hands back a canned final on every feed. The gate and
    the sidecar wire are proven in test_audio.py; this fake exists so the
    WS handler's bytes branch gets its first coverage."""

    def __init__(self, final="tighten the intro"):
        self.final = final
        self.feeds = 0

    async def feed(self, pcm16_bytes):
        self.feeds += 1
        return [self.final]

    def faulted(self):
        return False


def test_ws_binary_frames_drive_a_turn(tmp_path):
    # The mic path end-to-end at the socket seam: a binary frame becomes an
    # STT final becomes a full turn — RecordingLlm proves the final reached
    # the model, not just that the socket stayed open.
    state, _ = make_env(tmp_path)
    state.endpoint = VoiceEndpoint(ping_interval=2.0, lease_timeout=60.0)
    state.stt = VoiceStt()
    state.http = FakeHttp()
    state.ingestor = None
    with TestClient(create_app(state=state)) as client:
        with client.websocket_connect("/ws/voice?client=c1") as ws:
            assert ws.receive_json()["type"] == "hello"
            ws.send_bytes(b"\x00\x01" * 800)
            events = []
            while True:
                ev = ws.receive_json()
                events.append(ev)
                if ev["type"] == "assistant_text":
                    break
    types = [e["type"] for e in events]
    # the spoken words join the transcript ahead of the turn they open
    assert types[0] == "user_text"
    assert events[0]["text"] == "tighten the intro"
    assert "turn_started" in types
    joined = "\n".join(str(m.get("content") or "")
                       for m in state.llm.calls[0])
    assert "tighten the intro" in joined


# -- voice-out: holder-gated, sentence-level audio delivery (SPEC §3) -----
# The sitting speaks to the endpoint holder only (audio follows the
# mic); the wire is audio_start{sample_rate} → PCM16 binary → audio_end.
# Sentence-level TTS: the reply is spoken as it is generated — the gate
# pushes each completed sentence to the sidecar the moment it forms, so
# audio_start may precede assistant_text. That wire order is not the
# contract (both are reducer no-ops client-side); what is: audio_start
# before any PCM16, audio_end last, holder-only.


class ChunkyTtsStream:
    """TtsStream-shaped fake: each push queues one known chunk (silence
    costs no bytes), finish/abort end the stream. The pause between
    chunk yields is what makes "while speaking" observable from the
    test thread: audio_start proves the forwarder is inside the pause."""

    def __init__(self, owner, chunks, pause):
        self._owner = owner
        self._chunks = list(chunks)
        self._pause = pause
        self._q: asyncio.Queue = asyncio.Queue()
        self.pushed: list[str] = []
        self._ended = False

    async def push(self, text):
        if not text.strip():
            return
        self.pushed.append(text)
        if self._owner.timeline is not None:
            self._owner.timeline.append(f"push:{text[:8]}")
        if self._chunks:
            self._q.put_nowait(self._chunks.pop(0))

    async def finish(self):
        if not self._ended:
            self._ended = True
            self._q.put_nowait(None)

    async def abort(self):
        if not self._ended:
            self._ended = True
            self._q.put_nowait(None)

    async def chunks(self):
        first = True
        while True:
            item = await self._q.get()
            if item is None:
                return
            if not first and self._pause:
                await asyncio.sleep(self._pause)
            first = False
            yield item


class ChunkyTts:
    """Provider seam: stream() opens a handle (the no-holder test counts
    episodes through this); sample_rate rides audio_start."""

    sample_rate = 24000

    def __init__(self, chunks=(b"AB" * 4, b"CD" * 4), pause=0.0,
                 timeline=None):
        self.chunks = chunks
        self.pause = pause
        self.timeline = timeline
        self.requests = 0
        self.streams: list[ChunkyTtsStream] = []

    def stream(self):
        self.requests += 1
        s = ChunkyTtsStream(self, self.chunks, self.pause)
        self.streams.append(s)
        return s

    def faulted(self):
        return False


class FakeStreamLlm:
    """chat_stream-leg fake: replays a script of turns as the event
    stream the real client produces. Each turn is (pieces, valid_calls);
    a tool-call turn yields ("toolcalls",) after its preamble. chat() is
    the PHD_STREAM_TTS=0 leg: same script, one shot."""

    def __init__(self, script, pause=0.0, timeline=None):
        self.script = list(script)
        self.pause = pause
        self.timeline = timeline
        self.calls: list[list[dict]] = []
        self.stream_turns = 0
        self.oneshot_turns = 0

    def _next(self):
        return self.script.pop(0) if self.script else (["(end)"], [])

    async def chat_stream(self, messages, **kwargs):
        self.calls.append([dict(m) for m in messages])
        self.stream_turns += 1
        pieces, valid = self._next()
        text = "".join(pieces)
        msg = {"role": "assistant", "content": text or None}
        for piece in pieces:
            if self.pause:
                await asyncio.sleep(self.pause)
            if self.timeline is not None:
                self.timeline.append(f"delta:{piece.strip()[:8]}")
            yield ("text", piece)
        if valid:
            yield ("toolcalls",)
            msg["tool_calls"] = [
                {"id": c.id, "type": "function",
                 "function": {"name": c.name, "arguments": "{}"}}
                for c in valid]
        yield ("done", msg, valid, text)

    async def chat(self, messages, **kwargs):
        self.calls.append([dict(m) for m in messages])
        self.oneshot_turns += 1
        pieces, valid = self._next()
        text = "".join(pieces)
        return {"role": "assistant", "content": text or None}, valid, text


def drain(ws, until):
    """Frames in arrival order — ("ev", event) for JSON, ("bin", bytes)
    for binary — until the named event arrives (inclusive)."""
    frames = []
    while True:
        frame = ws.receive()
        if frame.get("text") is not None:
            ev = json.loads(frame["text"])
            frames.append(("ev", ev))
            if ev["type"] == until:
                return frames
        elif frame.get("bytes") is not None:
            frames.append(("bin", frame["bytes"]))


def drain_until_types(ws, wanted):
    """Frames until every named event type has arrived (binary frames
    pass through) — for when sends race and order is not the
    contract."""
    frames, seen = [], set(wanted)
    while seen:
        frame = ws.receive()
        if frame.get("text") is not None:
            ev = json.loads(frame["text"])
            frames.append(("ev", ev))
            seen.discard(ev["type"])
        elif frame.get("bytes") is not None:
            frames.append(("bin", frame["bytes"]))
    return frames


def test_the_holder_hears_the_reply_start_binary_end(tmp_path):
    state = ws_state(tmp_path, tts=ChunkyTts(),
                     llm=FakeStreamLlm([(["Tighten the intro. ",
                                          "Lead with the result."],
                                         [])]))
    seed_gists(state)
    with TestClient(create_app(state=state)) as client:
        with client.websocket_connect("/ws/voice?client=c1") as ws:
            assert ws.receive_json()["type"] == "hello"
            ws.send_json({"type": "arm"})
            assert ws.receive_json()["type"] == "armed"
            ws.send_json({"type": "typed", "text": "hello"})
            frames = drain(ws, "audio_end")
    types = [p["type"] for k, p in frames if k == "ev"]
    assert types[0] == "turn_started"
    assert types[-1] == "audio_end"
    assert {"assistant_text", "audio_start"} <= set(types)
    # With overlap the order of audio_start vs assistant_text is not
    # the contract; what is: audio_start opens before any PCM16 rides.
    start_at = next(i for i, (k, p) in enumerate(frames)
                    if k == "ev" and p["type"] == "audio_start")
    first_bin = next(i for i, (k, _) in enumerate(frames) if k == "bin")
    assert start_at < first_bin
    start = frames[start_at][1]
    assert start["sample_rate"] == 24000
    assert [p for k, p in frames if k == "bin"] == [b"AB" * 4, b"CD" * 4]


def test_audio_starts_before_the_reply_is_fully_generated(tmp_path):
    # The headline: generation and synthesis overlap. The first
    # sentence reaches the sidecar (and the socket) while the model is
    # still dictating the rest — audio_start lands before
    # assistant_text, and the shared timeline proves the push happened
    # mid-generation, not after it.
    timeline = []
    tts = ChunkyTts(timeline=timeline)
    llm = FakeStreamLlm([(["The intro is thin. ", "Widen the hook ",
                           "first. Then cut paragraph two."], [])],
                        pause=0.15, timeline=timeline)
    state = ws_state(tmp_path, tts=tts, llm=llm)
    seed_gists(state)
    with TestClient(create_app(state=state)) as client:
        with client.websocket_connect("/ws/voice?client=c1") as ws:
            assert ws.receive_json()["type"] == "hello"
            ws.send_json({"type": "arm"})
            assert ws.receive_json()["type"] == "armed"
            ws.send_json({"type": "typed", "text": "how?"})
            frames = drain(ws, "audio_start")
            types = [p["type"] for k, p in frames if k == "ev"]
            assert "assistant_text" not in types  # audio, mid-generation
            # let the dictation finish (shutdown would cancel it and
            # the timeline would never see the last piece)
            drain(ws, "assistant_text")
    assert timeline.index("push:The intr") < timeline.index("delta:first. T")


def test_sentences_are_pushed_as_they_complete(tmp_path):
    tts = ChunkyTts(chunks=(b"AB" * 4,) * 3)
    llm = FakeStreamLlm([(["Tighten the intro", " now. Then ",
                           "cut paragraph two. Done."], [])])
    state = ws_state(tmp_path, tts=tts, llm=llm)
    seed_gists(state)
    with TestClient(create_app(state=state)) as client:
        with client.websocket_connect("/ws/voice?client=c1") as ws:
            assert ws.receive_json()["type"] == "hello"
            ws.send_json({"type": "arm"})
            assert ws.receive_json()["type"] == "armed"
            ws.send_json({"type": "typed", "text": "hello"})
            drain(ws, "audio_end")
    assert tts.streams[0].pushed == ["Tighten the intro now.",
                                     "Then cut paragraph two.", "Done."]


def test_tool_call_preamble_is_spoken_in_one_episode(tmp_path):
    # Ratified: the agent says its preamble as it types it. The tool
    # call that follows silences the REST of that message (drop_tail),
    # and the final text turn rides the same audio episode — one
    # audio_start, one audio_end, the patch card arriving mid-episode.
    tts = ChunkyTts(chunks=(b"AB" * 4,) * 4)
    llm = FakeStreamLlm([
        (["Let me look at the intro. ", "Half a thought after"],
         [ValidCall(id="t1", name="section_write", args=WRITE)]),
        (["Proposed a tightening."], []),
    ])
    state = ws_state(tmp_path, tts=tts, llm=llm)
    seed_gists(state)
    with TestClient(create_app(state=state)) as client:
        with client.websocket_connect("/ws/voice?client=c1") as ws:
            assert ws.receive_json()["type"] == "hello"
            ws.send_json({"type": "arm"})
            assert ws.receive_json()["type"] == "armed"
            ws.send_json({"type": "typed", "text": "tighten it"})
            frames = drain(ws, "audio_end")
    assert tts.requests == 1                      # one episode, all turn
    assert tts.streams[0].pushed == ["Let me look at the intro.",
                                     "Proposed a tightening."]
    types = [p["type"] for k, p in frames if k == "ev"]
    assert "diff" in types                        # card rode mid-episode


def test_view_only_tabs_get_the_text_but_never_the_audio(tmp_path):
    state = ws_state(tmp_path, tts=ChunkyTts(),
                     llm=FakeStreamLlm([(["Hello there."], [])]))
    seed_gists(state)
    with TestClient(create_app(state=state)) as client:
        with client.websocket_connect("/ws/voice?client=c1") as holder:
            with client.websocket_connect("/ws/voice?client=c2") as viewer:
                assert holder.receive_json()["type"] == "hello"
                assert viewer.receive_json()["type"] == "hello"
                holder.send_json({"type": "arm"})
                assert holder.receive_json()["type"] == "armed"
                viewer.send_json({"type": "typed", "text": "hello"})
                # The holder's audio_end proves the turn is fully done;
                # the viewer's own ack proves its queue is drained —
                # audio frames would have been queued ahead of it.
                drain(holder, "audio_end")
                viewer.send_json({"type": "select_section",
                                  "section": "sections/intro.tex"})
                frames = drain(viewer, "section_selected")
    assert [p for k, p in frames if k == "bin"] == []
    types = [p["type"] for k, p in frames if k == "ev"]
    assert "assistant_text" in types  # text is everywhere (§7)
    assert "audio_start" not in types and "audio_end" not in types


def test_no_holder_means_no_synthesis(tmp_path):
    # Audio follows the mic: nobody armed, nobody hears it — the reply
    # stays on screen (the §8 TTS-down shape), and the sidecar is never
    # paid for audio no one will play. The holder is decided at the
    # first completed sentence; no holder means no episode all turn.
    tts = ChunkyTts()
    state = ws_state(tmp_path, tts=tts,
                     llm=FakeStreamLlm([(["Hello there."], [])]))
    seed_gists(state)
    with TestClient(create_app(state=state)) as client:
        with client.websocket_connect("/ws/voice?client=c1") as ws:
            assert ws.receive_json()["type"] == "hello"
            ws.send_json({"type": "typed", "text": "hello"})
            frames = drain(ws, "assistant_text")
    assert tts.requests == 0
    types = [p["type"] for k, p in frames if k == "ev"]
    assert "audio_start" not in types


def test_barge_in_during_overlap_cancels_the_turn(tmp_path):
    # With sentence-level TTS, speaking no longer implies
    # generation-done: the sitting can be mid-generation AND mid-audio
    # at once, so a barge landing while the first sentences are already
    # audible must kill the live turn (BargeGate's sustained-speech
    # requirement is the shield against false positives). The text was
    # not delivered, so this one interrupts loudly.
    tts = ChunkyTts(pause=0.5)
    llm = FakeStreamLlm([(["First sentence. ", "Second one. ",
                           "Third, never dictated."], [])], pause=0.3)
    state = ws_state(tmp_path, tts=tts, llm=llm)
    seed_gists(state)
    with TestClient(create_app(state=state)) as client:
        with client.websocket_connect("/ws/voice?client=c1") as ws:
            assert ws.receive_json()["type"] == "hello"
            ws.send_json({"type": "arm"})
            assert ws.receive_json()["type"] == "armed"
            ws.send_json({"type": "typed", "text": "hello"})
            frames = drain(ws, "audio_start")
            ws.send_json({"type": "barge_in"})
            # audio_end is guaranteed after tts_stopped + turn_inter-
            # rupted: abort() closes the episode it had started.
            frames += drain_until_types(ws, {"tts_stopped",
                                             "turn_interrupted",
                                             "audio_end"})
            # (checked before shutdown: the sitting's end folds the tail)
            assert "[interrupted]" in json.dumps(state.project.chat_tail())
    types = [p["type"] for k, p in frames if k == "ev"]
    assert "tts_stopped" in types and "turn_interrupted" in types
    assert "audio_end" in types   # abort closes: "no more is coming"


def test_barge_in_while_thinking_still_cancels(tmp_path):
    # The defensive branch: the client's gate only fires while audio
    # plays, but a barge_in during the thinking phase must still abort
    # the turn (§3: stop now).
    state = ws_state(tmp_path,
                     llm=FakeStreamLlm([(["Slow. ", "Later."], [])],
                                       pause=0.5))
    seed_gists(state)
    with TestClient(create_app(state=state)) as client:
        with client.websocket_connect("/ws/voice?client=c1") as ws:
            assert ws.receive_json()["type"] == "hello"
            ws.send_json({"type": "typed", "text": "hello"})
            frames = drain(ws, "turn_started")
            ws.send_json({"type": "barge_in"})
            seen, got = {"tts_stopped", "turn_interrupted"}, []
            while seen:  # the two sends race; order is not the contract
                ev = ws.receive_json()
                got.append(ev["type"])
                seen.discard(ev["type"])
    assert set(got) == {"tts_stopped", "turn_interrupted"}


def test_barge_in_during_the_drain_is_a_no_op(tmp_path):
    # The spec's tail case: the reply is fully composed (assistant_text
    # delivered) and only its audio is draining. The agent is no longer
    # composing — pause/resume still makes sense for audio that will
    # not change — so the control no-ops and the drain plays out to
    # audio_end; the client's own resume window governs the buffer.
    tts = ChunkyTts(pause=0.5)
    state = ws_state(tmp_path, tts=tts,
                     llm=FakeStreamLlm([(["One. ", "Two."], [])]))
    seed_gists(state)
    with TestClient(create_app(state=state)) as client:
        with client.websocket_connect("/ws/voice?client=c1") as ws:
            assert ws.receive_json()["type"] == "hello"
            ws.send_json({"type": "arm"})
            assert ws.receive_json()["type"] == "armed"
            ws.send_json({"type": "typed", "text": "hello"})
            frames = drain(ws, "assistant_text")
            ws.send_json({"type": "barge_in"})
            # the ack proves the queue drained past the barge: had it
            # stopped anything, tts_stopped would ride ahead of it
            ws.send_json({"type": "select_section",
                          "section": "sections/intro.tex"})
            frames += drain_until_types(ws, {"audio_end",
                                             "section_selected"})
    types = [p["type"] for k, p in frames if k == "ev"]
    assert "tts_stopped" not in types
    assert "turn_interrupted" not in types
    assert "audio_end" in types       # the drain played to its end
    assert tts.requests == 1


def test_barge_in_after_the_turn_is_done_is_a_no_op(tmp_path):
    # The same no-op one step later: a finished turn has nothing to
    # stop — the client's own stop covers its buffer. (Was: the
    # speaking-phase no-op; overlap moved it to "reply composed or
    # not", which draining above pins at the exact boundary.)
    # No holder means no audio tail: the turn is done at assistant_text.
    state = ws_state(tmp_path, llm=FakeStreamLlm([(["Done."], [])]))
    seed_gists(state)
    with TestClient(create_app(state=state)) as client:
        with client.websocket_connect("/ws/voice?client=c1") as ws:
            assert ws.receive_json()["type"] == "hello"
            ws.send_json({"type": "typed", "text": "hello"})
            drain(ws, "assistant_text")
            ws.send_json({"type": "barge_in"})
            # the ack proves the queue drained past the barge: had it
            # stopped anything, tts_stopped would ride ahead of it
            ws.send_json({"type": "select_section",
                          "section": "sections/intro.tex"})
            frames = drain(ws, "section_selected")
    types = [p["type"] for k, p in frames if k == "ev"]
    assert "tts_stopped" not in types
    assert "turn_interrupted" not in types


def test_holder_handoff_mid_stream_drops_the_rest(tmp_path):
    # Audio follows the mic: when the lease moves on, the ex-holder's
    # stream stops where it stood and the rest is text-only. audio_end
    # still arrives — it means "no more is coming", and a client left
    # with an open episode would keep its barge gate live on silence.
    tts = ChunkyTts(pause=0.5)
    llm = FakeStreamLlm([(["One. ", "Two. ", "Three."], [])])
    state = ws_state(tmp_path, tts=tts, llm=llm)
    seed_gists(state)
    with TestClient(create_app(state=state)) as client:
        with client.websocket_connect("/ws/voice?client=c1") as ws:
            assert ws.receive_json()["type"] == "hello"
            ws.send_json({"type": "arm"})
            assert ws.receive_json()["type"] == "armed"
            ws.send_json({"type": "typed", "text": "hello"})
            frames = drain(ws, "audio_start")
            state.endpoint._holder = "c2"  # the lease moved mid-stream
            time.sleep(0.7)  # chunk 2 would have arrived: dropped instead
            frames += drain(ws, "audio_end")
    assert [p for k, p in frames if k == "bin"] == [b"AB" * 4]
    types = [p["type"] for k, p in frames if k == "ev"]
    assert types[-2:] == ["audio_start", "audio_end"]


class BoomSocket:
    """A holder socket that dies exactly at a send."""

    async def send_text(self, text):
        raise RuntimeError("socket is dead")

    async def send_bytes(self, data):
        raise RuntimeError("socket is dead")


@pytest.mark.anyio
async def test_a_dying_holder_socket_is_a_detach_not_a_turn_fault(tmp_path):
    # The §8 error envelope must not fan an audio-send failure out to
    # every tab: one tab's audio pipe bursting is a detach, the reply
    # is already on screen, and the dead socket is pruned the way
    # Session.send prunes it.
    state, session = make_env(tmp_path)
    state.tts = ChunkyTts()
    session.state.endpoint.arm("c1", time.monotonic())
    events = []

    async def capture(event):
        events.append(event)

    session.send = capture
    session.attach("c1", BoomSocket())
    await session.run_turn("hello")
    assert not [e for e in events if e.get("type") == "error"]
    assert "c1" not in session.sockets  # pruned, keep talking to the rest


def test_a_final_during_audio_ends_the_spoken_turn_quietly(tmp_path):
    # Supersede (§3/§8): the barge-in's transcript opens a new turn,
    # which cancels the speaking one. The text was already logged and
    # shown, so the cancelled turn ends quietly — no second
    # [interrupted] in the history, no turn_interrupted on the wire.
    tts = ChunkyTts(pause=0.5)
    llm = FakeStreamLlm([(["First reply. ", "And another."], []),
                        (["Second reply."], [])])
    state = ws_state(tmp_path, tts=tts, llm=llm)
    seed_gists(state)
    with TestClient(create_app(state=state)) as client:
        with client.websocket_connect("/ws/voice?client=c1") as ws:
            assert ws.receive_json()["type"] == "hello"
            ws.send_json({"type": "arm"})
            assert ws.receive_json()["type"] == "armed"
            ws.send_json({"type": "typed", "text": "first"})
            drain(ws, "audio_start")
            ws.send_json({"type": "typed", "text": "second"})
            frames = drain(ws, "audio_end")  # the second turn's
    types = [p["type"] for k, p in frames if k == "ev"]
    assert "turn_interrupted" not in types
    assert tts.requests == 2
    assert "[interrupted]" not in json.dumps(state.project.chat_tail())


def test_a_stream_fault_mid_reply_errors_inline_and_stops_the_audio(tmp_path):
    # §8: vLLM dies mid-stream. The error envelope rides where="llm"
    # exactly as the one-shot leg's does (streaming changed the leg,
    # not the failure contract), nothing half-generated ships as text,
    # and the half-spoken episode still closes with audio_end: the
    # error path sends no tts_stopped, so audio_end is the only signal
    # that no more is coming — without it the client's barge gate stays
    # live on silence.
    class FlakyStreamLlm(FakeStreamLlm):
        async def chat_stream(self, messages, **kwargs):
            self.calls.append([dict(m) for m in messages])
            self.stream_turns += 1
            # a COMPLETED sentence first (the trailing "Then " confirms
            # the break): the episode opens and audio starts flowing —
            # then the connection dies mid-stream, as a real fault
            # would after some audio has already landed.
            yield ("text", "Half a sentence. Then ")
            await asyncio.sleep(0.2)   # let the forwarder open the wire
            raise LlmError("vLLM stream failed: connection reset")

    tts = ChunkyTts()
    state = ws_state(tmp_path, tts=tts, llm=FlakyStreamLlm([]))
    seed_gists(state)
    with TestClient(create_app(state=state)) as client:
        with client.websocket_connect("/ws/voice?client=c1") as ws:
            assert ws.receive_json()["type"] == "hello"
            ws.send_json({"type": "arm"})
            assert ws.receive_json()["type"] == "armed"
            ws.send_json({"type": "typed", "text": "hello"})
            frames = drain(ws, "error")
            ws.send_json({"type": "select_section",
                          "section": "sections/intro.tex"})
            frames += drain(ws, "section_selected")  # queue drained past
    types = [p["type"] for k, p in frames if k == "ev"]
    err = next(p for k, p in frames if k == "ev" and p["type"] == "error")
    assert err["where"] == "llm"
    assert "assistant_text" not in types   # nothing half-made shipped
    assert "audio_end" in types            # the episode closes anyway


def test_kill_switch_falls_back_to_one_shot_delivery(tmp_path):
    # PHD_STREAM_TTS=0: today's behavior — one-shot chat(), the whole
    # reply in a single push, through the SAME episode code (the flag
    # is nearly free, and the fallback keeps every wire semantic).
    tts = ChunkyTts()
    llm = FakeStreamLlm([(["The whole reply, ", "one shot."], [])])
    state = ws_state(tmp_path, tts=tts, llm=llm)
    state.config.stream_tts = False
    seed_gists(state)
    with TestClient(create_app(state=state)) as client:
        with client.websocket_connect("/ws/voice?client=c1") as ws:
            assert ws.receive_json()["type"] == "hello"
            ws.send_json({"type": "arm"})
            assert ws.receive_json()["type"] == "armed"
            ws.send_json({"type": "typed", "text": "hello"})
            frames = drain(ws, "audio_end")
            # (before shutdown: the sitting's end distills via chat() too)
            assert llm.oneshot_turns == 1 and llm.stream_turns == 0
    assert tts.streams[0].pushed == ["The whole reply, one shot."]
    types = [p["type"] for k, p in frames if k == "ev"]
    assert types == ["turn_started", "assistant_text", "audio_start",
                     "audio_end"]


# -- /health store probe (SPEC §8, issue #22) -------------------------------
# §8: "the backend health-probes each component so state is known, not
# discovered mid-turn" — a configured-but-faulting store must not report ok.


class OkLlm:
    async def healthy(self):
        return True


class ProbedStore:
    def __init__(self, ok=True):
        self.ok = ok
        self.probes = 0

    async def healthy(self):
        self.probes += 1
        return self.ok


def health_state(tmp_path, store):
    return SimpleNamespace(
        llm=OkLlm(), stt=StubStt(), tts=StubTts(), endpoint=VoiceEndpoint(),
        corpus=Corpus(tmp_path / "c"), corpus_store=store, http=FakeHttp(),
        ingestor=None, ingest_tasks=set())


def test_health_probes_the_store_not_just_the_config(tmp_path):
    store = ProbedStore(ok=True)
    with TestClient(create_app(state=health_state(tmp_path, store))) as c:
        assert c.get("/health").json()["corpus"] == "ok"
    assert store.probes == 1  # a real probe per call, not a config read


def test_health_reports_a_configured_but_faulting_store(tmp_path):
    # A LanceDB that is wired but faulting (locked file, corrupt
    # manifest) used to report "ok" until a search actually failed.
    with TestClient(create_app(
            state=health_state(tmp_path, ProbedStore(ok=False)))) as c:
        assert c.get("/health").json()["corpus"] == "faulted"


class ProbedProvider:
    """A sidecar adapter stand-in: /health must live-probe it, not assume."""

    def __init__(self, ok=True, faulted=False):
        self.ok = ok
        self._faulted = faulted

    async def healthy(self):
        return self.ok

    def faulted(self):
        return self._faulted

    async def aclose(self):
        pass


def test_health_stubs_answer_stub(tmp_path):
    state = health_state(tmp_path, None)
    with TestClient(create_app(state=state)) as c:
        body = c.get("/health").json()
    assert body["stt"] == "stub" and body["tts"] == "stub"


def test_health_probes_live_audio_sidecars(tmp_path):
    state = health_state(tmp_path, None)
    state.stt = ProbedProvider(ok=True)
    state.tts = ProbedProvider(ok=False)
    with TestClient(create_app(state=state)) as c:
        body = c.get("/health").json()
    assert body["stt"] == "ok" and body["tts"] == "faulted"


def test_health_fault_counter_beats_a_fresh_probe(tmp_path):
    # The counter is the adapter's own truth: it stays faulted until a
    # real turn succeeds, even if the probe endpoint has recovered.
    state = health_state(tmp_path, None)
    state.stt = ProbedProvider(ok=True, faulted=True)
    with TestClient(create_app(state=state)) as c:
        assert c.get("/health").json()["stt"] == "faulted"


def test_health_reports_paused_without_a_store(tmp_path):
    # No stack configured: indexing pauses, visibly (§8) — unchanged.
    with TestClient(create_app(state=health_state(tmp_path, None))) as c:
        assert c.get("/health").json()["corpus"] == "paused"


class ProbedEmbedder:
    """Corpus sidecar client stand-in: the store's own probe never
    encodes (lancedb_store.py), so /health must ask the embedder."""

    def __init__(self, ok=True, faulted=False):
        self.ok = ok
        self._faulted = faulted
        self.probes = 0

    def faulted(self):
        return self._faulted

    def healthy(self):  # sync — app.py offloads it via to_thread
        self.probes += 1
        return self.ok


def test_health_embedder_counter_flips_the_chip_without_network(tmp_path):
    store = ProbedStore(ok=True)
    emb = ProbedEmbedder(ok=True, faulted=True)
    store.embedder = emb
    with TestClient(create_app(state=health_state(tmp_path, store))) as c:
        assert c.get("/health").json()["corpus"] == "faulted"
    assert emb.probes == 0  # counter beats a fresh probe: zero network


def test_health_embedder_probe_fault_flips_the_corpus_chip(tmp_path):
    # Sidecar up on /health but the counter clean, yet the probe says
    # no (e.g. it answers 503 while loading): the chip must not lie ok.
    store = ProbedStore(ok=True)
    store.embedder = ProbedEmbedder(ok=False)
    with TestClient(create_app(state=health_state(tmp_path, store))) as c:
        assert c.get("/health").json()["corpus"] == "faulted"


def test_health_embedder_without_healthy_keeps_the_store_verdict(tmp_path):
    # The in-process classes (the 3090 slice) have neither faulted() nor
    # healthy(): the store's own verdict stands, unchanged.
    store = ProbedStore(ok=True)
    store.embedder = SimpleNamespace()
    with TestClient(create_app(state=health_state(tmp_path, store))) as c:
        assert c.get("/health").json()["corpus"] == "ok"


def test_health_store_fault_skips_the_embedder_probe(tmp_path):
    store = ProbedStore(ok=False)
    emb = ProbedEmbedder(ok=True)
    store.embedder = emb
    with TestClient(create_app(state=health_state(tmp_path, store))) as c:
        assert c.get("/health").json()["corpus"] == "faulted"
    assert emb.probes == 0  # already faulted upstream: nothing to ask


class ClosingClient:
    def __init__(self):
        self.closed = False

    def close(self):
        self.closed = True


def test_shutdown_closes_the_corpus_clients(tmp_path):
    store = ProbedStore(ok=True)
    emb, rr = ClosingClient(), ClosingClient()
    store.embedder, store.reranker = emb, rr
    with TestClient(create_app(state=health_state(tmp_path, store))):
        pass
    assert emb.closed and rr.closed  # sync close, owned clients only


# -- per-section gists wiring (SPEC §4) --------------------------------------


class GistLlm:
    """Chat without tools: one canned line per section, counted."""

    def __init__(self, fail_after=None):
        self.calls = 0
        self.fail_after = fail_after

    async def chat(self, messages, **kwargs):
        self.calls += 1
        if self.fail_after is not None and self.calls > self.fail_after:
            raise RuntimeError("vllm down")
        return {"role": "assistant", "content": "  a   one-liner. "}, [], \
            "  a   one-liner. "


def gist_env(tmp_path, llm):
    paper = tmp_path / "my-paper"
    (paper / "sections").mkdir(parents=True)
    (paper / "main.tex").write_text(
        "\title{My Paper}\n\begin{document}\n"
        "\input{sections/intro}\n\input{sections/method}\n"
        "\end{document}\n", encoding="utf-8")
    (paper / "sections" / "intro.tex").write_text(
        "\section{Introduction}\nIntro body prose.\n", encoding="utf-8")
    (paper / "sections" / "method.tex").write_text(
        "\section{Method}\nMethod body prose.\n", encoding="utf-8")
    state = SimpleNamespace(project=Project(paper), llm=llm,
                            ingest_tasks=set(), gist_task=None)
    return state


@pytest.mark.anyio
async def test_refresh_fills_the_cache_and_a_fresh_pass_spends_no_calls(
        tmp_path):
    from phd_helper.server.app import refresh_gists
    state = gist_env(tmp_path, GistLlm())
    await refresh_gists(state)
    cache = state.project.load_gists()
    assert set(cache) == {"sections/intro.tex", "sections/method.tex"}
    assert cache["sections/intro.tex"]["gist"] == "a one-liner."
    state.llm.calls = 0
    await refresh_gists(state)
    assert state.llm.calls == 0  # sha truth: nothing stale, nothing spent


@pytest.mark.anyio
async def test_llm_fault_keeps_the_gists_already_written(tmp_path):
    from phd_helper.server.app import refresh_gists
    state = gist_env(tmp_path, GistLlm(fail_after=1))
    await refresh_gists(state)  # must not raise out of the task
    cache = state.project.load_gists()
    assert len(cache) == 1  # first line survived, second section pending


@pytest.mark.anyio
async def test_empty_section_is_cached_without_burning_a_call(tmp_path):
    from phd_helper.server.app import refresh_gists
    state = gist_env(tmp_path, GistLlm())
    state.project.write_section("sections/method.tex", "")
    await refresh_gists(state)
    assert state.llm.calls == 1  # only intro got a call
    assert state.project.load_gists()["sections/method.tex"]["gist"] == ""


@pytest.mark.anyio
async def test_gists_reach_the_llm_and_the_selected_section_is_skipped(
        tmp_path):
    state, session = make_env(tmp_path, store=DocStore())
    # make_env's paper has one section; seed its gist as fresh.
    body = state.project.read_section("sections/intro.tex")
    from phd_helper.gists import body_sha
    state.project.save_gists({"sections/intro.tex":
                              {"sha": body_sha(body),
                               "gist": "the paper's opening claim"}})
    await session.run_turn("hello")
    assert "the paper's opening claim" in sent_text(session)
    session.select_section("sections/intro.tex")
    state.llm.calls.clear()
    await session.run_turn("again")
    assert "the paper's opening claim" not in sent_text(session)


@pytest.mark.anyio
async def test_a_turn_kicks_the_background_refresh_for_stale_gists(tmp_path):
    state, session = make_env(tmp_path, store=DocStore())
    await session.run_turn("hello")
    assert state.gist_task is not None
    await state.gist_task  # RecordingLlm answers "ok" as the gist line
    assert state.project.load_gists()["sections/intro.tex"]["gist"] == "ok"


@pytest.mark.anyio
async def test_spawn_gist_refresh_is_one_at_a_time(tmp_path):
    from phd_helper.server.app import spawn_gist_refresh
    state = gist_env(tmp_path, GistLlm())
    spawn_gist_refresh(state)
    first = state.gist_task
    spawn_gist_refresh(state)
    assert state.gist_task is first  # still running: no second pass
    await first


# -- paper memory wiring (SPEC §4/§7) ----------------------------------------


@pytest.mark.anyio
async def test_paper_memory_rides_the_context(tmp_path):
    state, session = make_env(tmp_path, store=DocStore())
    state.project.save_memory("- Decided: harrier for embeddings")
    await session.run_turn("hello")
    assert "harrier for embeddings" in sent_text(session)


@pytest.mark.anyio
async def test_memory_survives_the_drop_tiers(tmp_path):
    # §4: memory is never-droppable — a hopeless budget still keeps it.
    state, session = make_env(tmp_path, store=DocStore(), budget=1)
    state.project.save_memory("- The one decision that must survive")
    await session.run_turn("hello")
    assert "The one decision that must survive" in sent_text(session)


class MemoryLlm:
    """Chat seam for distillation: returns a markdown memory file."""

    def __init__(self):
        self.prompts = []

    async def chat(self, messages, **kwargs):
        self.prompts.append(messages[-1]["content"])
        text = "# Decisions\n- harrier for embeddings\n"
        return {"role": "assistant", "content": text}, [], text


@pytest.mark.anyio
async def test_distill_folds_the_session_into_memory(tmp_path):
    from phd_helper.server.app import distill_memory
    llm = MemoryLlm()
    state = gist_env(tmp_path, llm)
    await distill_memory(state, [
        {"role": "user", "content": "we decided on harrier"},
        {"role": "assistant", "content": "noted"}])
    assert state.project.load_memory() == "# Decisions\n- harrier for embeddings"
    assert "we decided on harrier" in llm.prompts[0]  # transcript rides
    assert "(empty)" in llm.prompts[0]  # old memory is offered to merge


@pytest.mark.anyio
async def test_distill_skips_empty_sittings_and_llm_faults(tmp_path):
    from phd_helper.server.app import distill_memory
    state = gist_env(tmp_path, GistLlm())
    await distill_memory(state, [{"role": "system", "content": "x"}])
    assert state.project.load_memory() == ""  # nothing to distill
    state2 = gist_env(tmp_path / "b", GistLlm(fail_after=0))
    await distill_memory(state2, [{"role": "user", "content": "hi"}])
    assert state2.project.load_memory() == ""  # fault: skip, no raise


def ws_state(tmp_path, tts=None, llm=None):
    """make_env's state plus what only the WS door reaches for. A
    caller-supplied llm is a streaming fake, so the voice leg is on
    for it; make_env keeps it off for the chat-only fakes."""
    state, _ = make_env(tmp_path)
    state.endpoint = VoiceEndpoint(ping_interval=2.0, lease_timeout=60.0)
    state.stt = StubStt()
    state.http = FakeHttp()
    state.ingestor = None
    if tts is not None:
        state.tts = tts
    if llm is not None:
        state.llm = llm
        state.config.stream_tts = True
    return state


def test_tab_close_is_not_a_session_end(tmp_path):
    # §7: a sitting ends on switch/shutdown/idle — not on a tab closing.
    # Closing the socket must not distill, and the sitting must survive
    # for the next reconnect; graceful shutdown is what folds it.
    state = ws_state(tmp_path)
    with TestClient(create_app(state=state)) as client:
        with client.websocket_connect("/ws/voice?client=c1") as ws:
            assert ws.receive_json()["type"] == "hello"
            ws.send_json({"type": "typed", "text": "hello there"})
            assert ws.receive_json()["type"] == "turn_started"
            while ws.receive_json()["type"] != "assistant_text":
                pass
        assert state.sitting is not None          # the sitting survived
        assert state.project.load_memory() == ""  # nothing distilled yet
        with client.websocket_connect("/ws/voice?client=c2") as ws2:
            assert ws2.receive_json()["type"] == "hello"
            assert state.sitting is not None      # same sitting, new tab
    # Graceful shutdown ended it: RecordingLlm's "ok" became the memory.
    assert state.sitting is None
    assert state.project.load_memory() == "ok"
    div = state.project.chat_divider()
    assert div and div["reason"] == "shutdown" and div["turns"] == 1


# -- rolling summaries wiring (SPEC §4/§7) -----------------------------------


@pytest.mark.anyio
async def test_distill_summaries_folds_the_sitting_per_section(tmp_path):
    from phd_helper.server.app import distill_summaries
    llm = MemoryLlm()
    state = gist_env(tmp_path, llm)
    await distill_summaries(state, [
        {"role": "user", "content": "tighten the hook",
         "section": "sections/intro.tex"},
        {"role": "assistant", "content": "did"}])
    summaries = state.project.load_summaries()
    entry = summaries["sections/intro.tex"][-1]
    assert entry["text"] == "# Decisions\n- harrier for embeddings"
    assert entry["date"]  # the dated session divider
    assert "tighten the hook" in llm.prompts[0]


@pytest.mark.anyio
async def test_distill_summaries_skips_unanchored_and_faults(tmp_path):
    from phd_helper.server.app import distill_summaries
    state = gist_env(tmp_path, MemoryLlm())
    await distill_summaries(state, [{"role": "user", "content": "hi",
                                      "section": ""}])
    assert state.project.load_summaries() == {}
    state2 = gist_env(tmp_path / "b", GistLlm(fail_after=0))
    await distill_summaries(state2, [{"role": "user", "content": "hi",
                                       "section": "sections/intro.tex"}])
    assert state2.project.load_summaries() == {}  # fault: skip, no raise


@pytest.mark.anyio
async def test_rolling_summaries_ride_the_context(tmp_path):
    state, session = make_env(tmp_path, store=DocStore())
    state.project.save_summaries({"sections/intro.tex": [
        {"date": "2026-09-29", "text": "hook was rewritten twice"}]})
    await session.run_turn("hello")
    assert "hook was rewritten twice" in sent_text(session)


def test_shutdown_distills_summaries_too(tmp_path):
    state = ws_state(tmp_path)
    with TestClient(create_app(state=state)) as client:
        with client.websocket_connect("/ws/voice?client=c1") as ws:
            assert ws.receive_json()["type"] == "hello"
            ws.send_json({"type": "select_section",
                          "section": "sections/intro.tex"})
            ws.receive_json()
            ws.send_json({"type": "typed", "text": "tighten it"})
            assert ws.receive_json()["type"] == "turn_started"
            while ws.receive_json()["type"] != "assistant_text":
                pass
    # RecordingLlm answers "ok"; the exchange was anchored, so intro
    # got a dated summary entry at the sitting's end.
    assert state.project.load_summaries()["sections/intro.tex"][-1]["text"] \
        == "ok"
    # The divider names the sections the sitting touched (§7).
    assert state.project.chat_divider()["sections"] == ["sections/intro.tex"]


# -- section tree door + nested selection (web shell, SPEC §1) --------------


def _nested_paper(tmp_path):
    paper = tmp_path / "my-paper"
    (paper / "sections").mkdir(parents=True)
    (paper / "main.tex").write_text(
        "\begin{document}\n\input{sections/intro}\n\end{document}\n",
        encoding="utf-8")
    (paper / "sections" / "intro.tex").write_text(
        "\section{Introduction}\n\input{sections/background}\n",
        encoding="utf-8")
    (paper / "sections" / "background.tex").write_text(
        "\section{Background}\nBody.\n", encoding="utf-8")
    return paper


def test_sections_endpoint_returns_the_nested_tree(tmp_path):
    # The shell's tree panel needs the parsed \input graph over HTTP;
    # clicking a node anchors the discussion (§1).
    state = SimpleNamespace(project=Project(_nested_paper(tmp_path)),
                            http=FakeHttp(), corpus=Corpus(tmp_path / "c"),
                            ingestor=None, ingest_tasks=set())
    with TestClient(create_app(state=state)) as client:
        tree = client.get("/sections").json()
    assert tree == [{"path": "sections/intro.tex", "title": "Introduction",
                     "children": [{"path": "sections/background.tex",
                                   "title": "Background",
                                   "children": []}]}]


def test_select_section_accepts_nested_nodes(tmp_path):
    # The tree is clickable at every depth (§1: "clicking a node
    # anchors"); the known-set must flatten, not just top level. The
    # system prompt's section list must carry nested paths too, or the
    # model can't be asked to write into them.
    state = SimpleNamespace(project=Project(_nested_paper(tmp_path)),
                            llm=RecordingLlm(), tts=StubTts(),
                            config=Config(), fetch=None,
                            corpus=Corpus(tmp_path / "c"),
                            corpus_store=None, crossref_mailto="",
                            openalex_mailto="",
                            ingest_tasks=set(), gist_task=None)
    session = Session(state)
    assert session.select_section("sections/background.tex")
    assert session.selected == "sections/background.tex"
    assert "sections/background.tex" in session.history[0]["content"]


# -- shell serving (web/app/dist, SPEC §1) ----------------------------------


def test_the_built_shell_is_served_and_api_routes_still_win(tmp_path):
    dist = tmp_path / "dist"
    (dist / "assets").mkdir(parents=True)
    (dist / "index.html").write_text(
        "<html><body><div id='root'>shell</div></body></html>",
        encoding="utf-8")
    (dist / "assets" / "index-abc.js").write_text("//hashed", encoding="utf-8")
    with TestClient(create_app(state=health_state(tmp_path, None),
                               web_dir=dist)) as c:
        assert "shell" in c.get("/").text
        assert c.get("/assets/index-abc.js").text == "//hashed"
        # The mount is registered last: API routes keep winning.
        assert c.get("/health").json()["corpus"] == "paused"


def test_no_shell_build_means_no_root_route(tmp_path):
    with TestClient(create_app(state=health_state(tmp_path, None),
                               web_dir=tmp_path / "missing")) as c:
        assert c.get("/").status_code == 404


@pytest.mark.anyio
async def test_the_sent_request_carries_exactly_one_system_message(tmp_path):
    # Live vLLM rejects a second system message ("System message must be
    # at the beginning", 400) — the §4 context message must merge into
    # the prompt one, not ride beside it. Fake LLMs never noticed.
    state, session = make_env(tmp_path, store=DocStore())
    session.select_section("sections/intro.tex")
    await session.run_turn("hello")
    msgs = state.llm.calls[0]
    assert [i for i, m in enumerate(msgs) if m["role"] == "system"] == [0]
    assert "phd-helper" in msgs[0]["content"]      # the prompt
    assert "Intro body prose" in msgs[0]["content"]  # the §4 context


# -- approval-window voice (SPEC §3) ------------------------------------------
# While a diff is pending, the next utterance is interpreted against the
# approval by the agent (never string-matched): the pending note rides the
# context, pending_decide resolves, clearly-neither keeps the diff pending.


def tool_step(name, args, cid="t1"):
    return ({"role": "assistant", "content": None,
             "tool_calls": [{"id": cid, "type": "function",
                             "function": {"name": name,
                                          "arguments": json.dumps(args)}}]},
            [ValidCall(id=cid, name=name, args=args)], "")


def text_step(text):
    return {"role": "assistant", "content": text}, [], text


class ScriptLlm:
    def __init__(self, script):
        self.script = list(script)
        self.calls: list[list[dict]] = []

    async def chat(self, messages, **kwargs):
        self.calls.append([dict(m) for m in messages])
        return self.script.pop(0) if self.script else text_step("(end)")


WRITE = {"section": "sections/intro.tex", "find": "Intro body prose.",
         "replace": "Intro prose, tightened."}


def seed_gists(state):
    """Fresh projects have no gists, and _build_context kicks a background
    refresh — which would steal ScriptLlm steps. Seed a fresh cache."""
    state.project.save_gists({p: {"sha": body_sha(b), "gist": "seeded"}
                              for p, b in state.project.files().items()})


def pending_env(tmp_path, script):
    state, session = make_env(tmp_path)
    state.llm = ScriptLlm(script)
    seed_gists(state)
    events: list[dict] = []

    async def capture(event):
        events.append(event)

    session.send = capture
    return state, session, events


@pytest.mark.anyio
async def test_voice_apply_resolves_the_pending_diff(tmp_path):
    state, session, events = pending_env(tmp_path, [
        tool_step("section_write", WRITE),
        text_step("Proposed a tightening."),
        tool_step("pending_decide", {"diff_id": "0000", "decision": "apply"},
                  "t2"),
        text_step("Applied."),
    ])
    await session.run_turn("tighten the intro line")
    diff = next(e for e in events if e["type"] == "diff")
    assert [(t["diff_id"], t["section"]) for t in session.pending_diffs] \
        == [(diff["diff_id"], "sections/intro.tex")]
    events.clear()

    await session.run_turn("apply it")

    # The pending note rode the turn: the model saw the id it addressed.
    assert diff["diff_id"] in state.llm.calls[2][0]["content"]
    resolved = next(e for e in events if e["type"] == "diff_resolved")
    assert resolved["diff_id"] == diff["diff_id"]
    assert resolved["applied"] is True
    assert session.pending_diffs == []
    assert "Intro prose, tightened." in \
        state.project.read_section("sections/intro.tex")


@pytest.mark.anyio
async def test_clearly_neither_keeps_the_diff_pending(tmp_path):
    state, session, events = pending_env(tmp_path, [
        tool_step("section_write", WRITE),
        text_step("Proposed a tightening."),
        text_step("It is a transformer architecture paper."),
    ])
    await session.run_turn("tighten the intro line")
    events.clear()

    await session.run_turn("what is this paper even about?")

    assert not [e for e in events if e["type"] == "diff_resolved"]
    assert len(session.pending_diffs) == 1  # still awaiting approval
    # the note (not just the prompt, which names pending_decide too)
    assert "PENDING DIFFS" in state.llm.calls[2][0]["content"]


@pytest.mark.anyio
async def test_voice_amend_proposes_the_replacement_then_discards(tmp_path):
    amended = dict(WRITE, replace="Intro prose, tightened differently.")
    state, session, events = pending_env(tmp_path, [
        tool_step("section_write", WRITE),
        text_step("Proposed a tightening."),
        tool_step("section_write", amended, "t2"),
        tool_step("pending_decide", {"diff_id": "0000",
                                     "decision": "discard"}, "t3"),
        text_step("Replaced with the amended version."),
    ])
    await session.run_turn("tighten the intro line")
    first = next(e for e in events if e["type"] == "diff")
    events.clear()

    await session.run_turn("actually make it tighter differently")

    resolved = next(e for e in events if e["type"] == "diff_resolved")
    assert resolved["diff_id"] == first["diff_id"]
    assert resolved["applied"] is False and resolved["reason"] == "discarded"
    new_diffs = [e for e in events if e["type"] == "diff"]
    assert [d["diff_id"] for d in new_diffs] == ["0001"]
    assert [(t["diff_id"], t["section"]) for t in session.pending_diffs] \
        == [("0001", "sections/intro.tex")]


@pytest.mark.anyio
async def test_voice_apply_bounce_keeps_the_diff_tracked(tmp_path):
    state, session, events = pending_env(tmp_path, [
        tool_step("section_write", WRITE),
        text_step("Proposed a tightening."),
        tool_step("pending_decide", {"diff_id": "0000", "decision": "apply"},
                  "t2"),
        text_step("That bounced — the file changed under it."),
    ])
    await session.run_turn("tighten the intro line")
    state.project.write_section("sections/intro.tex",
                                "User rewrote the file entirely.\n")
    events.clear()

    await session.run_turn("apply it")

    resolved = next(e for e in events if e["type"] == "diff_resolved")
    assert resolved["applied"] is False and resolved["reason"]
    # The diff is still pending on disk, so the note must still carry it.
    assert len(session.pending_diffs) == 1


def test_client_reject_clears_the_approval_window(tmp_path):
    state, _ = make_env(tmp_path)
    state.llm = ScriptLlm([
        tool_step("section_write", WRITE),
        text_step("Proposed a tightening."),
        text_step("The budget is 4096 tokens."),
    ])
    seed_gists(state)
    state.endpoint = VoiceEndpoint(ping_interval=2.0, lease_timeout=60.0)
    state.stt = StubStt()
    state.http = FakeHttp()
    state.ingestor = None
    with TestClient(create_app(state=state)) as client:
        with client.websocket_connect("/ws/voice?client=c1") as ws:
            assert ws.receive_json()["type"] == "hello"
            ws.send_json({"type": "typed", "text": "tighten the intro line"})
            while ws.receive_json()["type"] != "assistant_text":
                pass
            ws.send_json({"type": "reject", "section": "sections/intro.tex",
                          "diff_id": "0000"})
            assert ws.receive_json()["type"] == "diff_resolved"
            ws.send_json({"type": "typed", "text": "what is the budget?"})
            while ws.receive_json()["type"] != "assistant_text":
                pass
    # The turn after the client-side reject sees no pending note (the
    # prompt names pending_decide too, so check the note marker).
    assert "PENDING DIFFS" not in state.llm.calls[2][0]["content"]


@pytest.mark.anyio
async def test_voice_apply_all_only_touches_this_sessions_window(tmp_path):
    state, session, events = pending_env(tmp_path, [
        tool_step("section_write", WRITE),
        text_step("Proposed a tightening."),
        tool_step("pending_decide", {"diff_id": "all", "decision": "apply"},
                  "t2"),
        text_step("Applied."),
    ])
    await session.run_turn("tighten the intro line")
    # A diff from an earlier sitting: on disk, never shown in this window.
    state.project.propose_patch("sections/intro.tex", "Intro body prose.",
                                "Old sitting's idea.")
    events.clear()

    await session.run_turn("apply all")

    resolved = [e for e in events if e["type"] == "diff_resolved"]
    assert [e["diff_id"] for e in resolved] == ["0000"]
    assert state.project.list_pending("sections/intro.tex")  # 0001 untouched
    assert "Old sitting's idea." not in \
        state.project.read_section("sections/intro.tex")


# -- stepwise writing: spoken drafts, apply-from-draft ----------------------
# The paragraph is decoded WHILE spoken (a fenced draft in the reply text,
# not silent tool arguments); the apply turn is paperwork over words the
# user already heard (from_draft substitutes the captured draft), and the
# card stays the truth gate — including against the model itself: the
# approval window is snapshotted at turn start, so a model cannot approve
# a diff the user has not seen yet.

DRAFT_TURN = ("Here is a paragraph for the intro:\n\n"
              "```latex\nWe propose a sharper hook.\nIt leads with the "
              "result.\n```\nSay add when you want it in.")


@pytest.mark.anyio
async def test_a_fenced_draft_is_captured_and_survives_plain_turns(tmp_path):
    state, session, events = pending_env(tmp_path, [
        text_step(DRAFT_TURN),
        text_step("It cites the transformer paper."),  # interposed question
    ])
    await session.run_turn("add a hook to the intro")
    assert session.draft == ("We propose a sharper hook.\n"
                             "It leads with the result.")
    # the screen keeps the raw source — fences included
    ev = next(e for e in events if e["type"] == "assistant_text")
    assert "```latex" in ev["text"]
    events.clear()

    await session.run_turn("what does it cite?")
    assert session.draft  # a plain turn does not kill the draft


@pytest.mark.anyio
async def test_an_empty_fence_clears_the_draft_rather_than_keeping_it_stale(
        tmp_path):
    # a complete block REPLACES the draft — even when it is empty. The
    # user heard no new draft, so the old one must not survive to be
    # applied by a later "add it".
    state, session, events = pending_env(tmp_path, [
        text_step(DRAFT_TURN),
        text_step("Scratch that:\n\n```latex\n```\nNothing yet."),
    ])
    await session.run_turn("add a hook to the intro")
    assert session.draft
    await session.run_turn("actually, retract it")
    assert session.draft is None


@pytest.mark.anyio
async def test_the_apply_turn_substitutes_the_draft_into_the_card(tmp_path):
    state, session, events = pending_env(tmp_path, [
        text_step(DRAFT_TURN),
        tool_step("section_write",
                  {"section": "sections/intro.tex",
                   "find": "Intro body prose.", "replace": "",
                   "from_draft": True}),
        text_step("Card is up — say apply to land it."),
        tool_step("pending_decide", {"diff_id": "0000", "decision": "apply"},
                  "t2"),
        text_step("Landed."),
    ])
    await session.run_turn("add a hook to the intro")
    events.clear()

    await session.run_turn("add it")

    diff = next(e for e in events if e["type"] == "diff")
    # the card shows the bytes the user just heard, not the empty arg
    assert diff["replace"] == session.draft
    events.clear()

    await session.run_turn("apply it")

    assert any(e["type"] == "diff_resolved" and e["applied"]
               for e in events)
    assert "We propose a sharper hook." in \
        state.project.read_section("sections/intro.tex")


@pytest.mark.anyio
async def test_the_model_cannot_approve_its_own_fresh_diff(tmp_path):
    # Rubber-stamp guard: the window is snapshotted at turn start, so a
    # section_write and a pending_decide on its id in the SAME turn
    # bounce — the user has not seen that card yet.
    state, session, events = pending_env(tmp_path, [
        tool_step("section_write", WRITE),
        tool_step("pending_decide", {"diff_id": "0000", "decision": "apply"},
                  "t2"),
        text_step("Proposed — awaiting your approval."),
    ])
    await session.run_turn("tighten the intro")

    assert not [e for e in events if e["type"] == "diff_resolved"]
    assert len(session.pending_diffs) == 1
    assert "Intro prose, tightened." not in \
        state.project.read_section("sections/intro.tex")
    # the bounce teaches the model why (window, not missing diff)
    assert any(m["role"] == "tool" and "window" in m["content"]
               for m in session.history)


@pytest.mark.anyio
async def test_the_captured_draft_reaches_the_tool_validators(tmp_path):
    class CapturingLlm:
        def __init__(self):
            self.validators = []

        async def chat(self, messages, **kwargs):
            self.validators.append(kwargs["validators"])
            return {"role": "assistant", "content": "ok"}, [], "ok"

    state, session = make_env(tmp_path)
    state.llm = CapturingLlm()
    seed_gists(state)
    args = {"section": "sections/intro.tex", "find": "Intro body prose.",
            "replace": "", "from_draft": True}
    await session.run_turn("add it")
    # nothing captured yet: from_draft bounces, teaching the fenced flow
    assert state.llm.validators[0]["section_write"](args)
    session.draft = "We propose X."
    await session.run_turn("add it")
    assert state.llm.validators[1]["section_write"](args) is None


@pytest.mark.anyio
async def test_the_prompt_teaches_the_draft_flow(tmp_path):
    state, session = make_env(tmp_path)
    await session.run_turn("hello")
    prompt = state.llm.calls[0][0]["content"]
    assert "phd-helper" in prompt        # the identity, and rules 1-3
    assert "from_draft" in prompt        # rule 4: apply without retyping
    assert "```latex" in prompt          # rule 4: the draft rides the text
    assert "pending_decide" in prompt    # rule 5: the card, voice-decided


def test_fence_lines_never_reach_the_tts(tmp_path):
    tts = ChunkyTts(chunks=(b"AB" * 4,) * 4)
    llm = FakeStreamLlm([(["Here is the draft.\n\n", "```latex\n",
                           "We propose a hook. ", "It works.\n",
                           "```"], [])])
    state = ws_state(tmp_path, tts=tts, llm=llm)
    seed_gists(state)
    with TestClient(create_app(state=state)) as client:
        with client.websocket_connect("/ws/voice?client=c1") as ws:
            assert ws.receive_json()["type"] == "hello"
            ws.send_json({"type": "arm"})
            assert ws.receive_json()["type"] == "armed"
            ws.send_json({"type": "typed", "text": "draft a hook"})
            frames = drain(ws, "audio_end")
    pushed = tts.streams[0].pushed
    assert not any("```" in p for p in pushed)
    assert "We propose a hook." in pushed and "It works." in pushed
    # the screen keeps the raw fenced source
    ev = next(p for k, p in frames if k == "ev"
              and p["type"] == "assistant_text")
    assert "```latex" in ev["text"]


def test_latex_is_speechified_before_the_push(tmp_path):
    tts = ChunkyTts(chunks=(b"AB" * 4,) * 2)
    llm = FakeStreamLlm([(["Lead with the result \\cite{vaswani2017} and "
                           "widen. See \\ref{fig:hook} too."], [])])
    state = ws_state(tmp_path, tts=tts, llm=llm)
    seed_gists(state)
    with TestClient(create_app(state=state)) as client:
        with client.websocket_connect("/ws/voice?client=c1") as ws:
            assert ws.receive_json()["type"] == "hello"
            ws.send_json({"type": "arm"})
            assert ws.receive_json()["type"] == "armed"
            ws.send_json({"type": "typed", "text": "draft a hook"})
            frames = drain(ws, "audio_end")
    joined = " ".join(tts.streams[0].pushed)
    assert "citation" in joined and "\\cite" not in joined
    assert "reference" in joined and "\\ref{" not in joined
    ev = next(p for k, p in frames if k == "ev"
              and p["type"] == "assistant_text")
    assert "\\cite{vaswani2017}" in ev["text"]  # raw rides the transcript


def test_the_kill_switch_leg_speaks_the_same_filtered_text(tmp_path):
    tts = ChunkyTts()
    state = ws_state(tmp_path, tts=tts, llm=FakeStreamLlm([([DRAFT_TURN],
                                                            [])]))
    state.config.stream_tts = False  # ws_state turned it on for the fake
    seed_gists(state)
    with TestClient(create_app(state=state)) as client:
        with client.websocket_connect("/ws/voice?client=c1") as ws:
            assert ws.receive_json()["type"] == "hello"
            ws.send_json({"type": "arm"})
            assert ws.receive_json()["type"] == "armed"
            ws.send_json({"type": "typed", "text": "draft a hook"})
            drain(ws, "audio_end")
    # §8: the flag changes the pump, never the wire — and never the
    # filter either: one push, fences gone, LaTeX spoken as words
    assert tts.streams[0].pushed == [for_speech(strip_fences(DRAFT_TURN))]


@pytest.mark.anyio
async def test_note_follows_disk_truth_when_resolved_elsewhere(tmp_path):
    state, session, events = pending_env(tmp_path, [
        tool_step("section_write", WRITE),
        text_step("Proposed a tightening."),
        text_step("Nothing is pending now."),
    ])
    await session.run_turn("tighten the intro line")
    assert len(session.pending_diffs) == 1
    # Resolved outside this session (another tab's button, cleanup).
    state.project.reject_pending("sections/intro.tex", "0000")

    await session.run_turn("what about the results?")

    assert session.pending_diffs == []
    assert "PENDING DIFFS" not in state.llm.calls[2][0]["content"]


# -- sitting lifecycle + verbatim history (SPEC §7, #27) --------------------
# A sitting = one project sitting shared by every tab: it survives tab
# closes, resumes the verbatim JSONL after an abrupt end, ends on
# switch/shutdown/idle with a dated divider and a one-line recap.


@pytest.mark.anyio
async def test_abrupt_end_resumes_verbatim_turns(tmp_path):
    # The §8 crash shape: chat.jsonl has messages, no divider (the
    # process died before end_sitting). A fresh sitting resumes them
    # verbatim and the recap says so.
    state, _ = make_env(tmp_path)
    state.project.append_chat([
        {"role": "user", "content": "what was I saying?", "section": ""},
        {"role": "assistant", "content": "You were mid-thought."}])
    sitting = await ensure_sitting(state)
    assert sitting.recap.startswith("Previous session ended abruptly")
    await sitting.run_turn("and then?")
    joined = "\n".join(str(m.get("content") or "")
                       for m in state.llm.calls[0])
    assert "what was I saying?" in joined  # verbatim, not summarized


def test_clean_close_resumes_nothing_but_recaps(tmp_path):
    # A distilled sitting rides on as summaries + memory: the reopened
    # sitting starts with an empty verbatim window and a recap line.
    state, _ = make_env(tmp_path)
    state.project.append_chat([
        {"role": "user", "content": "old talk", "section": ""},
        {"role": "assistant", "content": "older"}])
    state.project.write_chat_divider(sessionlog.divider_record(
        "idle", 1, [], True))
    sitting = Session(state)
    assert sitting.history[1:] == []
    assert sitting.recap.startswith("Last sitting ended (idle)")
    assert sitting.recap.endswith("— distilled.")


def test_recap_event_reaches_the_first_tab(tmp_path):
    state = ws_state(tmp_path)
    state.project.append_chat([{"role": "user", "content": "x",
                                "section": ""}])
    state.project.write_chat_divider(sessionlog.divider_record(
        "shutdown", 1, [], True))
    with TestClient(create_app(state=state)) as client:
        with client.websocket_connect("/ws/voice?client=c1") as ws:
            assert ws.receive_json()["type"] == "hello"
            ev = ws.receive_json()
            assert ev["type"] == "recap"
            assert ev["text"].startswith("Last sitting ended (shutdown)")


def test_hello_carries_the_sitting_anchor(tmp_path):
    # §8 resync: the client's anchor is view state — hello re-sends the
    # server's truth, so a restart clears a stale one and a blip keeps it.
    state = ws_state(tmp_path)
    with TestClient(create_app(state=state)) as client:
        with client.websocket_connect("/ws/voice?client=c1") as ws:
            assert ws.receive_json() == {"type": "hello",
                                         "client_id": "c1",
                                         "section": None}
            ws.send_json({"type": "select_section",
                          "section": "sections/intro.tex"})
            assert ws.receive_json()["type"] == "section_selected"
        # tab blip: the sitting survived, and so does its anchor in hello
        with client.websocket_connect("/ws/voice?client=c1") as ws:
            hello = ws.receive_json()
            assert hello["type"] == "hello"
            assert hello["section"] == "sections/intro.tex"


def test_pending_diffs_reappear_on_reopen(tmp_path):
    # §7/§8: a proposed diff is disk truth — the next open re-presents
    # it through the reconcile path (card back, approval window re-armed
    # so voice can resolve it).
    state = ws_state(tmp_path)
    state.llm = ScriptLlm([tool_step("section_write", WRITE),
                           text_step("Proposed a tightening.")])
    seed_gists(state)
    with TestClient(create_app(state=state)) as client:
        with client.websocket_connect("/ws/voice?client=c1") as ws:
            ws.receive_json()
            ws.send_json({"type": "typed",
                          "text": "tighten the intro line"})
            while ws.receive_json()["type"] != "assistant_text":
                pass
        state.sitting = None  # the crash shape: sitting gone, no divider
        with client.websocket_connect("/ws/voice?client=c2") as ws:
            assert ws.receive_json()["type"] == "hello"
            # the crash recap rides first, then the re-presented card
            assert ws.receive_json()["text"].startswith(
                "Previous session ended abruptly")
            ev = ws.receive_json()
            assert ev["type"] == "diff" and ev["diff_id"] == "0000"
        # (inside the app: lifespan shutdown would end this sitting)
        assert state.sitting.pending_diffs == [
            {"diff_id": "0000", "section": "sections/intro.tex"}]


def test_heartbeat_does_not_resurrect_an_ended_sitting(tmp_path):
    # §7: ends are ends. Liveness is liveness — a pong must not open a
    # new sitting on a project nobody has re-opened.
    state = ws_state(tmp_path)
    with TestClient(create_app(state=state)) as client:
        with client.websocket_connect("/ws/voice?client=c1") as ws:
            assert ws.receive_json()["type"] == "hello"
            state.sitting = None  # the idle watch ended it
            ws.send_json({"type": "heartbeat", "rms": 0.1})
            assert ws.receive_json()["type"] == "pong"
            assert state.sitting is None


@pytest.mark.anyio
async def test_end_sitting_writes_divider_and_sends_receipt(tmp_path):
    state, session = make_env(tmp_path)
    seed_gists(state)
    events = []

    async def capture(event):
        events.append(event)

    session.send = capture
    state.sitting = session
    await session.run_turn("hi there")
    await end_sitting(state, "idle")
    assert state.sitting is None
    receipt = [e for e in events if e["type"] == "session_ended"]
    assert receipt and receipt[0]["reason"] == "idle"
    div = state.project.chat_divider()
    assert div["reason"] == "idle" and div["turns"] == 1
    assert div["distilled"] is True  # RecordingLlm answered
    assert state.project.chat_tail() == []  # verbatim window closed


@pytest.mark.anyio
async def test_distill_fault_keeps_the_tail_for_retry(tmp_path):
    # vLLM down at the end: the divider lands UNdistilled, so read_tail
    # does not close the verbatim window — the next sitting resumes the
    # talk and its end retries the fold (the promise the fault comments
    # make; §8 degrade, never lose).
    state, session = make_env(tmp_path)
    seed_gists(state)
    events = []

    async def capture(event):
        events.append(event)

    session.send = capture
    state.sitting = session
    await session.run_turn("hi there")

    class DownLlm:
        async def chat(self, messages, **kwargs):
            raise RuntimeError("vLLM down")

    state.llm = DownLlm()
    await end_sitting(state, "shutdown")
    div = state.project.chat_divider()
    assert div["distilled"] is False
    assert state.project.chat_tail()  # window open: the talk is retryable
    receipt = [e for e in events if e["type"] == "session_ended"]
    assert receipt[0]["recap"].endswith("distillation was skipped.")
    nxt = Session(state)
    assert nxt.recap.startswith("Last sitting's distillation was skipped")
    assert any(m.get("content") == "hi there" for m in nxt.history)


@pytest.mark.anyio
async def test_empty_model_answer_is_not_distilled(tmp_path):
    # An empty answer saved nothing — the divider must not claim
    # "distilled" (the same shape distill_memory already refuses).
    from phd_helper.server.app import distill_summaries
    state, _ = make_env(tmp_path)

    class EmptyLlm:
        async def chat(self, messages, **kwargs):
            return {"role": "assistant", "content": ""}, [], "   "

    state.llm = EmptyLlm()
    conv = [{"role": "user", "content": "talk",
             "section": "sections/intro.tex"}]
    assert await distill_summaries(state, conv) is False


@pytest.mark.anyio
async def test_message_during_end_waits_for_the_end(tmp_path):
    # The end race: a message arriving mid-distillation must not spawn a
    # sitting bound to the pre-end log — it waits, then opens the next
    # sitting on the closed (divided) file.
    state, session = make_env(tmp_path)
    seed_gists(state)
    state.sitting = session
    await session.run_turn("hi there")

    class SlowLlm:
        async def chat(self, messages, **kwargs):
            await asyncio.sleep(0.05)  # distillation in flight
            return {"role": "assistant", "content": "ok"}, [], "ok"

    state.llm = SlowLlm()
    end = asyncio.create_task(end_sitting(state, "idle"))
    await asyncio.sleep(0.01)  # the end is mid-await
    fresh = await ensure_sitting(state)
    assert fresh is not session
    assert state.project.chat_divider() is not None  # end completed first
    assert fresh.recap.startswith("Last sitting ended (idle)")
    await end


def test_detach_is_identity_checked(tmp_path):
    # A refresh reuses the client_id; the old socket's close must not
    # evict the new one from the fan-out.
    state, session = make_env(tmp_path)
    ws1, ws2 = object(), object()
    session.attach("c", ws1)
    session.attach("c", ws2)  # refresh: same id, new socket
    session.detach("c", ws1)  # the OLD handler's finally
    assert session.sockets.get("c") is ws2


@pytest.mark.anyio
async def test_idle_watch_survives_a_failing_end(tmp_path, monkeypatch):
    # §8: one locked file must not kill the watchdog for the server's
    # life — no sitting would ever idle-end again.
    import phd_helper.server.app as app_mod
    state, session = make_env(tmp_path)
    state.config = SimpleNamespace(session_idle_s=0.0)
    state.sitting = session
    tried = []

    async def boom(state, reason):
        tried.append(reason)
        raise RuntimeError("locked file")

    monkeypatch.setattr(app_mod, "end_sitting", boom)
    task = asyncio.create_task(app_mod.idle_watch(state, check_s=0.01))
    await asyncio.sleep(0.05)
    assert tried  # it fired
    assert not task.done()  # and the watchdog lived to fire again
    task.cancel()


@pytest.mark.anyio
async def test_turn_activity_moves_the_idle_clock(tmp_path):
    # §7: the idle clock rides turns, not the 2 s heartbeat — an open
    # tab with an absent writer is an idle sitting.
    state, session = make_env(tmp_path)
    seed_gists(state)
    session.last_active = 0.0
    await session.run_turn("hello")
    assert session.last_active > 0.0


def test_diff_resolution_fans_out_to_every_tab(tmp_path):
    # #27: the sitting speaks to all attached tabs — a button click in
    # one tab resolves the card in the other.
    state = ws_state(tmp_path)
    state.llm = ScriptLlm([
        tool_step("section_write", WRITE),
        text_step("Proposed a tightening.")])
    seed_gists(state)
    with TestClient(create_app(state=state)) as client:
        with client.websocket_connect("/ws/voice?client=c1") as a, \
             client.websocket_connect("/ws/voice?client=c2") as b:
            assert a.receive_json()["type"] == "hello"
            assert b.receive_json()["type"] == "hello"
            a.send_json({"type": "typed", "text": "tighten the intro line"})
            for ws in (a, b):  # both tabs see the turn and the diff
                while ws.receive_json()["type"] != "diff":
                    pass
                while ws.receive_json()["type"] != "assistant_text":
                    pass
            b.send_json({"type": "reject", "section": "sections/intro.tex",
                         "diff_id": "0000"})
            assert b.receive_json()["type"] == "diff_resolved"
            assert a.receive_json()["type"] == "diff_resolved"


def test_project_switch_ends_the_sitting_and_lands_there(tmp_path):
    state = ws_state(tmp_path)
    other = tmp_path / "other-paper"
    (other / "sections").mkdir(parents=True)
    (other / "main.tex").write_text(
        "\begin{document}\n\end{document}\n", encoding="utf-8")
    state.projects_root = tmp_path
    with TestClient(create_app(state=state)) as client:
        with client.websocket_connect("/ws/voice?client=c1") as ws:
            ws.receive_json()
            ws.send_json({"type": "typed", "text": "hi"})
            while ws.receive_json()["type"] != "assistant_text":
                pass
        listing = client.get("/projects").json()
        assert set(listing["projects"]) == {"my-paper", "other-paper"}
        assert listing["active"] == "my-paper"
        assert client.post("/projects/activate",
                           json={"name": "other-paper"}).json() \
            == {"active": "other-paper"}
    assert state.project.root.name == "other-paper"
    assert state.sitting is None  # the switch ended it
    # The divider and the distillation rode the OLD project (§7: switch
    # ends the session, distillation runs — before the swap).
    old = Project(tmp_path / "my-paper")
    assert old.chat_divider()["reason"] == "switch"
    assert old.load_memory() == "ok"
    assert (tmp_path / ".phd-helper-server" / "last_project").is_file()


def test_project_activate_rejects_traversal_and_strangers(tmp_path):
    state = ws_state(tmp_path)
    state.projects_root = tmp_path
    with TestClient(create_app(state=state)) as client:
        assert client.post("/projects/activate",
                           json={"name": "../secrets"}).status_code == 404
        assert client.post("/projects/activate",
                           json={"name": "nope"}).status_code == 404
    assert state.project.root.name == "my-paper"  # unchanged


# -- section_create card + file doors + project doors (issue #28) ---------

CREATE = {"section": "sections/related.tex",
          "content": "\\section{Related Work}\nPrior work.\n",
          "after": "sections/intro.tex"}


def test_create_diff_card_carries_created_and_approves(tmp_path):
    state = ws_state(tmp_path)
    state.llm = ScriptLlm([tool_step("section_create", CREATE),
                           text_step("Proposed the new part.")])
    seed_gists(state)
    with TestClient(create_app(state=state)) as client:
        with client.websocket_connect("/ws/voice?client=c1") as ws:
            ws.receive_json()
            ws.send_json({"type": "typed", "text": "add a related work part"})
            while (card := ws.receive_json())["type"] != "diff":
                pass
            assert card["section"] == "main.tex"  # the wiring is the patch
            assert card["created"] == {"path": "sections/related.tex",
                                       "content": CREATE["content"]}
            ws.send_json({"type": "approve", "section": card["section"],
                          "diff_id": card["diff_id"]})
            while (res := ws.receive_json())["type"] != "diff_resolved":
                pass  # the turn's closing assistant_text may land first
            assert res["applied"]
    root = state.project.root
    assert (root / "sections" / "related.tex").is_file()
    assert ("\\input{sections/intro}\n\\input{sections/related}"
            in (root / "main.tex").read_text(encoding="utf-8"))


def test_pending_create_replays_with_created_on_reconnect(tmp_path):
    # §7/§8 reopen: a create diff that outlives the tab comes back as a
    # card with its file bytes, not just the wiring line.
    state = ws_state(tmp_path)
    state.llm = ScriptLlm([tool_step("section_create", CREATE),
                           text_step("Proposed the new part.")])
    seed_gists(state)
    with TestClient(create_app(state=state)) as client:
        with client.websocket_connect("/ws/voice?client=c1") as ws:
            ws.receive_json()
            ws.send_json({"type": "typed", "text": "add a related work part"})
            while ws.receive_json()["type"] != "diff":
                pass
        with client.websocket_connect("/ws/voice?client=c2") as ws2:
            ws2.receive_json()  # hello
            card = ws2.receive_json()
            assert card["type"] == "diff"
            assert card["created"]["path"] == "sections/related.tex"


def test_file_upload_lands_wired_at_the_end(tmp_path):
    state = ws_state(tmp_path)
    with TestClient(create_app(state=state)) as client:
        r = client.post("/project/files?name=method.tex",
                        content=b"We measure things.\n")
        assert r.status_code == 200
        assert r.json() == {"path": "sections/method.tex"}
        main = (state.project.root / "main.tex").read_text(encoding="utf-8")
        assert "\\input{sections/method}\n\\end{document}" in main
        linked = {f["path"]: f["linked"]
                  for f in client.get("/project/files").json()["files"]}
        assert linked["sections/method.tex"] is True


def test_file_upload_guards(tmp_path):
    state = ws_state(tmp_path)
    with TestClient(create_app(state=state)) as client:
        assert client.post("/project/files?name=../evil.tex",
                           content=b"x").status_code == 400
        assert client.post("/project/files?name=notes.txt",
                           content=b"x").status_code == 400
        assert client.post("/project/files?name=intro.tex",  # exists
                           content=b"x").status_code == 409
    assert not (state.project.root / "sections" / "intro.tex"
                ).read_text(encoding="utf-8").startswith("x")


def test_files_listing_marks_linked_and_skips_the_spine(tmp_path):
    state = ws_state(tmp_path)
    (state.project.root / "sections" / "orphan.tex").write_text(
        "x\n", encoding="utf-8")
    with TestClient(create_app(state=state)) as client:
        files = client.get("/project/files").json()["files"]
    by = {f["path"]: f["linked"] for f in files}
    assert by["sections/intro.tex"] is True
    assert by["sections/orphan.tex"] is False
    assert "main.tex" not in by  # the spine is not a file-door target


def test_file_remove_is_soft_and_restorable(tmp_path):
    state = ws_state(tmp_path)
    original = (state.project.root / "sections" / "intro.tex").read_text(
        encoding="utf-8")
    with TestClient(create_app(state=state)) as client:
        assert client.post("/project/files/remove",
                           json={"path": "sections/intro.tex"}
                           ).status_code == 200
        root = state.project.root
        assert not (root / "sections" / "intro.tex").exists()
        main = (root / "main.tex").read_text(encoding="utf-8")
        assert "\\input{sections/intro}" not in main
        listing = client.get("/project/files").json()
        assert listing["trash"] == ["sections/intro.tex"]
        assert client.post("/project/files/restore",
                           json={"path": "sections/intro.tex"}
                           ).status_code == 200
    assert (root / "sections" / "intro.tex").read_text(
        encoding="utf-8") == original
    assert "\\input{sections/intro}" in (
        root / "main.tex").read_text(encoding="utf-8")


def test_file_remove_guards(tmp_path):
    state = ws_state(tmp_path)
    with TestClient(create_app(state=state)) as client:
        assert client.post("/project/files/remove",
                           json={"path": "main.tex"}).status_code == 400
        # Case-insensitive filesystems (the deployment platform): "Main.tex"
        # IS the spine — the guard compares casefolded, not by exact string.
        assert client.post("/project/files/remove",
                           json={"path": "Main.tex"}).status_code == 400
        assert client.post("/project/files/remove",
                           json={"path": "../x.tex"}).status_code == 400
        assert client.post("/project/files/remove",
                           json={"path": "sections/nope.tex"}
                           ).status_code == 404
    assert (state.project.root / "sections" / "intro.tex").exists()
    assert (state.project.root / "main.tex").is_file()  # the spine stays put


def test_projects_new_scaffolds_and_activates(tmp_path):
    state = ws_state(tmp_path)
    state.projects_root = tmp_path
    with TestClient(create_app(state=state)) as client:
        with client.websocket_connect("/ws/voice?client=c1") as ws:
            ws.receive_json()
            ws.send_json({"type": "typed", "text": "hi"})
            while ws.receive_json()["type"] != "assistant_text":
                pass
        assert client.post("/projects/new", json={"name": "next-paper"}).json() \
            == {"active": "next-paper"}
    assert state.project.root.name == "next-paper"
    assert state.sitting is None  # the switch ended the old sitting
    new = tmp_path / "next-paper"
    assert (new / "main.tex").is_file()
    assert (new / "sections" / "intro.tex").is_file()
    assert (new / "refs.bib").is_file()
    assert Project(tmp_path / "my-paper").chat_divider()["reason"] == "switch"


def test_projects_new_guards(tmp_path):
    state = ws_state(tmp_path)
    state.projects_root = tmp_path
    with TestClient(create_app(state=state)) as client:
        assert client.post("/projects/new",
                           json={"name": "my-paper"}).status_code == 409
        assert client.post("/projects/new",
                           json={"name": "../escape"}).status_code == 400
        assert client.post("/projects/new",
                           json={"name": ""}).status_code == 400
    assert state.project.root.name == "my-paper"  # unchanged


def _zip(members: dict) -> bytes:
    import io
    import zipfile
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        for name, body in members.items():
            z.writestr(name, body)
    return buf.getvalue()


def test_projects_import_zip_lands_and_activates(tmp_path):
    state = ws_state(tmp_path)
    state.projects_root = tmp_path
    blob = _zip({"main.tex": "\\begin{document}\n\\input{sections/a}\n"
                             "\\end{document}\n",
                 "sections/a.tex": "\\section{A}\nA.\n",
                 "refs.bib": ""})
    with TestClient(create_app(state=state)) as client:
        assert client.post("/projects/import?name=imported",
                           content=blob).json() == {"active": "imported"}
    assert state.project.root.name == "imported"
    assert (tmp_path / "imported" / "sections" / "a.tex").is_file()


def test_projects_import_accepts_one_top_level_folder(tmp_path):
    state = ws_state(tmp_path)
    state.projects_root = tmp_path
    blob = _zip({"paper/main.tex": "\\begin{document}\n\\end{document}\n",
                 "paper/sections/a.tex": "A.\n"})
    with TestClient(create_app(state=state)) as client:
        assert client.post("/projects/import?name=flat",
                           content=blob).status_code == 200
    assert (tmp_path / "flat" / "main.tex").is_file()
    assert (tmp_path / "flat" / "sections" / "a.tex").is_file()


def test_projects_import_guards(tmp_path):
    state = ws_state(tmp_path)
    state.projects_root = tmp_path
    with TestClient(create_app(state=state)) as client:
        assert client.post("/projects/import?name=slip", content=_zip(
            {"main.tex": "\\end{document}\n", "../evil.tex": "x"}
        )).status_code == 400
        assert not (tmp_path / "evil.tex").exists()
        assert client.post("/projects/import?name=exe", content=_zip(
            {"main.tex": "\\end{document}\n", "tool.exe": "MZ"}
        )).status_code == 400
        assert client.post("/projects/import?name=nohead", content=_zip(
            {"sections/a.tex": "A.\n"})).status_code == 400
        assert client.post("/projects/import?name=my-paper", content=_zip(
            {"main.tex": "\\end{document}\n"})).status_code == 409
        assert client.post("/projects/import?name=../out", content=b"zip"
                           ).status_code == 400
        assert client.post("/projects/import?name=notazip",
                           content=b"junk").status_code == 400
    assert state.project.root.name == "my-paper"  # unchanged
    assert not (tmp_path / "slip").exists()  # refused zips leave nothing


# -- read view + project download (the simple-variant slice) ----------------


def test_document_returns_sections_in_document_order(tmp_path):
    state = ws_state(tmp_path)
    with TestClient(create_app(state=state)) as client:
        doc = client.get("/document").json()
    paths = [s["path"] for s in doc["sections"]]
    assert paths == ["main.tex", "sections/intro.tex"]
    assert doc["sections"][0]["blocks"][0] == {
        "kind": "heading", "level": 0, "text": "My Paper"}
    assert doc["sections"][1]["title"] == "Introduction"
    assert [b["kind"] for b in doc["sections"][1]["blocks"]] == [
        "heading", "paragraph"]


def test_document_order_is_depth_first(tmp_path):
    # main -> [intro, related] with intro -> i2: the nested child lands
    # between its parent and the next sibling, as LaTeX puts it.
    state = ws_state(tmp_path)
    root = state.project.root
    (root / "main.tex").write_text(
        "\\begin{document}\n\\input{sections/intro}\n"
        "\\input{sections/related}\n\\end{document}\n", encoding="utf-8")
    (root / "sections" / "intro.tex").write_text(
        "\\section{I}\n\\input{sections/i2}\n", encoding="utf-8")
    (root / "sections" / "i2.tex").write_text(
        "\\subsection{I2}\n", encoding="utf-8")
    (root / "sections" / "related.tex").write_text(
        "\\section{R}\n", encoding="utf-8")
    with TestClient(create_app(state=state)) as client:
        paths = [s["path"] for s in client.get("/document").json()["sections"]]
    assert paths == ["main.tex", "sections/intro.tex", "sections/i2.tex",
                     "sections/related.tex"]


def test_document_degrades_around_an_unreadable_file(tmp_path):
    # One file the door cannot read (here: a directory wearing a .tex
    # name — read_text raises OSError) costs its own blocks, not the
    # whole view: the rest of the paper still reads.
    state = ws_state(tmp_path)
    root = state.project.root
    (root / "sections" / "broken.tex").mkdir()
    (root / "main.tex").write_text(
        "\\begin{document}\n\\input{sections/intro}\n"
        "\\input{sections/broken}\n\\end{document}\n", encoding="utf-8")
    with TestClient(create_app(state=state)) as client:
        r = client.get("/document")
    assert r.status_code == 200
    by_path = {s["path"]: s for s in r.json()["sections"]}
    assert by_path["sections/broken.tex"]["blocks"] == []
    assert by_path["sections/intro.tex"]["blocks"] != []


def test_projects_download_zip(tmp_path):
    import io
    import zipfile
    state = ws_state(tmp_path)
    state.projects_root = tmp_path
    root = state.project.root
    (root / "refs.bib").write_text("@article{a,\n title={T},\n}\n",
                                   encoding="utf-8")
    (root / "notes.txt").write_text("not a project file", encoding="utf-8")
    (root / ".phd-helper").mkdir()
    (root / ".phd-helper" / "memory.md").write_text("agent state",
                                                    encoding="utf-8")
    with TestClient(create_app(state=state)) as client:
        r = client.get("/projects/download", params={"name": "my-paper"})
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("application/zip")
    assert r.headers["content-disposition"] == \
        'attachment; filename="my-paper.zip"'
    names = sorted(zipfile.ZipFile(io.BytesIO(r.content)).namelist())
    assert names == ["main.tex", "refs.bib", "sections/intro.tex"]


def test_projects_download_guards(tmp_path):
    state = ws_state(tmp_path)
    state.projects_root = tmp_path
    with TestClient(create_app(state=state)) as client:
        assert client.get("/projects/download",
                          params={"name": "nope"}).status_code == 404
        assert client.get("/projects/download",
                          params={"name": "../out"}).status_code == 400
        assert client.get("/projects/download",
                          params={"name": ""}).status_code == 400


def test_download_round_trips_through_import(tmp_path):
    state = ws_state(tmp_path)
    state.projects_root = tmp_path
    (state.project.root / "refs.bib").write_text("", encoding="utf-8")
    with TestClient(create_app(state=state)) as client:
        blob = client.get("/projects/download",
                          params={"name": "my-paper"}).content
        assert client.post("/projects/import?name=copy",
                           content=blob).status_code == 200
    src, copy = tmp_path / "my-paper", tmp_path / "copy"
    assert (copy / "main.tex").read_text(
        encoding="utf-8") == (src / "main.tex").read_text(encoding="utf-8")
    assert (copy / "sections" / "intro.tex").is_file()
    assert (copy / "refs.bib").is_file()
