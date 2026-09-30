"""App assembly at the HTTP seam (SPEC §1): startup/shutdown wiring and
turn-error containment. The voice loop itself is proven end-to-end against
live vLLM (§2); this file covers what only the app owns.
"""

import pytest
from fastapi.testclient import TestClient
from types import SimpleNamespace

from phd_helper.corpus import Corpus
from phd_helper.endpoint import VoiceEndpoint
from phd_helper.project import Project
from phd_helper.server.app import Session, create_app
from phd_helper.server.config import Config
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
                            tts=None, config=Config(), fetch=None,
                            corpus=Corpus(tmp_path / "c"),
                            corpus_store=None, crossref_mailto="",
                            openalex_mailto="",
                            ingest_tasks=set(), gist_task=None)
    session = Session(state, "c1")
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
                                context_budget_tokens=budget),
                            fetch=None, corpus=corpus, corpus_store=store,
                            crossref_mailto="", openalex_mailto="",
                            ingest_tasks=set(), gist_task=None)
    return state, Session(state, "c1")


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


def test_health_reports_paused_without_a_store(tmp_path):
    # No stack configured: indexing pauses, visibly (§8) — unchanged.
    with TestClient(create_app(state=health_state(tmp_path, None))) as c:
        assert c.get("/health").json()["corpus"] == "paused"


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


@pytest.mark.anyio
async def test_disconnect_distills_the_sitting(tmp_path):
    import asyncio
    state, _ = make_env(tmp_path)
    state.endpoint = VoiceEndpoint(ping_interval=2.0, lease_timeout=60.0)
    state.stt = StubStt()
    state.http = FakeHttp()
    state.ingestor = None
    with TestClient(create_app(state=state)) as client:
        with client.websocket_connect("/ws/voice?client=c1") as ws:
            assert ws.receive_json()["type"] == "hello"
            ws.send_json({"type": "typed", "text": "hello there"})
            assert ws.receive_json()["type"] == "turn_started"
    await asyncio.gather(*state.ingest_tasks)  # the distill task
    # RecordingLlm answers "ok" — that became the distilled memory.
    assert state.project.load_memory() == "ok"


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


@pytest.mark.anyio
async def test_disconnect_distills_summaries_too(tmp_path):
    import asyncio
    state, _ = make_env(tmp_path)
    state.endpoint = VoiceEndpoint(ping_interval=2.0, lease_timeout=60.0)
    state.stt = StubStt()
    state.http = FakeHttp()
    state.ingestor = None
    with TestClient(create_app(state=state)) as client:
        with client.websocket_connect("/ws/voice?client=c1") as ws:
            assert ws.receive_json()["type"] == "hello"
            ws.send_json({"type": "select_section",
                          "section": "sections/intro.tex"})
            ws.receive_json()
            ws.send_json({"type": "typed", "text": "tighten it"})
            assert ws.receive_json()["type"] == "turn_started"
    await asyncio.gather(*state.ingest_tasks)
    # RecordingLlm answers "ok"; the exchange was anchored, so intro
    # got a dated summary entry.
    assert state.project.load_summaries()["sections/intro.tex"][-1]["text"] \
        == "ok"
