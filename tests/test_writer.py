"""Background writer (docs/background-writer.md): a second agent session
drafting paper text while the conversation keeps talking. The writer is
silent — its only outputs are the §5 approval cards (the same gate the
voice loop's patches ride) and the notices that name them. This file
proves the loop seam (fake LLM), the one-at-a-time guard, the project
guard, the restricted toolset, and the delegation tool. The harness is
test_app's shape restated here (house idiom: repeated setup over
cross-file imports)."""

import asyncio
import json

import pytest
from fastapi.testclient import TestClient
from types import SimpleNamespace

from phd_helper.corpus import Corpus
from phd_helper.endpoint import VoiceEndpoint
from phd_helper.gists import body_sha
from phd_helper.project import Project
from phd_helper.server.app import (Session, WRITER_OFFERED, WRITER_TOOLS,
                                   create_app, run_writer, spawn_writer)
from phd_helper.server.config import Config
from phd_helper.server.llm import LlmError
from phd_helper.server.voice import StubTts
from phd_helper.toolcall import ValidCall
from phd_helper.tools import TOOL_SCHEMAS, execute_async, make_validators


@pytest.fixture
def anyio_backend():
    return "asyncio"


class FakeHttp:
    async def aclose(self):
        pass


class ScriptLlm:
    def __init__(self, script):
        self.script = list(script)
        self.calls: list[list[dict]] = []

    async def chat(self, messages, **kwargs):
        self.calls.append([dict(m) for m in messages])
        return self.script.pop(0) if self.script else text_step("(end)")


# The routing marker only WRITER_PROMPT carries — SYSTEM_PROMPT (and the
# §4 context riding it) must not read as a writer leg.
WRITER_MARKER = "phd-helper's background writer"


class RoutingLlm:
    """Two scripts, routed by the system message: the conversation and
    the writer run concurrently and would consume a shared script in a
    nondeterministic order."""

    def __init__(self, talker, writer):
        self.talker = ScriptLlm(talker)
        self.writer = ScriptLlm(writer)

    async def chat(self, messages, **kwargs):
        if WRITER_MARKER in messages[0]["content"]:
            return await self.writer.chat(messages, **kwargs)
        return await self.talker.chat(messages, **kwargs)


def tool_step(name, args, cid="t1"):
    return ({"role": "assistant", "content": None,
             "tool_calls": [{"id": cid, "type": "function",
                             "function": {"name": name,
                                          "arguments": json.dumps(args)}}]},
            [ValidCall(id=cid, name=name, args=args)], "")


def text_step(text):
    return {"role": "assistant", "content": text}, [], text


WRITE = {"section": "sections/intro.tex", "find": "Intro body prose.",
         "replace": "Intro prose, tightened."}


def seed_gists(state):
    """The talker's _build_context kicks a gist refresh that would steal
    script steps (test_app's lesson) — seed a fresh cache."""
    state.project.save_gists({p: {"sha": body_sha(b), "gist": "seeded"}
                              for p, b in state.project.files().items()})


def writer_env(tmp_path, script, sitting=True):
    """Paper + state for the writer seam: the writer reads section,
    skeleton and memory directly (no _build_context, so no gist steps to
    defend). ``sitting`` attaches one whose send() captures events —
    writer fan-out is guarded on it exactly like the librarian notices."""
    paper = tmp_path / "my-paper"
    (paper / "sections").mkdir(parents=True)
    (paper / "main.tex").write_text(
        "\\title{My Paper}\n\\begin{document}\n"
        "\\input{sections/intro}\n\\end{document}\n", encoding="utf-8")
    (paper / "sections" / "intro.tex").write_text(
        "\\section{Introduction}\nIntro body prose.\n", encoding="utf-8")
    state = SimpleNamespace(
        project=Project(paper), llm=ScriptLlm(script), tts=StubTts(),
        config=Config(stream_tts=False), fetch=None,
        corpus=Corpus(tmp_path / "c"), corpus_store=None,
        crossref_mailto="", openalex_mailto="", ingestor=None,
        ingest_tasks=set(), gist_task=None, sitting=None,
        endpoint=VoiceEndpoint(), git_run=None, http=FakeHttp(),
        writer_task=None, writer_info=None)
    events: list[dict] = []

    async def capture(event):
        events.append(event)

    session = None
    if sitting:
        session = Session(state)
        session.send = capture
        state.sitting = session
    return state, session, events


# -- the writer loop ----------------------------------------------------------


@pytest.mark.anyio
async def test_the_writer_drafts_into_an_approval_card(tmp_path):
    state, session, events = writer_env(tmp_path, [
        tool_step("section_write", WRITE, "w1"),
        text_step("Tightened the opening."),
    ])
    await run_writer(state, state.project, "sections/intro.tex",
                     "tighten the opening line")

    card = next(e for e in events if e["type"] == "diff")
    assert card["section"] == "sections/intro.tex"
    assert card["find"] == WRITE["find"]
    assert card["replace"] == WRITE["replace"]
    # The card enters the sitting's approval window: the next utterance
    # can decide it by voice exactly like a card the talker proposed.
    assert [(t["diff_id"], t["section"]) for t in session.pending_diffs] \
        == [(card["diff_id"], "sections/intro.tex")]
    # Disk truth, applying through the ordinary §5 door.
    r = state.project.apply_pending("sections/intro.tex", card["diff_id"])
    assert r.applied
    assert "Intro prose, tightened." in \
        state.project.read_section("sections/intro.tex")

    notices = [e["text"] for e in events if e["type"] == "notice"]
    assert any("drafting" in n.lower() for n in notices)   # opened
    assert any("1 patch" in n for n in notices)            # closed honest


@pytest.mark.anyio
async def test_the_ladder_reads_the_brief_section_and_paper(tmp_path):
    state, _, _ = writer_env(tmp_path, [text_step("Nothing to change.")])
    await run_writer(state, state.project, "sections/intro.tex",
                     "tighten the opening, keep it two sentences")
    sent = state.llm.calls[0]
    assert WRITER_MARKER in sent[0]["content"]  # the writer's own prompt
    user = sent[1]["content"]
    assert "tighten the opening" in user        # the brief
    assert "Intro body prose." in user          # the target body
    assert "intro" in user.lower()              # the skeleton names it
    # One conversation of its own: the talker's chat JSONL stays clean.
    assert state.project.chat_tail() == []


@pytest.mark.anyio
async def test_the_writer_only_gets_the_drafting_tools(tmp_path):
    # No approval, no voice, no user-hands tools: the writer's toolset is
    # propose-only. pending_decide would rubber-stamp, memory/git/undo
    # are the user's hands, and delegate_write would recurse.
    for banned in ("pending_decide", "memory_write", "git_commit",
                   "undo_last", "delegate_write"):
        assert banned not in WRITER_OFFERED
    names = {s["function"]["name"] for s in WRITER_TOOLS}
    assert {"section_read", "section_write", "section_create", "cite_add",
            "web_search", "corpus_search", "corpus_doc"} == names


@pytest.mark.anyio
async def test_a_second_delegation_bounces_while_one_runs(tmp_path):
    state, _, _ = writer_env(tmp_path, [])
    gate = asyncio.Event()
    state.writer_task = asyncio.create_task(gate.wait())
    try:
        assert spawn_writer(state, state.project, "sections/intro.tex",
                            "anything") is False
    finally:
        gate.set()
        await state.writer_task
    # A finished run frees the slot: running truth is the task, not a
    # stored flag — a crashed writer can never keep the door closed.
    assert spawn_writer(state, state.project, "sections/intro.tex",
                        "later") is True
    await state.writer_task


@pytest.mark.anyio
async def test_a_project_switch_mutes_the_writer_but_keeps_its_work(
        tmp_path):
    state, _session, events = writer_env(tmp_path, [
        tool_step("section_write", WRITE, "w1"),
        text_step("Done."),
    ])
    old = state.project
    (tmp_path / "other").mkdir()
    state.project = Project(tmp_path / "other")  # switched mid-pass

    await run_writer(state, old, "sections/intro.tex", "tighten")

    # Nothing speaks into the new paper's conversation (§7)…
    assert events == []
    # …but the proposal is disk truth for the paper it was drafted for.
    assert len(old.pending.list_all()) == 1


@pytest.mark.anyio
async def test_a_model_fault_notices_and_stops(tmp_path):
    state, _, events = writer_env(tmp_path, [])

    class Boom(ScriptLlm):
        async def chat(self, messages, **kwargs):
            raise LlmError("vLLM 500: no capacity")

    state.llm = Boom([])
    await run_writer(state, state.project, "sections/intro.tex", "go")
    notices = [e["text"] for e in events if e["type"] == "notice"]
    assert any("model" in n for n in notices)  # the writer names its fault
    assert state.project.pending.list_all() == []  # proposes nothing


@pytest.mark.anyio
async def test_no_sitting_no_fanout_the_diff_still_lands(tmp_path):
    # The sitting is decoration: a writer running between sittings (or
    # after a switch) still drafts — its cards re-present through the
    # next attach's reconcile, because disk is the truth (§7/§8).
    state, _session, events = writer_env(tmp_path, [
        tool_step("section_write", WRITE, "w1"),
        text_step("Done."),
    ], sitting=False)
    await run_writer(state, state.project, "sections/intro.tex", "tighten")
    assert events == []
    assert len(state.project.pending.list_all()) == 1


# -- delegation through the live conversation ----------------------------------


@pytest.mark.anyio
async def test_delegation_lets_the_writer_run_while_talking_continues(
        tmp_path):
    # The feature's shape end-to-end at the seam: one talker turn spawns
    # the writer and RETURNS; the writer drafts on its own leg, and its
    # card lands in the same sitting's window afterwards.
    state, session, events = writer_env(tmp_path, [])
    state.llm = RoutingLlm(
        talker=[
            tool_step("delegate_write",
                      {"section": "sections/intro.tex",
                       "brief": "tighten the opening line"}),
            text_step("Drafting it in the background."),
        ],
        writer=[
            tool_step("section_write", WRITE, "w1"),
            text_step("Tightened the opening."),
        ])
    seed_gists(state)

    await session.run_turn("get the intro tightened while we keep talking")
    # The talker is done (its turn ended on the delegation paperwork);
    # the writer may still be mid-flight — awaiting it is the test's
    # join, not the conversation's.
    assert state.writer_task is not None
    await state.writer_task

    assert any(e["type"] == "assistant_text"
               and "background" in e["text"] for e in events)
    card = next(e for e in events if e["type"] == "diff")
    assert card["find"] == WRITE["find"]
    assert len(state.llm.writer.calls) >= 1  # the writer rode its own leg
    r = state.project.apply_pending("sections/intro.tex", card["diff_id"])
    assert r.applied


# -- the delegate_write tool ----------------------------------------------------


@pytest.mark.anyio
async def test_delegate_write_without_a_writer_wiring_bounces(tmp_path):
    # execute_async's default kwargs (a mis-wired caller): the model gets
    # a bounce, never a TypeError.
    state, _, _ = writer_env(tmp_path, [])
    r = await execute_async(
        ValidCall(id="t", name="delegate_write",
                  args={"section": "sections/intro.tex", "brief": "x"}),
        state.project)
    assert "error" in r


@pytest.mark.anyio
async def test_delegate_write_starts_one_and_the_second_says_already(
        tmp_path):
    state, _, _ = writer_env(tmp_path, [])
    seen = []

    def spawn(section, brief):
        seen.append((section, brief))
        return True

    r = await execute_async(
        ValidCall(id="t1", name="delegate_write",
                  args={"section": "sections/intro.tex", "brief": "x"}),
        state.project, spawn_writer=spawn)
    assert r["status"] == "delegated"
    assert seen == [("sections/intro.tex", "x")]

    r = await execute_async(
        ValidCall(id="t2", name="delegate_write",
                  args={"section": "sections/intro.tex", "brief": "again"}),
        state.project, spawn_writer=lambda section, brief: False)
    assert "already" in r["error"]


def test_delegate_write_schema_is_strict():
    s = next(s for s in TOOL_SCHEMAS
             if s["function"]["name"] == "delegate_write")
    p = s["function"]["parameters"]
    assert p["additionalProperties"] is False
    assert set(p["required"]) == set(p["properties"])  # OpenAI strict


def test_the_brief_validators(tmp_path):
    state, _, _ = writer_env(tmp_path, [])
    v = make_validators(state.project)["delegate_write"]
    assert v({"section": "sections/intro.tex", "brief": "go"}) is None
    assert "does not exist" in v({"section": "sections/nope.tex",
                                  "brief": "go"})
    assert "brief" in v({"section": "sections/intro.tex", "brief": "  "})


# -- the status door -------------------------------------------------------------


class NotDone:
    """The door only asks done() (the related-panel guard's shape)."""

    def done(self):
        return False


def test_the_writer_door_answers_idle_then_running(tmp_path):
    state, _, _ = writer_env(tmp_path, [])
    with TestClient(create_app(state=state)) as client:
        assert client.get("/project/writer").json() == {
            "running": False, "section": "", "brief": ""}
        state.writer_task = NotDone()
        state.writer_info = {"section": "sections/related.tex",
                             "brief": "cover retrieval papers",
                             "at": "2026-10-30 12:00"}
        assert client.get("/project/writer").json() == {
            "running": True, "section": "sections/related.tex",
            "brief": "cover retrieval papers"}
