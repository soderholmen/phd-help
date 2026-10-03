# chat_stream: the SSE leg of the sentence-level TTS slice. MockTransport
# carries canned SSE bodies; the events the generator yields are the
# contract run_turn pumps sentences from. (LlmClient had no direct tests
# before this — chat() stays covered through the app-level fakes.)
import json

import httpx
import pytest

from phd_helper.server.config import Config
from phd_helper.server.llm import LlmClient, LlmError


@pytest.fixture
def anyio_backend():
    return "asyncio"


def sse(chunks):
    body = "".join(f"data: {json.dumps(c)}\n\n" for c in chunks)
    return (body + "data: [DONE]\n\n").encode()


def delta(content):
    return {"choices": [{"delta": {"content": content}}]}


def tool_delta(fragment):
    return {"choices": [{"delta": {"tool_calls": [fragment]}}]}


def make_client(chunks_per_request, captured=None):
    """One canned SSE body per POST, in order; captures request bodies."""
    state = {"n": 0}

    def handler(request):
        if captured is not None:
            captured.append(json.loads(request.content))
        i = min(state["n"], len(chunks_per_request) - 1)
        state["n"] += 1
        return httpx.Response(200, content=sse(chunks_per_request[i]))

    http = httpx.AsyncClient(base_url="http://llm.test",
                             transport=httpx.MockTransport(handler))
    return LlmClient(Config(), http=http)


async def collect(client, **kw):
    return [ev async for ev in client.chat_stream(
        [{"role": "user", "content": "hi"}], **kw)]


@pytest.mark.anyio
async def test_chat_stream_yields_text_deltas_then_done():
    captured = []
    client = make_client([[delta("Hel"), delta("lo"),
                           {"choices": [{"delta": {},
                                         "finish_reason": "stop"}]}]],
                         captured)
    events = await collect(client)
    assert [e[0] for e in events] == ["text", "text", "done"]
    assert events[0][1] == "Hel" and events[1][1] == "lo"
    _, msg, valid, text = events[-1]
    assert msg == {"role": "assistant", "content": "Hello"}
    assert valid == [] and text == "Hello"
    assert captured[0]["stream"] is True          # the slice's whole point
    assert captured[0]["messages"][0]["content"] == "hi"


@pytest.mark.anyio
async def test_chat_stream_ignores_reasoning_deltas():
    # vLLM templates carry thinking as delta.reasoning (newer) or
    # delta.reasoning_content (older); neither may reach the gate or
    # the assembled message.
    client = make_client([[
        {"choices": [{"delta": {"reasoning": "think ",
                                "reasoning_content": "hmm"}}]},
        delta("Hi"),
        {"choices": [{"delta": {"reasoning_content": "more thinking"}}]},
    ]])
    events = await collect(client)
    assert [e[0] for e in events] == ["text", "done"]
    _, msg, _, text = events[-1]
    assert text == "Hi"
    assert "reasoning" not in msg and "reasoning_content" not in msg


@pytest.mark.anyio
async def test_chat_stream_stops_text_on_tool_calls_and_reassembles():
    # vLLM fragments tool calls across deltas: id/name arrive with the
    # first, arguments string-concatenate by index. Text after the first
    # fragment is suppressed from the event stream (the message keeps it).
    client = make_client([[
        delta("Let me "), delta("check."),
        tool_delta({"index": 0, "id": "call_1",
                    "function": {"name": "noop", "arguments": "{\"a"}}),
        tool_delta({"index": 0, "function": {"arguments": "\": 1}"}}),
        delta(" trailing after calls"),
    ]])
    events = await collect(client, offered={"noop": {}})
    assert [e[0] for e in events] == ["text", "text", "toolcalls", "done"]
    _, msg, valid, text = events[-1]
    assert msg["tool_calls"] == [{
        "id": "call_1", "type": "function",
        "function": {"name": "noop", "arguments": '{"a": 1}'}}]
    assert text == "Let me check. trailing after calls"
    assert [(v.name, v.args) for v in valid] == [("noop", {"a": 1})]


@pytest.mark.anyio
async def test_chat_stream_bounces_invalid_calls_then_succeeds():
    # First attempt: a call missing a required param. The bounce replays
    # the assistant turn and tells the model what failed — the same §2
    # contract chat() has, with a ("restart",) event so the caller
    # drops its sentence gate.
    captured = []
    client = make_client([
        [delta("Sure. "),
         tool_delta({"index": 0, "id": "c1",
                     "function": {"name": "patch", "arguments": "{}"}})],
        [delta("Recovered.")],
    ], captured)
    events = await collect(client, offered={"patch": {"find": "string"}})
    kinds = [e[0] for e in events]
    assert kinds == ["text", "toolcalls", "restart", "text", "done"]
    assert len(captured) == 2
    replayed = captured[1]["messages"]
    assert replayed[-2]["role"] == "assistant"       # the bounced turn
    assert "[tool-call validation]" in replayed[-1]["content"]
    assert "find" in replayed[-1]["content"]          # names the failure
    assert events[-1][3] == "Recovered."


@pytest.mark.anyio
async def test_chat_stream_raises_llm_error_on_non_200():
    def handler(request):
        return httpx.Response(500, text="engine crashed")

    http = httpx.AsyncClient(base_url="http://llm.test",
                             transport=httpx.MockTransport(handler))
    client = LlmClient(Config(), http=http)
    with pytest.raises(LlmError) as exc:
        [e async for e in client.chat_stream([{"role": "user",
                                               "content": "hi"}])]
    assert "vLLM 500" in str(exc.value)


@pytest.mark.anyio
async def test_chat_stream_done_message_passes_validate_tool_calls_unchanged():
    # A per-tool validator rejection bounces exactly like chat()'s path;
    # the accepted attempt hands back ValidCall objects unchanged.
    seen_args = []

    def check(args):
        seen_args.append(args)
        return "anchor missing" if args["find"] == "gone" else None

    client = make_client([
        [tool_delta({"index": 0, "id": "c1",
                     "function": {"name": "patch",
                                  "arguments": '{"find": "gone"}'}})],
        [tool_delta({"index": 0, "id": "c2",
                     "function": {"name": "patch",
                                  "arguments": '{"find": "here"}'}})],
    ])
    events = await collect(client, offered={"patch": {"find": "string"}},
                           validators={"patch": check})
    _, msg, valid, _ = events[-1]
    assert seen_args == [{"find": "gone"}, {"find": "here"}]
    assert [(v.id, v.args) for v in valid] == [("c2", {"find": "here"})]
