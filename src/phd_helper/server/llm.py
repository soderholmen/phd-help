"""vLLM chat client (SPEC §2).

Two legs: chat() is the one-shot (the §2 preference for tool-decision
turns, the internal thinking=False calls, and the PHD_STREAM_TTS=0
fallback); chat_stream() is the SSE leg the sentence-level TTS slice
pumps prose from. Both carry the same contract: mandatory validation of
every returned tool call, bounce-back retry on failure (validation
errors go back to the model, never to the user), never
tool_choice="required".
"""

import json

import httpx

from phd_helper.toolcall import validate_tool_calls

MAX_BOUNCES = 2


class LlmError(Exception):
    pass


def _request_body(config, msgs, body_tools, max_tokens, thinking,
                  stream=False):
    body = {"model": config.llm_model, "messages": msgs,
            "max_tokens": max_tokens, **config.sampling(thinking)}
    if stream:
        body["stream"] = True
    if body_tools:
        body["tools"] = body_tools
        body["tool_choice"] = "auto"
    return body


def _bounce(msgs, msg, errors):
    # Bounce: replay the assistant turn, then tell the model what
    # failed (SPEC §2: never surface to the user). Both legs share the
    # exact wording — it is tuned against the live model.
    msgs.append(msg)
    for e in errors:
        msgs.append({"role": "user",
                     "content": f"[tool-call validation] {e}. "
                                "Re-issue the tool call correctly."})


class LlmClient:
    def __init__(self, config, http: httpx.AsyncClient | None = None):
        self.config = config
        headers = {"Content-Type": "application/json"}
        if config.llm_api_key:
            headers["Authorization"] = f"Bearer {config.llm_api_key}"
        self._http = http or httpx.AsyncClient(
            base_url=config.llm_base_url, headers=headers, timeout=httpx.Timeout(
                300.0, connect=10.0))

    async def healthy(self) -> bool:
        try:
            r = await self._http.get("/v1/models", timeout=5.0)
            return r.status_code == 200
        except httpx.HTTPError:
            return False

    async def chat(self, messages, tools=None, offered=None,
                   validators=None, thinking=True, max_tokens=4096):
        """One agent turn. Returns (assistant_message, valid_calls, text).

        Rejected tool calls are bounced back to the model up to
        MAX_BOUNCES times; the model's final accepted turn is returned.
        """
        msgs = list(messages)
        body_tools = tools or None
        for attempt in range(MAX_BOUNCES + 1):
            body = _request_body(self.config, msgs, body_tools, max_tokens,
                                 thinking)
            r = await self._http.post("/v1/chat/completions", json=body)
            if r.status_code != 200:
                raise LlmError(f"vLLM {r.status_code}: {r.text[:200]}")
            choice = r.json()["choices"][0]
            msg = choice["message"]
            raw_calls = msg.get("tool_calls") or []
            if not raw_calls or not offered:
                return msg, [], msg.get("content") or ""
            valid, errors = validate_tool_calls(raw_calls, offered, validators)
            if not errors:
                return msg, valid, msg.get("content") or ""
            if attempt == MAX_BOUNCES:
                # Still failing: hand back what passed, errors to the caller.
                return msg, valid, msg.get("content") or ""
            _bounce(msgs, msg, errors)
        raise LlmError("unreachable")

    async def chat_stream(self, messages, tools=None, offered=None,
                          validators=None, thinking=True, max_tokens=4096):
        """The streaming leg: an async generator of events —

          ("text", delta)     content deltas (reasoning never appears)
          ("toolcalls",)      once, at the first tool_call fragment
          ("restart",)        invalid calls bounced; per-attempt state reset
          ("done", msg, valid_calls, text)   the chat() triple, final

        Validation and MAX_BOUNCES mirror chat(): the message is only
        assembled and validated at [DONE]. Once a tool_call fragment
        appears, later text deltas stop being events (the assembled
        message still carries them) — the caller drops its sentence
        gate at ("toolcalls",) and ("restart",).
        """
        msgs = list(messages)
        body_tools = tools or None
        for attempt in range(MAX_BOUNCES + 1):
            body = _request_body(self.config, msgs, body_tools, max_tokens,
                                 thinking, stream=True)
            content = []
            calls: dict[int, dict] = {}   # index -> OpenAI-shaped fragment
            saw_calls = False
            try:
                async with self._http.stream(
                        "POST", "/v1/chat/completions", json=body) as r:
                    if r.status_code != 200:
                        await r.aread()
                        raise LlmError(
                            f"vLLM {r.status_code}: {r.text[:200]}")
                    async for line in r.aiter_lines():
                        if not line.startswith("data:"):
                            continue
                        payload = line[len("data:"):].strip()
                        if payload == "[DONE]":
                            break
                        choice = ((json.loads(payload).get("choices")
                                   or [{}])[0])
                        d = choice.get("delta") or {}
                        for frag in d.get("tool_calls") or []:
                            if not saw_calls:
                                saw_calls = True
                                yield ("toolcalls",)
                            acc = calls.setdefault(frag.get("index", 0), {
                                "id": "", "type": "function",
                                "function": {"name": "", "arguments": ""}})
                            if frag.get("id"):
                                acc["id"] = frag["id"]
                            fn = frag.get("function") or {}
                            if fn.get("name"):
                                acc["function"]["name"] += fn["name"]
                            if fn.get("arguments"):
                                acc["function"]["arguments"] += \
                                    fn["arguments"]
                        piece = d.get("content")
                        if piece:
                            content.append(piece)
                            if not saw_calls:
                                yield ("text", piece)
            except httpx.HTTPError as e:
                raise LlmError(f"vLLM stream failed: {e}") from e
            msg = {"role": "assistant", "content": "".join(content) or None}
            if calls:
                msg["tool_calls"] = [calls[i] for i in sorted(calls)]
            raw_calls = msg.get("tool_calls") or []
            if not raw_calls or not offered:
                yield ("done", msg, [], msg.get("content") or "")
                return
            valid, errors = validate_tool_calls(raw_calls, offered,
                                                validators)
            if not errors or attempt == MAX_BOUNCES:
                yield ("done", msg, valid, msg.get("content") or "")
                return
            _bounce(msgs, msg, errors)
            yield ("restart",)

    async def aclose(self):
        await self._http.aclose()
