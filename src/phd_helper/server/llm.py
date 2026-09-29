"""vLLM chat client (SPEC §2).

Non-streaming for tool-decision turns (the §2 preference), with mandatory
validation of every returned tool call and bounce-back retry on failure —
validation errors go back to the model, never to the user. Never relies on
tool_choice="required"; a "must call a tool" nudge retry lives in the
caller.
"""

import httpx

from phd_helper.toolcall import validate_tool_calls

MAX_BOUNCES = 2


class LlmError(Exception):
    pass


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
            body = {"model": self.config.llm_model, "messages": msgs,
                    "max_tokens": max_tokens,
                    **self.config.sampling(thinking)}
            if body_tools:
                body["tools"] = body_tools
                body["tool_choice"] = "auto"
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
            # Bounce: replay the assistant turn, then tell the model what
            # failed (SPEC §2: never surface to the user).
            msgs.append(msg)
            for e in errors:
                msgs.append({"role": "user",
                             "content": f"[tool-call validation] {e}. "
                                        "Re-issue the tool call correctly."})
        raise LlmError("unreachable")

    async def aclose(self):
        await self._http.aclose()
