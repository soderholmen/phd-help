"""FastAPI assembly (SPEC §1): one process serving the web app, the voice
WebSocket, and the agent loop against vLLM.

Milestone-1 scaffold: typed input drives real LLM turns over the same
WebSocket the mic will use; STT/TTS are stubs (voice.py). No auth —
ZeroTier membership is the access control (§1).
"""

import asyncio
import json
import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from phd_helper.cascade import ArxivRateLimited
from phd_helper.endpoint import VoiceEndpoint
from phd_helper.project import Project
from phd_helper.server.config import REPO_ROOT, load as load_config
from phd_helper.server.http import HttpFetcher
from phd_helper.server.llm import LlmClient, LlmError
from phd_helper.server.voice import StubStt, StubTts, parse_control
from phd_helper.tools import (OFFERED, TOOL_SCHEMAS, execute_async,
                              make_validators)

WEB_DIR = Path(__file__).resolve().parents[3] / "web"

# SPEC §2: these two rules are verbatim — they fixed every probe failure.
SYSTEM_PROMPT = (
    "You are phd-helper, a voice-first co-writing agent for LaTeX papers.\n"
    "1. If the user asks to edit text already in your context, emit the "
    "`section_write` patch directly; do NOT search first to verify wording "
    "— search only for new facts/citations/corpus content.\n"
    "2. Questions about the writing of the current section (flow, clarity, "
    "comparisons within it) are answered directly from context, with no "
    "tool call.\n"
    "When speaking, pre-normalize math to words (say 'E equals m c "
    "squared', not symbols) — the TTS engine hallucinates on dense symbol "
    "strings.\n")


class Session:
    """One browser tab's voice connection (§1: one WebSocket per tab)."""

    def __init__(self, app_state, client_id: str):
        self.state = app_state
        self.client_id = client_id
        self.ws: WebSocket | None = None
        self.turn_task: asyncio.Task | None = None
        sections = ", ".join(n.path for n in app_state.project.section_tree())
        self.history: list[dict] = [{"role": "system", "content":
                                     SYSTEM_PROMPT +
                                     f"\nProject sections: {sections}"}]

    async def send(self, event: dict):
        if self.ws is not None:
            await self.ws.send_text(json.dumps(event))

    async def run_turn(self, user_text: str):
        # No queue: one in-flight turn, last utterance wins (§8).
        current = asyncio.current_task()
        if (self.turn_task is not None and self.turn_task is not current
                and not self.turn_task.done()):
            self.turn_task.cancel()
        self.history.append({"role": "user", "content": user_text})
        await self.send({"type": "turn_started"})
        project = self.state.project
        try:
            for _ in range(5):  # tool loop; the model ends with a text turn
                msg, valid_calls, text = await self.state.llm.chat(
                    self.history, tools=TOOL_SCHEMAS, offered=OFFERED,
                    validators=make_validators(project))
                if not valid_calls:
                    self.history.append({"role": "assistant",
                                         "content": text or ""})
                    await self.send({"type": "assistant_text", "text": text})
                    # TTS seam: sentence-by-sentence synthesis lands with
                    # the 3090 stack; the stub counts the request.
                    async for _chunk in self.state.tts.synthesize(text):
                        pass  # binary audio frames go out here
                    return
                self.history.append(msg)  # assistant turn with tool_calls
                for vc in valid_calls:
                    result = await execute_async(
                        vc, project, fetch=self.state.fetch,
                        mailto=self.state.config.crossref_mailto,
                        openalex_mailto=self.state.config.openalex_mailto)
                    if result.get("status") == "pending":
                        # One-at-a-time diff awaiting approval (§5). The
                        # result's find/replace are the final ones (cite_add
                        # rewrites the \\cite key); section_write has none.
                        await self.send({"type": "diff",
                                         "diff_id": result["diff_id"],
                                         "section": result["section"],
                                         "find": result.get("find",
                                                            vc.args["find"]),
                                         "replace": result.get("replace",
                                                               vc.args["replace"])})
                    self.history.append({"role": "tool",
                                         "tool_call_id": vc.id,
                                         "content": json.dumps(result)})
            await self.send({"type": "error", "where": "loop",
                             "message": "tool loop budget exhausted"})
        except asyncio.CancelledError:
            # Barge-in while thinking: abort, keep history consistent.
            self.history.append({"role": "assistant",
                                 "content": "[interrupted]"})
            await self.send({"type": "turn_interrupted"})
        except LlmError as e:
            # vLLM down, backend up: error inline in chat (§8 matrix).
            await self.send({"type": "error", "where": "llm",
                             "message": str(e)})

    def cancel_turn(self):
        if self.turn_task is not None and not self.turn_task.done():
            self.turn_task.cancel()


class AppState:
    def __init__(self):
        self.config = load_config()
        self.llm = LlmClient(self.config)
        self.stt = StubStt()
        self.tts = StubTts()
        # Cascade HTTP, arXiv-spaced (SPEC §6 politeness).
        self.http = HttpFetcher()
        self.fetch = ArxivRateLimited(self.http)
        self.endpoint = VoiceEndpoint(
            ping_interval=self.config.ping_interval_s,
            lease_timeout=self.config.lease_timeout_s)
        # One active project at a time (§7); scaffold ships the sample paper.
        self.project = Project(REPO_ROOT / "sample_paper")


def create_app(state: "AppState | None" = None) -> FastAPI:
    state = state or AppState()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        yield
        await state.http.aclose()  # release the shared client on shutdown

    app = FastAPI(title="phd-helper", lifespan=lifespan)
    app.state.phd = state

    @app.get("/health")
    async def health():
        return {
            "vllm": await state.llm.healthy(),
            "stt": "faulted" if state.stt.faulted() else "stub",
            "tts": "faulted" if state.tts.faulted() else "stub",
            "endpoint_holder": state.endpoint.endpoint(time.monotonic()),
        }

    @app.websocket("/ws/voice")
    async def voice(ws: WebSocket):
        await ws.accept()
        client_id = ws.query_params.get("client", "anon")
        session = Session(state, client_id)
        session.ws = ws
        now = time.monotonic()
        state.endpoint.heartbeat(client_id, now)
        await session.send({"type": "hello", "client_id": client_id})
        try:
            while True:
                frame = await ws.receive()
                if frame["type"] == "websocket.disconnect":
                    break
                if (data := frame.get("bytes")) is not None:
                    for final in state.stt.feed(data):
                        session.turn_task = asyncio.create_task(
                            session.run_turn(final))
                    continue
                if (text := frame.get("text")) is None:
                    continue
                try:
                    msg = parse_control(text)
                except ValueError as e:
                    await session.send({"type": "error", "where": "control",
                                        "message": str(e)})
                    continue
                await dispatch(session, msg)
        except WebSocketDisconnect:
            pass
        finally:
            session.cancel_turn()
            session.ws = None

    async def dispatch(session: Session, msg: dict):
        now = time.monotonic()
        kind = msg["type"]
        if kind == "heartbeat":
            state.endpoint.heartbeat(session.client_id, now)
            # One mechanism, three consumers (§8): liveness, meter, watchdog.
            await session.send({"type": "pong", "rms": msg.get("rms", 0.0)})
        elif kind == "arm":
            ok = state.endpoint.arm(session.client_id, now)
            await session.send({"type": "armed", "ok": ok,
                                "holder": state.endpoint.endpoint(now)})
        elif kind == "disarm":
            session.cancel_turn()
            await session.send({"type": "disarmed"})
        elif kind == "typed":
            text = str(msg.get("text", "")).strip()
            if text:
                session.turn_task = asyncio.create_task(
                    session.run_turn(text))
        elif kind == "barge_in":
            # Qualifying interrupt: stop TTS now, abort thinking (§3).
            session.cancel_turn()
            await session.send({"type": "tts_stopped"})
        elif kind in ("approve", "reject"):
            section, diff_id = msg.get("section"), msg.get("diff_id")
            if kind == "approve":
                result = state.project.apply_pending(section, diff_id)
                await session.send({
                    "type": "diff_resolved", "diff_id": diff_id,
                    "applied": result.applied, "reason": result.reason,
                    "text": result.text})
            else:
                state.project.reject_pending(section, diff_id)
                await session.send({"type": "diff_resolved",
                                    "diff_id": diff_id, "applied": False,
                                    "reason": "discarded", "text": None})

    if WEB_DIR.is_dir():
        @app.get("/")
        async def index():
            return FileResponse(WEB_DIR / "index.html")
        app.mount("/static", StaticFiles(directory=WEB_DIR), name="static")

    # Private-CA root cert for devices to install (public half only; the CA
    # key never leaves certs/, which is gitignored).
    ca_pem = REPO_ROOT / "certs" / "ca.pem"
    if ca_pem.is_file():
        @app.get("/ca.pem")
        async def ca_pem_download():
            return FileResponse(ca_pem, media_type="application/x-pem-file")
    return app


app = create_app()
