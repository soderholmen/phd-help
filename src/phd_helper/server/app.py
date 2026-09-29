"""FastAPI assembly (SPEC §1): one process serving the web app, the voice
WebSocket, and the agent loop against vLLM.

Milestone-1 scaffold: typed input drives real LLM turns over the same
WebSocket the mic will use; STT/TTS are stubs (voice.py). No auth —
ZeroTier membership is the access control (§1).
"""

import asyncio
import json
import time
from pathlib import Path

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from phd_helper.endpoint import VoiceEndpoint
from phd_helper.server.config import load as load_config
from phd_helper.server.llm import LlmClient, LlmError
from phd_helper.server.voice import StubStt, StubTts, parse_control

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
        self.history: list[dict] = [{"role": "system",
                                     "content": SYSTEM_PROMPT}]

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
        try:
            msg, valid_calls, text = await self.state.llm.chat(self.history)
            self.history.append({"role": "assistant",
                                 "content": text or ""})
            await self.send({"type": "assistant_text", "text": text})
            # TTS seam: sentence-by-sentence synthesis lands with the
            # 3090 stack; the stub counts the request so health is real.
            async for _chunk in self.state.tts.synthesize(text):
                pass  # binary audio frames go out here
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
        self.endpoint = VoiceEndpoint(
            ping_interval=self.config.ping_interval_s,
            lease_timeout=self.config.lease_timeout_s)


def create_app() -> FastAPI:
    app = FastAPI(title="phd-helper")
    state = AppState()
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

    if WEB_DIR.is_dir():
        @app.get("/")
        async def index():
            return FileResponse(WEB_DIR / "index.html")
        app.mount("/static", StaticFiles(directory=WEB_DIR), name="static")
    return app


app = create_app()
