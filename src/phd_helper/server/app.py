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

from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from phd_helper.cascade import ArxivRateLimited
from phd_helper.corpus import Corpus, CorpusError
from phd_helper.endpoint import VoiceEndpoint
from phd_helper.ingest import IngestError, Ingestor
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
                        openalex_mailto=self.state.config.openalex_mailto,
                        corpus=self.state.corpus,
                        store=self.state.corpus_store,
                        autojoin=lambda ids: autojoin(self.state, ids))
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
        except Exception as e:
            # Last line: no bug anywhere in a tool may silently kill the
            # fire-and-forget turn task — the client always hears back.
            await self.send({"type": "error", "where": "turn",
                             "message": f"unexpected error: {e}"})

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
        # One global corpus across projects (§6). The registry is live
        # always; the heavy stack is config-gated (PHD_CORPUS_STACK):
        # "off" degrades per the §8 matrix, "local" runs MinerU + harrier
        # + LanceDB on this machine. Imports stay inside the branch so the
        # off path never pays for lancedb/torch.
        self.corpus = Corpus(REPO_ROOT / "corpus_data")
        self.corpus_store = None
        extractor = None
        if self.config.corpus_stack == "local":
            from phd_helper.server.embed import HarrierEmbedder
            from phd_helper.server.lancedb_store import LanceStore
            from phd_helper.server.mineru import MinerUExtractor
            from phd_helper.server.rerank import QwenReranker
            self.corpus_store = LanceStore(
                REPO_ROOT / "corpus_data" / "lancedb", HarrierEmbedder(),
                reranker=QwenReranker())  # lazy: loads on first search
            extractor = MinerUExtractor()
        # PDF fetch is arXiv-spaced too (§6 politeness covers all arXiv
        # access, not just the bibtex cascade).
        self.fetch_pdf = ArxivRateLimited(self.http.fetch_bytes)
        self.ingestor = Ingestor(self.corpus, extractor=extractor,
                                 store=self.corpus_store,
                                 fetch_pdf=self._fetch_pdf_or_raise)
        self.ingest_tasks: set[asyncio.Task] = set()

    async def _fetch_pdf_or_raise(self, url: str) -> bytes:
        status, body = await self.fetch_pdf(url)
        if status != 200 or not body:
            raise IngestError(f"PDF fetch failed ({status})")
        return body


# Background ingest plumbing, state-agnostic so the HTTP seam can be
# tested against a bare namespace (the tests' harness style).

def spawn(state, coro):
    """Track background tasks so they are never garbage-collected
    mid-flight (the ingest pipeline runs as one per PDF)."""
    task = asyncio.create_task(coro)
    state.ingest_tasks.add(task)
    task.add_done_callback(state.ingest_tasks.discard)


def spawn_ingest(state, doc_id: str) -> None:
    if state.ingestor is not None and state.ingestor.runnable():
        spawn(state, state.ingestor.ingest(doc_id))
    # not runnable: the doc stays queued — paused, visible (§8)


async def autojoin(state, arxiv_ids: list[str]) -> None:
    """§6: search hits with an openly downloadable PDF auto-join the
    corpus. Fire-and-forget: the agent's turn never waits on indexing."""
    if state.ingestor is None or not state.ingestor.runnable():
        return
    for aid in arxiv_ids:
        spawn(state, state.ingestor.fetch_arxiv(aid))


def create_app(state: "AppState | None" = None) -> FastAPI:
    state = state or AppState()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        # Startup resume (§6): crashed extractions go back to the queue and
        # everything queued drains — if the pipeline is runnable at all.
        state.corpus.reconcile_startup()
        if state.ingestor is not None and state.ingestor.runnable():
            spawn(state, state.ingestor.drain_queued())
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
            # §8: indexing pauses (queued docs wait visibly) without the store.
            "corpus": "ok" if state.corpus_store is not None else "paused",
            "endpoint_holder": state.endpoint.endpoint(time.monotonic()),
        }

    # -- corpus doors and status (SPEC §6): upload is door 1, the agent
    # fetch/auto-join is door 2; the UI reads status, never searches.

    @app.post("/corpus/upload")
    async def corpus_upload(request: Request, title: str = "",
                            arxiv: str = "", doi: str = "", year: str = ""):
        data = await request.body()  # raw PDF bytes (no multipart dep)
        if not data:
            return JSONResponse({"error": "empty PDF body"}, status_code=400)
        rec, new = state.corpus.add_pdf(
            data, title=title, arxiv=arxiv, doi=doi, year=year,
            source="upload")
        if new:
            spawn_ingest(state, rec.doc_id)
        return {"doc_id": rec.doc_id, "status": rec.status, "new": new}

    @app.get("/corpus/docs")
    async def corpus_docs():
        # Per-PDF status so failures are visible, not rot (§6).
        return [{"doc_id": d.doc_id, "title": d.title, "status": d.status,
                 "error": d.error, "arxiv": d.arxiv, "doi": d.doi,
                 "year": d.year, "source": d.source,
                 "pinned_in": list(d.pinned_in),
                 "chunk_count": d.chunk_count}
                for d in state.corpus.list()]

    @app.post("/corpus/{doc_id}/retry")
    async def corpus_retry(doc_id: str):
        try:
            rec = state.corpus.retry(doc_id)
        except CorpusError as e:
            return JSONResponse({"error": str(e)}, status_code=409)
        spawn_ingest(state, doc_id)
        return {"doc_id": doc_id, "status": rec.status}

    # Pins attach a doc to the active project (§5/§6): the boost the
    # agent's boost_pinned rides on. The UI's pin button is this door.

    @app.post("/corpus/{doc_id}/pin")
    async def corpus_pin(doc_id: str):
        try:
            state.corpus.pin(doc_id, state.project.root.name)
        except CorpusError as e:
            return JSONResponse({"error": str(e)}, status_code=409)
        return {"doc_id": doc_id,
                "pinned_in": list(state.corpus.get(doc_id).pinned_in)}

    @app.post("/corpus/{doc_id}/unpin")
    async def corpus_unpin(doc_id: str):
        state.corpus.unpin(doc_id, state.project.root.name)
        rec = state.corpus.get(doc_id)
        return {"doc_id": doc_id,
                "pinned_in": list(rec.pinned_in) if rec else []}

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
