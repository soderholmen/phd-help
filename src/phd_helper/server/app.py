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
from datetime import date
from pathlib import Path

from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from phd_helper.cascade import ArxivRateLimited
from phd_helper.context import (ContextInputs, PinnedSource, Turn,
                                approx_tokens, assemble_context,
                                group_exchanges)
from phd_helper.corpus import Corpus, CorpusError
from phd_helper.endpoint import VoiceEndpoint
from phd_helper.gists import body_sha, flatten, render_gists, stale_sections
from phd_helper.ingest import IngestError, Ingestor
from phd_helper.project import Project
from phd_helper.server.config import REPO_ROOT, load as load_config
from phd_helper.server.http import HttpFetcher
from phd_helper.server.llm import LlmClient, LlmError
from phd_helper.server.voice import StubStt, StubTts, parse_control
from phd_helper.summaries import by_section, render as render_summaries
from phd_helper.tools import (OFFERED, TOOL_SCHEMAS, execute_async,
                              make_validators)

WEB_DIR = Path(__file__).resolve().parents[3] / "web"
DIST_DIR = WEB_DIR / "app" / "dist"  # the built React shell (npm run build)

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
        # §1: clicking a tree node anchors the section-scoped discussion;
        # the anchored section's body then rides every turn's context (§4).
        self.selected: str | None = None
        # The tree is clickable at every depth (§1), so the prompt's
        # section list flattens rather than stopping at the top level.
        sections = ", ".join(flatten(app_state.project.section_tree()))
        self.history: list[dict] = [{"role": "system", "content":
                                     SYSTEM_PROMPT +
                                     f"\nProject sections: {sections}"}]

    def select_section(self, path: str) -> bool:
        if not path:
            self.selected = None  # deselect: the anchor must be clearable
            return True
        known = set(flatten(self.state.project.section_tree()))
        if path not in known:
            return False
        self.selected = path
        return True

    async def _build_context(self):
        """Assemble this turn's context (§4) and locate the conversation
        tail that survived the drop tiers.

        Returns (context message or None, index into the conversation
        where the kept tail starts). The full log stays in ``history``;
        only the sent message list is trimmed."""
        project = self.state.project
        section = ""
        if self.selected:
            try:
                body = project.read_section(self.selected)
            except OSError:
                body = ""  # file vanished since selection: degrade (§8)
            if body:
                section = f"Selected section ({self.selected}):\n{body}"
        pinned = await pinned_sources(self.state.corpus,
                                      self.state.corpus_store,
                                      project.root.name)
        # §4 far-section gists: render what's cached, kick a background
        # refresh for the stale ones — this turn degrades honestly, the
        # next turn has them. The selected section is skipped: its full
        # body rides the turn already.
        tree = project.section_tree()
        gist_cache = project.load_gists()
        distant_gists = render_gists(tree, gist_cache,
                                     skip=self.selected or "")
        if stale_sections(tree, project.files(), gist_cache):
            spawn_gist_refresh(self.state)
        # §4 paper memory: never drops, rides every turn once distilled.
        raw_memory = project.load_memory()
        exchanges = group_exchanges(self.history[1:])
        inputs = ContextInputs(
            section=section, skeleton=project.skeleton(),
            memory=f"Paper memory:\n{raw_memory}" if raw_memory else "",
            distant_gists=distant_gists,
            rolling_summary=render_summaries(project.load_summaries()),
            turns=tuple(Turn("user", text) for text, _ in exchanges),
            pinned=pinned)
        ctx = assemble_context(
            inputs, budget=self.state.config.context_budget_tokens,
            count_tokens=approx_tokens)
        # The current request is never droppable: clamp to the last
        # exchange even when the assembly's tiers emptied the conversation.
        kept = max(ctx.conversation_kept, 1) if exchanges else 0
        tail = exchanges[len(exchanges) - kept:] if kept else []
        start = sum(len(msgs) for _, msgs in exchanges) - \
            sum(len(msgs) for _, msgs in tail)
        # Conversation parts are the tail of ctx.parts; the rest is the
        # context message (skeleton, gists, memory, section, pinned,
        # summary — empties filtered).
        head = ctx.parts[:len(ctx.parts) - ctx.conversation_kept]
        content = "\n\n".join(p for p in head if p)
        ctx_msg = {"role": "system", "content": content} if content else None
        return ctx_msg, start

    async def send(self, event: dict):
        if self.ws is not None:
            await self.ws.send_text(json.dumps(event))

    async def run_turn(self, user_text: str):
        # No queue: one in-flight turn, last utterance wins (§8).
        current = asyncio.current_task()
        if (self.turn_task is not None and self.turn_task is not current
                and not self.turn_task.done()):
            self.turn_task.cancel()
        # The §7 rolling summaries group a sitting by the section each
        # exchange was anchored to — tag the user message with it.
        self.history.append({"role": "user", "content": user_text,
                             "section": self.selected or ""})
        await self.send({"type": "turn_started"})
        project = self.state.project
        try:
            ctx_msg, conv_start = await self._build_context()
            for _ in range(5):  # tool loop; the model ends with a text turn
                # One system message, always: vLLM 400s on a second one
                # ("System message must be at the beginning"), so the
                # §4 context merges into the prompt message.
                sys_msg = self.history[0]
                if ctx_msg is not None:
                    sys_msg = {"role": "system", "content":
                               sys_msg["content"] + "\n\n" + ctx_msg["content"]}
                msgs = [sys_msg]
                msgs += self.history[1:][conv_start:]
                msg, valid_calls, text = await self.state.llm.chat(
                    msgs, tools=TOOL_SCHEMAS, offered=OFFERED,
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
        self.gist_task: asyncio.Task | None = None

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
    return task


def spawn_ingest(state, doc_id: str) -> None:
    if state.ingestor is not None and state.ingestor.runnable():
        spawn(state, state.ingestor.ingest(doc_id))
    # not runnable: the doc stays queued — paused, visible (§8)


GIST_PROMPT = (
    "Summarize this LaTeX section in one sentence of at most 30 words, "
    "saying what it covers — the writer sees this line instead of "
    "opening the file. Reply with the sentence only.")
GIST_BODY_CAP = 12000  # chars of section body sent to the model


async def refresh_gists(state) -> None:
    """§4: regenerate stale per-section gists in the background. The
    turn that found them renders with what's cached (stale this turn,
    present next); an LLM fault stops the pass and the next turn
    retries — gists are context polish, never an error to surface (§8).
    Saved after each line so a mid-pass fault keeps the progress."""
    project = state.project
    tree = project.section_tree()
    files = project.files()
    cache = project.load_gists()
    for path in stale_sections(tree, files, cache):
        body = files.get(path, "")
        if not body.strip():
            cache[path] = {"sha": body_sha(body), "gist": ""}
            project.save_gists(cache)
            continue  # nothing to gist; don't burn a call every turn
        try:
            _, _, line = await state.llm.chat(
                [{"role": "system", "content": GIST_PROMPT},
                 {"role": "user",
                  "content": f"{path}:\n\n{body[:GIST_BODY_CAP]}"}],
                thinking=False, max_tokens=120)
        except Exception:
            return  # vLLM down or faulting: keep the gists we have
        line = " ".join((line or "").split())[:300]
        if line:
            cache[path] = {"sha": body_sha(body), "gist": line}
            project.save_gists(cache)


def spawn_gist_refresh(state) -> None:
    """One refresh at a time — stale gists are not an emergency."""
    task = getattr(state, "gist_task", None)
    if task is not None and not task.done():
        return
    state.gist_task = spawn(state, refresh_gists(state))


MEMORY_PROMPT = (
    "You maintain the paper's memory file: the decisions, claims, "
    "terminology and TODOs a co-writer must not lose between sessions. "
    "Rewrite the file to fold in this session's conversation: add what "
    "was decided, remove only what was explicitly decided away, keep it "
    "under ~300 words of markdown bullets under Decisions / Claims / "
    "Terminology / TODOs. Reply with the file content only.")
MEMORY_TRANSCRIPT_CAP = 12000  # chars of transcript sent to the model


async def distill_memory(state, conversation) -> None:
    """§4/§7: fold a finished sitting into paper memory. A fault skips
    the distillation — memory is polish, never an error to surface (§8);
    the next session end retries. The voice door is the memory_write
    tool; this is the session-end door."""
    transcript = "\n".join(
        f"{m['role']}: {m['content']}"
        for m in conversation
        if m.get("role") in ("user", "assistant") and m.get("content"))
    if not transcript.strip():
        return  # an empty sitting has nothing to distill
    old = state.project.load_memory()
    try:
        _, _, text = await state.llm.chat(
            [{"role": "system", "content": MEMORY_PROMPT},
             {"role": "user",
              "content": f"Current memory:\n{old or '(empty)'}\n\n"
                         f"Session transcript:\n{transcript[:MEMORY_TRANSCRIPT_CAP]}"}],
            thinking=False, max_tokens=900)
    except Exception:
        return  # vLLM down at disconnect: skip, retry next session end
    text = (text or "").strip()
    if text:
        state.project.save_memory(text)


SUMMARY_PROMPT = (
    "Merge the existing rolling summary of this paper section with the "
    "new session's conversation about it into one summary of at most 80 "
    "words: what was written, decided and left open. Reply with the "
    "summary text only.")
SUMMARY_TRANSCRIPT_CAP = 8000  # chars of per-section transcript sent


async def distill_summaries(state, conversation) -> None:
    """§7: fold a finished sitting into per-section rolling summaries —
    the residue of conversation that fell out of the verbatim window.
    Sections discussed with nothing selected are nobody's summary; a
    fault stops the pass, the next session end retries (§8)."""
    groups = by_section(conversation)
    if not groups:
        return
    summaries = state.project.load_summaries()
    for section, lines in groups.items():
        entries = summaries.get(section, [])
        old = entries[-1]["text"] if entries else "(none yet)"
        try:
            _, _, text = await state.llm.chat(
                [{"role": "system", "content": SUMMARY_PROMPT},
                 {"role": "user",
                  "content": f"Section: {section}\n"
                             f"Existing summary:\n{old}\n\n"
                             f"This session:\n{lines[:SUMMARY_TRANSCRIPT_CAP]}"}],
                thinking=False, max_tokens=300)
        except Exception:
            return  # keep what landed; retry the rest next session end
        text = (text or "").strip()
        if text:
            summaries.setdefault(section, []).append(
                {"date": date.today().isoformat(), "text": text})
            state.project.save_summaries(summaries)




async def _doc_info(store, doc_id: str):
    if store is None:
        return None
    try:
        return await store.doc(doc_id)
    except Exception:
        return None  # faulting store degrades per §8, never kills the turn


async def pinned_sources(corpus, store, project_name: str):
    """§4: pinned sources always contribute abstract + headings. With the
    store down or faulting they degrade to the registry title (§8) — a pin
    that silently vanishes from context is worse than a thin one. The
    doc() lookups fan out concurrently: each is a blocking LanceDB query,
    and sequential awaits would stack N round-trips onto every turn."""
    recs = [r for r in corpus.list()  # attach order: recent pins drop first
            if project_name in r.pinned_in]
    infos = await asyncio.gather(*(_doc_info(store, r.doc_id) for r in recs))
    out = []
    for rec, info in zip(recs, infos):
        label = f"Pinned source: {rec.title or rec.doc_id} (doc {rec.doc_id})"
        if info is not None:
            out.append(PinnedSource(rec.doc_id, f"{label}\n{info.abstract}",
                                    "\n".join(info.headings)))
        else:
            out.append(PinnedSource(rec.doc_id, label, ""))
    return tuple(out)


async def autojoin(state, hits) -> None:
    """§6: search hits with an openly downloadable PDF auto-join the
    corpus, carrying the title/year the search already returned (§6's
    embed prefix is `paper title » section`). Fire-and-forget: the
    agent's turn never waits on indexing."""
    if state.ingestor is None or not state.ingestor.runnable():
        return
    for h in hits:
        spawn(state, state.ingestor.fetch_arxiv(h.arxiv, title=h.title,
                                                year=h.year))


def create_app(state: "AppState | None" = None,
               web_dir: Path | None = None) -> FastAPI:
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
        # §8: the store is probed, not config-read — a wired-but-faulting
        # LanceDB (locked file, corrupt manifest) must read faulted, not ok.
        corpus = "paused"
        if state.corpus_store is not None:
            corpus = "ok" if await state.corpus_store.healthy() else "faulted"
        return {
            "vllm": await state.llm.healthy(),
            "stt": "faulted" if state.stt.faulted() else "stub",
            "tts": "faulted" if state.tts.faulted() else "stub",
            # No store configured: indexing pauses (queued docs wait visibly).
            "corpus": corpus,
            "endpoint_holder": state.endpoint.endpoint(time.monotonic()),
        }

    @app.get("/sections")
    async def sections():
        # The shell's tree panel: the parsed \input graph (§1), nested —
        # clicking a node sends select_section over the voice socket.
        def node(n):
            return {"path": n.path, "title": n.title,
                    "children": [node(c) for c in n.children]}
        return [node(n) for n in state.project.section_tree()]

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
        # pinned_here is the server-side join with the active project —
        # the UI's pin button can't know the project name itself (issue #21).
        here = state.project.root.name
        return [{"doc_id": d.doc_id, "title": d.title, "status": d.status,
                 "error": d.error, "arxiv": d.arxiv, "doi": d.doi,
                 "year": d.year, "source": d.source,
                 "pinned_in": list(d.pinned_in),
                 "pinned_here": here in d.pinned_in,
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
            # §7: the sitting ends here until the web shell owns session
            # semantics (project switch, 30-min idle) — distill memory
            # and per-section rolling summaries in the background; a tab
            # refresh just folds a shorter sitting.
            sitting = session.history[1:]
            spawn(state, distill_memory(state, sitting))
            spawn(state, distill_summaries(state, sitting))

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
        elif kind == "select_section":
            # §1: clicking a tree node anchors the section-scoped
            # discussion; the body then rides every later turn (§4).
            path = str(msg.get("section", ""))
            if session.select_section(path):
                await session.send({"type": "section_selected",
                                    "section": session.selected})
            else:
                await session.send({"type": "error", "where": "control",
                                    "message": f"no such section: {path}"})
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

    # Private-CA root cert for devices to install (public half only; the CA
    # key never leaves certs/, which is gitignored).
    ca_pem = REPO_ROOT / "certs" / "ca.pem"
    if ca_pem.is_file():
        @app.get("/ca.pem")
        async def ca_pem_download():
            return FileResponse(ca_pem, media_type="application/x-pem-file")

    # The built React shell (web/app/dist): hashed assets, the capture
    # worklet, index.html for the app itself. Mounted last and at "/", so
    # every API route above keeps winning; the mount only catches the
    # rest (html=True serves index.html for "/"). No build: no root
    # route — dev runs the Vite server against the proxy instead.
    dist = web_dir if web_dir is not None else DIST_DIR
    if dist.is_dir():
        app.mount("/", StaticFiles(directory=dist, html=True), name="shell")
    return app


app = create_app()
