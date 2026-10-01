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
from datetime import date, datetime
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
from phd_helper import sessionlog
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
    """One sitting in a project (SPEC §7): the conversation, the approval
    window and the §4 anchor, shared by every tab attached to it. Tabs
    attach and detach freely — a blip or a refresh loses nothing; the
    sitting ends on project switch, graceful shutdown or ~30 min idle,
    never on a tab closing. The conversation is the §7 verbatim JSONL."""

    def __init__(self, app_state):
        self.state = app_state
        self.sockets: dict[str, WebSocket] = {}
        self.turn_task: asyncio.Task | None = None
        # §1: clicking a tree node anchors the section-scoped discussion;
        # the anchored section's body then rides every turn's context (§4).
        # Sitting-level, not per-tab: one conversation, one anchor (the
        # shell keeps its own tree highlight as §7 per-client view state).
        self.selected: str | None = None
        # §3 approval window: diffs this sitting proposed that are still
        # pending. While any ride, the next utterance is interpreted
        # against the approval by the agent (the note in _build_context).
        self.pending_diffs: list[dict] = []
        # §7/§8 reopen: diffs already pending when this sitting opened,
        # reconciled once (re-anchored or bounced) and re-presented to
        # every socket that attaches — disk is the truth, the cards ride it.
        self.pending_sent = False
        self.live_pending: list = []
        self.last_active = time.monotonic()
        self.recap_sent = False
        project = app_state.project
        # §7: reopening resumes the same history — the verbatim turns
        # after the last divider. A clean close leaves none (that talk
        # rides on as rolling summaries); a crash leaves them all.
        tail = project.chat_tail()
        self.recap = sessionlog.recap_line(project.chat_divider(),
                                           len(tail))
        # The tree is clickable at every depth (§1), so the prompt's
        # section list flattens rather than stopping at the top level.
        sections = ", ".join(flatten(project.section_tree()))
        self.history: list[dict] = [
            {"role": "system",
             "content": SYSTEM_PROMPT + f"\nProject sections: {sections}"}
        ] + tail

    def attach(self, client_id: str, ws: WebSocket) -> None:
        self.sockets[client_id] = ws

    def detach(self, client_id: str,
               ws: WebSocket | None = None) -> None:
        # Identity-checked pop: a refresh reuses the client_id and the
        # new socket can attach before the old handler's finally runs —
        # the old close must not evict the live socket.
        if ws is None or self.sockets.get(client_id) is ws:
            self.sockets.pop(client_id, None)

    def log(self, msg: dict) -> None:
        """Append to the in-memory conversation and the §7 verbatim JSONL
        — the file is the spine, memory is its working copy."""
        self.history.append(msg)
        self.state.project.append_chat(
            [dict(msg, ts=datetime.now().isoformat(timespec="seconds"))])

    def sync_pending(self) -> None:
        """Re-read the approval window from disk truth, one pass: a bounced
        apply leaves its diff pending, and a diff resolved elsewhere (other
        tab, cleanup) must drop out — tracking follows the registry, never
        the decisions made."""
        live = {d.id for d in self.state.project.pending.list_all()}
        self.pending_diffs = [t for t in self.pending_diffs
                              if t["diff_id"] in live]

    def _pending_note(self) -> str:
        """The §3 approval-window note: what is pending, and how to read
        the next utterance against it. Interpretation is the agent's job —
        the note carries the ids, never a string-matching rule. The lines
        are read from disk truth, so a bounced apply plus user edits can
        never show a find anchor that no longer exists."""
        if not self.pending_diffs:
            return ""
        tracked = {t["diff_id"] for t in self.pending_diffs}
        lines = [
            f"- diff {d.id} ({d.section_path}): "
            f"find: {d.patch.find[:120]!r} -> "
            f"replace: {d.patch.replace[:120]!r}"
            for d in self.state.project.pending.list_all()
            if d.id in tracked]
        if not lines:
            return ""
        return ("PENDING DIFFS AWAITING APPROVAL:\n" + "\n".join(lines) +
                "\nThe user's utterance may be about these — interpret it "
                "against the approval, never by string matching: apply -> "
                "pending_decide(diff_id, 'apply'); discard -> "
                "pending_decide(diff_id, 'discard'); 'apply all' -> "
                "pending_decide('all', 'apply'). To amend: propose the "
                "amended patch with section_write first; once it is "
                "pending, discard the superseded diff with "
                "pending_decide(diff_id, 'discard'). If the utterance is "
                "clearly about neither, just answer — the diffs stay "
                "pending.")

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
        # The approval-window note rides outside the budget: it is small,
        # capped, and dropping it would silently close the window (§3).
        note = self._pending_note()
        if note:
            content = f"{content}\n\n{note}" if content else note
        ctx_msg = {"role": "system", "content": content} if content else None
        return ctx_msg, start

    async def send(self, event: dict):
        # The sitting speaks to every attached tab (§7/#27): a diff
        # resolved by voice or by another tab's button lands everywhere.
        text = json.dumps(event)
        for cid, ws in list(self.sockets.items()):
            try:
                await ws.send_text(text)
            except Exception:
                self.sockets.pop(cid, None)  # dead socket: prune, keep talking

    async def run_turn(self, user_text: str):
        # No queue: one in-flight turn, last utterance wins (§8).
        self.last_active = time.monotonic()  # §7: turns, not heartbeats,
        # keep the sitting alive — an open tab with an absent writer idles.
        current = asyncio.current_task()
        if (self.turn_task is not None and self.turn_task is not current
                and not self.turn_task.done()):
            self.turn_task.cancel()
        # The window may have closed since the last turn (another tab's
        # button, cleanup) — the note must read disk truth, not a mirror.
        self.sync_pending()
        # The §7 rolling summaries group a sitting by the section each
        # exchange was anchored to — tag the user message with it.
        self.log({"role": "user", "content": user_text,
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
                    self.log({"role": "assistant", "content": text or ""})
                    await self.send({"type": "assistant_text", "text": text})
                    # TTS seam: sentence-by-sentence synthesis lands with
                    # the 3090 stack; the stub counts the request.
                    async for _chunk in self.state.tts.synthesize(text):
                        pass  # binary audio frames go out here
                    return
                self.log(msg)  # assistant turn with tool_calls
                for vc in valid_calls:
                    result = await execute_async(
                        vc, project, fetch=self.state.fetch,
                        mailto=self.state.config.crossref_mailto,
                        openalex_mailto=self.state.config.openalex_mailto,
                        corpus=self.state.corpus,
                        store=self.state.corpus_store,
                        autojoin=lambda ids: autojoin(self.state, ids),
                        # Voice decides only inside this session's window:
                        # the user is asked to approve what they have seen.
                        window={t["diff_id"]
                                for t in self.pending_diffs})
                    if result.get("status") == "pending":
                        # One-at-a-time diff awaiting approval (§5). The
                        # result's find/replace are the final ones (cite_add
                        # rewrites the \\cite key); section_write has none.
                        find = result.get("find", vc.args["find"])
                        replace = result.get("replace",
                                            vc.args["replace"])
                        await self.send({"type": "diff",
                                         "diff_id": result["diff_id"],
                                         "section": result["section"],
                                         "find": find, "replace": replace})
                        # The approval window opens: later turns' notes
                        # carry this diff until it resolves (§3). Only the
                        # id/section are mirrored — the note reads the
                        # patch itself from disk truth.
                        self.pending_diffs.append(
                            {"diff_id": result["diff_id"],
                             "section": result["section"]})
                    if result.get("status") == "resolved":
                        # §3: the agent interpreted the utterance against
                        # the approval — the card resolves exactly as if
                        # the button had been clicked.
                        for r in result["resolutions"]:
                            await self.send({"type": "diff_resolved",
                                             "diff_id": r["diff_id"],
                                             "applied": r["applied"],
                                             "reason": r["reason"],
                                             "text": r["text"]})
                        self.sync_pending()
                    self.log({"role": "tool",
                              "tool_call_id": vc.id,
                              "content": json.dumps(result)})
            await self.send({"type": "error", "where": "loop",
                             "message": "tool loop budget exhausted"})
        except asyncio.CancelledError:
            # Barge-in while thinking: abort, keep history consistent.
            self.log({"role": "assistant", "content": "[interrupted]"})
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


def load_last_project(root: Path) -> Path | None:
    """§7: the app lands in the last-used project, server-side, surviving
    restarts. A stale or removed path degrades to the caller's default."""
    try:
        raw = (root / ".phd-helper-server" / "last_project").read_text(
            encoding="utf-8").strip()
    except OSError:
        return None
    p = (Path(raw) if Path(raw).is_absolute() else root / raw).resolve()
    return p if p.is_relative_to(root) and (p / "main.tex").is_file() else None


def save_last_project(root: Path, path: Path) -> None:
    d = root / ".phd-helper-server"
    d.mkdir(parents=True, exist_ok=True)
    (d / "last_project").write_text(str(path), encoding="utf-8")


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
        # One active project at a time (§7): the app lands in the
        # last-used one; the scaffold ships the sample paper as default.
        self.projects_root = REPO_ROOT
        self.project = Project(load_last_project(REPO_ROOT)
                               or REPO_ROOT / "sample_paper")
        # The project's current sitting (§7), shared by every attached
        # tab; None between sittings (ended, or nobody has opened yet).
        self.sitting: Session | None = None
        # Set while end_sitting distills: ensure_sitting awaits it, so a
        # message arriving mid-end can't bind the pre-end project/log.
        self.end_fut: asyncio.Future | None = None
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
        # Voice stack, same gate shape (PHD_AUDIO_STACK): "off" keeps the
        # stubs (§8 honest silence); "local" swaps in thin clients over the
        # py3.12 sidecars — the models never load here, because Smart App
        # Control blocks torch in this venv (docs/audio-stack.md).
        if self.config.audio_stack == "local":
            from phd_helper.segmenting import EnergyVad
            from phd_helper.server.stt import SidecarStt
            from phd_helper.server.tts import MossTts
            self.stt = SidecarStt(
                self.config.stt_url,
                vad=EnergyVad(threshold=self.config.vad_threshold),
                hangover_s=self.config.conversation_hangover_s,
                min_utterance_s=self.config.min_utterance_s)
            self.tts = MossTts(self.config.tts_url,
                               self.config.tts_prompt_wav)
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


async def distill_memory(state, conversation) -> bool:
    """§4/§7: fold a finished sitting into paper memory. A fault skips
    the distillation — memory is polish, never an error to surface (§8);
    the next session end retries. The voice door is the memory_write
    tool; this is the session-end door. The bool feeds the §7 divider's
    distilled flag — the next open's recap says whether it ran."""
    transcript = "\n".join(
        f"{m['role']}: {m['content']}"
        for m in conversation
        if m.get("role") in ("user", "assistant") and m.get("content"))
    if not transcript.strip():
        return True  # an empty sitting has nothing to distill: done
    old = state.project.load_memory()
    try:
        _, _, text = await state.llm.chat(
            [{"role": "system", "content": MEMORY_PROMPT},
             {"role": "user",
              "content": f"Current memory:\n{old or '(empty)'}\n\n"
                         f"Session transcript:\n{transcript[:MEMORY_TRANSCRIPT_CAP]}"}],
            thinking=False, max_tokens=900)
    except Exception:
        return False  # vLLM down at the end: skip, retry next session end
    text = (text or "").strip()
    if not text:
        return False  # the model gave nothing: the divider must not
                      # claim a distilled sitting (§7 recap honesty)
    try:
        state.project.save_memory(text)
    except OSError:
        return False  # §8: a locked file (Syncthing mid-sync) skips the
                      # fold; the undistilled tail stays retryable
    return True


SUMMARY_PROMPT = (
    "Merge the existing rolling summary of this paper section with the "
    "new session's conversation about it into one summary of at most 80 "
    "words: what was written, decided and left open. Reply with the "
    "summary text only.")
SUMMARY_TRANSCRIPT_CAP = 8000  # chars of per-section transcript sent


async def distill_summaries(state, conversation) -> bool:
    """§7: fold a finished sitting into per-section rolling summaries —
    the residue of conversation that fell out of the verbatim window.
    Sections discussed with nothing selected are nobody's summary; a
    fault stops the pass, the next session end retries (§8). The bool
    feeds the §7 divider's distilled flag."""
    groups = by_section(conversation)
    if not groups:
        return True  # nothing anchored: nothing to roll up
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
            return False  # keep what landed; retry the rest next end
        text = (text or "").strip()
        if not text:
            return False  # empty for this section: nothing landed, so
                          # the divider must not claim "distilled"
        summaries.setdefault(section, []).append(
            {"date": date.today().isoformat(), "text": text})
        try:
            state.project.save_summaries(summaries)
        except OSError:
            return False  # §8: locked file — undistilled, retryable
    return True


async def ensure_sitting(state) -> Session:
    """The project's current sitting (§7: starts on open), created on
    first use. An ended sitting is replaced by the next message's call —
    but never mid-end: a message arriving while end_sitting distills
    waits for it, so the replacement binds the ended sitting's project
    and post-divider log, never the pre-end ones."""
    end = getattr(state, "end_fut", None)
    if end is not None and not end.done():
        try:
            await end
        except Exception:
            pass  # the ending caller reports its own faults; here the
                  # end simply finished and the fresh sitting follows
    sitting = getattr(state, "sitting", None)
    if sitting is None:
        sitting = state.sitting = Session(state)
    return sitting


async def revalidate_pending(state, sitting: Session,
                             ws: WebSocket) -> None:
    """§7/§8 reopen: pending diffs are disk truth, so an attaching socket
    gets them back through the apply-time hash check / re-anchor path —
    bounced ones resolve with their reason, live ones become cards again
    and re-enter the sitting's approval window, so the next utterance is
    interpreted against them (§3). Reconcile runs once per sitting; the
    cards are sent to every socket that attaches."""
    if not sitting.pending_sent:
        sitting.pending_sent = True
        by_sec: dict[str, list] = {}
        for d in state.project.pending.list_all():
            by_sec.setdefault(d.section_path, []).append(d)
        for section, ds in by_sec.items():
            try:
                text = state.project.read_section(section)
            except OSError:
                for d in ds:  # the file itself is gone: nothing to
                    state.project.reject_pending(section, d.id)  # re-anchor
                    await ws.send_text(json.dumps(
                        {"type": "diff_resolved", "diff_id": d.id,
                         "applied": False,
                         "reason": "section file is gone", "text": None}))
                continue
            for out in state.project.pending.reconcile(section, text):
                if out.bounce_reason:
                    await ws.send_text(json.dumps(
                        {"type": "diff_resolved", "diff_id": out.diff.id,
                         "applied": False, "reason": out.bounce_reason,
                         "text": None}))
                else:
                    sitting.live_pending.append(out.diff)
                    if not any(t["diff_id"] == out.diff.id
                               for t in sitting.pending_diffs):
                        sitting.pending_diffs.append(
                            {"diff_id": out.diff.id, "section": section})
    for d in sitting.live_pending:
        await ws.send_text(json.dumps(
            {"type": "diff", "diff_id": d.id, "section": d.section_path,
             "find": d.patch.find, "replace": d.patch.replace}))


async def sync_socket(state, sitting: Session, client_id: str,
                      ws: WebSocket, on_connect: bool = False) -> None:
    """§8 resync on (re)open — plain refetch, server is the source of
    truth: the recap of how the last sitting ended (§7, once per
    sitting), the anchor (hello carries it on connect; a re-attach to a
    replacement sitting gets it as an event), and the pending diffs."""
    sitting.attach(client_id, ws)
    if sitting.recap and not sitting.recap_sent:
        # §7: the one-line on-screen recap — on screen only, the
        # reply-only voice rule holds.
        sitting.recap_sent = True
        await ws.send_text(json.dumps({"type": "recap",
                                       "text": sitting.recap}))
    if not on_connect:
        await ws.send_text(json.dumps({"type": "section_selected",
                                       "section": sitting.selected}))
    await revalidate_pending(state, sitting, ws)


def idle_expired(last_active: float, now: float, timeout_s: float) -> bool:
    return now - last_active >= timeout_s


async def end_sitting(state, reason: str) -> None:
    """§7: the sitting ends — distill, dated divider, one-line receipt to
    any tab still attached. A sitting that never saw a user turn leaves
    no trace. The §8 crash path never reaches here: no divider is
    written, so the next open resumes the sitting verbatim. A distill
    FAULT does write the divider, but flagged undistilled — read_tail
    only closes the verbatim window on a distilled one, so the talk
    stays resumable and the next sitting's end retries the fold (the
    promise the distill-fault comments make)."""
    sitting = getattr(state, "sitting", None)
    if sitting is None:
        return
    state.sitting = None
    end = asyncio.get_running_loop().create_future()
    state.end_fut = end  # messages arriving mid-distill wait (ensure_sitting)
    try:
        sitting.cancel_turn()
        conversation = sitting.history[1:]
        turns = [m for m in conversation if m.get("role") == "user"]
        if not turns:
            return
        mem_ok = await distill_memory(state, conversation)
        sum_ok = await distill_summaries(state, conversation)
        sections = {m["section"] for m in turns if m.get("section")}
        divider = sessionlog.divider_record(reason, len(turns), sections,
                                            mem_ok and sum_ok)
        try:
            state.project.write_chat_divider(divider)
        except OSError:
            pass  # §8: a locked file loses the divider, not the sitting
                  # — with no divider the tail stays verbatim anyway
        await sitting.send({"type": "session_ended", "reason": reason,
                            "recap": sessionlog.recap_line(
                                divider["divider"], 0)})
    finally:
        state.end_fut = None
        if not end.done():
            end.set_result(None)


async def idle_watch(state, check_s: float = 5.0) -> None:
    """§7: ~30 min idle ends the sitting (distillation runs). Turn
    activity — not heartbeats — resets the clock: an open tab with an
    absent writer is an idle sitting."""
    while True:
        await asyncio.sleep(check_s)
        sitting = getattr(state, "sitting", None)
        if sitting is not None and idle_expired(
                sitting.last_active, time.monotonic(),
                state.config.session_idle_s):
            try:
                await end_sitting(state, "idle")
            except Exception:
                pass  # §8: the watchdog outlives one failed ending —
                      # a dead watchdog means no sitting ever ends again




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
        # A plain task, not spawn(): the watchdog must not join the
        # ingest-task set that tests (and shutdown) await to completion.
        watch = (asyncio.create_task(idle_watch(state))
                 if getattr(state, "config", None) is not None else None)
        yield
        # §7: graceful shutdown ends the sitting (distillation runs,
        # divider lands). An ungraceful crash never reaches here — the
        # next open's recap notes the abrupt end, resumed verbatim.
        if watch is not None:
            watch.cancel()
        try:
            await end_sitting(state, "shutdown")
        finally:
            # A faulting ending must not leak the shared HTTP client.
            await state.http.aclose()  # release the shared client on shutdown
            # Audio clients exist only on the local stack; the
            # SimpleNamespace test states have neither the providers nor
            # their aclose, hence both guards.
            for provider in (getattr(state, "stt", None),
                             getattr(state, "tts", None)):
                if provider is not None and hasattr(provider, "aclose"):
                    await provider.aclose()

    app = FastAPI(title="phd-helper", lifespan=lifespan)
    app.state.phd = state

    @app.get("/health")
    async def health():
        # §8: the store is probed, not config-read — a wired-but-faulting
        # LanceDB (locked file, corrupt manifest) must read faulted, not ok.
        corpus = "paused"
        if state.corpus_store is not None:
            corpus = "ok" if await state.corpus_store.healthy() else "faulted"

        async def audio_status(p):
            # Stubs answer "stub"; sidecar adapters answer with a live
            # probe. Duck-typed on healthy() so the stub-injecting tests'
            # SimpleNamespace states keep answering without change.
            if p.faulted():
                return "faulted"        # counter is truth until a turn heals it
            if not hasattr(p, "healthy"):
                return "stub"
            return "ok" if await p.healthy() else "faulted"

        return {
            "vllm": await state.llm.healthy(),
            "stt": await audio_status(state.stt),
            "tts": await audio_status(state.tts),
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
        state.endpoint.heartbeat(client_id, time.monotonic())
        sitting = None
        try:
            sitting = await ensure_sitting(state)
            # hello carries the sitting's anchor: the client's own copy
            # is view state that a server restart can have invalidated —
            # the server's answer is the truth it refetches to (§8).
            await ws.send_text(json.dumps({"type": "hello",
                                           "client_id": client_id,
                                           "section": sitting.selected}))
            await sync_socket(state, sitting, client_id, ws,
                              on_connect=True)
            while True:
                frame = await ws.receive()
                if frame["type"] == "websocket.disconnect":
                    break
                if (data := frame.get("bytes")) is not None:
                    finals = await state.stt.feed(data)
                    if finals:
                        # Voice opens a sitting exactly like text does:
                        # after an idle-end the mic must not run a turn
                        # on the ended sitting — re-resolve, then attach.
                        sitting = await ensure_sitting(state)
                        if sitting.sockets.get(client_id) is not ws:
                            await sync_socket(state, sitting, client_id, ws)
                        for final in finals:
                            sitting.turn_task = asyncio.create_task(
                                sitting.run_turn(final))
                    continue
                if (text := frame.get("text")) is None:
                    continue
                try:
                    msg = parse_control(text)
                except ValueError as e:
                    await ws.send_text(json.dumps(
                        {"type": "error", "where": "control",
                         "message": str(e)}))
                    continue
                if msg["type"] == "heartbeat":
                    # One mechanism, three consumers (§8): liveness,
                    # meter, watchdog. Liveness needs no sitting — a
                    # heartbeat must not resurrect an ended one — but a
                    # live sitting keeps its tabs attached (fan-out),
                    # and a tab whose sitting was replaced under it
                    # (project switch) re-syncs here without speaking.
                    state.endpoint.heartbeat(client_id, time.monotonic())
                    cur = getattr(state, "sitting", None)
                    if cur is not None and \
                            cur.sockets.get(client_id) is not ws:
                        sitting = cur
                        await sync_socket(state, cur, client_id, ws)
                    await ws.send_text(json.dumps(
                        {"type": "pong", "rms": msg.get("rms", 0.0)}))
                    continue
                # The sitting may have ended under this tab (idle,
                # switch); the next message opens the next sitting.
                sitting = await ensure_sitting(state)
                if sitting.sockets.get(client_id) is not ws:
                    await sync_socket(state, sitting, client_id, ws)
                await dispatch(state, sitting, msg, client_id, ws)
        except WebSocketDisconnect:
            pass
        finally:
            # §7: a tab closing is not a session end. The sitting — and
            # any in-flight turn — survives for the other tabs and the
            # next reconnect; only the socket detaches.
            if sitting is not None:
                sitting.detach(client_id, ws)

    async def dispatch(state, sitting: Session, msg: dict, client_id: str,
                       ws: WebSocket):
        now = time.monotonic()
        kind = msg["type"]
        if kind == "arm":
            ok = state.endpoint.arm(client_id, now)
            await ws.send_text(json.dumps(
                {"type": "armed", "ok": ok,
                 "holder": state.endpoint.endpoint(now)}))
        elif kind == "disarm":
            sitting.cancel_turn()
            await ws.send_text(json.dumps({"type": "disarmed"}))
        elif kind == "typed":
            text = str(msg.get("text", "")).strip()
            if text:
                sitting.turn_task = asyncio.create_task(
                    sitting.run_turn(text))
        elif kind == "barge_in":
            # Qualifying interrupt: stop TTS now, abort thinking (§3).
            # Fan-out: every tab's playback stops with the sitting's turn.
            sitting.cancel_turn()
            await sitting.send({"type": "tts_stopped"})
        elif kind == "select_section":
            # §1: clicking a tree node anchors the section-scoped
            # discussion; the body then rides every later turn (§4).
            # Fan-out: one sitting, one anchor — every tree follows.
            path = str(msg.get("section", ""))
            if sitting.select_section(path):
                await sitting.send({"type": "section_selected",
                                    "section": sitting.selected})
            else:
                await ws.send_text(json.dumps(
                    {"type": "error", "where": "control",
                     "message": f"no such section: {path}"}))
        elif kind in ("approve", "reject"):
            section, diff_id = msg.get("section"), msg.get("diff_id")
            if kind == "approve":
                result = state.project.apply_pending(section, diff_id)
                await sitting.send({
                    "type": "diff_resolved", "diff_id": diff_id,
                    "applied": result.applied, "reason": result.reason,
                    "text": result.text})
            else:
                state.project.reject_pending(section, diff_id)
                await sitting.send({"type": "diff_resolved",
                                    "diff_id": diff_id, "applied": False,
                                    "reason": "discarded", "text": None})
            # The button closed (or bounced) the window — the note follows
            # disk truth, same as the voice path (§3).
            sitting.sync_pending()

    # -- projects (SPEC §7): server-side list, one active at a time;
    # switching ends the sitting (distillation runs) and lands there,
    # remembered across restarts. New/Import are later slices.

    @app.get("/projects")
    async def projects():
        root = getattr(state, "projects_root", REPO_ROOT)
        names = sorted(d.name for d in root.iterdir()
                       if d.is_dir() and not d.name.startswith((".", "_"))
                       and (d / "main.tex").is_file())
        return {"projects": names, "active": state.project.root.name}

    @app.post("/projects/activate")
    async def project_activate(request: Request):
        root = getattr(state, "projects_root", REPO_ROOT)
        body = await request.json()
        name = str(body.get("name", ""))
        target = (root / name).resolve() if name else None
        if (target is None or not target.is_relative_to(root)
                or not (target / "main.tex").is_file()):
            return JSONResponse({"error": f"no such project: {name}"},
                                status_code=404)
        await end_sitting(state, "switch")  # distills before the swap
        state.project = Project(target)
        save_last_project(root, target)
        return {"active": target.name}

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
