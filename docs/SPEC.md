# phd-helper — Build-Ready Spec

The consolidated spec for **phd-helper**: a personal local web app where the user converses hands-free (voice in, voice out) with an agent running on their own vLLM-served Qwen model over LAN, co-writing LaTeX papers section by section, backed by a hybrid-search PDF reference corpus and academic-first web search.

Every decision here was made on the [wayfinder map](https://github.com/soderholmen/phd-help/issues/1); each section cites its source ticket. Vocabulary is normative in [`CONTEXT.md`](../CONTEXT.md). This spec is the handoff artifact: the build effort executes it and measures against it; changing a decision here means reopening it against the map.

**Out of scope, by standing decision:** local LaTeX compilation (deferred past v1), cloud model APIs (local models only), multi-user anything (single user).

---

## 1. Architecture and topology

*(Source: [App architecture and topology](https://github.com/soderholmen/phd-help/issues/7), [Document the serving setup](https://github.com/soderholmen/phd-help/issues/13))*

- **One backend process, on the vLLM server.** A single FastAPI/uvicorn Python process hosts: the agent loop, STT, TTS, embeddings, LanceDB, and the web app itself. The LLM stays on vLLM on the same box.
- **Clients are browsers only**: the Windows box and the phone. No native apps.
- **Access is ZeroTier-only.** The backend binds to the ZeroTier network; no public port, no app-level auth — ZeroTier membership *is* the access control.
- **Audio transport**: browser `getUserMedia` (with `echoCancellation`) → AudioWorklet → PCM chunks over **one WebSocket per tab**; TTS audio streams back on the same socket into a Web Audio scheduling queue, which implements barge-in stop/resume.
- **Stack**: FastAPI + uvicorn backend; React + Vite frontend; CodeMirror 6 for the section view and inline diffs.
- **Files are server-first**: paper projects, the reference corpus, and LanceDB live on the server's filesystem. **Syncthing** replicates paper folders and the corpus to the Windows box for local access and backup. The section tree is parsed from the root file's `\input`/`\include` graph; clicking a node anchors the section-scoped discussion.
- **One voice endpoint** (glossary): the device that arms conversation mode captures and plays audio; all other clients are view-only. The phone is a first-class **voice + review client**; heavy LaTeX editing stays on the desktop.

### GPU layout (server)

*(Source: [Document the serving setup](https://github.com/soderholmen/phd-help/issues/13) — supersedes the 96 GB co-residency fallback ladder from #7, which is retired.)*

- **96 GB RTX PRO 6000 (Blackwell)**: LLM + KV cache only.
- **3090 (24 GB)**: the audio stack — STT (silero-vad + nemotron-streaming + parakeet), TTS (MOSS-TTS-Realtime), harrier-oss embeddings, and MinerU extraction (batch/offline). ~7–8 GB live + extraction headroom.
- No cross-architecture tensor-parallel pooling; separate processes on one machine, one server topology.

## 2. Serving setup

*(Source: [Document the serving setup](https://github.com/soderholmen/phd-help/issues/13), [Qwen3 tool-calling on vLLM](https://github.com/soderholmen/phd-help/issues/4). Status: **deployed 2026-09-29** at the live endpoint below; one deviation open — MTP is ON on the running server, relaunch with it OFF pending server shell access.)*

- **Model**: `local-inference-lab/Qwen3.8-Flash-Next-NVFP4` (N-gram/PLE table FP4-quantized, offloaded to ~27 GiB host RAM; host has 64 GB).
- **Serving path**: docker image `ghcr.io/local-inference-lab/vllm:karmic-kraken-beta`, turnkey profile `qwen38-flash-next`, TP1. **Plain `pip install vllm` cannot serve this model.**
- **Launch flags**: `--enable-auto-tool-choice --tool-call-parser qwen3_coder --reasoning-parser qwen3 --tool-strict-level function --max-model-len 262144`, env `VLLM_PLE_CPU_OFFLOAD=1`. (`qwen3_xml` is the same parser class in vLLM ≥0.30.)
- **MTP / speculative decoding: OFF at launch** (open tool-call corruption bug, vllm#56077 family). Re-enabling is a post-build measurement, not a launch option. **Deviation (2026-09-29):** the deployed server runs with MTP ON (spec-decode active, ~74/53/39% per-position acceptance); a 6-turn multi-turn tool-call probe found no corruption, but the OFF decision stands — relaunch pending server shell access.
- **Endpoint**: `http://10.147.242.50:8888` (ZeroTier, API-key auth — key in the local key store, never in the repo). Context 262K. First-run measurements 2026-09-29 (non-thinking, n=3 medians): decode **44.6 tok/s**, TTFT **~388 ms @1K in**. The earlier 173–190 tok/s / ~190 ms figures did not reproduce on this deployment; cause unknown until server-side checks land.
- **Fallback order**: (1) decided NVFP4 path; (2) `Qwen3.8-27B-FP8` via pip vLLM ≥0.17, same flags with `qwen3_xml`, ~2× slower; (3) official `vllm/vllm-openai:qwen38-flash-next` image + patch stack + +16 GB RAM, only if both fail.
- **Sampling (agentic, thinking mode)**: `temperature=1.0, top_p=0.95, top_k=20, min_p=0.0, presence_penalty=0.0, repetition_penalty=1.0`; generous max_tokens (never starve thinking — truncated thinking drops tool calls). Non-thinking mode: `temp 0.7 / top_p 0.8 / presence_penalty 1.5`.
- **Pending first-run facts** (build effort measures, not decisions): image digest and free-VRAM check — still open, blocked on server shell access; MTP re-validation — superseded by the deviation note above (server must be relaunched with MTP OFF, then re-measure decode/TTFT).

### Tool-calling contract

*(Sources: [Qwen3 tool-calling on vLLM](https://github.com/soderholmen/phd-help/issues/4), [Tool-call reliability prototype](https://github.com/soderholmen/phd-help/issues/8) — 21-turn probe on the served model: parser held; all misses were prompt-policy failures.)*

- Define every tool with OpenAI strict-style schemas (`additionalProperties: false`, all fields required, optional as `["type","null"]`); the server-side `--tool-strict-level function` floor grammar-constrains the envelope.
- **Mandatory client-side validation** of every returned tool call before execution: name ∈ offered set, dedupe, JSON-parse `arguments`, required params present, patch `find` actually exists in the section. On failure: **bounce back to the model** (retry with the validation error), never surface to the user, never `eval` unparseable arguments.
- **Never rely on `tool_choice="required"`** for control flow (open vLLM bug); enforce "must call a tool" in the loop with a nudge retry.
- Prefer non-streaming for the tool-decision turn, or re-validate streamed JSON; tolerate empty content deltas; re-issue malformed streamed turns non-streaming.
- Check both `content` and `reasoning` for stray tool markup — literal tool markup in content is a **known corruption vector** (the #57699 family); appendix-example use cases need a code-block-safe path.
- **System prompt must carry these two rules verbatim** (they fixed every probe failure):
  1. "If the user asks to edit text already in your context, emit the `section_write` patch directly; do NOT search first to verify wording — search only for new facts/citations/corpus content."
  2. "Questions about the writing of the current section (flow, clarity, comparisons within it) are answered directly from context, with no tool call."

## 3. Voice pipeline

### Capture and STT

*(Sources: [STT for hands-free conversation and dictation](https://github.com/soderholmen/phd-help/issues/6) — model picks; placement **superseded** to the server by [App architecture and topology](https://github.com/soderholmen/phd-help/issues/7); [Voice loop prototype](https://github.com/soderholmen/phd-help/issues/11) — proven loop.)*

- **Two-tier hybrid**: silero-vad v6 capture gating; `nvidia/nemotron-speech-streaming-en-0.6b` (560 ms chunks) for replaceable live partials (UI transcript + barge-in detection); `nvidia/parakeet-tdt-0.6b-v3` for the single authoritative final per utterance. Runs on the server's 3090 inside the backend process — server placement is required, not just convenient: every client is a browser (including the phone), so the STT models must live where the backend does.
- **AGC on the capture path + a persistent input-level meter** in the UI; silero's threshold stays fixed post-AGC. (The prototype's "crash/cut-off" was low mic level vs a fixed threshold — metering makes it self-diagnosing.)
- Measured floors: STT finals 270–540 ms; the prototype proved the full loop (browser mic → VAD → finals → TTS) at ~1 s perceived dead air.

### TTS

*(Source: [Conversational TTS that runs locally](https://github.com/soderholmen/phd-help/issues/3).)*

- **Primary: MOSS-TTS-Realtime** (Apache-2.0, ~1.9B, ~4–5 GB VRAM, 180 ms TTFB) — chosen for multi-turn text+acoustic context conditioning; served on the 3090, OpenAI-compatible streaming endpoint.
- **Baseline/fallback: Kokoro-82M** (phoneme-level pronunciation control for academic terms — the escape hatch for names/Greek letters). Runner-up for permissive voice cloning: Chatterbox-Turbo.
- **Pre-normalize math in the LLM's output stage** ("E equals m c squared") — AR TTS hallucinates on dense symbol strings; the prompt controls this cheaply.
- The agent speaks **sentence-by-sentence off the token stream**, never waiting for the full reply.

### Turn-taking

*(Source: [Conversation-mode turn-taking](https://github.com/soderholmen/phd-help/issues/5).)*

- **Arming**: dedicated toggle only — never auto-armed, never push-to-talk, no wake word. On-screen toggle + keyboard shortcut + voice command all flip it. Capture = toggle AND tab-focused (tabbing away pauses capture, preserves toggle state). Toggle off ⇒ typed chat + editor, screen-only replies.
- **Endpointing**: 0.6 s hangover for conversation turns (≈1 s dead air, validated); ~1.0 s for dictation segments. Config settings.
- **Barge-in while speaking**: full duplex, mic hot during TTS with browser AEC; interrupt requires speech sustained ~150–200 ms; qualifying interrupt stops TTS immediately; an interrupt yielding no final transcript within a short window resumes TTS where it stopped.
- **Barge-in while acting**: voice interrupts thinking and speaking instantly; in-flight tool calls abort-and-discard; **atomic writes are never interrupted mid-apply** — the interrupting utterance queues and is heard right after the write lands.
- **Narration**: backchannel only (one short filler when an operation plausibly exceeds ~2 s); details on screen.
- **Voice vs screen**: voice carries dialogue only — replies, approval prompts, backchannels. Long prose, diffs, search results are screen-only with a one-line voice pointer ("three candidates on screen"). Ratified exception: **drafts are spoken** — a fenced draft is read sentence-by-sentence as it generates, because the draft's decode wait is exactly where the voice earns its place (stepwise writing).
- **Dictation vs conversation**: the agent classifies intent every turn (the cleanup pass does the routing); explicit voice overrides ("take this down" / "back to chat"); the screen always shows the current mode.
- **Approval window**: while a diff is pending, the next utterance is interpreted against the approval (approve/reject/amend) by the agent, not string-matching; clearly-neither ⇒ diff stays pending, conversation resumes.
- **Proactive speech**: voice is reply-only in v1; non-reply events go to the screen notification tray. The one exception is **Alerts** (§8).

### Mobile

*(Source: [Mobile browser voice pipeline support](https://github.com/soderholmen/phd-help/issues/15); mitigations + 5-item on-device verification checklist in `docs/research/mobile-browser-voice.md` on branch `research/mobile-browser-voice`.)*

- **Android Chrome = full voice endpoint**, including screen-off (capture survives via `kCapturingAudio`/foreground-service exemptions). Wildcard: OEM battery managers — on-device test required at build.
- **iOS Safari = foreground-only endpoint**: WebKit mutes the mic on page hide and closes the WebSocket. The endpoint lease is foreground-only on iOS: auto-pause on page hide, explicit tap-to-resume. (If hidden >60 s the lease expires per §8 — re-arm on return.) Push-to-talk does not help.
- **Both**: HTTPS secure context with a **phone-trusted cert** on the ZeroTier host, and a **hostname (not IP) for the wss URL** (WebKit reconnect bug #308073). One user-gesture tap arms AudioContext + mic.
- Phone barge-in mitigations: AEC warmup preroll, barge-in word-count gating (self-echo worst on first utterance).

## 4. Agent behavior and context

*(Source: [Agent context and write-back design](https://github.com/soderholmen/phd-help/issues/10).)*

### Per-turn context assembly

- Selected **section in full** + **paper skeleton** (title, abstract, all headings, one-line gist per section; gists auto-regenerate whenever a section file changes) + **paper memory** (persistent decisions/claims/terminology/TODOs; distilled at session end and on voice command; user-editable side panel; a few hundred tokens) + conversation history (verbatim until budget, then per-section rolling summaries).
- **Pinned sources** attached to the section always contribute abstract + headings; chunk-searched for detail. New corpus/web searches are agent tool calls, never auto-stuffed.
- **Fixed drop-priority** (assembly is a priority list, window-agnostic): never drop the selected section, paper memory, or skeleton; drop in order — pinned-paper abstracts/headings → conversation beyond the rolling summary → far-away section gists.

### Cleanup pass

A dedicated fast transformation (fixed-format task, fast sampling config) turns raw dictation transcript into clean instruction/prose **before** the agent's turn; the cleaned text is what enters history. Shown as the user's chat message, **not gated** — mishears are corrected by voice next turn.

### Agent tools

`corpus_search`, `corpus_doc` (page/block locators), `web_search`, `section_read`, `section_write` (anchored patch). *(Probe: [Tool-call reliability prototype](https://github.com/soderholmen/phd-help/issues/8).)*

## 5. Writing into the paper

*(Sources: [Agent context and write-back design](https://github.com/soderholmen/phd-help/issues/10), [Editing UX and undo](https://github.com/soderholmen/phd-help/issues/17), [In-app LaTeX preview depth](https://github.com/soderholmen/phd-help/issues/16).)*

### Write mechanism

- Default write: **anchored find/replace patch** — untouched prose stays byte-identical. Full-section rewrite only on explicit user command.
- Every patch shows as a **one-at-a-time diff requiring approval**, voice-approvable ("apply" / "discard" / "apply all" for trusted batches).
- **Deterministic LaTeX lint** runs on the resulting file *before* the diff is shown (brace balance, matched begin/end, citations to unknown `.bib` keys); failures bounce back to the agent, never reaching the user's approval attention.
- **Clobber protection**: apply-time hash check; if the file changed since the patch was computed, deterministically fuzzy-re-anchor (unique close match) or reject on ambiguity with the reason shown. The model is never asked to reconcile the user's edit away.

### Editing surface

- **Desktop**: CodeMirror source mode with **debounced autosave** (~0.5–1 s after typing pauses, saved/dirty indicator) — keeps the server file matching the screen so the apply-time hash check doesn't become daily friction.
- **Phone**: source view is **read-only**; small fixes go through voice → anchored patch through normal approval.
- Persistent history covers **agent applies and full-section rewrites only**; user typing is covered by editor-session undo. The tool never reverts the user's own edits.

### Undo and history

- **Snapshot** of the file's prior bytes server-side before every agent apply / full-section rewrite; last N per file (default ~20); survives restarts; no git.
- **Undo = inverse anchored patch** through the same re-anchor + lint machinery. Any past apply is undoable; bare "undo" = most recent; a descriptive voice pointer ("undo the citation change") inverts *that* apply. User-commanded undo **lands directly, no approval window**.
- **Section history** list (timestamp + gist + restore) exposes blunt snapshot **revert** — discards everything since, including the user's edits, so it requires an **on-screen confirmation**; voice may initiate, never confirm.
- Git rejected as the undo mechanism (interleaves with Syncthing-arriving VS Code edits; ceremony).

### Section view

- **Read view** (rendered, read-only) is the default on every device; toggle to CodeMirror source for editing. One client-side LaTeX→HTML renderer, one contract, both devices; no server render endpoint.
- **Fidelity contract (normative)**: *never drop content — anything the renderer can't render shows as raw source, never vanishes.* Allowlist rendered: sectioning, text formatting, inline+display math (KaTeX), lists, `quote`, footnotes inline-bracketed, figures/tables as caption boxes. Raw: `\ref`/`\eqref`, unknown macros, everything off-allowlist. No preamble handling (section files are `\input` fragments).
- `\cite`/`\citep`/`\citet` render as `[Author, year]` via deterministic `refs.bib` lookup; unknown keys fall back to raw.
- Math renders **only** in the Read view; the editor stays pure highlighted source.
- **Diffs stay raw source diff everywhere** — a rendered diff risks approving a render artifact.

## 6. Corpus, search, and citations

*(Sources: [PDF extraction + hybrid search stack](https://github.com/soderholmen/phd-help/issues/9), [Corpus and citation flow](https://github.com/soderholmen/phd-help/issues/12), [arXiv and Semantic Scholar APIs](https://github.com/soderholmen/phd-help/issues/2).)*

### Corpus pipeline (all server-side, on the 3090)

- **Two doors, one pipeline**: web-UI upload + agent-fetched PDFs. **No drop folder.** Async indexing with visible per-PDF status (queued / extracting / indexed / failed); papers from search with an openly downloadable PDF **auto-join** the corpus; paywalled papers get a bib entry only. Dedup by content hash; a new arXiv version of an owned paper is re-fetched on request, not tracked automatically.
- **Extraction**: MinerU 4.0 standard tier (page/block citation locators). Figure locators are MinerU's real `page:N/block:M`; text/table/equation/heading blocks carry a synthesized page-local block id (1-based, reset per page) — stable against a reparse of a single page, not against a re-segmentation of one. Quality ceiling if ever needed: Chandra 2 served on the server.
- **Chunking**: section-aware on MinerU's block tree; heading-path prefix prepended (`paper title » section`); tables/figures as standalone chunks; ~512–1024 token cap.
- **Embeddings**: harrier-oss-v1-0.6b (MIT); upgrade path harrier-oss-v1-27b or Qwen3-Embedding-8B served on the server.
- **Store**: embedded **LanceDB** — Tantivy FTS (BM25) + vector + RRF fusion in one library call, no server; rerank (Qwen3-Reranker-0.6B over the fused top-50) rides as a second stage on the store side, since LanceDB has no native reranker. Brute-force vectors are fine at this scale (~10⁵ chunks).
- **One global index** across projects; pinned docs boosted, not filtered.
- **Agent-facing only**: `corpus_search(query, k, boost_pinned)` → ranked chunks with page/block locators + doc ids; `corpus_doc(doc_id)` → abstract, headings, bib status. Locators are what let the agent cite precisely and say "§3.2 of the parakeet paper" in voice. **No user-facing search UI** — the user asks the agent by voice.

### Academic-first search and the cite loop

- **Discovery**: OpenAlex `/works?search=` with free key (primary, ~1,000 searches/day); arXiv API (keyless, **1 req/3 s is a rule**) for preprint/category-scoped search; Semantic Scholar `paper/search` only behind a key + exponential backoff, secondary.
- **Citation graph**: OpenAlex `cites`/`cited_by` by default (cheap, CC0); S2 `/citations` + `/references` for richer edges when the key is configured.
- **BibTeX resolution cascade** (stop at first success): arXiv id → `arxiv.org/bibtex/<id>`; CS venue → DBLP search → `dblp.org/rec/<key>.bib` (browser-like UA; Anubis anti-bot ⇒ graceful fallback); has DOI → Crossref `x-bibtex` transform (add `mailto=` for the polite pool); last resort → build from OpenAlex metadata (CC0).
- **Politeness defaults baked in**: arXiv 3 s delay single-connection; OpenAlex key + `per-page=100`; Crossref `mailto`; S2 backoff on 429, never circumvent; DBLP trivial volume, never defeat the challenge.
- **Cite loop**: single `refs.bib` per project; bib entry + prose + `\cite` land in **one anchored patch** — one voice approval covers all three; pre-approval lint rejects unresolved/duplicate keys, so hallucinated keys can't ship. Keys like `vaswani2023attention`, deduped; re-citing an owned paper reuses its key (the cascade only runs for new entries).
- **Keys**: read from server-side env (gitignored, e.g. `~/.config/phd-helper/.env`), never committed. Storage location recorded on [Request a Semantic Scholar API key](https://github.com/soderholmen/phd-help/issues/14) (open: request submitted, awaiting key; S2 keys prune after ~60 days inactive).

## 7. Sessions, persistence, projects

*(Source: [Session persistence](https://github.com/soderholmen/phd-help/issues/18).)*

- A **session** = one sitting in a project: starts on open, ends on project switch, graceful server shutdown, or ~30 min idle. A paper spans many sessions; sessions reset nothing — reopening resumes the same history (recent turns verbatim, older in rolling summaries) plus paper memory.
- Idle/shutdown endings distill paper memory; one-line tray receipt; next open shows a one-line **on-screen** recap (reply-only voice rule honored).
- **Persistent per project in `.phd-helper/`** inside the project folder (rides Syncthing, user-inspectable, one gitignore line): JSONL verbatim history, per-section rolling summaries with dated session dividers, paper memory, snapshots, **pending diffs** — re-presented on reopen through the apply-time hash check / re-anchor path (stale-but-re-anchorable re-anchor; ambiguous bounce with reason).
- **Never auto-rearm**: the server never re-arms a microphone after a restart; re-arming is explicit.
- **View state** (selected section, Read/source, scroll) is per-client localStorage.
- **Projects**: server-side list, one active at a time; app lands in last-used (server-side, survives restarts); switch ends the session (distillation runs); New = scaffold from template; Import = point at an existing server-side LaTeX folder.

## 8. Failure and latency envelope

*(Sources: [Failure and latency envelope](https://github.com/soderholmen/phd-help/issues/19), [App architecture and topology](https://github.com/soderholmen/phd-help/issues/7), [Voice loop prototype](https://github.com/soderholmen/phd-help/issues/11).)*

### Latency acceptance bar

End-of-speech → first spoken word: **target ≤1.5 s, hard ceiling 2.5 s**. Shares: STT final ≤0.6 s, LLM time-to-first-speakable-sentence ~1.0 s, TTS-first ~0.2 s. The build measures against this bar (prototype floors: STT 270–540 ms, TTS-first 50–102 ms, hangover 0.6 s).

### Connection health

- **One heartbeat**: each client pings every ~2 s carrying mic RMS — one mechanism, three consumers: liveness, input-level meter feed, silent-mic watchdog. Two missed round trips (~4–5 s) ⇒ connection declared dead.
- **Connection blip** (<60 s): the armed voice endpoint and the **endpoint lease** survive; reconnect resumes capture silently, mid-conversation. **Blip audio is dropped, not replayed** — the transcript shows a visible gap marker.
- **Endpoint lease** expires after 60 s of dead connection; the toggle becomes available on every client; arming elsewhere is the explicit handoff. A returning departed device is a view-only client with a tray notice ("voice endpoint moved to <device>") — never a surprise re-arm.
- Server restart or lease expiry ⇒ explicit re-arm (§7's never-auto-rearm rule).

### Alerts — the one exception to reply-only voice

An **Alert** is one short spoken sentence, spoken **only** when the voice endpoint is armed *and* the failure itself breaks the voice loop (mic dead, server unreachable, STT/TTS down). Spoken once per incident, then banner-only. Everything else stays tray-only.

### Server outage

Persistent "server unreachable — reconnecting" banner on every client; auto-reconnect with backoff; on reconnect the client refetches section files + pending diffs + transcript tail (server is source of truth; resync = plain refetch). Accepted losses: ≤1 s of typing in the autosave debounce window; the in-flight agent turn. Pending diffs survive (§7). An ungraceful crash skips that sitting's memory distillation — the next open's recap notes the session ended abruptly.

### Partial-failure matrix

Principle: **every component failure degrades to the next-lower modality, never to a dead app.** The backend health-probes each component so state is known, not discovered mid-turn.

| Failure | Behavior |
|---|---|
| STT down | Voice toggle visibly faults (red, tooltip); typed chat + TTS replies continue |
| TTS down | Replies screen-only (the view-only path); toggle shows degraded |
| Embeddings/LanceDB down | `corpus_search`/`corpus_doc` return tool errors; agent says one line and continues via web search; indexing pauses with per-PDF status |
| vLLM down, backend up | Chat shows error inline; if armed, one spoken Alert; files, Read view, history stay browsable |
| Audio GPU gone | STT+TTS+embeddings at once: three tray entries, one spoken Alert |

### Mic hardware events

Track mute/ended, device change, unplug ⇒ **auto-reacquire and stay armed** (arming is tied to the toggle, not a stream); device switch gets a one-line tray notice; failed reacquisition after a couple of tries faults the toggle red + one spoken Alert. **RMS watchdog**: armed + ~10 s of near-zero RMS ⇒ banner pointing at the input meter.

### Latency overrun and jitter

- **No queue**: one in-flight turn, last utterance wins (barge-in already aborts thinking, so utterances can't pile up).
- Ceiling blown: backchannel once at ~2 s; still not speaking by ~8 s ⇒ tray notice "still working"; talking again always aborts. No adaptive endpointing, no load-shedding — the 3090 bought the headroom; measure, don't design around its absence.
- **Cellular jitter**: fixed ~250–300 ms TTS jitter buffer; underrun = clean pause at the buffer edge, resume on arrival, never time-compressed catch-up; barge-in stays live during underrun; persistent underrun >5 s ⇒ degraded-connection tray notice, not a failure.

## 9. Handoff to the build effort

Open items the build inherits (none are decisions):

1. **API keys** — [Request a Semantic Scholar API key](https://github.com/soderholmen/phd-help/issues/14): request submitted, awaiting the key (~1-month backlog possible); record storage location on resolution. OpenAlex key: instant, same env file.
2. **First-run serving measurements** — image digest and free-VRAM check still open (need server shell access); endpoint + decode/TTFT measured 2026-09-29 and recorded in §2. **Relaunch the serving profile with MTP OFF** when shell access is available (§2 deviation).
3. **On-device mobile verification** — the 5-item checklist in `docs/research/mobile-browser-voice.md` (branch `research/mobile-browser-voice`), especially the Android OEM battery-manager wildcard and iOS cert/hostname setup.
4. **Prototype assets** (throwaway, decision sources only): branches `proto/voice-loop` (latency dashboard, hangover slider) and `proto/tool-reliability` (probe harness).

Provenance note: [Corpus and citation flow](https://github.com/soderholmen/phd-help/issues/12) lost its resolution comment on the tracker (a `gh issue close --comment @-` mishap posted a literal `@-`); the full text was recovered from the local session transcript and re-posted to the ticket 2026-09-29.
