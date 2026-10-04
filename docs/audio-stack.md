# Audio stack: sidecar topology and ops runbook

## Why sidecars

SPEC §3 puts silero-vad + parakeet in the backend process. That is not
possible on this box: Smart App Control blocks `torch._C` in the main venv
(Python 3.14), and NeMo needs 3.12 anyway. So the models live in separate
py3.12 venvs as plain HTTP sidecars — the same MinerU convention: ops
starts the sidecar, the backend holds a thin client, and a dead sidecar
degrades per §8 (typed chat and screen-only replies keep working).

Two honest deviations from SPEC §3, both contained behind the Protocols
in `server/voice.py`:

- **Endpointing: silero verdicts over HTTP, not in-process** (SPEC
  wanted silero in the backend; torch can't load there). The ASR
  sidecar answers `POST /vad` per mic blob — one verdict per 30 ms
  frame, silero on CPU, no VRAM — and `RemoteVad` (`server/stt.py`)
  feeds the verdicts to the same `UtteranceGate` (`segmenting.py`, pure
  Python, fully tested). `EnergyVad` stays the per-frame fallback: a
  `/vad` fault degrades endpointing quality, never drops an utterance.
  The shipped energy gate had cut real speech at micro-pauses (the live
  listen-test heard "Yeah." where a sentence was said); with silero the
  0.6 s hangover finally means a real pause. `PHD_VAD=energy` is the
  kill switch back to the pure gate.
- **Live partials ride a third sidecar, not NeMo.** NeMo 3.0 dropped
  the stateful per-chunk streaming API the nemotron streaming model
  needs, so partials come from `scripts/partials_server.py` — plain
  transformers RNNT streaming in the same `.venv-asr`, CPU by decision
  (measured below). Parakeet stays the authoritative transcript: a
  partial is ghost text, never a turn.

## One-time setup

```powershell
# ASR sidecar deps (the venv already has NeMo 3.0 + torch cu128)
.venv-asr/Scripts/pip.exe install fastapi uvicorn silero-vad

# Partials sidecar deps (same venv; the RNNT class needs transformers 5.18)
.venv-asr/Scripts/pip.exe install "transformers>=5.18"

# TTS sidecar: vendored clone + weights (already done on this box)
git clone https://github.com/OpenMOSS/MOSS-TTS .probe/MOSS-TTS
# weights in .probe/models/MOSS-TTS-Realtime and MOSS-Audio-Tokenizer
# (hf download with HF_HUB_DISABLE_XET=1 — xet left dangling symlinks here)
```

The TTS launcher needs FFmpeg on disk (winget Gyan build; the path is the
`PHD_FFMPEG` default) and `triton-windows==3.5.1` (3.8 breaks torch 2.9.1's
inductor; absent means TritonMissing).

## Start / stop

```powershell
.venv-asr/Scripts/python.exe scripts/asr_server.py --port 8090   # loads lazily, ~7 s first call
.venv-tts/Scripts/python.exe scripts/tts_server.py --port 8083   # ~9 s to /health
.venv-asr/Scripts/python.exe scripts/partials_server.py --port 8092  # lazy load, CPU fp32
# backend
set PHD_AUDIO_STACK=local
```

`/health` then reports `"stt": "ok", "tts": "ok"` (live-probed; `"stub"`
while the gate is off, `"faulted"` after three consecutive failures —
which self-heal on the next success).

**VRAM budget (16.3 GB):** MOSS ~11.5 + parakeet bf16 ~2.5 + MinerU ~2.8
— all three do not fit. Stop MinerU during voice sessions; OOM surfaces
as sidecar 500 → `faulted`, never as a silent lie.

## Verify

```powershell
.venv/Scripts/python.exe scripts/smoke_audio.py
```

Writes `.probe/smoke_tts_out.wav` (listen), prints the parakeet
transcript of `.probe/probe_speech_16k.wav`, and streams the same file
through `/vad` the way the mic does. The file is two concatenated
sentences, so the property the energy gate failed must hold: exactly
ONE hangover-length (0.6 s) silence cut inside the speech — the real
inter-sentence pause. The energy gate produced two, the second
mid-sentence (the "Yeah." shape). Exit 0 only if all three legs worked.

```powershell
.venv/Scripts/python.exe scripts/smoke_partials.py
```

The ghost-text leg: cumulative prefixes of the same file through
`/stream`, one request at a time. Green means the printed hypothesis
grows every second and never shrinks.

## Measured on the 5070 Ti (probe, 2026-10-01)

| component | number |
| --- | --- |
| parakeet-tdt-0.6b-v3 final | 0.46 s for 11.8 s of audio (26× realtime) |
| parakeet load | 7.4 s, 5.1 GB fp32 / ~2.5 GB bf16 |
| silero-vad (CPU, reference) | 0.12 s per 11.8 s |
| silero `/vad` shipped (HTTP, 392 frames) | 1.19 s per 11.8 s audio; 1 cut (the real pause) vs energy's 2 (one mid-sentence), speech fraction 0.81 vs 0.36 |
| MOSS-TTS server up | ~9 s |
| MOSS-TTS TTFB | ~13 s warm (test-stack accepted; SPEC's 180 ms is not met) |
| MOSS-TTS turn 2 | ~30 s TTFB (suspected per-shape torch.compile recompiles — resolved 2026-10-03 below: it was the one-long-text shape, not the turn count) |

## Measured on the 5070 Ti (sentence-TTS probe, 2026-10-03)

| question | number |
| --- | --- |
| vLLM thinking-stream delta keys | `{role, content}` only — no reasoning key, content is speakable (default-on safe) |
| sidecar TTFB, one long push | 13.4 s (the 2026-10-01 shape) |
| sidecar TTFB, sentence pushes | **1.0 s**, identical total bytes — short pushes cost no extra recompiles |
| backend TTF-audio, `PHD_STREAM_TTS=1` | **2.2 s**, and `audio_start` landed 0.4 s BEFORE `assistant_text` |
| backend TTF-audio, `PHD_STREAM_TTS=0` | 5.2 s, text 2.7 s ahead (warm sidecar; the cold one-long shape is the 13.4 s row) |
| chunk gaps while streaming | max 0.98 s over 51 chunks — over the 275 ms jitter buffer, so an occasional clean underrun pause is honest behavior, not a fault |

## Voice-out (shipped)

The endpoint holder hears replies **as they are generated**: `run_turn`
pumps the vLLM SSE stream through `SentenceGate` (`sentences.py` — the
boundary rules bend around LaTeX: braces, `$…$`, abbreviations,
decimals), pushes each completed sentence to the MOSS session
(`/tts/session/push`, `is_final=False`), and an `AudioEpisode`
forwarder streams the PCM16 to the holder's socket between
`audio_start{sample_rate}` and `audio_end` (view-only tabs get text
only — audio follows the mic). Generation and synthesis overlap, so
the first audio lands ~2 s into the reply (measured, above) instead of
~13 s after the whole turn. The shell's `PcmPlayer` plays it through a
jitter-buffered scheduling queue (`web/app/src/voice/`).
`PHD_STREAM_TTS=0` is the kill switch: one-shot `chat()` + one push,
through the same episode code (same wire, same semantics).
No holder armed ⇒ no synthesis at all (screen-only, the §8 TTS-down
shape). Disarming stops playback.

Before the gate rides `saytext.py`: `Unfence` keeps the ``` delimiter
lines of a fenced draft out of the audio (the draft's prose itself is
spoken — the ratified §3 carve-out, stepwise writing), and `for_speech`
turns LaTeX into words (`\cite` → "citation", `$…$` → "formula", …)
because the engine hallucinates on symbol strings. The screen and the
history keep the raw source; the kill-switch leg applies the same
filter one-shot (`strip_fences`).

Barge-in: 200 ms of sustained AEC'd mic while playing pauses the
playhead instantly and sends the control; while the agent is still
composing the server **cancels the turn** and fans out `tts_stopped`,
which hard-stops every player and kills its resume window. The
cancelled turn's sidecar session is closed too (`abort()` → `/close`):
the worker is single-threaded, so an abandoned turn would otherwise
keep synthesizing its queued sentences — and force-finish into a
garbled tail — ahead of the next turn on the same queue; deleting the
session hands the next turn a fresh worker. Once the
reply is fully composed — audio draining, or the turn already done —
there is nothing to un-ring: the control is a no-op and the client's
2.5 s resume window governs the buffered audio (SPEC §3).

Honest deviations, stated not hidden:

- **Tool-call preambles are spoken** (ratified): the agent says "let
  me look at the intro…" as it types it; the tool call that follows
  silences the rest of that message, and the final text turn rides
  the same audio episode. A validation bounce can supersede a spoken
  preamble — the retry's text replaces what the first attempt said.
- **The holder is decided at the first completed sentence**: arming
  the mic mid-turn misses that turn's audio (the episode was already
  dead, and the sidecar was never paid for it).
- **A barge during the overlap kills a live turn** — the accepted
  cost of overlap: with sentence-level TTS, "speaking" no longer
  implies "generation done", so while the agent composes, pause/resume
  cannot un-ring sentences it keeps writing. The no-op survives
  exactly where it always made sense — after composition ends, while
  only audio drains. BargeGate's sustained-speech requirement is the
  shield against false positives.
- **Barge-in is an energy gate, not a VAD** (browser RMS in the shell's
  `BargeGate` — a different job from endpointing, which is silero over
  the sidecar again). A false trip is recoverable by the resume window
  — but only for the no-live-turn case; a trip during a live turn
  cancels it.
- **Mid-stream holder handoff drops the remaining audio** (the text is
  on every screen; the audio is ephemeral, the sendAudio-blip
  precedent). `audio_end` still closes the episode — it means "no more
  is coming", not "you heard it all".
- **Jitter is fixed at 275 ms** (SPEC's 250-300 ms band, one knob in
  `playback.ts`).
- **SPEC's phone barge-in mitigations stay on the phone** (SPEC:97):
  AEC warmup preroll and word-count gating are mobile-slice work; on
  desktop the browser AEC is already on and the resume window recovers
  from a false trip.
- **No persistent-underrun tray notice** (SPEC:226): underrun behaves
  as the spec's clean pause; the >5 s "degraded connection" notice is
  deferred — the sidecar streams faster than realtime, so mid-stream
  starvation is a cellular-shape problem.

## Live partials (shipped)

Words appear in the composer as **ghost text while you are still
speaking**. The mic feed already flows through the backend; mid-utterance
`SidecarStt` pumps the utterance-so-far to the partials sidecar, and
each growing hypothesis arrives as `user_partial{text}` — a transient
`partial` in shell state that REPLACES (it is the whole utterance-so-
far, not a delta) and never joins the transcript. The parakeet final
lands as `user_text` and clears the ghost: the final is the record,
the partial is a liveness cue.

**The probe (2026-10-04) decided the ladder — GREEN, CPU by decision:**

| question | number |
| --- | --- |
| CPU fp32 RTF (nemotron-speech-streaming-en-0.6b, `chunked_limited`) | **0.24** — under the 0.5 gate |
| GPU fit | moot: 15359/16303 MiB already used (TTS + parakeet + contexts); the 2.47 GB fp32 model does not fit alongside |
| verdict | CPU sidecar, `PHD_PARTIALS` default **on**; `PHD_PARTIALS=0` is the kill switch |
| live smoke (`scripts/smoke_partials.py`) | 11 cumulative prefixes, the hypothesis grew every second; streaming-model word errors present mid-sentence ("Riranka" for the reranker) — the reason the final, not the partial, is the record |

Two transformer-5.18 gotchas the probe paid for in crashes, recorded so
the next reader doesn't: the streaming dispatch is
`isinstance(input_features, GeneratorType)` — `iter(list)` is a
list_iterator and **silently falls through to the offline path**, so
the chunk feed must be a real generator; and `num_lookahead_tokens` is
not in the model config — it must be set on the processor AND passed to
`generate()` (one `LOOKAHEAD` knob in the sidecar feeds both).

**Stateless per request by design:** every `/stream` re-encodes the
utterance-so-far through the chunked path (first 49 mel frames, then
56, tail zero-padded — the model's own validator's sizes). The backend
keeps **one request in flight** and skips while one runs, so the O(n²)
of re-encoding stays bounded by the probe's measurement shape, and the
sidecar holds no per-utterance state to leak across sittings. A failed
partial is no partial (silent); finals are never touched by any of it.
A generation counter kills a late hypothesis at the close, and the
close always clears the ghost — including a close without a final, and
a discarded blip, which never closes at all (the gate drops it below
`min_utterance` without a verdict; `feed` notices the gone `pending()`
and clears through the same door).

Honest deviations, stated not hidden:

- **`user_partial` is holder-only on the wire** — the deliberate
  asymmetry of `user_text`, which fans to every tab. Another tab
  showing your mic's ghost text is wrong; the ghost is also
  `aria-hidden`, because the final transcript is the accessible
  record.
- **No client-side barge-in clear.** The barge utterance IS the
  partial's utterance — clearing on barge would flicker live ghost
  text mid-sentence. The stale cases are covered: the close sends
  `""` (and so does a discarded blip), while disarm and a dropped
  connection clear client-side (no close reaches the server when the
  mic stops mid-utterance).
- **A dead partials sidecar is invisible by design** — no `/health`
  key, no faulted state: no ghost text, exactly the pre-partial
  behavior. Ghost text is a garnish, never a dependency.
- **The ghost yields to typed text** — if you type while the mic is
  open, the typed input wins the overlay.

## Deferred (seams intact)

nothing in the audio stack — endpointing, voice-out and live partials
all shipped. The phone slice (SPEC:97 barge-in mitigations) stays
separate work.
