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

- **Endpointing is an energy gate, not silero** (`segmenting.py`, pure
  Python, fully tested). `UtteranceGate` takes any `speech(frame)->bool`,
  so a sidecar-hosted silero can replace `EnergyVad` without touching the
  state machine.
- **No live partials yet.** NeMo 3.0 dropped the stateful per-chunk
  streaming API the nemotron partials needed.

## One-time setup

```powershell
# ASR sidecar deps (the venv already has NeMo 3.0 + torch cu128)
.venv-asr/Scripts/pip.exe install fastapi uvicorn

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

Writes `.probe/smoke_tts_out.wav` (listen) and prints the parakeet
transcript of `.probe/probe_speech_16k.wav`. Exit 0 only if both legs
worked.

## Measured on the 5070 Ti (probe, 2026-10-01)

| component | number |
| --- | --- |
| parakeet-tdt-0.6b-v3 final | 0.46 s for 11.8 s of audio (26× realtime) |
| parakeet load | 7.4 s, 5.1 GB fp32 / ~2.5 GB bf16 |
| silero-vad (CPU, reference) | 0.12 s per 11.8 s |
| MOSS-TTS server up | ~9 s |
| MOSS-TTS TTFB | ~13 s warm (test-stack accepted; SPEC's 180 ms is not met) |
| MOSS-TTS turn 2 | ~30 s TTFB (suspected per-shape torch.compile recompiles) |

## Voice-out (shipped)

The endpoint holder hears replies **as they are generated**: `run_turn`
pumps the vLLM SSE stream through `SentenceGate` (`sentences.py` — the
boundary rules bend around LaTeX: braces, `$…$`, abbrevials, decimals),
pushes each completed sentence to the MOSS session
(`/tts/session/push`, `is_final=False`), and an `AudioEpisode`
forwarder streams the PCM16 to the holder's socket between
`audio_start{sample_rate}` and `audio_end` (view-only tabs get text
only — audio follows the mic). Generation and synthesis overlap, so
the first audio lands seconds into the reply instead of ~13 s after
the whole turn. The shell's `PcmPlayer` plays it through a
jitter-buffered scheduling queue (`web/app/src/voice/`).
`PHD_STREAM_TTS=0` is the kill switch: one-shot `chat()` + one push,
through the same episode code (same wire, same semantics).
No holder armed ⇒ no synthesis at all (screen-only, the §8 TTS-down
shape). Disarming stops playback.

Barge-in: 200 ms of sustained AEC'd mic while playing pauses the
playhead instantly and sends the control; while the agent is still
composing the server **cancels the turn** and fans out `tts_stopped`,
which hard-stops every player and kills its resume window. Once the
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
- **The gate is energy, not a VAD** (the endpointing deviation again).
  A false trip is recoverable by the resume window — but only for the
  no-live-turn case; a trip during a live turn cancels it.
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

## Deferred (seams intact)

silero-in-sidecar WS endpointing; live partials.
