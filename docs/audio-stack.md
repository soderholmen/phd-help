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
- **No live partials and no browser audio-out yet.** NeMo 3.0 dropped the
  stateful per-chunk streaming API the nemotron partials needed, and no
  player exists for the TTS stream the backend already consumes.

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

## Deferred (seams intact)

Browser PCM16 player + binary-out + barge-in; silero-in-sidecar WS
endpointing; live partials; sentence-level TTS off the token stream.
