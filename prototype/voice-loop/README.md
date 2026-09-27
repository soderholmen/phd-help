# ⚠ PROTOTYPE — voice loop latency probe (throwaway)

Resolves the wayfinder ticket **[Voice loop prototype](https://github.com/soderholmen/phd-help/issues/11)**.
This is not the app — it's the cheapest thing that proves the hands-free round trip
and shows **where the latency actually goes**.

## The question

What does the full loop feel like — VAD → STT → (stubbed LLM echo) → TTS — and
which stage eats the time?

## Run it

```powershell
powershell -File prototype\voice-loop\run.ps1
```

First run creates a throwaway `.venv` and downloads torch (cu128) + model weights
(parakeet-tdt-0.6b-v3 ~1.7 GB, Kokoro-82M ~330 MB). Then open
**http://127.0.0.1:8756**, click **Start mic**, and talk. Each turn appends a row
to the per-stage latency table.

## The loop

Browser mic (getUserMedia + AudioWorklet, downsampled to 16 kHz PCM over a
WebSocket) → **silero-vad** endpointing (server, custom hangover so the endpoint
delay is measured, not hidden) → **parakeet-tdt-0.6b-v3** finals (the STT decision
from ticket #6, minus the streaming-partials tier) → **echo stub** (stand-in for
the LLM turn; set `STUB_DELAY_S` to feel a slow model) → **Kokoro-82M** TTS (the
zero-risk baseline from ticket #3) streamed sentence-by-sentence back to the browser.

## What to watch

- **VAD endpoint** — hangover you must wait out before anything starts (tune `HANGOVER_S`).
- **STT** — parakeet finals on the 5070 Ti.
- **TTS first audio** — Kokoro time-to-first-audio; the real answer (MOSS-TTS-Realtime)
  claims ~180 ms TTFB, compare against this floor.
- **dead air** — perceived silence from end-of-speech to first sound; the number the
  conversation-mode ticket (#5) should be argued from.

Env knobs: `VAD_THRESHOLD`, `HANGOVER_S`, `STUB_DELAY_S`, `TTS_VOICE`.

Wipe me: delete this folder and the `.venv` inside it.
