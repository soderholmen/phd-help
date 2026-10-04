"""Live proof for the partials sidecar — stdlib + httpx only, so it runs
in the main venv without touching torch (docs/audio-stack.md).

With the sidecar up (.venv-asr/Scripts/python.exe scripts/partials_server.py
--port 8092):
  .venv/Scripts/python.exe scripts/smoke_partials.py
Feeds cumulative prefixes of the probe speech file to /stream the way
SidecarStt does — the whole utterance-so-far per request, one in flight —
and prints the hypothesis at each second. Green means readable words
that grow; the unit tests already pin the chunking. Nonzero exit on any
failure — CI-adjacent honest.
"""

import os
import sys
import wave
from pathlib import Path

import httpx

REPO = Path(__file__).resolve().parents[1]
PARTIALS_URL = os.environ.get("PHD_PARTIALS_URL", "http://127.0.0.1:8092")
SPEECH_WAV = REPO / ".probe" / "probe_speech_16k.wav"


def main() -> int:
    with wave.open(str(SPEECH_WAV), "rb") as w:
        assert w.getframerate() == 16000 and w.getnchannels() == 1
        assert w.getsampwidth() == 2, "PCM16 expected"
        pcm = w.readframes(w.getnframes())

    sr = 16000
    last = ""
    grew = True
    with httpx.Client(timeout=120.0) as c:   # first call pays the lazy load
        for secs in range(1, len(pcm) // (sr * 2) + 1):
            r = c.post(f"{PARTIALS_URL}/stream", content=pcm[: secs * sr * 2])
            r.raise_for_status()
            text = r.json()["partial"]
            print(f"{secs:>3}s: {text!r}", flush=True)
            if len(text) < len(last):
                grew = False
            last = text

    ok = grew and bool(last)
    print("VERDICT:", "GREEN - hypothesis grows" if ok else "RED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
