"""Live proof for the audio sidecars — stdlib + httpx only, so it runs in
the main venv without touching torch (docs/audio-stack.md).

With both sidecars up:
  .venv/Scripts/python.exe scripts/smoke_audio.py
Writes .probe/smoke_tts_out.wav (listen to it) and prints the transcript
of the probe speech file. Nonzero exit on any failure — CI-adjacent honest.
"""

import os
import sys
import time
import wave
from pathlib import Path

import httpx

REPO = Path(__file__).resolve().parents[1]
TTS_URL = os.environ.get("PHD_TTS_URL", "http://127.0.0.1:8083")
STT_URL = os.environ.get("PHD_STT_URL", "http://127.0.0.1:8090")
PROMPT = REPO / ".probe" / "MOSS-TTS" / "assets" / "audio" / "reference_en_0.mp3"
SPEECH_WAV = REPO / ".probe" / "probe_speech_16k.wav"
OUT_WAV = REPO / ".probe" / "smoke_tts_out.wav"
TEXT = ("We chose harrier embeddings for the retrieval stack, "
        "because the reranker saturates on raw cosine boost.")


def tts_smoke() -> bool:
    sid = "smoke"
    with httpx.Client(timeout=180.0) as client:
        client.post(f"{TTS_URL}/tts/session/start",
                    json={"session_id": sid,
                          "prompt_audio": str(PROMPT)}).raise_for_status()
        t0 = time.time()
        client.post(f"{TTS_URL}/tts/session/push",
                    json={"session_id": sid, "text": TEXT,
                          "is_final": True}).raise_for_status()
        ttfb, chunks, sr = None, [], 24000
        with client.stream("GET", f"{TTS_URL}/tts/session/{sid}/audio") as r:
            r.raise_for_status()
            sr = int(r.headers.get("X-Audio-Sample-Rate", "24000"))
            for chunk in r.iter_bytes():
                if ttfb is None:
                    ttfb = time.time() - t0
                chunks.append(chunk)
        client.post(f"{TTS_URL}/tts/session/close",
                    json={"session_id": sid})
    pcm = b"".join(chunks)
    if not pcm:
        print("TTS: NO AUDIO")
        return False
    with wave.open(str(OUT_WAV), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(pcm)
    dur = len(pcm) / 2 / sr
    print(f"TTS: TTFB {ttfb:.1f} s, {dur:.1f} s of audio -> {OUT_WAV}")
    return True


def stt_smoke() -> bool:
    with wave.open(str(SPEECH_WAV), "rb") as w:
        assert w.getnchannels() == 1 and w.getframerate() == 16000
        frames = w.readframes(w.getnframes())
    t0 = time.time()
    r = httpx.post(f"{STT_URL}/transcribe", content=frames, timeout=120.0)
    r.raise_for_status()
    print(f"STT: {len(frames) / 2 / 16000:.1f} s of audio in "
          f"{time.time() - t0:.2f} s -> {r.json()['text']!r}")
    return True


if __name__ == "__main__":
    ok = False
    try:
        ok = tts_smoke() and stt_smoke()
    except httpx.HTTPError as e:
        print(f"sidecar unreachable: {e}")
    sys.exit(0 if ok else 1)
