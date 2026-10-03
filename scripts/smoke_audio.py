"""Live proof for the audio sidecars — stdlib + httpx only, so it runs in
the main venv without touching torch (docs/audio-stack.md).

With both sidecars up:
  .venv/Scripts/python.exe scripts/smoke_audio.py
Writes .probe/smoke_tts_out.wav (listen to it), prints the transcript and
the endpointing verdicts for the probe speech file. Nonzero exit on any
failure — CI-adjacent honest.
"""

import os
import re
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


def vad_smoke() -> bool:
    """Stream the probe speech through /vad the way the mic does — 100 ms
    blobs, offset-tracked — and check the property the energy gate failed.
    The file is two concatenated TTS sentences, so the ground truth is
    EXACTLY ONE hangover-length (20 frames = 0.6 s) silence run inside
    the speech: the real inter-sentence pause. The energy gate produced
    two — the second mid-sentence, where a 0.21 s pause at unvoiced
    consonants inflated past the hangover. That extra cut is the "Yeah."
    bug."""
    with wave.open(str(SPEECH_WAV), "rb") as w:
        assert w.getnchannels() == 1 and w.getframerate() == 16000
        frames = w.readframes(w.getnframes())
    blob, stream, verdicts = 3200, "smoke-vad", []
    t0 = time.time()
    for off in range(0, len(frames), blob):
        r = httpx.post(f"{STT_URL}/vad", params={"stream": stream,
                                                 "off": off},
                       content=frames[off:off + blob], timeout=30.0)
        r.raise_for_status()
        verdicts += r.json()["speech"]
    dt = time.time() - t0
    n_frames = len(frames) // 2 // 480             # segmenting.py's grid
    runs = "".join("1" if v else "0" for v in verdicts)
    first, last = runs.find("1"), len(runs) - 1 - runs[::-1].find("1")
    cuts = len(re.findall(r"0{20,}", runs[first:last + 1]))
    frac = sum(verdicts) / max(1, len(verdicts))
    ok = len(verdicts) == n_frames and 0.2 < frac < 0.95 and cuts == 1
    print(f"VAD: {len(verdicts)}/{n_frames} frames, speech fraction "
          f"{frac:.2f}, {cuts} hangover-length cut(s) inside (want 1) "
          f"in {dt:.2f} s -> {'ok' if ok else 'FAIL'}")
    return ok


if __name__ == "__main__":
    ok = False
    try:
        ok = tts_smoke() and stt_smoke() and vad_smoke()
    except httpx.HTTPError as e:
        print(f"sidecar unreachable: {e}")
    sys.exit(0 if ok else 1)
