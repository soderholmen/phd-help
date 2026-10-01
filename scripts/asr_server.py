"""parakeet STT sidecar (docs/audio-stack.md).

Standalone on purpose: it runs in .venv-asr (py3.12 + NeMo + torch), which
does not have phd_helper installed — and must never import it, so the
backend venv (where Smart App Control blocks torch) can never be tempted
to load a model in-process. Pinned by a grep test in tests/test_audio.py.

Run:  .venv-asr/Scripts/python.exe scripts/asr_server.py --port 8090
"""

import argparse
import asyncio
import os
import tempfile
import threading
import wave
from pathlib import Path

from fastapi import FastAPI, Request

MODEL = os.environ.get("PHD_ASR_MODEL", "nvidia/parakeet-tdt-0.6b-v3")
# The probe eyeballed accuracy on fp32; bf16 halves the VRAM bill (2.5 vs
# 5.1 GB) and is the default so MinerU can coexist. PHD_ASR_DTYPE=fp32 is
# the escape hatch if a transcript starts to drift.
DTYPE = os.environ.get("PHD_ASR_DTYPE", "bf16")

app = FastAPI(title="phd-helper ASR sidecar")

_model = None
_load_lock = threading.Lock()      # double-checked lazy load (embed.py shape)
_infer_lock = threading.Lock()     # one GPU: one transcribe at a time


def get_model():
    global _model
    if _model is None:
        with _load_lock:
            if _model is None:
                from nemo.collections.asr.models import ASRModel
                asr = ASRModel.from_pretrained(MODEL).to("cuda")
                if DTYPE == "bf16":
                    asr = asr.bfloat16()
                asr.eval()
                _model = asr    # a raise above leaves None: next call retries
    return _model


@app.get("/health")
def health():
    return {"status": "ok", "model": MODEL, "dtype": DTYPE,
            "loaded": _model is not None}


def _final_text(out) -> str:
    """One hypothesis per audio, and NeMo has shipped three shapes for it:
    a plain str, a Hypothesis with .text, and an ASRTranscription whose
    .text is a per-segment list. str() of the dataclass is never the text
    — the live smoke proved that the honest way."""
    text = getattr(out, "text", None)
    if text is None:                    # a list of per-audio results
        first = out[0] if len(out) else ""
        text = getattr(first, "text", first)
    if isinstance(text, (list, tuple)):
        text = " ".join(str(t) for t in text)
    return str(text).strip()


def _transcribe_wav(body: bytes) -> str:
    """Blocking half: temp WAV in, text out. The NeMo transcribe API
    takes file paths, so the utterance lands in a temp WAV first."""
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        path = tmp.name
        with wave.open(tmp, "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(16000)
            w.writeframes(body)
    try:
        model = get_model()
        with _infer_lock:
            out = model.transcribe([path])
        return _final_text(out)
    finally:
        Path(path).unlink(missing_ok=True)


@app.post("/transcribe")
async def transcribe(request: Request):
    """Raw 16 kHz mono PCM16 in, {"text": ...} out."""
    body = await request.body()
    if not body:
        return {"text": ""}
    # to_thread, not inline: the ~7 s lazy load and the inference lock on
    # the event loop would stall this sidecar's own /health — a merely
    # busy sidecar must not read faulted to the backend.
    return {"text": await asyncio.to_thread(_transcribe_wav, body)}


if __name__ == "__main__":
    import uvicorn
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8090)
    args = parser.parse_args()
    uvicorn.run(app, host="127.0.0.1", port=args.port)
