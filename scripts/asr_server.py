"""parakeet STT + silero VAD sidecar (docs/audio-stack.md).

Standalone on purpose: it runs in .venv-asr (py3.12 + NeMo + torch +
silero-vad), which does not have phd_helper installed — and must never
import it, so the backend venv (where Smart App Control blocks torch) can
never be tempted to load a model in-process. Pinned by an AST test in
tests/test_audio.py.

Run:  .venv-asr/Scripts/python.exe scripts/asr_server.py --port 8090
"""

import argparse
import array
import asyncio
import os
import tempfile
import threading
import wave
from collections import OrderedDict
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


# --- /vad: silero endpointing verdicts ------------------------------------
#
# The backend owns the endpointing state machine (segmenting.py); this leg
# only answers "is this 30 ms frame speech" from silero-vad on CPU (no
# VRAM — it coexists with parakeet and MOSS by design). The frame grid is
# GLOBAL to a stream: the backend sends the byte offset it has fed, and a
# mismatch (a lost POST) resyncs the stream — fresh model state, the
# un-startable frame skipped — so a fault costs that blob's verdicts (the
# backend's energy fallback answers them), never a permanent misalignment.
# A frame is speech if ANY overlapping silero chunk cleared the threshold:
# conservative on purpose — longer speech regions mean fewer mid-utterance
# cuts, which is the whole reason silero replaced the energy gate. A chunk
# still straddling the POST's end is not scored yet, so a frame's verdict
# can miss at most that one chunk: a <=1-frame (30 ms) edge effect per
# POST, at a speech-region boundary the 0.6 s hangover absorbs anyway.

VAD_FRAME_SAMPLES = 480        # 30 ms at 16 kHz — segmenting.py's grid,
                               # duplicated on purpose: scripts must never
                               # import phd_helper (the AST pin)
VAD_CHUNK_SAMPLES = 512        # silero v6's fixed chunk size at 16 kHz
# Silero's probability floor — deliberately NOT PHD_VAD_THRESHOLD, which
# is the backend's energy-RMS floor (config.py): one knob, one name.
VAD_THRESHOLD = float(os.environ.get("PHD_SILERO_THRESHOLD", "0.5"))
VAD_STREAMS_MAX = 8            # LRU; endpoint handoffs must not leak

_vad_lock = threading.Lock()   # one CPU model: one verdict call at a time
_vad_streams: "OrderedDict[str, dict]" = OrderedDict()


def load_vad():
    # torch lives only here: the mapping below stays pure Python, so the
    # backend's test venv (no torch, by SAC) can still unit-test the
    # offset/resync/trim grid with an injected fake model.
    import torch
    from silero_vad import load_silero_vad
    model = load_silero_vad()

    def call(chunk):                     # array('h') of 512 int16 -> prob
        t = torch.tensor(list(chunk), dtype=torch.float32) / 32768.0
        with torch.no_grad():
            return float(model(t.reshape(1, -1), 16000))
    return call


def _vad_verdicts(stream: str, off: int, body: bytes) -> list:
    st = _vad_streams.get(stream)
    if st is None or st["end"] != off:
        # new stream, or resync after a lost POST: a fresh model instance
        # (silero's RNN state must not straddle the gap), and the first
        # frame whose start we missed is skipped — the backend's energy
        # fallback answers exactly that one frame.
        st = {"pos": off, "end": off, "buf": b"", "model": load_vad(),
              "probs": [], "fed": 0}
        st["next_frame"] = -(-off // 2 // VAD_FRAME_SAMPLES)   # ceil
        _vad_streams[stream] = st
        while len(_vad_streams) > VAD_STREAMS_MAX:
            _vad_streams.popitem(last=False)
    else:
        _vad_streams.move_to_end(stream)

    st["buf"] += body
    st["end"] += len(body)
    gstart = st["pos"] // 2                       # global sample position
    n_avail = len(st["buf"]) // 2

    # run silero over the newly available 512-sample chunks (grid anchored
    # at the stream's resync point; the model's state carries across)
    fed = st["fed"]
    while n_avail - fed >= VAD_CHUNK_SAMPLES:
        chunk = array.array("h")
        chunk.frombytes(st["buf"][fed * 2:(fed + VAD_CHUNK_SAMPLES) * 2])
        st["probs"].append((gstart + fed, gstart + fed + VAD_CHUNK_SAMPLES,
                            st["model"](chunk) > VAD_THRESHOLD))
        fed += VAD_CHUNK_SAMPLES
    st["fed"] = fed

    # verdicts for every global frame this POST completes
    out = []
    k = st["next_frame"]
    while (k + 1) * VAD_FRAME_SAMPLES <= gstart + n_avail:
        fs, fe = k * VAD_FRAME_SAMPLES, (k + 1) * VAD_FRAME_SAMPLES
        out.append(any(p for cs, ce, p in st["probs"]
                       if cs < fe and ce > fs and p))
        st["probs"] = [(cs, ce, p) for cs, ce, p in st["probs"] if ce > fe]
        k += 1
    st["next_frame"] = k

    # trim consumed bytes: nothing before the earlier of (model-fed,
    # next frame's start) is ever looked at again
    keep_from = min(gstart + fed, k * VAD_FRAME_SAMPLES)
    if keep_from > gstart:
        st["buf"] = st["buf"][(keep_from - gstart) * 2:]
        st["pos"] = keep_from * 2
        st["fed"] = fed - (keep_from - gstart)
    return out


@app.post("/vad")
async def vad(request: Request, stream: str = "default", off: int = 0):
    """Raw 16 kHz mono PCM16 in, {"speech": [bool per 30 ms frame]} out."""
    body = await request.body()
    if not body:
        return {"speech": []}
    # to_thread like /transcribe: the first silero load must not stall the
    # event loop (and /health) — and the lock above must never be held on
    # it.
    def run():
        with _vad_lock:
            return _vad_verdicts(stream, off, body)
    return {"speech": await asyncio.to_thread(run)}


if __name__ == "__main__":
    import uvicorn
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8090)
    args = parser.parse_args()
    uvicorn.run(app, host="127.0.0.1", port=args.port)
