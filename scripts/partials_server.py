"""nemotron live-partials sidecar (docs/audio-stack.md).

Standalone on purpose (the asr_server.py rule: never import phd_helper —
AST-pinned): it runs in .venv-asr (py3.12 + transformers 5.18 + torch).

CPU fp32 by decision, not by accident: the probe measured RTF 0.24 on
CPU fp32 (2026-10-04, .probe/probe_rnnt_stream.py) — under the 0.5 the
plan set — and the GPU is full (TTS + parakeet + contexts measured
15359/16303 MiB; the 2.47 GB fp32 model does not fit alongside).

Stateless per request by design: every /stream re-encodes the
utterance-so-far through the model's chunked_limited streaming path
(a generator of fixed mel chunks — first 1+8*6=49 frames, then
8*(6+1)=56, the final short chunk zero-padded; the encoder caches
carry state across chunks inside one generate). The backend sends one
request at a time and skips while one is in flight, so the O(n^2) of
re-encoding stays bounded by the probe's own measurement shape.

Run:  .venv-asr/Scripts/python.exe scripts/partials_server.py --port 8092
"""

import argparse
import asyncio
import os
import threading

from fastapi import FastAPI, Request

MODEL = os.environ.get("PHD_PARTIALS_MODEL",
                       "nvidia/nemotron-speech-streaming-en-0.6b")
# The chunked_limited right attention context. NOT in the model config:
# the probe found it must be set on the processor AND passed to
# generate() explicitly — one knob here feeds both (docs/audio-stack.md).
LOOKAHEAD = int(os.environ.get("PHD_PARTIALS_LOOKAHEAD", "6"))

app = FastAPI(title="phd-helper partials sidecar")

_adapter = None
_load_lock = threading.Lock()      # double-checked lazy load (embed.py shape)
_infer_lock = threading.Lock()     # one CPU model: one generate at a time


def load_model():
    # torch/transformers live only here: the mapping below stays pure
    # Python over a small adapter contract, so the backend's test venv
    # (no torch, by SAC) unit-tests the chunking + hypothesis path with
    # an injected fake — the load_vad trick from asr_server.py.
    import numpy as np
    import torch
    from transformers import AutoModelForRNNT, AutoProcessor

    proc = AutoProcessor.from_pretrained(MODEL)
    proc.set_num_lookahead_tokens(LOOKAHEAD)
    model = AutoModelForRNNT.from_pretrained(MODEL, dtype=torch.float32
                                             ).eval()
    sub = model.config.encoder_config.subsampling_factor

    class Adapter:
        first = 1 + sub * LOOKAHEAD          # mel frames, per the
        step = sub * (LOOKAHEAD + 1)         # model's own validator

        def feats(self, pcm: bytes):
            audio = np.frombuffer(pcm, dtype=np.int16)
            audio = audio.astype(np.float32) / 32768.0
            return proc(audio, sampling_rate=16000,
                        return_tensors="pt")["input_features"]

        def slice(self, feats, a: int, b: int, pad_to: int = 0):
            t = feats[:, a:b]
            if pad_to:
                pad = torch.zeros(t.shape[0], pad_to - t.shape[1],
                                  t.shape[2])
                t = torch.cat([t, pad], dim=1)
            return t

        def generate(self, chunks) -> str:
            # A REAL generator, not iter(list): the streaming dispatch
            # is isinstance(input_features, GeneratorType), and a
            # list_iterator silently falls through to the offline path
            # (the probe learned this the crash way).
            def feed():
                yield from chunks

            with torch.no_grad():
                out = model.generate(feed(), num_lookahead_tokens=LOOKAHEAD)
            return proc.tokenizer.decode(out[0].tolist(),
                                         skip_special_tokens=True)

    return Adapter()


def _as_text(out) -> str:
    """Decode output has shipped as a str and as a one-element list of
    str (the probe's log shows the list shape) — str() of the wrong
    shape is how a sidecar starts speaking in brackets."""
    if isinstance(out, (list, tuple)):
        out = " ".join(str(x) for x in out)
    return str(out).strip()


def _hypothesis(adapter, body: bytes) -> str:
    """Raw 16 kHz mono PCM16 (the utterance so far) in, growing
    hypothesis out. Empty under one first chunk of mel frames: there is
    nothing causal to say yet."""
    if not body:
        return ""
    feats = adapter.feats(body)
    n = feats.shape[1]
    if n < adapter.first:
        return ""
    bounds = [0, adapter.first]
    while bounds[-1] + adapter.step <= n:
        bounds.append(bounds[-1] + adapter.step)
    chunks = [adapter.slice(feats, bounds[i], bounds[i + 1])
              for i in range(len(bounds) - 1)]
    if n > bounds[-1]:                       # the validator demands
        chunks.append(adapter.slice(feats, bounds[-1], n,                    # exact sizes:
                                    pad_to=adapter.step))                     # pad the tail
    return _as_text(adapter.generate(chunks))


@app.get("/health")
def health():
    return {"status": "ok", "model": MODEL, "lookahead": LOOKAHEAD,
            "loaded": _adapter is not None}


@app.post("/stream")
async def stream(request: Request):
    """Raw PCM16 of the utterance-so-far in, {"partial": text} out."""
    body = await request.body()

    def run():
        global _adapter
        if _adapter is None:
            with _load_lock:
                if _adapter is None:
                    _adapter = load_model()   # a raise leaves None: retry
        with _infer_lock:
            return _hypothesis(_adapter, body)

    # to_thread, not inline: the multi-second lazy load must not stall
    # this sidecar's own /health (the asr_server lesson — a merely busy
    # sidecar must not read faulted).
    return {"partial": await asyncio.to_thread(run)}


if __name__ == "__main__":
    import uvicorn
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8092)
    args = parser.parse_args()
    uvicorn.run(app, host="127.0.0.1", port=args.port)
