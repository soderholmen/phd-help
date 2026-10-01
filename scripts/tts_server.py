"""MOSS-TTS-Realtime sidecar launcher (docs/audio-stack.md).

Promotes the probe's launcher, including its two Windows workarounds:
torchaudio 2.9 routes audio I/O through torchcodec, whose FFmpeg-4..8
DLLs cannot load against the installed FFmpeg 9 — so load/info/save are
shimmed through the FFmpeg CLI; and Inductor's %TEMP% cache plus 64-char
hashes plus long kernel names overflow MAX_PATH, so caches go to C:\\ti.

Run (from anywhere; it chdirs into the vendored clone itself):
  .venv-tts/Scripts/python.exe scripts/tts_server.py --port 8083
"""

import argparse
import os
import subprocess
import sys
import wave
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
MOSS_DIR = Path(os.environ.get(
    "PHD_MOSS_DIR", str(REPO / ".probe" / "MOSS-TTS" / "moss_tts_realtime")))
FFMPEG = os.environ.get(
    "PHD_FFMPEG",
    r"C:\Users\Gustaf\AppData\Local\Microsoft\WinGet\Packages"
    r"\Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe"
    r"\ffmpeg-9.0.2-full_build\bin\ffmpeg.exe")

os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR", r"C:\ti")
os.environ.setdefault("TRITON_CACHE_DIR", r"C:\ti\triton")
os.environ.setdefault("MOSS_TTS_DEVICE", "cuda:0")
os.environ.setdefault("MOSS_TTS_ATTN_IMPL", "sdpa")
os.environ.setdefault("MOSS_TTS_MODEL_PATH",
                      str(REPO / ".probe" / "models" / "MOSS-TTS-Realtime"))
os.environ.setdefault("MOSS_TTS_CODEC_MODEL_PATH",
                      str(REPO / ".probe" / "models" / "MOSS-Audio-Tokenizer"))

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torchaudio  # noqa: E402


def _load(path, *a, **k):
    out = subprocess.run(
        [FFMPEG, "-v", "error", "-i", str(path), "-f", "f32le", "-acodec",
         "pcm_f32le", "-ac", "1", "-ar", "48000", "-"],
        capture_output=True, check=True,
    ).stdout
    wav = torch.from_numpy(np.frombuffer(out, dtype=np.float32)).unsqueeze(0)
    return wav, 48000


class _Info:
    def __init__(self, sr):
        self.sample_rate = sr


def _info(path, *a, **k):
    return _Info(48000)


def _save(path, wav, sample_rate, *a, **k):
    pcm = (wav[0].detach().cpu().clamp(-1, 1).numpy() * 32767).astype(np.int16)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(int(sample_rate))
        w.writeframes(pcm.tobytes())


torchaudio.load = _load
torchaudio.info = _info
torchaudio.save = _save

# NB: the codec's remote code upcasts activations to fp32 internally, so
# it must load fp32 (its config's dtype) — bf16 weights clash with its own
# float buffers. The server handles this; nothing to patch here.

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8083)
    args = parser.parse_args()
    os.chdir(MOSS_DIR)               # fast_api resolves its assets relatively
    sys.path.insert(0, str(MOSS_DIR))
    import fast_api  # noqa: E402
    import uvicorn  # noqa: E402
    uvicorn.run(fast_api.app, host="127.0.0.1", port=args.port)
