"""PROTOTYPE — phd-helper voice loop latency probe. Throwaway; see README.md.

Browser mic -> WebSocket -> silero-vad endpointing -> parakeet STT -> echo stub
-> Kokoro TTS, with per-stage latency measurement. Run via run.ps1.

Question it answers: what does the full hands-free round trip feel like, and
where does the latency actually go?
"""
import asyncio
import base64
import io
import json
import os
import time
import wave
from collections import deque
from pathlib import Path

import numpy as np
import torch
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from silero_vad import load_silero_vad
from transformers import pipeline
from kokoro import KPipeline

SR = 16000
CHUNK = 512  # 32 ms at 16 kHz — silero's expected frame size
VAD_THRESHOLD = float(os.environ.get("VAD_THRESHOLD", "0.5"))
HANGOVER_S = float(os.environ.get("HANGOVER_S", "0.6"))  # silence after speech before endpoint
PREROLL_S = 0.3
STUB_DELAY_S = float(os.environ.get("STUB_DELAY_S", "0"))  # fake LLM think-time, to feel a slow turn
TTS_SR = 24000
TTS_VOICE = os.environ.get("TTS_VOICE", "af_heart")

print("[proto] loading models (first run downloads weights, be patient)...", flush=True)
VAD = load_silero_vad()
ASR = pipeline(
    "automatic-speech-recognition",
    model="nvidia/parakeet-tdt-0.6b-v3",
    torch_dtype=torch.float16,
    device="cuda:0",
)
TTS = KPipeline(lang_code="a")


def _warmup():
    with torch.no_grad():
        VAD(torch.zeros(CHUNK), SR)
    ASR({"array": np.zeros(SR, dtype=np.float32), "sampling_rate": SR})
    next(iter(TTS("Hello there.", voice=TTS_VOICE)))[2]


_warmup()
print("[proto] models warm.", flush=True)

app = FastAPI()


@app.get("/")
async def index():
    return FileResponse(Path(__file__).parent / "index.html")


def run_stt(audio: np.ndarray) -> str:
    peak = float(np.max(np.abs(audio))) if len(audio) else 0.0
    if peak > 1e-4:
        audio = audio * (0.9 / peak)  # quiet mics shouldn't starve the ASR either
    out = ASR({"array": audio, "sampling_rate": SR})
    return (out.get("text") or "").strip()


def synth_all(text: str):
    """Synthesize the whole response; returns (t_first_chunk, [wav bytes per sentence])."""
    t_first = None
    chunks = []
    for _, _, audio in TTS(text, voice=TTS_VOICE):
        if t_first is None:
            t_first = time.time()
        pcm = (audio.float().cpu().numpy() * 32767).astype(np.int16)
        buf = io.BytesIO()
        with wave.open(buf, "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(TTS_SR)
            w.writeframes(pcm.tobytes())
        chunks.append(buf.getvalue())
    return t_first, chunks


@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket):
    await ws.accept()
    print("[proto] client connected", flush=True)

    preroll = deque(maxlen=int(PREROLL_S * SR / CHUNK) + 2)
    speech_chunks: list[np.ndarray] = []
    in_speech = False
    busy = False
    cum_end = 0
    last_voiced_wall = 0.0
    last_voiced_cum = 0
    speech_start_cum = 0
    hangover_s = HANGOVER_S  # live-tunable from the client slider
    MAX_UTTERANCE_S = 15.0
    n_chunks = 0
    n_pings = 0

    async def process_turn(audio: np.ndarray, vad_ms: float):
        nonlocal busy
        try:
            await _process_turn(audio, vad_ms)
        except Exception:
            import traceback
            traceback.print_exc()
            try:
                await ws.send_json({"type": "error", "msg": "turn failed, see server console"})
            except Exception:
                pass
        finally:
            busy = False
            try:
                await ws.send_json({"type": "state", "state": "listening"})
            except Exception:
                pass

    async def _process_turn(audio: np.ndarray, vad_ms: float):
        nonlocal busy
        t_ready = time.time()
        text = await asyncio.to_thread(run_stt, audio)
        t_stt = time.time()
        if len(audio) < 0.4 * SR or not text:
            response = "I didn't catch that, could you repeat?"
        else:
            response = f"I heard you say: {text}"
        if STUB_DELAY_S:
            await asyncio.sleep(STUB_DELAY_S)
        t_llm = time.time()
        t_first, chunks = await asyncio.to_thread(synth_all, response)
        t_synth = time.time()
        for seq, wav in enumerate(chunks):
            await ws.send_json({"type": "tts", "seq": seq, "b64": base64.b64encode(wav).decode()})
        t_sent = time.time()
        await ws.send_json({
            "type": "timing",
            "vad_ms": round(vad_ms, 1),
            "stt_ms": round((t_stt - t_ready) * 1000, 1),
            "llm_ms": round((t_llm - t_stt) * 1000, 1),
            "tts_first_ms": round((t_first - t_llm) * 1000, 1),
            "tts_total_ms": round((t_synth - t_llm) * 1000, 1),
            "send_ms": round((t_sent - t_synth) * 1000, 1),
            "transcript": text,
            "response": response,
        })
        print(
            f"[turn] vad={vad_ms:.0f} stt={(t_stt - t_ready) * 1000:.0f} "
            f"llm={(t_llm - t_stt) * 1000:.0f} tts_first={(t_first - t_llm) * 1000:.0f} "
            f"tts_total={(t_synth - t_llm) * 1000:.0f} :: {text!r}",
            flush=True,
        )

    try:
        while True:
            msg = await ws.receive()
            if msg.get("text") is not None:
                try:
                    ctrl = json.loads(msg["text"])
                    if ctrl.get("type") == "config":
                        hangover_s = float(ctrl["hangover_s"])
                        print(f"[proto] hangover -> {hangover_s:.2f}s", flush=True)
                    elif ctrl.get("type") == "ping":
                        n_pings += 1
                        if n_pings % 5 == 0:  # every ~10 s
                            print(f"[ping] chunks={n_chunks} {ctrl}", flush=True)
                except (ValueError, KeyError):
                    pass
                continue
            if msg.get("type") == "websocket.disconnect":
                print(f"[proto] client disconnected (code={msg.get('code')})", flush=True)
                return
            if msg.get("bytes") is None:
                continue
            n_chunks += 1
            if msg.get("bytes") is None:
                continue
            pcm = np.frombuffer(msg["bytes"], dtype=np.int16).astype(np.float32) / 32767.0
            if busy:
                continue  # discard while the agent is thinking/speaking
            with torch.no_grad():
                prob = float(VAD(torch.from_numpy(pcm), SR))
            cum_end += len(pcm)
            preroll.append(pcm)
            voiced = prob > VAD_THRESHOLD
            started_now = False
            if voiced:
                if not in_speech:
                    in_speech = True
                    started_now = True
                    speech_start_cum = cum_end
                    speech_chunks = list(preroll)  # already includes this chunk
                    await ws.send_json({"type": "state", "state": "heard"})
                last_voiced_wall = time.time()
                last_voiced_cum = cum_end
            if in_speech:
                if not started_now:
                    speech_chunks.append(pcm)
                too_long = cum_end - speech_start_cum > MAX_UTTERANCE_S * SR
                if cum_end - last_voiced_cum >= hangover_s * SR or too_long:
                    # endpoint: the utterance is over
                    in_speech = False
                    busy = True
                    vad_ms = (time.time() - last_voiced_wall) * 1000
                    audio = np.concatenate(speech_chunks)
                    speech_chunks = []
                    await ws.send_json({"type": "endpoint"})
                    await ws.send_json({"type": "state", "state": "busy"})
                    asyncio.create_task(process_turn(audio, vad_ms))
    except (WebSocketDisconnect, RuntimeError):
        print("[proto] client disconnected", flush=True)


if __name__ == "__main__":
    import uvicorn

    print("[proto] open http://127.0.0.1:8756", flush=True)
    uvicorn.run(app, host="127.0.0.1", port=8756, log_level="warning")
