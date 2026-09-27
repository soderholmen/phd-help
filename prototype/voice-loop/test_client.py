"""PROTOTYPE helper — drives the voice loop without a mic. Throwaway.

Synthesizes a test sentence with Kokoro, streams it downsampled to 16 kHz as if
from the browser worklet, then silence, and prints what comes back.
Run with server.py already listening:  .venv/python.exe test_client.py
"""
import asyncio, json, time
import numpy as np
import websockets
from kokoro import KPipeline

UTTERANCE = "This is a test of the voice loop."
SR = 16000


def utterance_pcm16():
    pipe = KPipeline(lang_code="a", repo_id="hexgrad/Kokoro-82M")
    audio = np.concatenate([a.float().cpu().numpy() for _, _, a in pipe(UTTERANCE, voice="af_heart")])
    # 24 kHz -> 16 kHz linear interp
    x_old = np.linspace(0, len(audio) - 1, len(audio))
    x_new = np.linspace(0, len(audio) - 1, int(len(audio) * SR / 24000))
    pcm = np.interp(x_new, x_old, audio)
    return (np.clip(pcm, -1, 1) * 32767).astype(np.int16)


async def main():
    speech = utterance_pcm16()
    print(f"[test] utterance: {len(speech)/SR:.2f} s of audio", flush=True)
    async with websockets.connect("ws://127.0.0.1:8756/ws") as ws:
        # drain initial state messages while we stream
        async def reader():
            out = []
            try:
                while True:
                    m = json.loads(await asyncio.wait_for(ws.recv(), timeout=60))
                    out.append(m)
                    if m["type"] == "timing":
                        return out
            except asyncio.TimeoutError:
                return out

        recv = asyncio.create_task(reader())
        for i in range(0, len(speech) - 512, 512):
            await ws.send(speech[i:i + 512].tobytes())
            await asyncio.sleep(0.032)
        silence = np.zeros(512, dtype=np.int16)
        for _ in range(int(1.0 * SR / 512)):  # 1 s of trailing silence
            await ws.send(silence.tobytes())
            await asyncio.sleep(0.032)
        msgs = await recv

    for m in msgs:
        if m["type"] == "tts":
            print(f"[test] tts chunk seq={m['seq']} ({len(m['b64'])//1365} ms audio)")
        else:
            print("[test]", json.dumps(m))
    timing = [m for m in msgs if m["type"] == "timing"]
    assert timing, "no timing message — loop did not close!"
    t = timing[0]
    assert UTTERANCE.lower().rstrip(".") in t["transcript"].lower(), f"bad transcript: {t['transcript']!r}"
    print("[test] LOOP_OK")


if __name__ == "__main__":
    asyncio.run(main())
