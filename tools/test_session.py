"""tools/test_session.py -- headless end-to-end check of a running server (no browser, no mic).

    venv/bin/python tools/test_session.py [PORT] [SECONDS]      (defaults 8998, 40)

Streams tools/test_question.wav (20 s of speech, then silence) to the server in real time,
counts the audio + video messages that come back, then reads the session capture the server
saved and prints how much speech it heard, how much it said, its step time and what it said.
PASS = it heard the question, answered with speech, and sent video at ~25 fps.
"""
import asyncio, glob, json, os, sys, time
import numpy as np, torchaudio, websockets

HERE = os.path.dirname(os.path.abspath(__file__))
PORT = sys.argv[1] if len(sys.argv) > 1 else "8998"
SECONDS = float(sys.argv[2]) if len(sys.argv) > 2 else 40.0
SR, FRAME = 24000, 1920
CAPTURES = os.path.join(os.path.dirname(HERE), "captures")

wav, sr = torchaudio.load(os.path.join(HERE, "test_question.wav"))
wav = wav.mean(0) if wav.shape[0] > 1 else wav[0]
if sr != SR:
    wav = torchaudio.functional.resample(wav, sr, SR)
p = wav.numpy(); need = int(SECONDS * SR)
p = np.concatenate([p, 1e-4 * np.random.randn(max(0, need - len(p))).astype(np.float32)])[:need]
i16 = (np.clip(p, -1, 1) * 32767).astype(np.int16)


async def main():
    cnt = {"a": 0, "v": 0}
    ws = await websockets.connect(f"ws://127.0.0.1:{PORT}/ws/conversation", max_size=None, ping_interval=None)
    ready = json.loads(await ws.recv())
    vws = await websockets.connect(f"ws://127.0.0.1:{PORT}{ready.get('video_ws_path')}", max_size=None, ping_interval=None)

    async def rx(sock, key):
        try:
            async for _ in sock:
                cnt[key] += 1
        except Exception:
            pass
    tasks = [asyncio.create_task(rx(ws, "a")), asyncio.create_task(rx(vws, "v"))]
    await ws.send(json.dumps({"type": "start", "sample_rate": SR}))
    await asyncio.sleep(1.0)
    t0 = time.perf_counter(); n = len(i16) // FRAME
    for i in range(n):
        await ws.send(b"\x03" + i16[i * FRAME:(i + 1) * FRAME].tobytes())
        dt = t0 + (i + 1) * FRAME / SR - time.perf_counter()
        if dt > 0:
            await asyncio.sleep(dt)
    await asyncio.sleep(3)
    for t in tasks:
        t.cancel()
    for s in (ws, vws):
        try:
            await s.close()
        except Exception:
            pass
    return cnt, time.perf_counter() - t0

before = set(glob.glob(os.path.join(CAPTURES, "*")))
cnt, el = asyncio.run(main())
print(f"sent {SECONDS:.0f}s of audio | received {cnt['a']} audio msgs, {cnt['v']} video frames ({cnt['v'] / el:.1f} fps)")
cap = None
for _ in range(30):
    new = sorted(set(glob.glob(os.path.join(CAPTURES, "*"))) - before)
    if new and os.path.exists(os.path.join(new[-1], "persona_events.jsonl")):
        cap = new[-1]; break
    time.sleep(1)
if cap is None:
    sys.exit("FAIL: no session capture was saved (see the server log)")
time.sleep(2)
ev = [json.loads(l) for l in open(os.path.join(cap, "persona_events.jsonl"))]
heard = sum(e["input_rms"] > 0.01 for e in ev) * 0.08
said = sum(e["reply_rms"] > 0.01 for e in ev) * 0.08
steps = sorted(e["total_ms"] for e in ev[5:]) or [0]
text = json.load(open(os.path.join(cap, "manifest.json"))).get("persona_audio_text", "").strip()
print(f"heard {heard:.1f}s of speech | said {said:.1f}s | step time median {steps[len(steps) // 2]:.1f} ms (budget 80)")
print(f"model said: {text}")
ok = heard > 5 and said > 3 and cnt["v"] / el > 15
print("PASS" if ok else "FAIL")
sys.exit(0 if ok else 1)
