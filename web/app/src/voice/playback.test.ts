// The scheduling queue SPEC §3 names: chunks decode to AudioBuffers and
// start back-to-back on a monotonic cursor; an underrun resets the cursor
// to now (a clean pause, never a catch-up storm); pause/resume move the
// playhead. The AudioContext is a constructor argument, so the tests
// inject a hand-rolled fake and read the schedule off the fake sources.
import { describe, expect, it } from "vitest";
import { JITTER_S, PcmPlayer } from "./playback";

class FakeSource {
  buffer: unknown = null;
  onended: (() => void) | null = null;
  startedAt: number | null = null;
  startOffset = 0;
  stopped = false;
  connect(): void {}
  start(when: number, offset = 0): void {
    this.startedAt = when;
    this.startOffset = offset;
  }
  stop(): void {
    this.stopped = true;
  }
}

class FakeCtx {
  currentTime = 0;
  sources: FakeSource[] = [];
  destination = {};
  createBuffer(_ch: number, len: number, rate: number) {
    const data = new Float32Array(len);
    return {
      duration: len / rate,
      length: len,
      sampleRate: rate,
      copyToChannel: (src: Float32Array) => data.set(src),
      getChannelData: () => data,
    };
  }
  createBufferSource() {
    const s = new FakeSource();
    this.sources.push(s);
    return s;
  }
}

// 0.2 s of 24 kHz = 4800 samples = 9600 bytes of PCM16
const CHUNK = new Uint8Array(9600);
const pcm = () => CHUNK.buffer.slice(0);

const make = () => {
  const ctx = new FakeCtx();
  return { ctx, player: new PcmPlayer(ctx as unknown as AudioContext) };
};

describe("PcmPlayer scheduling", () => {
  it("opens the first chunk after the jitter lead, then packs back-to-back", () => {
    const { ctx, player } = make();
    player.start(24000);
    player.feed(pcm());
    expect(ctx.sources[0].startedAt).toBeCloseTo(JITTER_S);
    player.feed(pcm());
    // second chunk starts exactly where the first ends: monotonic cursor
    expect(ctx.sources[1].startedAt).toBeCloseTo(JITTER_S + 0.2);
  });

  it("an underrun resets the cursor to now — a clean pause, no catch-up", () => {
    const { ctx, player } = make();
    player.start(24000);
    player.feed(pcm());
    ctx.currentTime = 5; // the queue starved while the network stalled
    player.feed(pcm());
    // not 0.475 (the stale cursor): the buffer plays now + jitter
    expect(ctx.sources[1].startedAt).toBeCloseTo(5 + JITTER_S);
  });

  it("carries an odd trailing byte across chunks", () => {
    const { ctx, player } = make();
    player.start(24000);
    // 0x4000 little-endian = 16384 → 0.5; three bytes = one sample + carry
    player.feed(new Uint8Array([0x00, 0x40, 0x00]).buffer);
    expect(ctx.sources).toHaveLength(1);
    player.feed(new Uint8Array([0x40]).buffer); // completes the carried byte
    expect(ctx.sources).toHaveLength(2);
    const second = ctx.sources[1].buffer as { getChannelData(): Float32Array; length: number };
    expect(second.length).toBe(1);
    expect(second.getChannelData()[0]).toBeCloseTo(0.5);
  });

  it("stop discards the queue, stops live sources, and ignores late feeds", () => {
    const { ctx, player } = make();
    player.start(24000);
    player.feed(pcm());
    player.feed(pcm());
    player.stop();
    expect(ctx.sources.every((s) => s.stopped)).toBe(true);
    expect(player.isSpeaking).toBe(false);
    player.feed(pcm()); // a frame that lost the race with stop: dropped
    expect(ctx.sources).toHaveLength(2);
  });

  it("pause stops at the playhead; resume continues from the offset", () => {
    const { ctx, player } = make();
    player.start(24000);
    player.feed(pcm()); // A: scheduled at 0.275
    player.feed(pcm()); // B: scheduled at 0.475
    ctx.currentTime = 0.375; // A has played 0.1 s
    player.pause();
    expect(ctx.sources[0].stopped).toBe(true);
    expect(player.isSpeaking).toBe(true); // paused, not stopped
    ctx.currentTime = 2; // the barge-in window
    player.resume();
    const resumed = ctx.sources[2];
    expect(resumed.startedAt).toBeCloseTo(2 + JITTER_S);
    expect(resumed.startOffset).toBeCloseTo(0.1); // where the ear left off
    // B re-queues behind the remainder: 2.275 + 0.1 of A left = 2.375
    expect(ctx.sources[3].startedAt).toBeCloseTo(2.375);
  });

  it("isSpeaking covers the drain after end, and clears when drained", () => {
    const { ctx, player } = make();
    player.start(24000);
    player.feed(pcm());
    expect(player.isSpeaking).toBe(true);
    player.end();
    expect(player.isSpeaking).toBe(true); // still audible: the drain phase
    ctx.sources[0].onended?.();
    expect(player.isSpeaking).toBe(false);
  });

  it("a stop makes a pending resume inert", () => {
    // The App's resume timer is not cancelled on every path; the
    // invariant lives here: stop clears the resumePoint, so a resume
    // that fires after a supersede or disarm schedules nothing.
    const { ctx, player } = make();
    player.start(24000);
    player.feed(pcm());
    ctx.currentTime = 0.375;
    player.pause();
    player.stop(); // superseded / disarmed while the timer was armed
    ctx.currentTime = 2;
    player.resume(); // the stale timer firing
    expect(ctx.sources).toHaveLength(1); // no third source
    expect(player.isSpeaking).toBe(false);
  });

  it("a new start supersedes a leftover stream", () => {
    const { ctx, player } = make();
    player.start(24000);
    player.feed(pcm());
    player.start(16000); // next reply, different rate
    expect(ctx.sources[0].stopped).toBe(true);
    player.feed(new Uint8Array(3200).buffer); // 0.1 s at 16 kHz
    const buf = ctx.sources[1].buffer as { sampleRate: number };
    expect(buf.sampleRate).toBe(16000);
  });
});
