// Voice-out playback (SPEC §3): the scheduling queue the spec names.
// PCM16 chunks arrive as binary WS frames between audio_start and
// audio_end; each decodes to an AudioBuffer and starts on a monotonic
// cursor, back-to-back. An underrun resets the cursor to now — a clean
// pause, never a catch-up storm of past-due starts. pause/resume move
// the playhead: pause stops the live source and remembers where the ear
// was; resume re-schedules the remainder from there. The AudioContext
// is a constructor argument (the tests inject a fake; the app passes a
// real one unlocked by the arm gesture).
export const JITTER_S = 0.275; // SPEC §3: ~250-300 ms jitter buffer

type Scheduled = { src: AudioBufferSourceNode; buf: AudioBuffer; start: number };

export class PcmPlayer {
  private queue: AudioBuffer[] = [];
  private scheduled: Scheduled[] = [];
  private cursor = 0;
  private carry: Uint8Array | null = null;
  private rate = 24000;
  private active = false;
  private paused = false;
  private ended = false;
  private resumePoint: { buf: AudioBuffer; offset: number } | null = null;
  private readonly ctx: AudioContext;
  private readonly jitter: number;

  constructor(ctx: AudioContext, jitter: number = JITTER_S) {
    this.ctx = ctx;
    this.jitter = jitter;
  }

  /** The arm toggle's gesture unlocks autoplay (the MicCapture pattern);
   *  audio only ever flows to an armed holder anyway. */
  unlock(): void {
    void this.ctx.resume();
  }

  /** True while audio is playing, paused mid-stream, or draining —
   *  the window in which a barge-in is meaningful. */
  get isSpeaking(): boolean {
    return (
      this.active &&
      (!this.ended || this.queue.length > 0 || this.scheduled.length > 0 ||
        this.resumePoint !== null)
    );
  }

  start(sampleRate: number): void {
    this.stop(); // a new reply supersedes any leftover stream
    this.active = true;
    this.rate = sampleRate;
  }

  feed(pcm: ArrayBuffer): void {
    if (!this.active) return; // a frame that lost the race with stop
    const bytes = new Uint8Array(pcm);
    let data = bytes;
    if (this.carry) {
      data = new Uint8Array(this.carry.length + bytes.length);
      data.set(this.carry);
      data.set(bytes, this.carry.length);
      this.carry = null;
    }
    const usable = data.length - (data.length % 2); // PCM16: whole samples
    if (usable < data.length) this.carry = data.slice(usable);
    if (!usable) return;
    const view = new DataView(data.buffer, data.byteOffset, usable);
    const f = new Float32Array(usable / 2);
    for (let i = 0; i < f.length; i++) f[i] = view.getInt16(i * 2, true) / 0x8000;
    const buf = this.ctx.createBuffer(1, f.length, this.rate);
    buf.copyToChannel(f, 0);
    this.queue.push(buf);
    this.schedule();
  }

  /** The stream is complete: isSpeaking stays true through the drain. */
  end(): void {
    this.ended = true;
  }

  /** Hard stop and discard — barge-in while thinking, supersede, disarm.
   *  Also makes a pending resume inert: stop clears the resumePoint, so
   *  a resume timer that fires after a stop schedules nothing. */
  stop(): void {
    this.stopAll();
    this.queue = [];
    this.resumePoint = null;
    this.carry = null;
    this.active = false;
    this.paused = false;
    this.ended = false;
    this.cursor = 0;
  }

  pause(): void {
    if (!this.active || this.paused) return;
    this.paused = true;
    const now = this.ctx.currentTime;
    let audible: Scheduled | undefined;
    const future: Scheduled[] = [];
    for (const s of this.scheduled) {
      if (s.start <= now) audible = s;
      else future.push(s);
    }
    this.stopAll();
    this.resumePoint = audible
      ? {
          buf: audible.buf,
          offset: Math.min(Math.max(now - audible.start, 0), audible.buf.duration),
        }
      : null;
    // not-yet-started buffers re-queue, in order, behind nothing
    this.queue = future.map((s) => s.buf).concat(this.queue);
  }

  resume(): void {
    if (!this.active || !this.paused) return;
    this.paused = false;
    const now = this.ctx.currentTime;
    if (this.resumePoint) {
      const { buf, offset } = this.resumePoint;
      this.resumePoint = null;
      // the playhead picks up where the ear left off
      this.cursor = this.play(buf, now + this.jitter, offset);
    }
    this.schedule();
  }

  private stopAll(): void {
    for (const s of this.scheduled) {
      try {
        s.src.stop();
      } catch {
        /* already ended: the honest race */
      }
    }
    this.scheduled = [];
  }

  /** One source, connected and started; returns where its audio ends. */
  private play(buf: AudioBuffer, at: number, offset = 0): number {
    const src = this.ctx.createBufferSource();
    src.buffer = buf;
    src.connect(this.ctx.destination);
    src.onended = () => this.untick(src);
    src.start(at, offset);
    this.scheduled.push({ src, buf, start: at });
    return at + (buf.duration - offset);
  }

  private untick(src: AudioBufferSourceNode): void {
    this.scheduled = this.scheduled.filter((s) => s.src !== src);
  }

  private schedule(): void {
    if (this.paused) return;
    while (this.queue.length) {
      const buf = this.queue.shift()!;
      // max() is the whole contract: back-to-back off the cursor, but a
      // starved cursor (underrun) restarts at now + jitter — a clean
      // pause, never a burst of past-due catch-up starts.
      this.cursor = this.play(buf, Math.max(this.cursor, this.ctx.currentTime + this.jitter));
    }
  }
}
