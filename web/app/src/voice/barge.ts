// The barge-in gate (SPEC §3): ~150-200 ms of sustained speech while
// TTS plays. Pure over the worklet's per-100 ms RMS levels — no clock,
// no DOM — so the contract is testable at the seam. It fires once per
// speaking episode; the app's reaction is pause + bargeIn() + the
// resume window. Energy gating, not a VAD: a false trip is recoverable
// (no final transcript inside the window resumes playback), which is
// exactly why the resume half of §3 exists. Word-count gating is
// deferred (docs/audio-stack.md).
export const BARGE_RMS = 0.04; // sustained-speech floor on the AEC'd mic
export const BARGE_CHUNKS = 2; // 2 x 100 ms worklet chunks ≈ 200 ms
export const RESUME_WINDOW_MS = 2500; // no final in this window → resume

export interface GateOpts {
  threshold: number;
  chunks: number;
  speaking: () => boolean;
  onFire: () => void;
}

export class BargeGate {
  private streak = 0;
  private fired = false;
  private readonly opts: GateOpts;

  constructor(opts: GateOpts) {
    this.opts = opts;
  }

  level(rms: number): void {
    if (!this.opts.speaking()) {
      // Not playing: nothing to interrupt. Reset so the next reply's
      // first sustained utterance can fire again.
      this.streak = 0;
      this.fired = false;
      return;
    }
    if (this.fired) return; // one barge-in per episode
    if (rms > this.opts.threshold) {
      if (++this.streak >= this.opts.chunks) {
        this.fired = true;
        this.streak = 0;
        this.opts.onFire();
      }
    } else {
      this.streak = 0;
    }
  }
}
