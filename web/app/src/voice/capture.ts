// Mic capture (SPEC §1/§3): getUserMedia (echoCancellation) -> worklet ->
// PCM16 chunks. Arming is explicit — this only ever runs after the user
// toggles voice on; capture also pauses when the tab loses focus (§3).
export class MicCapture {
  private ctx: AudioContext | null = null;
  rms = 0;
  onChunk: ((pcm: ArrayBuffer) => void) | null = null;

  /** Lazy one-time setup (needs the user gesture), then resume. */
  async start(): Promise<void> {
    if (!this.ctx) {
      this.ctx = new AudioContext();
      await this.ctx.audioWorklet.addModule("/capture.js");
      const stream = await navigator.mediaDevices.getUserMedia({
        audio: { echoCancellation: true, noiseSuppression: true, autoGainControl: false },
      });
      const src = this.ctx.createMediaStreamSource(stream);
      const node = new AudioWorkletNode(this.ctx, "capture");
      node.port.onmessage = (ev: MessageEvent<ArrayBuffer>) => {
        this.onChunk?.(ev.data);
        const view = new Int16Array(ev.data);
        let sum = 0;
        for (let i = 0; i < view.length; i += 8) sum += view[i] * view[i];
        this.rms = Math.sqrt(sum / (view.length / 8)) / 0x7fff; // meter feed
      };
      src.connect(node);
    }
    await this.ctx.resume();
  }

  /** Disarmed or tab-hidden: capture pauses, the toggle state survives. */
  suspend(): void {
    void this.ctx?.suspend();
  }
}
