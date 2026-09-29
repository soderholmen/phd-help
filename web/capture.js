// AudioWorklet capture processor (SPEC §1): downsample to 16 kHz mono
// PCM16 and post 100 ms chunks to the main thread for WebSocket transport.
class CaptureProcessor extends AudioWorkletProcessor {
  constructor() {
    super();
    this.ratio = sampleRate / 16000;
    this.blockStart = 0; // global index of the current block's first sample
    this.next = 0;       // global index of the next sample to take
    this.buf = new Int16Array(1600); // 100 ms at 16 kHz
    this.n = 0;
  }
  process(inputs) {
    const ch = inputs[0] && inputs[0][0];
    if (!ch) return true;
    const end = this.blockStart + ch.length;
    for (let g = this.next; g < end; g += this.ratio) {
      const s = Math.max(-1, Math.min(1, ch[g - this.blockStart]));
      this.buf[this.n++] = s * 0x7fff;
      if (this.n === this.buf.length) {
        this.port.postMessage(this.buf.buffer, [this.buf.buffer]);
        this.buf = new Int16Array(1600);
        this.n = 0;
      }
    }
    // Advance the take-position past this block, preserving phase.
    while (this.next < end) this.next += this.ratio;
    this.blockStart = end;
    return true;
  }
}
registerProcessor('capture', CaptureProcessor);
