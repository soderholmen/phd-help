// The socket's demux (SPEC §3): binary frames are PCM16 for the
// player; audio_start/audio_end bookend it; turn_started and
// tts_stopped hard-stop the buffer (supersede / thinking-phase stop —
// the player's stop() also makes a pending resume inert, which is why
// the resume window needs no cancel here). Everything else is the
// reducer's, and every event dispatches so the reducer stays total
// over the wire.
import type { PcmPlayer } from "../voice/playback";
import type { ServerEvent, ShellEvent } from "../types";

export interface Sink {
  dispatch: (event: ShellEvent) => void;
  // the four methods the demux touches, tied to PcmPlayer's real
  // signatures — a rename there lands here, not silently in App.tsx
  player: Pick<PcmPlayer, "start" | "feed" | "end" | "stop">;
}

export function routeIncoming(data: string | ArrayBuffer, sink: Sink): void {
  if (typeof data !== "string") {
    sink.player.feed(data);
    return;
  }
  const ev = JSON.parse(data) as ServerEvent;
  switch (ev.type) {
    case "audio_start":
      sink.player.start(ev.sample_rate);
      break;
    case "audio_end":
      sink.player.end();
      break;
    case "tts_stopped":
    case "turn_started":
      sink.player.stop();
      break;
  }
  sink.dispatch(ev);
}
