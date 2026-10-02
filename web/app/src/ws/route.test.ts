// The socket's demux, as a pure function: binary feeds the player,
// the audio bookends drive it, turn_started/tts_stopped hard-stop it,
// and everything else is the reducer's. useConnection is thin because
// this holds the contract.
import { describe, expect, it } from "vitest";
import { routeIncoming } from "./route";
import type { ServerEvent, ShellEvent } from "../types";

function makeSink() {
  const calls: string[] = [];
  const events: ShellEvent[] = [];
  const sink = {
    dispatch: (e: ShellEvent) => {
      events.push(e);
      calls.push(`dispatch:${e.type}`);
    },
    player: {
      start: (rate: number) => void calls.push(`start:${rate}`),
      feed: (pcm: ArrayBuffer) => void calls.push(`feed:${pcm.byteLength}`),
      end: () => void calls.push("end"),
      stop: () => void calls.push("stop"),
    },
  };
  return { sink, calls, events };
}

const json = (ev: ServerEvent) => JSON.stringify(ev);

describe("routeIncoming", () => {
  it("binary feeds the player and nothing else", () => {
    const { sink, calls, events } = makeSink();
    routeIncoming(new Uint8Array(9600).buffer, sink);
    expect(calls).toEqual(["feed:9600"]);
    expect(events).toHaveLength(0);
  });

  it("audio_start opens the player at the sidecar's rate", () => {
    const { sink, calls } = makeSink();
    routeIncoming(json({ type: "audio_start", sample_rate: 24000 }), sink);
    expect(calls).toEqual(["start:24000", "dispatch:audio_start"]);
  });

  it("audio_end closes the stream (the drain keeps playing)", () => {
    const { sink, calls } = makeSink();
    routeIncoming(json({ type: "audio_end" }), sink);
    expect(calls).toEqual(["end", "dispatch:audio_end"]);
  });

  it("tts_stopped and turn_started hard-stop the buffer", () => {
    const { sink, calls } = makeSink();
    routeIncoming(json({ type: "tts_stopped" }), sink);
    routeIncoming(json({ type: "turn_started" }), sink);
    expect(calls).toEqual([
      "stop",
      "dispatch:tts_stopped",
      "stop",
      "dispatch:turn_started",
    ]);
  });

  it("ordinary events reach only the reducer", () => {
    const { sink, calls } = makeSink();
    routeIncoming(json({ type: "assistant_text", text: "hi" }), sink);
    routeIncoming(json({ type: "hello", client_id: "c1", section: null }), sink);
    expect(calls).toEqual(["dispatch:assistant_text", "dispatch:hello"]);
  });
});
