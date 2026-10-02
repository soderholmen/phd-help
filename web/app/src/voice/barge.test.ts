// The sustained-speech gate SPEC §3 asks for (~150-200 ms of speech
// while TTS plays, not a 2 s heartbeat, not a single noisy chunk).
// Pure over per-100 ms RMS levels; the app feeds it from the capture
// worklet and reacts by pausing the player.
import { describe, expect, it } from "vitest";
import { BargeGate } from "./barge";

const make = (speaking = () => true) => {
  const fired: number[] = [];
  const gate = new BargeGate({
    threshold: 0.04,
    chunks: 2,
    speaking,
    onFire: () => fired.push(1),
  });
  return { gate, fired };
};

describe("BargeGate", () => {
  it("needs sustained speech: one loud chunk is a cough, two fire", () => {
    const { gate, fired } = make();
    gate.level(0.5);
    expect(fired).toHaveLength(0);
    gate.level(0.5);
    expect(fired).toHaveLength(1);
  });

  it("fires once per speaking episode — a held utterance is one barge-in", () => {
    const { gate, fired } = make();
    for (let i = 0; i < 10; i++) gate.level(0.5);
    expect(fired).toHaveLength(1);
  });

  it("silence resets the streak", () => {
    const { gate, fired } = make();
    gate.level(0.5);
    gate.level(0.001); // the gap between syllables is not a sustained utterance
    gate.level(0.5);
    expect(fired).toHaveLength(0);
    gate.level(0.5);
    expect(fired).toHaveLength(1);
  });

  it("is closed while nothing is playing — quiet-room noise never barges", () => {
    const { gate, fired } = make(() => false);
    for (let i = 0; i < 10; i++) gate.level(0.5);
    expect(fired).toHaveLength(0);
  });

  it("re-arms for the next speaking episode", () => {
    let speaking = true;
    const { gate, fired } = make(() => speaking);
    gate.level(0.5);
    gate.level(0.5);
    expect(fired).toHaveLength(1);
    speaking = false; // this reply ended (stopped, drained, or superseded)
    gate.level(0.0);
    speaking = true; // the next reply starts
    gate.level(0.5);
    gate.level(0.5);
    expect(fired).toHaveLength(2);
  });
});
