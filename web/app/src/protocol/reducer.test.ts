// The reducer is the shell's deterministic core — every server event
// shape from app.py gets a fold test here.
import { describe, expect, it } from "vitest";
import { initialState, reducer } from "./reducer";
import type { ShellState } from "./types";

const fold = (...events: Parameters<typeof reducer>[1][]) =>
  events.reduce(reducer, initialState);

describe("connection", () => {
  it("tracks open/close and clears turn + arming on close", () => {
    const open = fold({ type: "connection_opened" }, { type: "hello", client_id: "c1" });
    expect(open).toMatchObject({ connected: true, clientId: "c1" });
    const busy = fold(
      { type: "connection_opened" },
      { type: "armed", ok: true, holder: "c1" },
      { type: "turn_started" },
      { type: "connection_closed" },
    );
    expect(busy).toMatchObject({ connected: false, turnActive: false, armed: false, rms: 0 });
  });

  it("close drops the anchor and pending diffs — the server's Session is fresh per socket", () => {
    const s = fold(
      { type: "section_selected", section: "sections/intro.tex" },
      {
        type: "diff",
        diff_id: "d1",
        section: "sections/intro.tex",
        find: "a",
        replace: "b",
      },
      { type: "connection_closed" },
    );
    // app.py builds a new Session per WS connect: selected=None, no
    // pending diffs re-presented yet (§7 slice). Keeping them client-
    // side would claim an anchor and an approval the server doesn't have.
    expect(s.selected).toBeNull();
    expect(s.pendingDiffs).toEqual([]);
    // The visible transcript stays (the user's record), but a notice
    // says the agent's own context reset — until §7 resume lands.
    expect(s.messages.at(-1)).toMatchObject({
      role: "notice",
      text: "Connection lost — the agent's context resets on reconnect",
    });
  });
});

describe("turn taking", () => {
  it("user_send appends with the current anchor; turn_started marks busy", () => {
    const s = fold(
      { type: "section_selected", section: "sections/intro.tex" },
      { type: "user_send", text: "tighten it" },
      { type: "turn_started" },
    );
    expect(s.turnActive).toBe(true);
    expect(s.messages).toEqual([{ id: 1, role: "user", text: "tighten it", section: "sections/intro.tex" }]);
  });

  it("assistant_text appends and ends the turn", () => {
    const s = fold({ type: "turn_started" }, { type: "assistant_text", text: "done" });
    expect(s.turnActive).toBe(false);
    expect(s.messages.at(-1)).toMatchObject({ role: "assistant", text: "done" });
  });

  it("turn_interrupted ends the turn with a notice", () => {
    const s = fold({ type: "turn_started" }, { type: "turn_interrupted" });
    expect(s.turnActive).toBe(false);
    expect(s.messages.at(-1)).toMatchObject({ role: "notice", text: "Turn interrupted" });
  });

  it("error ends the turn and lands inline (§8)", () => {
    const s = fold(
      { type: "turn_started" },
      { type: "error", where: "llm", message: "vLLM down" },
    );
    expect(s.turnActive).toBe(false);
    expect(s.messages.at(-1)).toMatchObject({ role: "error", text: "llm: vLLM down" });
  });
});

describe("diffs", () => {
  const diff = {
    type: "diff" as const,
    diff_id: "d1",
    section: "sections/intro.tex",
    find: "old",
    replace: "new",
  };

  it("diff queues a card", () => {
    const s = fold(diff);
    expect(s.pendingDiffs).toEqual([
      { diff_id: "d1", section: "sections/intro.tex", find: "old", replace: "new" },
    ]);
  });

  it("approve resolves the card with the section named", () => {
    const s = fold(diff, {
      type: "diff_resolved",
      diff_id: "d1",
      applied: true,
      reason: "applied",
      text: null,
    });
    expect(s.pendingDiffs).toEqual([]);
    expect(s.messages.at(-1)).toMatchObject({ role: "notice", text: "Applied to sections/intro.tex" });
  });

  it("a bounced apply names the reason", () => {
    const s = fold(diff, {
      type: "diff_resolved",
      diff_id: "d1",
      applied: false,
      reason: "section changed since proposal",
      text: null,
    });
    expect(s.pendingDiffs).toEqual([]);
    expect(s.messages.at(-1)).toMatchObject({
      role: "notice",
      text: "Not applied (section changed since proposal)",
    });
  });

  it("resolving an unknown diff_id still clears nothing and notices", () => {
    const s: ShellState = fold({
      type: "diff_resolved",
      diff_id: "ghost",
      applied: false,
      reason: "discarded",
      text: null,
    });
    expect(s.pendingDiffs).toEqual([]);
    expect(s.messages.at(-1)?.role).toBe("notice");
  });
});

describe("voice endpoint", () => {
  it("arming succeeds", () => {
    const s = fold({ type: "armed", ok: true, holder: "c1" });
    expect(s.armed).toBe(true);
  });

  it("a held endpoint refuses with a notice naming the holder", () => {
    const s = fold({ type: "armed", ok: false, holder: "phone" });
    expect(s.armed).toBe(false);
    expect(s.messages.at(-1)).toMatchObject({ role: "notice", text: "Voice endpoint held by phone" });
  });

  it("disarm and pong", () => {
    const s = fold(
      { type: "armed", ok: true, holder: "c1" },
      { type: "disarmed" },
      { type: "pong", rms: 0.4 },
    );
    expect(s).toMatchObject({ armed: false, rms: 0.4 });
  });
});

describe("anchor", () => {
  it("select and clear", () => {
    const s = fold(
      { type: "section_selected", section: "sections/intro.tex" },
      { type: "section_selected", section: null },
    );
    expect(s.selected).toBeNull();
  });
});

it("an unknown event never wedges the shell", () => {
  const s = reducer(initialState, { type: "from_the_future" } as never);
  expect(s).toBe(initialState);
});
