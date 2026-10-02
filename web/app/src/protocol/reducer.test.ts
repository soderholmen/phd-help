// The reducer is the shell's deterministic core — every server event
// shape from app.py gets a fold test here.
import { describe, expect, it } from "vitest";
import { initialState, reducer } from "./reducer";
import type { ShellState } from "./types";

const fold = (...events: Parameters<typeof reducer>[1][]) =>
  events.reduce(reducer, initialState);

describe("connection", () => {
  it("tracks open/close and clears turn + arming on close", () => {
    const open = fold(
      { type: "connection_opened" },
      { type: "hello", client_id: "c1", section: null },
    );
    expect(open).toMatchObject({ connected: true, clientId: "c1" });
    const busy = fold(
      { type: "connection_opened" },
      { type: "armed", ok: true, holder: "c1" },
      { type: "turn_started" },
      { type: "connection_closed" },
    );
    expect(busy).toMatchObject({ connected: false, turnActive: false, armed: false, rms: 0 });
  });

  it("close keeps the anchor and cards — the sitting is server-side (§7)", () => {
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
    // The sitting survives a blip: reconnect resumes the same
    // conversation, anchor and approval window. Dropping them client-
    // side would deny a resume the server is honoring.
    expect(s.selected).toBe("sections/intro.tex");
    expect(s.pendingDiffs).toHaveLength(1);
    expect(s.messages.at(-1)).toMatchObject({
      role: "notice",
      text: "Connection lost — reconnecting; the sitting resumes",
    });
  });

  it("a recap rides the transcript as a notice (§7)", () => {
    const s = fold({
      type: "recap",
      text: "Previous session ended abruptly — resumed 2 verbatim turns.",
    });
    expect(s.messages.at(-1)).toMatchObject({
      role: "notice",
      text: "Previous session ended abruptly — resumed 2 verbatim turns.",
    });
  });

  it("session_ended clears the anchor, keeps the cards, notices the recap", () => {
    const s = fold(
      { type: "section_selected", section: "sections/intro.tex" },
      {
        type: "diff",
        diff_id: "d1",
        section: "sections/intro.tex",
        find: "a",
        replace: "b",
      },
      {
        type: "session_ended",
        reason: "idle",
        recap: "Last sitting ended (idle) on 2026-10-01: 3 turns — distilled.",
      },
    );
    // The anchor was sitting-level (gone with it); the diff is pending
    // on disk and its button rides disk truth, so the card stays.
    expect(s.selected).toBeNull();
    expect(s.pendingDiffs).toHaveLength(1);
    expect(s.messages.at(-1)?.text).toContain("Last sitting ended (idle)");
  });

  it("a switch drops the old project's cards — they can never resolve here", () => {
    const s = fold(
      {
        type: "diff",
        diff_id: "d1",
        section: "sections/intro.tex",
        find: "a",
        replace: "b",
      },
      {
        type: "session_ended",
        reason: "switch",
        recap: "Last sitting ended (switch) on 2026-10-01: 2 turns — distilled.",
      },
    );
    expect(s.pendingDiffs).toEqual([]);
  });

  it("hello carries the sitting's anchor — the server's answer is truth (§8)", () => {
    const stale = fold({ type: "section_selected", section: "sections/old.tex" });
    // A server restart kept the client's view state but not the sitting:
    // hello's null clears the stale anchor; a blip's hello re-sends it.
    const restarted = reducer(stale, { type: "hello", client_id: "c1", section: null });
    expect(restarted.selected).toBeNull();
    const resumed = reducer(stale, {
      type: "hello",
      client_id: "c1",
      section: "sections/intro.tex",
    });
    expect(resumed.selected).toBe("sections/intro.tex");
  });
});

describe("reopen re-presentation", () => {
  const card = {
    type: "diff" as const,
    diff_id: "d1",
    section: "sections/intro.tex",
    find: "old",
    replace: "new",
  };

  it("a re-presented diff never doubles a card already on screen", () => {
    const s = fold(card, card);
    expect(s.pendingDiffs).toHaveLength(1);
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

  it("user_text (a spoken utterance) joins as a user message", () => {
    // Voice has no optimistic add — the server echo is the only path,
    // and it carries the anchor the turn was logged with.
    const s = fold({ type: "user_text", text: "tighten it", section: "sections/intro.tex" });
    expect(s.messages).toEqual([
      { id: 1, role: "user", text: "tighten it", section: "sections/intro.tex" },
    ]);
  });

  it("local_notice surfaces a client-side failure as a notice", () => {
    // The refused-mic path never reaches the server; the reducer still
    // folds it like any other notice line.
    const s = fold({
      type: "local_notice",
      text: "Mic needs HTTPS (or localhost)",
    });
    expect(s.messages).toEqual([
      { id: 1, role: "notice", text: "Mic needs HTTPS (or localhost)" },
    ]);
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

describe("audio bookends", () => {
  it("fold to no-ops — playback is the player's, not state", () => {
    const s = fold(
      { type: "turn_started" },
      { type: "audio_start", sample_rate: 24000 },
      { type: "audio_end" },
      { type: "tts_stopped" },
    );
    // turnActive cleared on assistant_text, never on audio events
    // (audio_end never reaches non-holders; clearing here would wedge
    // "thinking…" for them).
    expect(s.turnActive).toBe(true);
    expect(s.messages).toHaveLength(0);
  });
});

it("an unknown event never wedges the shell", () => {
  const s = reducer(initialState, { type: "from_the_future" } as never);
  expect(s).toBe(initialState);
});
