// The shell's deterministic core: every WS event and local action folds
// into state here, as a pure function. The WS hook and components stay
// thin because all the turn-taking, diff and anchor logic lives here.
import type { ShellEvent } from "../types";
import type { DiffCard, Message, ShellState } from "./types";

export const initialState: ShellState = {
  connected: false,
  clientId: null,
  turnActive: false,
  messages: [],
  pendingDiffs: [],
  selected: null,
  armed: false,
  rms: 0,
  nextId: 1,
};

function withMessage(
  state: ShellState,
  role: Message["role"],
  text: string,
  section?: string,
): ShellState {
  const msg: Message = section
    ? { id: state.nextId, role, text, section }
    : { id: state.nextId, role, text };
  return { ...state, messages: [...state.messages, msg], nextId: state.nextId + 1 };
}

export function reducer(state: ShellState, event: ShellEvent): ShellState {
  switch (event.type) {
    case "connection_opened":
      return { ...state, connected: true };
    case "connection_closed":
      // A dropped socket ends the turn and the arming optimistically;
      // reconnect resumes (§8 connection blips are the server's story).
      return { ...state, connected: false, turnActive: false, armed: false, rms: 0 };
    case "hello":
      return { ...state, clientId: event.client_id };
    case "user_send":
      return withMessage(state, "user", event.text, state.selected ?? undefined);
    case "turn_started":
      return { ...state, turnActive: true };
    case "assistant_text":
      return withMessage({ ...state, turnActive: false }, "assistant", event.text);
    case "diff": {
      const card: DiffCard = {
        diff_id: event.diff_id,
        section: event.section,
        find: event.find,
        replace: event.replace,
      };
      return { ...state, pendingDiffs: [...state.pendingDiffs, card] };
    }
    case "diff_resolved": {
      const card = state.pendingDiffs.find((d) => d.diff_id === event.diff_id);
      const without = state.pendingDiffs.filter((d) => d.diff_id !== event.diff_id);
      const notice = event.applied
        ? `Applied to ${card?.section ?? "?"}`
        : `Not applied (${event.reason ?? "rejected"})`;
      return withMessage({ ...state, pendingDiffs: without }, "notice", notice);
    }
    case "turn_interrupted":
      return withMessage({ ...state, turnActive: false }, "notice", "Turn interrupted");
    case "error":
      // §8: errors land inline in the chat, never a silent spinner.
      return withMessage(
        { ...state, turnActive: false },
        "error",
        `${event.where}: ${event.message}`,
      );
    case "pong":
      return { ...state, rms: event.rms };
    case "armed":
      if (!event.ok) {
        // Another device holds the endpoint (glossary: voice endpoint).
        return withMessage(
          { ...state, armed: false },
          "notice",
          `Voice endpoint held by ${event.holder ?? "another device"}`,
        );
      }
      return { ...state, armed: true };
    case "disarmed":
      return { ...state, armed: false };
    case "tts_stopped":
      return state; // audio playback lands with the 3090 stack
    case "section_selected":
      return { ...state, selected: event.section };
    default:
      return state; // unknown events must never wedge the shell
  }
}
