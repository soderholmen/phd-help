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
  partial: "",
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
      // §7: the sitting is server-side and survives a blip — the next
      // connect resumes the same conversation, anchor and approval
      // window. So the client keeps claiming them; only the socket is
      // gone. (If the server itself died, the next hello+recap tells it.)
      return withMessage(
        {
          ...state,
          connected: false,
          turnActive: false,
          armed: false,
          rms: 0,
          partial: "",
        },
        "notice",
        "Connection lost — reconnecting; the sitting resumes",
      );
    case "recap":
      // §7: the one-line on-screen recap of how the last sitting ended.
      return withMessage(state, "notice", event.text);
    case "session_ended":
      // The sitting ended (switch/shutdown/idle). The anchor was
      // sitting-level, so it clears; the diff cards stay — the diffs are
      // pending on disk and the buttons ride disk truth (§5). Except on
      // a project switch: those cards belong to the old project's disk
      // and can never resolve here, so they'd only be noise.
      return withMessage(
        {
          ...state,
          selected: null,
          pendingDiffs:
            event.reason === "switch" ? [] : state.pendingDiffs,
        },
        "notice",
        event.recap,
      );
    case "hello":
      // The server's hello carries the sitting's anchor: the client's
      // copy is view state a server restart can have invalidated, so
      // the refetch overwrites it (§8 resync).
      return { ...state, clientId: event.client_id, selected: event.section };
    case "user_send":
      return withMessage(state, "user", event.text, state.selected ?? undefined);
    case "local_notice":
      // §8 honesty for client-side failures: a refused mic never reaches
      // the server, so the reason has to come back as a transcript line.
      return withMessage(state, "notice", event.text);
    case "user_partial":
      // Ghost text, holder-only on the wire: each partial REPLACES the
      // last (it is the whole utterance-so-far, not a delta), and it
      // never joins messages — the final transcript is the record.
      return { ...state, partial: event.text };
    case "user_text":
      // A spoken utterance joins the transcript like a typed one; the
      // server carries the anchor it was logged with (§7 tagging). The
      // ghost it was forming into has now landed — it clears with it.
      return withMessage(
        { ...state, partial: "" },
        "user",
        event.text,
        event.section || undefined,
      );
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
        ...(event.created ? { created: event.created } : {}),
      };
      // A reconnect re-presents the sitting's pending diffs (§7 reopen);
      // a card already on screen must not double.
      if (state.pendingDiffs.some((d) => d.diff_id === event.diff_id)) {
        return state;
      }
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
      // The mic stops mid-utterance: no close reaches the server, so
      // this is the only clear left for a ghost that is still showing.
      return { ...state, armed: false, partial: "" };
    case "audio_start":
    case "audio_end":
    case "tts_stopped":
      // Playback is the PcmPlayer ref-singleton's business (App wires
      // it to the socket router), never reducer state: audio must not
      // re-render the shell, and non-holders never see these events.
      return state;
    case "section_selected":
      return { ...state, selected: event.section };
    default:
      return state; // unknown events must never wedge the shell
  }
}
