// The wire contract with server/app.py — events in, control frames out.
// Keep in lockstep with dispatch() and run_turn() there.

// Server → client events.
export type ServerEvent =
  | { type: "hello"; client_id: string; section: string | null }
  // What the mic heard, fanned to every tab. Typed text is NOT echoed —
  // the typing tab adds its own optimistically (LocalEvent user_send).
  | { type: "user_text"; text: string; section: string }
  | { type: "turn_started" }
  | { type: "assistant_text"; text: string }
  | {
      type: "diff";
      diff_id: string;
      section: string;
      find: string;
      replace: string;
      // section_create (#28): a new file's bytes ride the same approval;
      // the find/replace above is then the \input wiring into main.tex.
      created?: { path: string; content: string };
    }
  | {
      type: "diff_resolved";
      diff_id: string;
      applied: boolean;
      reason: string | null;
      text: string | null;
    }
  | { type: "turn_interrupted" }
  | { type: "error"; where: string; message: string }
  | { type: "pong"; rms: number }
  | { type: "armed"; ok: boolean; holder: string | null }
  | { type: "disarmed" }
  | { type: "audio_start"; sample_rate: number }
  | { type: "audio_end" }
  | { type: "tts_stopped" }
  | { type: "section_selected"; section: string | null }
  | { type: "recap"; text: string }
  | { type: "session_ended"; reason: string; recap: string };

// Client → server control frames (voice.py CONTROL_TYPES).
export type ControlFrame =
  | { type: "heartbeat"; rms: number }
  | { type: "arm" }
  | { type: "disarm" }
  | { type: "typed"; text: string }
  | { type: "barge_in" }
  | { type: "select_section"; section: string }
  | { type: "approve"; section: string; diff_id: string }
  | { type: "reject"; section: string; diff_id: string };

// Client-internal events the reducer also folds.
export type LocalEvent =
  | { type: "connection_opened" }
  | { type: "connection_closed" }
  | { type: "user_send"; text: string }
  | { type: "local_notice"; text: string };

export type ShellEvent = ServerEvent | LocalEvent;

// The section tree from GET /sections (sections.py SectionNode).
export interface SectionNode {
  path: string;
  title: string | null;
  children: SectionNode[];
}

// A corpus doc from GET /corpus/docs.
export interface CorpusDoc {
  doc_id: string;
  title: string;
  status: string;
  error: string | null;
  arxiv: string;
  doi: string;
  year: string;
  source: string;
  pinned_in: string[];
  pinned_here: boolean;
  chunk_count: number;
}

// /health shape.
export interface Health {
  vllm: string;
  stt: string;
  tts: string;
  corpus: string;
  endpoint_holder: string | null;
}
