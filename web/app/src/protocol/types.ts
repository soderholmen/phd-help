// Shell state shapes the reducer folds (wire types live in ../types).

export interface Message {
  id: number;
  role: "user" | "assistant" | "error" | "notice";
  text: string;
  section?: string; // the anchor the exchange was made under
}

export interface DiffCard {
  diff_id: string;
  section: string;
  find: string;
  replace: string;
  // section_create (#28): the new file the approval lands alongside the
  // wiring patch.
  created?: { path: string; content: string };
}

export interface ShellState {
  connected: boolean;
  clientId: string | null;
  turnActive: boolean;
  messages: Message[];
  pendingDiffs: DiffCard[];
  selected: string | null;
  armed: boolean;
  rms: number;
  // The live partial: words forming mid-speech, holder-only on the
  // wire. Transient by contract — user_partial replaces, user_text and
  // disarmed clear; it never joins messages (the final is the record).
  partial: string;
  nextId: number;
}
