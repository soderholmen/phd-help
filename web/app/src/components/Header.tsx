// Voice arming (explicit, never automatic — glossary: Conversation mode),
// the input meter, and the health chip (§8: state is known, not discovered
// mid-turn).
import type { Health } from "../types";

interface Props {
  connected: boolean;
  armed: boolean;
  rms: number;
  health: Health | null;
  onToggleVoice: () => void;
}

export function Header({ connected, armed, rms, health, onToggleVoice }: Props) {
  return (
    <header className="bar">
      <button
        id="toggle"
        className={armed ? "on" : ""}
        onClick={onToggleVoice}
        aria-pressed={armed}
      >
        Voice: {armed ? "ON" : "OFF"}
      </button>
      <div className="meter" title="input level">
        <div style={{ width: `${Math.min(100, rms * 400)}%` }} />
      </div>
      <span className={`conn ${connected ? "up" : "down"}`}>
        {connected ? "connected" : "reconnecting…"}
      </span>
      {health && (
        <span className="health" title="component health (§8)">
          llm:{health.vllm} corpus:{health.corpus}
        </span>
      )}
    </header>
  );
}
