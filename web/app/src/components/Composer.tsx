// Typed input — the door that works while STT is stubbed.
import { useState } from "react";

export function Composer({ onSend }: { onSend: (text: string) => void }) {
  const [text, setText] = useState("");
  const submit = () => {
    const t = text.trim();
    if (!t) return;
    onSend(t);
    setText("");
  };
  return (
    <form
      className="composer"
      onSubmit={(e) => {
        e.preventDefault();
        submit();
      }}
    >
      <input
        className="input"
        aria-label="message"
        value={text}
        placeholder="type, or arm voice and talk…"
        onChange={(e) => setText(e.target.value)}
      />
      {/* Send waits for text: the disabled state says "nothing to
          send" before the click, instead of a silent no-op after. */}
      <button className="btn btn-primary" type="submit" disabled={!text.trim()}>
        Send
      </button>
    </form>
  );
}
