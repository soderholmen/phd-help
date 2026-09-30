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
        aria-label="message"
        value={text}
        placeholder="type, or arm voice and talk…"
        onChange={(e) => setText(e.target.value)}
      />
      <button type="submit">Send</button>
    </form>
  );
}
