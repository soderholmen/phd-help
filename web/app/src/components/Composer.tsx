// Typed input — the door that works while STT is stubbed. The ghost
// overlay (live partials) rides on top: words forming while the
// sentence is still spoken.
import { useState } from "react";

export function Composer({
  onSend,
  partial = "",
}: {
  onSend: (text: string) => void;
  partial?: string;
}) {
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
      <div className="composer-field">
        <input
          className="input"
          aria-label="message"
          value={text}
          placeholder="type, or arm voice and talk…"
          onChange={(e) => setText(e.target.value)}
        />
        {/* An overlay, not the placeholder: the placeholder is the
            resting hint and must survive; the ghost is transient and
            aria-hidden — the final transcript is the accessible
            record, this is a liveness cue for the holder's eyes only
            (it never leaves their socket). It yields to typed text —
            the ghost is the one that steps aside. */}
        {partial && !text ? (
          <div className="ghost" aria-hidden="true">
            {partial}
          </div>
        ) : null}
      </div>
      {/* Send waits for text: the disabled state says "nothing to
          send" before the click, instead of a silent no-op after. */}
      <button className="btn btn-primary" type="submit" disabled={!text.trim()}>
        Send
      </button>
    </form>
  );
}
