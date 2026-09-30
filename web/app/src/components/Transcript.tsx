// The conversation: user/assistant turns, inline errors (§8), notices.
import type { Message } from "../protocol/types";

export function Transcript({ messages }: { messages: Message[] }) {
  return (
    <ul className="transcript" aria-live="polite">
      {messages.map((m) => (
        <li key={m.id} className={`msg ${m.role}`}>
          {m.section && <span className="anchor">{m.section}</span>}
          <span className="text">{m.text}</span>
        </li>
      ))}
    </ul>
  );
}
