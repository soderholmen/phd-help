// The conversation: user/assistant turns, inline errors (§8), notices.
// A fenced draft (stepwise writing) renders as a pre block — the voice
// speaks the draft while it generates, the screen keeps its shape.
// The conversation reads bottom-up, so a new message pulls the view to
// the newest; an empty transcript says so instead of showing a void.
import { useEffect, useRef } from "react";
import type { Message } from "../protocol/types";

type Part = { kind: "text" | "draft"; body: string };

function parts(text: string): Part[] {
  const out: Part[] = [];
  const closed = /```[^\n]*\n([\s\S]*?)```/g;
  let last = 0;
  for (const m of text.matchAll(closed)) {
    if (m.index > last) out.push({ kind: "text", body: text.slice(last, m.index) });
    out.push({ kind: "draft", body: m[1].replace(/\n$/, "") });
    last = m.index + m[0].length;
  }
  const rest = text.slice(last);
  const open = rest.indexOf("```");
  if (open >= 0) {
    // streaming: the closing fence has not arrived — the draft still
    // reads as a draft, fence marker hidden
    if (open > 0) out.push({ kind: "text", body: rest.slice(0, open) });
    out.push({ kind: "draft", body: rest.slice(open).replace(/^```[^\n]*\n?/, "") });
  } else if (rest) {
    out.push({ kind: "text", body: rest });
  }
  return out;
}

export function Transcript({ messages }: { messages: Message[] }) {
  const ref = useRef<HTMLUListElement | null>(null);
  // Length, not content: token streaming would otherwise fight the
  // user's scroll on every token; the message landing is the event.
  useEffect(() => {
    const el = ref.current;
    if (el) el.scrollTop = el.scrollHeight;
  }, [messages.length]);
  return (
    <>
      {messages.length === 0 && (
        <p className="transcript-empty">
          nothing said yet — type below, or arm the voice
        </p>
      )}
      <ul ref={ref} className="transcript" aria-live="polite">
      {messages.map((m) => (
        <li key={m.id} className={`msg ${m.role}`}>
          {m.section && <span className="anchor">{m.section}</span>}
          {parts(m.text).map((p, i) =>
            p.kind === "draft" ? (
              <pre key={i} className="draft">{p.body}</pre>
            ) : (
              <span key={i} className="text">{p.body}</span>
            ),
          )}
        </li>
      ))}
      </ul>
    </>
  );
}
