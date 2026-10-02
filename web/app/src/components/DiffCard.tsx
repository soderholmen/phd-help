// A pending anchored patch (§5): raw find/replace — diffs are never
// rendered, approving a render artifact is the failure mode. Approval
// is on-screen; voice may initiate, never confirm (§3). A create card
// (#28) shows the new file's bytes above the wiring patch it rides.
import { useState } from "react";
import type { DiffCard as DiffCardData } from "../protocol/types";

interface Props {
  card: DiffCardData;
  onApprove: (card: DiffCardData) => void;
  onReject: (card: DiffCardData) => void;
}

export function DiffCard({ card, onApprove, onReject }: Props) {
  // One decision per card: a double-click's second frame would hit an
  // already-consumed pending diff and bounce a confusing notice.
  const [decided, setDecided] = useState(false);
  return (
    <li className="diff" data-diff-id={card.diff_id}>
      <header>
        <code>{card.section}</code>
        {card.created && (
          <span className="creates">creates {card.created.path}</span>
        )}
      </header>
      {card.created && (
        <pre className="creates-body">
          <ins>{card.created.content}</ins>
        </pre>
      )}
      <pre className="find">
        <del>{card.find}</del>
      </pre>
      <pre className="replace">
        <ins>{card.replace}</ins>
      </pre>
      <div className="acts">
        <button
          disabled={decided}
          onClick={() => {
            setDecided(true);
            onApprove(card);
          }}
        >
          Apply
        </button>
        <button
          disabled={decided}
          onClick={() => {
            setDecided(true);
            onReject(card);
          }}
        >
          Discard
        </button>
      </div>
    </li>
  );
}
