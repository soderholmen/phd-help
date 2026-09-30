// A pending anchored patch (§5): raw find/replace — diffs are never
// rendered, approving a render artifact is the failure mode. Approval
// is on-screen; voice may initiate, never confirm (§3).
import type { DiffCard as DiffCardData } from "../protocol/types";

interface Props {
  card: DiffCardData;
  onApprove: (card: DiffCardData) => void;
  onReject: (card: DiffCardData) => void;
}

export function DiffCard({ card, onApprove, onReject }: Props) {
  return (
    <li className="diff" data-diff-id={card.diff_id}>
      <header>
        <code>{card.section}</code>
      </header>
      <pre className="find">
        <del>{card.find}</del>
      </pre>
      <pre className="replace">
        <ins>{card.replace}</ins>
      </pre>
      <div className="acts">
        <button onClick={() => onApprove(card)}>Apply</button>
        <button onClick={() => onReject(card)}>Discard</button>
      </div>
    </li>
  );
}
