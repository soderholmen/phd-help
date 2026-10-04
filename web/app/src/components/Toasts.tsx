// Toasts: the notice surface every view has. The transcript keeps the
// record (and its aria-live), but in Read/Source the transcript is
// unmounted — a notice nobody can see is §8's failure mode, so the
// last notice/error also floats, bottom-right, above every view.
// Dumb by house rule: App hands in the toast, the timer lives here,
// onDismiss goes out.
import { useEffect, useRef } from "react";

export interface Toast {
  id: number;
  kind: "notice" | "error";
  text: string;
}

const AUTO_DISMISS_MS = 6000;

export function Toasts({
  toast,
  onDismiss,
}: {
  toast: Toast | null;
  onDismiss: () => void;
}) {
  // The timer re-arms per toast, not per render: the 3 s tick
  // re-renders the shell, and a timer keyed on the callback's
  // identity would restart every tick and never fire. The callback
  // rides a ref for the same reason.
  const dismiss = useRef(onDismiss);
  dismiss.current = onDismiss;
  useEffect(() => {
    if (!toast) return;
    const t = window.setTimeout(() => dismiss.current(), AUTO_DISMISS_MS);
    return () => window.clearTimeout(t);
  }, [toast]);

  if (!toast) return null;
  return (
    <div className="toasts">
      <div className={`toast ${toast.kind}`}>
        <span
          className="toast-text"
          role={toast.kind === "error" ? "alert" : "status"}
        >
          {toast.text}
        </span>
        <button
          className="btn btn-ghost btn-sm"
          onClick={onDismiss}
          aria-label="Dismiss notice"
        >
          ✕
        </button>
      </div>
    </div>
  );
}
