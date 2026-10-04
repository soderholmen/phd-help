// The raw-LaTeX source editor: the honest surface for everything the
// read view can't patch safely (lists, math, citations, macros). One
// textarea over the whole file; Save is the same patch door as inline
// prose editing, just start 0 / end len. The editor owns its text —
// no tick gating — and staleness surfaces as the 409 on save, which
// keeps the editor open so nothing typed is lost.
import { useState } from "react";

interface Props {
  path: string;
  text: string;
  onSave: (draft: string) => Promise<void>;
  onCancel: () => void;
}

export function SourceEditor({ path, text, onSave, onCancel }: Props) {
  const [draft, setDraft] = useState(text);
  const [saving, setSaving] = useState(false);

  const save = async () => {
    setSaving(true);
    try {
      await onSave(draft);
    } catch {
      /* the notice rode the transcript; the draft stays open */
    } finally {
      setSaving(false);
    }
  };

  return (
    <div className="source-editor" aria-label={`Source editor for ${path}`}>
      <header className="sec-head">
        <span className="src-path">{path}</span>
        <span className="edit-acts">
          <button onClick={() => void save()} disabled={saving}>
            Save
          </button>
          <button onClick={onCancel} disabled={saving}>
            Discard
          </button>
        </span>
      </header>
      <textarea
        value={draft}
        onChange={(e) => setDraft(e.target.value)}
        spellCheck={false}
        aria-label={`LaTeX source for ${path}`}
      />
    </div>
  );
}
