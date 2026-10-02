// The file doors (issue #28) for the active project: upload a .tex (the
// server auto-\inputs it at the end), remove (a soft delete into the
// server's trash — the door IS the undo), and restore. These are the
// user's own hand, not the agent's: direct writes, no approval card.
// Presentational — App owns the polling and the API calls.
import { useState } from "react";
import type { ProjectFile } from "../api/files";

interface Props {
  files: ProjectFile[];
  trash: string[];
  onUpload: (name: string, bytes: ArrayBuffer) => Promise<unknown>;
  onRemove: (path: string) => void;
  onRestore: (path: string) => void;
}

export function FilesPanel({ files, trash, onUpload, onRemove, onRestore }: Props) {
  const [file, setFile] = useState<File | null>(null);
  const [error, setError] = useState("");
  const [busy, setBusy] = useState(false);

  // CorpusPanel's discipline: the pick clears only once the door
  // accepts; a failed upload keeps it and shows why.
  const submit = async () => {
    if (!file) return;
    setBusy(true);
    setError("");
    try {
      await onUpload(file.name, await file.arrayBuffer());
      setFile(null);
    } catch (e) {
      setError(e instanceof Error ? e.message : "upload failed");
    } finally {
      setBusy(false);
    }
  };

  return (
    <section className="files" aria-label="Files">
      <h2>Files</h2>
      <form
        className="upload"
        onSubmit={(e) => {
          e.preventDefault();
          void submit();
        }}
      >
        <input
          type="file"
          accept=".tex"
          aria-label="TeX file"
          onChange={(e) => setFile(e.target.files?.[0] ?? null)}
        />
        <button type="submit" disabled={!file || busy}>
          {busy ? "Uploading…" : "Upload"}
        </button>
      </form>
      {error && <div className="err">{error}</div>}
      <ul>
        {files.length === 0 && <li className="empty">no files yet</li>}
        {files.map((f) => (
          <li key={f.path} className="file">
            <span className="ttl">{f.path}</span>
            {!f.linked && <span className="meta"> (not in the document)</span>}
            <span className="acts">
              <button onClick={() => onRemove(f.path)}>Remove</button>
            </span>
          </li>
        ))}
      </ul>
      {trash.length > 0 && (
        <ul className="trash" aria-label="Trash">
          {trash.map((t) => (
            <li key={t} className="file">
              <span className="meta">{t}</span>
              <span className="acts">
                <button onClick={() => onRestore(t)}>Restore</button>
              </span>
            </li>
          ))}
        </ul>
      )}
    </section>
  );
}
