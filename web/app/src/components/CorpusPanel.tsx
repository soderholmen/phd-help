// The corpus surface (SPEC §6, issue #21): upload door, per-PDF status
// (failures visible, not rot), retry, and the pin toggle. Presentational
// — App owns the polling and the API calls.
import { useState } from "react";
import type { CorpusDoc } from "../types";
import type { UploadMeta } from "../api/corpus";

interface Props {
  docs: CorpusDoc[];
  onUpload: (bytes: ArrayBuffer, meta: UploadMeta) => Promise<unknown>;
  onRetry: (docId: string) => void;
  onPin: (docId: string) => void;
  onUnpin: (docId: string) => void;
}

export function CorpusPanel({ docs, onUpload, onRetry, onPin, onUnpin }: Props) {
  const [file, setFile] = useState<File | null>(null);
  const [title, setTitle] = useState("");
  const [arxiv, setArxiv] = useState("");
  const [error, setError] = useState("");
  const [busy, setBusy] = useState(false);

  // The selection clears only once the door accepts — a failed upload
  // keeps the user's pick and shows why (the harness did this).
  const submit = async () => {
    if (!file) return;
    setBusy(true);
    setError("");
    try {
      await onUpload(await file.arrayBuffer(), { title: title.trim(), arxiv: arxiv.trim() });
      setFile(null);
      setTitle("");
      setArxiv("");
    } catch (e) {
      setError(e instanceof Error ? e.message : "upload failed");
    } finally {
      setBusy(false);
    }
  };

  return (
    <section className="corpus" aria-label="Corpus">
      <h2>Corpus</h2>
      <form
        className="upload"
        onSubmit={(e) => {
          e.preventDefault();
          void submit();
        }}
      >
        <input
          type="file"
          accept="application/pdf"
          aria-label="PDF file"
          onChange={(e) => setFile(e.target.files?.[0] ?? null)}
        />
        <input
          className="meta"
          placeholder="title (optional)"
          value={title}
          onChange={(e) => setTitle(e.target.value)}
        />
        <input
          className="meta"
          placeholder="arXiv id (optional)"
          value={arxiv}
          onChange={(e) => setArxiv(e.target.value)}
        />
        <button type="submit" disabled={!file || busy}>
          {busy ? "Uploading…" : "Upload"}
        </button>
      </form>
      {error && <div className="err">{error}</div>}
      <ul>
        {docs.length === 0 && <li className="empty">no papers yet</li>}
        {docs.map((d) => (
          <li key={d.doc_id} className="doc">
            <div className="row">
              <span className={`st ${d.status}`}>{d.status}</span>
              <span className="ttl">
                {d.title || d.doc_id}
                {d.year ? ` (${d.year})` : ""}
              </span>
              <span className="acts">
                {d.status === "failed" && (
                  <button onClick={() => onRetry(d.doc_id)}>Retry</button>
                )}
                {d.pinned_here ? (
                  <button onClick={() => onUnpin(d.doc_id)}>Unpin</button>
                ) : (
                  <button onClick={() => onPin(d.doc_id)}>Pin</button>
                )}
              </span>
            </div>
            {d.status === "failed" && d.error && <div className="err">{d.error}</div>}
            {d.status === "indexed" && (
              <div className="meta">{d.chunk_count} chunks</div>
            )}
          </li>
        ))}
      </ul>
    </section>
  );
}
