// The corpus doors (SPEC §6): upload is door 1, status is read-only —
// the UI never searches, the agent does.
import type { CorpusDoc } from "../types";
import { json } from "./http";

export const listDocs = (): Promise<CorpusDoc[]> =>
  fetch("/corpus/docs").then(json<CorpusDoc[]>);

export interface UploadMeta {
  title?: string;
  arxiv?: string;
  doi?: string;
  year?: string;
}

export function uploadPdf(
  bytes: ArrayBuffer,
  meta: UploadMeta = {},
): Promise<{ doc_id: string; status: string; new: boolean }> {
  const q = new URLSearchParams();
  for (const [k, v] of Object.entries(meta)) if (v) q.set(k, v);
  const query = q.toString(); // .size is too new; truthiness is universal
  const suffix = query ? `?${query}` : "";
  return fetch(`/corpus/upload${suffix}`, { method: "POST", body: bytes }).then(
    json<{ doc_id: string; status: string; new: boolean }>,
  );
}

export const retry = (docId: string): Promise<{ doc_id: string; status: string }> =>
  fetch(`/corpus/${docId}/retry`, { method: "POST" }).then(
    json<{ doc_id: string; status: string }>,
  );

export const pin = (docId: string): Promise<{ doc_id: string; pinned_in: string[] }> =>
  fetch(`/corpus/${docId}/pin`, { method: "POST" }).then(
    json<{ doc_id: string; pinned_in: string[] }>,
  );

export const unpin = (docId: string): Promise<{ doc_id: string; pinned_in: string[] }> =>
  fetch(`/corpus/${docId}/unpin`, { method: "POST" }).then(
    json<{ doc_id: string; pinned_in: string[] }>,
  );
