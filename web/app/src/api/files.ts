// The file doors (issue #28) for the active project: list (with the
// linked/unlinked mark), upload (auto-\input at the end), soft-remove
// into the server's trash, and restore. These are the user's own hand —
// direct writes, no approval round-trip.
import { json } from "./http";

export interface ProjectFile {
  path: string;
  linked: boolean;
}

export interface FileList {
  files: ProjectFile[];
  trash: string[];
}

export const listFiles = (): Promise<FileList> =>
  fetch("/project/files").then(json<FileList>);

export function uploadFile(
  name: string,
  bytes: ArrayBuffer,
): Promise<{ path: string }> {
  const q = new URLSearchParams({ name });
  return fetch(`/project/files?${q}`, { method: "POST", body: bytes }).then(
    json<{ path: string }>,
  );
}

const post = (url: string, body: unknown) =>
  fetch(url, {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify(body),
  });

export const removeFile = (path: string): Promise<{ removed: string }> =>
  post("/project/files/remove", { path }).then(json<{ removed: string }>);

export const restoreFile = (path: string): Promise<{ restored: string }> =>
  post("/project/files/restore", { path }).then(json<{ restored: string }>);
