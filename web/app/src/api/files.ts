// The file doors (issue #28) for the active project: list (with the
// linked/unlinked mark), upload (auto-\input at the end), soft-remove
// into the server's trash, restore — and the edit doors: content (the
// source editor's load) and patch (both editors' save). These are the
// user's own hand — direct writes, no approval round-trip.
import type { FilePatch } from "../types";
import { json } from "./http";

export interface ProjectFile {
  path: string;
  linked: boolean;
}

export interface FileList {
  files: ProjectFile[];
  trash: string[];
}

export interface FileContent {
  path: string;
  text: string;
  hash: string;
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

export const fetchFileContent = (path: string): Promise<FileContent> =>
  fetch(`/project/files/content?${new URLSearchParams({ path })}`).then(
    json<FileContent>,
  );

// A stale base hash answers 409 ("changed since you opened it") — the
// caller keeps the draft open and lets the user reload, never clobbers.
export const patchFile = (p: FilePatch): Promise<{ path: string }> =>
  post("/project/files/patch", p).then(json<{ path: string }>);
