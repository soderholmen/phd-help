// The project doors (issue #28): list/switch, scaffold a new paper,
// import one as a zip (a remote device cannot browse the server's
// disk), and the git doors: every project a repo — status/commit/push,
// with the remote URL stored once per project.
import { json, throwIfError } from "./http";

export interface ProjectList {
  projects: string[];
  active: string;
}

export const listProjects = (): Promise<ProjectList> =>
  fetch("/projects").then(json<ProjectList>);

export const activateProject = (name: string): Promise<{ active: string }> =>
  fetch("/projects/activate", {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify({ name }),
  }).then(json<{ active: string }>);

export const newProject = (name: string): Promise<{ active: string }> =>
  fetch("/projects/new", {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify({ name }),
  }).then(json<{ active: string }>);

export function importProject(
  name: string,
  bytes: ArrayBuffer,
): Promise<{ active: string }> {
  const q = new URLSearchParams({ name });
  return fetch(`/projects/import?${q}`, { method: "POST", body: bytes }).then(
    json<{ active: string }>,
  );
}

// Download is the one door that answers with bytes, not JSON: the shared
// error discipline (throwIfError) first, then a blob click-through. The
// zip is the project's .tex/.bib files — the import door's own
// allowlist, so it round-trips.
export async function downloadProject(name: string): Promise<void> {
  const q = new URLSearchParams({ name });
  const r = await fetch(`/projects/download?${q}`);
  await throwIfError(r);
  const url = URL.createObjectURL(await r.blob());
  const a = document.createElement("a");
  a.href = url;
  a.download = `${name}.zip`;
  a.click();
  URL.revokeObjectURL(url);
}

// The git doors. Status never 500s — a git fault arrives as an `error`
// string on a 200, so the chip rides the 3 s tick like health does.
export interface GitStatus {
  initialized: boolean;
  dirty: boolean;
  has_remote: boolean;
  last: { sha: string; date: string; subject: string } | null;
  error?: string;
}

const post = (path: string, body?: unknown) =>
  fetch(path, {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: body === undefined ? "{}" : JSON.stringify(body),
  });

export const gitStatus = (): Promise<GitStatus> =>
  fetch("/project/git/status").then(json<GitStatus>);

export const gitCommit = (
  message: string,
): Promise<{ ok: boolean; commit: string | null }> =>
  post("/project/git/commit", { message }).then(
    json<{ ok: boolean; commit: string | null }>,
  );

export const gitPush = (): Promise<{ ok: boolean }> =>
  post("/project/git/push").then(json<{ ok: boolean }>);

export const gitSetRemote = (url: string): Promise<{ ok: boolean }> =>
  post("/project/git/remote", { url }).then(json<{ ok: boolean }>);
