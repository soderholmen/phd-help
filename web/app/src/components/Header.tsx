// Voice arming (explicit, never automatic — glossary: Conversation mode),
// the input meter, the health chip (§8: state is known, not discovered
// mid-turn), the project doors (#28): switch by name, scaffold a new
// paper, import one as a zip, and the git doors: Commit (a prompt like
// New's) and Push (an on-screen gesture — it publishes outward, so
// voice never owns it; the remote URL is pasted once per project).
// Switching ends the sitting server-side; the session_ended event
// clears the old project's cards.
import { useRef } from "react";
import type { Health } from "../types";
import type { GitStatus } from "../api/projects";

interface Props {
  connected: boolean;
  armed: boolean;
  rms: number;
  health: Health | null;
  projects: string[];
  active: string;
  gitStatus: GitStatus | null;
  onToggleVoice: () => void;
  onActivate: (name: string) => void;
  onNew: (name: string) => void;
  onImport: (name: string, bytes: ArrayBuffer) => void;
  onDownload: (name: string) => void;
  onCommit: (message: string) => void;
  onPush: (remoteUrl?: string) => void;
}

export function Header({
  connected,
  armed,
  rms,
  health,
  projects,
  active,
  gitStatus,
  onToggleVoice,
  onActivate,
  onNew,
  onImport,
  onDownload,
  onCommit,
  onPush,
}: Props) {
  const zipRef = useRef<HTMLInputElement | null>(null);

  const pickZip = async (f: File) => {
    // The zip's own name is the honest default for the project name.
    const name = window.prompt("Import project as", f.name.replace(/\.zip$/i, ""));
    if (!name) return;
    onImport(name.trim(), await f.arrayBuffer());
  };

  const commit = () => {
    // Cancel (null) asks for nothing; the empty string is the
    // door's auto-date convention, same as the voice tool's.
    const m = window.prompt("Commit message (empty to auto-date)");
    if (m === null) return;
    onCommit(m.trim());
  };

  const push = () => {
    if (gitStatus && !gitStatus.has_remote) {
      const url = window.prompt("Remote URL (https://…, git@host:…, or a path)");
      if (!url?.trim()) return;
      onPush(url.trim()); // the App stores it, then pushes
      return;
    }
    onPush();
  };

  // The chip is the repo's state of mind, same discipline as health:
  // known, not discovered mid-save. A git fault shows as "git?", not
  // a dead poll. The class rides the fact (`dirty`), never the glyph.
  const chip = !gitStatus
    ? null
    : gitStatus.error
      ? "git?"
      : !gitStatus.initialized
        ? "no repo"
        : gitStatus.dirty
          ? "● dirty"
          : "clean";
  const dirty =
    !!gitStatus &&
    !gitStatus.error &&
    gitStatus.initialized &&
    gitStatus.dirty;

  return (
    <header className="bar">
      <button
        id="toggle"
        className={armed ? "on" : ""}
        onClick={onToggleVoice}
        aria-pressed={armed}
      >
        Voice: {armed ? "ON" : "OFF"}
      </button>
      <div className="meter" title="input level">
        <div style={{ width: `${Math.min(100, rms * 400)}%` }} />
      </div>
      <span className={`conn ${connected ? "up" : "down"}`}>
        {connected ? "connected" : "reconnecting…"}
      </span>
      {health && (
        <span className="health" title="component health (§8)">
          llm:{health.vllm} corpus:{health.corpus}
        </span>
      )}
      <span className="projects">
        <select
          aria-label="Project"
          value={active}
          onChange={(e) => onActivate(e.target.value)}
        >
          {projects.map((p) => (
            <option key={p} value={p}>
              {p}
            </option>
          ))}
        </select>
        <button
          onClick={() => {
            const name = window.prompt("New project name");
            if (name?.trim()) onNew(name.trim());
          }}
        >
          New
        </button>
        <button onClick={() => zipRef.current?.click()}>Import</button>
        <input
          ref={zipRef}
          type="file"
          accept=".zip"
          hidden
          aria-label="Project zip"
          onChange={(e) => {
            const f = e.target.files?.[0];
            e.target.value = ""; // re-picking the same zip must re-fire
            if (f) void pickZip(f);
          }}
        />
        {/* Download any project without switching to it — the select
            can't double as the picker, its onChange activates. */}
        <details className="download">
          <summary>Download</summary>
          <ul>
            {projects.map((p) => (
              <li key={p}>
                <button onClick={() => onDownload(p)}>{p}</button>
              </li>
            ))}
          </ul>
        </details>
        {chip && (
          <span
            className="git"
            title={
              gitStatus?.error ||
              (gitStatus?.last
                ? `${gitStatus.last.sha}: ${gitStatus.last.subject}`
                : "no commits yet")
            }
          >
            <span className={`git-chip ${dirty ? "dirty" : ""}`}>
              {chip}
            </span>
            {gitStatus?.initialized && !gitStatus.error && (
              <span
                className="git-remote"
                title={
                  gitStatus.has_remote
                    ? "remote set"
                    : "no remote — Push will ask for one"
                }
              >
                {gitStatus.has_remote ? "↗" : "+remote"}
              </span>
            )}
          </span>
        )}
        <button onClick={commit}>Commit</button>
        <button onClick={push}>Push</button>
      </span>
    </header>
  );
}
