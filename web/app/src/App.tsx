// The shell (§1/§3): section tree | transcript + diffs + composer |
// corpus, over one WebSocket per tab. All turn-taking logic lives in
// the reducer; this file only wires doors to it.
import { useEffect, useReducer, useRef, useState } from "react";
import { reducer, initialState } from "./protocol/reducer";
import { approve, arm, bargeIn, disarm, reject, selectSection, typed } from "./protocol/frames";
import { useConnection } from "./ws/useConnection";
import { MicCapture } from "./voice/capture";
import { PcmPlayer } from "./voice/playback";
import { BARGE_CHUNKS, BARGE_RMS, BargeGate, RESUME_WINDOW_MS } from "./voice/barge";
import { fetchHealth, fetchTree } from "./api/sections";
import { listDocs, pin, retry, unpin, uploadPdf, type UploadMeta } from "./api/corpus";
import {
  activateProject,
  downloadProject,
  gitCommit,
  gitPush,
  gitSetRemote,
  gitStatus,
  importProject,
  listProjects,
  newProject,
  undoLast,
  type GitStatus,
} from "./api/projects";
import {
  fetchFileContent,
  listFiles,
  patchFile,
  removeFile,
  restoreFile,
  uploadFile,
  type ProjectFile,
} from "./api/files";
import { fetchDocument } from "./api/document";
import { findRelated, getRelated } from "./api/related";
import type {
  CorpusDoc,
  DocSection,
  FilePatch,
  Health,
  RelatedEntry,
  RelatedPass,
  SectionNode,
} from "./types";
import { Header } from "./components/Header";
import { SectionTree } from "./components/SectionTree";
import { Transcript } from "./components/Transcript";
import { Composer } from "./components/Composer";
import { DiffCard } from "./components/DiffCard";
import { CorpusPanel } from "./components/CorpusPanel";
import { RelatedPanel } from "./components/RelatedPanel";
import { FilesPanel } from "./components/FilesPanel";
import { ReadView } from "./components/ReadView";
import { SourceEditor } from "./components/SourceEditor";
import { Toasts, type Toast } from "./components/Toasts";
import { useTheme } from "./useTheme";

export default function App() {
  const [state, dispatch] = useReducer(reducer, initialState);
  const micRef = useRef<MicCapture | null>(null);
  if (micRef.current === null) micRef.current = new MicCapture();
  const mic = micRef.current;
  // Voice-out (§3): the player is a ref-singleton like the mic — audio
  // must never re-render the shell. The AudioContext starts suspended
  // (autoplay policy); the arm gesture unlocks it, and audio only ever
  // flows to an armed holder anyway.
  const playerRef = useRef<PcmPlayer | null>(null);
  if (playerRef.current === null) playerRef.current = new PcmPlayer(new AudioContext());
  const player = playerRef.current;
  const conn = useConnection(dispatch, () => mic.rms, player);
  const connRef = useRef(conn);
  connRef.current = conn;
  // Barge-in (§3): sustained AEC'd mic while playing → pause the
  // playhead, tell the server (it only acts on a thinking turn), and
  // arm the resume window. A final transcript opens a new turn whose
  // turn_started stops the player — which also makes a pending resume
  // inert (stop() clears the pause), so the window needs no cancel.
  const resumeTimer = useRef<number | undefined>(undefined);
  const gateRef = useRef<BargeGate | null>(null);
  if (gateRef.current === null) {
    gateRef.current = new BargeGate({
      threshold: BARGE_RMS,
      chunks: BARGE_CHUNKS,
      speaking: () => playerRef.current!.isSpeaking,
      onFire: () => {
        playerRef.current!.pause();
        connRef.current.send(bargeIn());
        window.clearTimeout(resumeTimer.current);
        resumeTimer.current = window.setTimeout(
          () => playerRef.current!.resume(),
          RESUME_WINDOW_MS,
        );
      },
    });
  }

  const [tree, setTree] = useState<SectionNode[]>([]);
  const [docs, setDocs] = useState<CorpusDoc[]>([]);
  const [health, setHealth] = useState<Health | null>(null);
  const [projects, setProjects] = useState<string[]>([]);
  const [active, setActive] = useState("");
  const [files, setFiles] = useState<ProjectFile[]>([]);
  const [trash, setTrash] = useState<string[]>([]);
  // The git chip rides the same 3 s tick as health: local `git status`
  // is milliseconds, and the state must be known, not discovered at
  // the Commit click (§8).
  const [git, setGit] = useState<GitStatus | null>(null);
  // Related work (SPEC §6, issue #29): the agent's searches record
  // themselves server-side; the panel just reads them on the same tick.
  // `searching` mirrors the door's one-at-a-time guard, never a local
  // guess — the server owns whether a pass is running.
  const [related, setRelated] = useState<RelatedEntry[]>([]);
  const [relatedSearching, setRelatedSearching] = useState(false);
  // What the last pass was asked for (issue #29) — the door's stored
  // truth, not a local echo: a pass started elsewhere still shows.
  const [relatedPass, setRelatedPass] = useState<RelatedPass | null>(null);
  // The read view (SPEC §Section view): Talk/Read is a local view toggle
  // — the transcript keeps living underneath. /document rides the tick
  // only while reading, so the paper stays fresh as the agent writes.
  const [reading, setReading] = useState(false);
  const [doc, setDoc] = useState<DocSection[]>([]);
  const readingRef = useRef(false);
  readingRef.current = reading;
  // The source editor (issue: edit the paper): the loaded file plus
  // the hash its bytes had at load — the patch door's clobber check.
  // The editor owns its text; staleness surfaces as the 409 on save.
  const [editing, setEditing] = useState<{
    path: string;
    text: string;
    base: string;
  } | null>(null);
  const [theme, toggleTheme] = useTheme();
  // The toast: the newest notice/error floats above EVERY view — in
  // Read/Source the transcript is unmounted, and a notice nobody can
  // see is §8's failure mode. The transcript keeps the record (and
  // its aria-live); the ref guard means one toast per message, not
  // one per render.
  const [toast, setToast] = useState<Toast | null>(null);
  const toastId = useRef(-1);
  useEffect(() => {
    for (let i = state.messages.length - 1; i >= 0; i--) {
      const m = state.messages[i];
      if (m.role === "notice" || m.role === "error") {
        if (m.id !== toastId.current) {
          toastId.current = m.id;
          setToast({ id: m.id, kind: m.role, text: m.text });
        }
        break;
      }
    }
  }, [state.messages]);

  // One 3 s poll for tree + health + corpus status + projects + files +
  // related work:
  // indexing is async and must be visible (§6), and the agent can
  // restructure the paper (a new \input) so the tree is not fetch-once.
  // A backgrounded tab stays quiet and catches up on focus (§3's
  // visibility discipline); the doors are independent, so they fan out.
  useEffect(() => {
    let stop = false;
    const tick = async () => {
      const [t, h, d, p, f, g, doc, rel] = await Promise.allSettled([
        fetchTree(),
        fetchHealth(),
        listDocs(),
        listProjects(),
        listFiles(),
        gitStatus(),
        readingRef.current ? fetchDocument() : Promise.resolve(null),
        getRelated(),
      ]);
      if (stop) return;
      if (t.status === "fulfilled") setTree(t.value);
      if (h.status === "fulfilled") setHealth(h.value);
      else setHealth(null);
      if (d.status === "fulfilled") setDocs(d.value);
      if (p.status === "fulfilled") {
        setProjects(p.value.projects);
        setActive(p.value.active);
      }
      if (f.status === "fulfilled") {
        setFiles(f.value.files);
        setTrash(f.value.trash);
      }
      if (g.status === "fulfilled") setGit(g.value);
      if (doc.status === "fulfilled" && doc.value) setDoc(doc.value.sections);
      if (rel.status === "fulfilled") {
        setRelated(rel.value.entries);
        setRelatedSearching(rel.value.searching);
        setRelatedPass(rel.value.last_pass);
      }
    };
    const go = () => {
      if (!document.hidden) void tick();
    };
    go();
    const timer = window.setInterval(go, 3000);
    document.addEventListener("visibilitychange", go);
    return () => {
      stop = true;
      window.clearInterval(timer);
      document.removeEventListener("visibilitychange", go);
    };
  }, []);

  // Opening the read view fetches at once (the 3 s tick keeps it fresh
  // only while open); a failure leaves the last document on screen.
  useEffect(() => {
    if (!reading) return;
    let stop = false;
    void fetchDocument()
      .then((d) => !stop && setDoc(d.sections))
      .catch(() => {});
    return () => {
      stop = true;
    };
  }, [reading]);

  // §3: capture pauses with the tab; the toggle state survives.
  useEffect(() => {
    const onVis = () => {
      if (document.hidden) mic.suspend();
      else if (state.armed) void mic.start();
    };
    document.addEventListener("visibilitychange", onVis);
    return () => document.removeEventListener("visibilitychange", onVis);
  }, [mic, state.armed]);

  // PCM rides the socket only while armed (arming is explicit, §3).
  // The level feed is unconditional: the gate itself is closed unless
  // audio is playing, and capture only runs while armed anyway.
  useEffect(() => {
    mic.onChunk = (pcm) => {
      if (state.armed) conn.sendAudio(pcm);
    };
    mic.onLevel = (rms) => gateRef.current!.level(rms);
  }, [mic, conn, state.armed]);

  const toggleVoice = async () => {
    if (!state.armed) {
      try {
        await mic.start(); // needs the gesture; denial never arms
      } catch (e) {
        // §8: the button must not look dead. Off-localhost plain HTTP
        // is not a secure context — the browser refuses getUserMedia
        // before the server is ever involved.
        const name = e instanceof DOMException ? e.name : "";
        const why =
          name === "NotAllowedError"
            ? "Mic permission denied — arm the mic in browser settings to go voice"
            : name === "NotFoundError"
              ? "No microphone found on this device"
              : !navigator.mediaDevices?.getUserMedia
                ? "Mic needs HTTPS (or localhost): this page is not a secure context — use typed chat, or open https://…"
                : `Mic unavailable: ${name || "unknown error"}`;
        dispatch({ type: "local_notice", text: why });
        return;
      }
      player.unlock(); // same gesture: the reply's AudioContext autoplay
      conn.send(arm());
    } else {
      mic.suspend();
      player.stop(); // the off switch means silence, mid-reply included
      conn.send(disarm());
    }
  };

  const send = (text: string) => {
    dispatch({ type: "user_send", text }); // optimistic: the echo is ours
    conn.send(typed(text));
  };

  // Project and file doors (#28). Switching ends the sitting
  // server-side — the session_ended event clears the old project's
  // cards, and the 3 s tick refreshes tree/docs/files. Failures ride
  // the transcript like every other §8 notice (FilesPanel keeps its
  // own error line for uploads, CorpusPanel's discipline).
  const notice =
    (label: string) =>
    (e: unknown) =>
      dispatch({
        type: "local_notice",
        text: `${label}: ${e instanceof Error ? e.message : "failed"}`,
      });

  // Cite from a related card is a typed turn (§5): the words go through
  // the same approval flow as if the user had spoken them — the panel
  // itself never writes to the paper.
  const citeText = (e: RelatedEntry, section: string | null) =>
    `Cite "${e.title}"${
      e.arxiv ? ` (arXiv ${e.arxiv})` : e.doi ? ` (DOI ${e.doi})` : ""
    } in ${section ?? "the right place"}.`;

  const onActivate = (name: string) =>
    void activateProject(name)
      .then((r) => setActive(r.active))
      .catch(notice(`Switch to ${name}`));
  const onNew = (name: string) =>
    void newProject(name)
      .then((r) => setActive(r.active))
      .catch(notice(`New project ${name}`));
  const onImport = (name: string, bytes: ArrayBuffer) =>
    void importProject(name, bytes)
      .then((r) => setActive(r.active))
      .catch(notice(`Import ${name}`));
  const onRemove = (path: string) =>
    void removeFile(path).catch(notice(`Remove ${path}`));
  const onRestore = (path: string) =>
    void restoreFile(path).catch(notice(`Restore ${path}`));
  const onDownload = (name: string) =>
    void downloadProject(name).catch(notice(`Download ${name}`));

  // The git doors: failures ride the transcript like every other
  // notice; the chip refreshes after any write so it tells the truth
  // the moment the door answers.
  const refreshGit = () => void gitStatus().then(setGit).catch(() => {});
  const onCommit = (message: string) =>
    void gitCommit(message).then(refreshGit).catch(notice("Commit"));
  const onPush = async (remoteUrl?: string) => {
    try {
      // The first push pastes the remote (Header's prompt): store it
      // before pushing, or the door still sees "no remote set".
      if (remoteUrl) await gitSetRemote(remoteUrl);
      await gitPush();
      refreshGit();
    } catch (e) {
      notice(remoteUrl ? "Set remote" : "Push")(e);
    }
  };

  // Undo (SPEC:137): the door names the section it restored, and the
  // toast says exactly that — honest about what went back (refs.bib
  // is not reverted; the door's response is the truth we tell).
  // No document_changed event: refreshDoc now + the 3 s tick is the
  // freshness story, so the read view can lag a voice-side undo ≤3 s.
  const onUndo = () =>
    void undoLast()
      .then((r) => {
        dispatch({
          type: "local_notice",
          text: r.applied
            ? `Undid the last change in ${r.section}`
            : r.reason || "nothing to undo",
        });
        if (r.applied) {
          refreshDoc();
          refreshGit(); // the tree moved: the chip tells the truth now
        }
      })
      .catch(notice("Undo"));

  // The edit doors: the user's own hand writes directly (ratified in
  // #28 — no approval card, no lint gate; the base hash is the conflict
  // story). A failed patch rethrows after the notice so the open draft
  // stays open — a 409 must never look like a saved edit.
  const refreshDoc = () =>
    void fetchDocument()
      .then((d) => setDoc(d.sections))
      .catch(() => {});
  const onEditSource = (path: string) =>
    void fetchFileContent(path)
      .then((f) => setEditing({ path: f.path, text: f.text, base: f.hash }))
      .catch(notice(`Open ${path}`));
  const patch = (p: FilePatch, label: string) =>
    patchFile(p).catch((e: unknown) => {
      notice(label)(e);
      return Promise.reject(e); // the caller's draft stays open
    });
  const onPatch = (p: FilePatch) =>
    patch(p, `Edit ${p.path}`).then(refreshDoc);
  const onSaveSource = (draft: string) => {
    if (!editing) return Promise.resolve();
    return patch(
      {
        path: editing.path,
        start: 0,
        end: editing.text.length,
        base: editing.base,
        text: draft,
      },
      `Save ${editing.path}`,
    ).then(() => {
      setEditing(null);
      refreshDoc();
    });
  };

  return (
    <div className="shell">
      <Header
        connected={state.connected}
        armed={state.armed}
        rms={state.rms}
        health={health}
        projects={projects}
        active={active}
        gitStatus={git}
        theme={theme}
        onToggleVoice={() => void toggleVoice()}
        onActivate={onActivate}
        onNew={onNew}
        onImport={onImport}
        onDownload={onDownload}
        onCommit={onCommit}
        onPush={(url) => void onPush(url)}
        onUndo={onUndo}
        onToggleTheme={toggleTheme}
      />
      <div className="panels">
        <aside className="left">
          <h2>Sections</h2>
          <SectionTree tree={tree} selected={state.selected} onSelect={(p) => conn.send(selectSection(p))} />
        </aside>
        <main className="center">
          <div className="view-toggle">
            <button
              onClick={() => {
                setReading(false);
                setEditing(null);
              }}
              aria-pressed={!reading}
            >
              Talk
            </button>
            <button
              onClick={() => {
                setReading(true);
                setEditing(null);
              }}
              aria-pressed={reading}
            >
              Read
            </button>
          </div>
          {editing ? (
            <SourceEditor
              path={editing.path}
              text={editing.text}
              onSave={onSaveSource}
              onCancel={() => setEditing(null)}
            />
          ) : reading ? (
            <ReadView
              sections={doc}
              onEditSource={onEditSource}
              onPatch={onPatch}
            />
          ) : (
            <>
              <Transcript messages={state.messages} />
              {state.turnActive && <div className="thinking">thinking…</div>}
              <ul className="diffs">
                {state.pendingDiffs.map((c) => (
                  <DiffCard
                    key={c.diff_id}
                    card={c}
                    onApprove={(card) => conn.send(approve(card.section, card.diff_id))}
                    onReject={(card) => conn.send(reject(card.section, card.diff_id))}
                  />
                ))}
              </ul>
            </>
          )}
          <Composer onSend={send} partial={state.partial} />
        </main>
        <aside className="right">
          <CorpusPanel
            docs={docs}
            onUpload={(bytes, meta: UploadMeta) => uploadPdf(bytes, meta)}
            onRetry={(id) => void retry(id).catch(() => {})}
            onPin={(id) => void pin(id).catch(() => {})}
            onUnpin={(id) => void unpin(id).catch(() => {})}
          />
          <RelatedPanel
            entries={related}
            searching={relatedSearching}
            onPin={(id) => void pin(id).catch(() => {})}
            onUnpin={(id) => void unpin(id).catch(() => {})}
            onCite={(e) => send(citeText(e, state.selected))}
            lastPass={relatedPass}
            // The one search trigger (issue #29): the door answers at
            // once and the pass reports through the notice event; the
            // optimistic flag is confirmed by the next tick. The steer
            // box's keywords and focus ride along — they seed the
            // agent's planning, the agent still runs every search.
            onFind={(keywords, focus) => {
              setRelatedSearching(true);
              void findRelated(keywords, focus).catch(notice("Find papers"));
            }}
          />
          <FilesPanel
            files={files}
            trash={trash}
            onUpload={(name, bytes) => uploadFile(name, bytes)}
            onRemove={onRemove}
            onRestore={onRestore}
          />
        </aside>
      </div>
      <Toasts toast={toast} onDismiss={() => setToast(null)} />
    </div>
  );
}
