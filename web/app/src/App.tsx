// The shell (§1/§3): section tree | transcript + diffs + composer |
// corpus, over one WebSocket per tab. All turn-taking logic lives in
// the reducer; this file only wires doors to it.
import { useEffect, useReducer, useRef, useState } from "react";
import { reducer, initialState } from "./protocol/reducer";
import { approve, arm, disarm, reject, selectSection, typed } from "./protocol/frames";
import { useConnection } from "./ws/useConnection";
import { MicCapture } from "./voice/capture";
import { fetchHealth, fetchTree } from "./api/sections";
import { listDocs, pin, retry, unpin, uploadPdf, type UploadMeta } from "./api/corpus";
import type { CorpusDoc, Health, SectionNode } from "./types";
import { Header } from "./components/Header";
import { SectionTree } from "./components/SectionTree";
import { Transcript } from "./components/Transcript";
import { Composer } from "./components/Composer";
import { DiffCard } from "./components/DiffCard";
import { CorpusPanel } from "./components/CorpusPanel";

export default function App() {
  const [state, dispatch] = useReducer(reducer, initialState);
  const micRef = useRef<MicCapture | null>(null);
  if (micRef.current === null) micRef.current = new MicCapture();
  const mic = micRef.current;
  const conn = useConnection(dispatch, () => mic.rms);

  const [tree, setTree] = useState<SectionNode[]>([]);
  const [docs, setDocs] = useState<CorpusDoc[]>([]);
  const [health, setHealth] = useState<Health | null>(null);

  // The tree changes only when the project does; health and corpus
  // status poll — indexing is async and must be visible (§6).
  useEffect(() => {
    fetchTree().then(setTree).catch(() => setTree([]));
  }, []);
  useEffect(() => {
    let stop = false;
    const tick = async () => {
      try {
        const h = await fetchHealth();
        if (!stop) setHealth(h);
      } catch {
        if (!stop) setHealth(null);
      }
      try {
        const d = await listDocs();
        if (!stop) setDocs(d);
      } catch {
        /* tray stays as-is; the next tick retries */
      }
    };
    void tick();
    const timer = window.setInterval(() => void tick(), 3000);
    return () => {
      stop = true;
      window.clearInterval(timer);
    };
  }, []);

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
  useEffect(() => {
    mic.onChunk = (pcm) => {
      if (state.armed) conn.sendAudio(pcm);
    };
  }, [mic, conn, state.armed]);

  const toggleVoice = async () => {
    if (!state.armed) {
      try {
        await mic.start(); // needs the gesture; denial never arms
      } catch {
        return;
      }
      conn.send(arm());
    } else {
      mic.suspend();
      conn.send(disarm());
    }
  };

  const send = (text: string) => {
    dispatch({ type: "user_send", text }); // optimistic: the echo is ours
    conn.send(typed(text));
  };

  return (
    <div className="shell">
      <Header
        connected={state.connected}
        armed={state.armed}
        rms={state.rms}
        health={health}
        onToggleVoice={() => void toggleVoice()}
      />
      <div className="panels">
        <aside className="left">
          <h2>Sections</h2>
          <SectionTree tree={tree} selected={state.selected} onSelect={(p) => conn.send(selectSection(p))} />
        </aside>
        <main className="center">
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
          <Composer onSend={send} />
        </main>
        <aside className="right">
          <CorpusPanel
            docs={docs}
            onUpload={(bytes, meta: UploadMeta) => void uploadPdf(bytes, meta).catch(() => {})}
            onRetry={(id) => void retry(id).catch(() => {})}
            onPin={(id) => void pin(id).catch(() => {})}
            onUnpin={(id) => void unpin(id).catch(() => {})}
          />
        </aside>
      </div>
    </div>
  );
}
