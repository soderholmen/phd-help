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

  // One 3 s poll for tree + health + corpus status: indexing is async
  // and must be visible (§6), and the agent can restructure the paper
  // (a new \input) so the tree is not fetch-once. A backgrounded tab
  // stays quiet and catches up on focus (§3's visibility discipline);
  // the three doors are independent, so they fan out.
  useEffect(() => {
    let stop = false;
    const tick = async () => {
      const [t, h, d] = await Promise.allSettled([fetchTree(), fetchHealth(), listDocs()]);
      if (stop) return;
      if (t.status === "fulfilled") setTree(t.value);
      if (h.status === "fulfilled") setHealth(h.value);
      else setHealth(null);
      if (d.status === "fulfilled") setDocs(d.value);
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
            onUpload={(bytes, meta: UploadMeta) => uploadPdf(bytes, meta)}
            onRetry={(id) => void retry(id).catch(() => {})}
            onPin={(id) => void pin(id).catch(() => {})}
            onUnpin={(id) => void unpin(id).catch(() => {})}
          />
        </aside>
      </div>
    </div>
  );
}
