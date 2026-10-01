// One WebSocket per tab (SPEC §1), thin over the reducer: connect with
// reconnect, parse server events into dispatch, and carry the 2 s
// heartbeat that feeds liveness, the meter and the lease watchdog (§8).
// A blip loses only the socket: the §7 sitting is server-side and the
// reconnect resumes it (hello re-sends the anchor, cards ride disk truth).
import { useEffect, useRef } from "react";
import { heartbeat } from "../protocol/frames";
import type { ControlFrame, ServerEvent, ShellEvent } from "../types";

export interface Connection {
  send: (frame: ControlFrame) => void;
  sendAudio: (chunk: ArrayBuffer) => void;
}

function clientId(): string {
  let id = sessionStorage.getItem("phd-client");
  if (!id) {
    id = `${Date.now()}-${Math.random().toString(36).slice(2, 8)}`;
    sessionStorage.setItem("phd-client", id);
  }
  return id;
}

export function useConnection(
  dispatch: (event: ShellEvent) => void,
  getRms: () => number,
): Connection {
  const wsRef = useRef<WebSocket | null>(null);
  const dispatchRef = useRef(dispatch);
  dispatchRef.current = dispatch;
  const rmsRef = useRef(getRms);
  rmsRef.current = getRms;

  useEffect(() => {
    let ws: WebSocket;
    let closedByUs = false;
    let retryTimer: number | undefined;

    const connect = () => {
      const proto = location.protocol === "https:" ? "wss" : "ws";
      ws = new WebSocket(`${proto}://${location.host}/ws/voice?client=${clientId()}`);
      ws.binaryType = "arraybuffer";
      wsRef.current = ws;
      ws.onopen = () => dispatchRef.current({ type: "connection_opened" });
      ws.onclose = () => {
        // A discarded socket's close (StrictMode double-mount, a
        // superseded retry) must not mark the live one closed.
        if (wsRef.current !== ws) return;
        dispatchRef.current({ type: "connection_closed" });
        if (!closedByUs) retryTimer = window.setTimeout(connect, 2000);
      };
      ws.onmessage = (ev) => {
        if (typeof ev.data === "string") {
          dispatchRef.current(JSON.parse(ev.data) as ServerEvent);
        }
      };
    };
    connect();

    const hb = window.setInterval(() => {
      if (wsRef.current?.readyState === WebSocket.OPEN) {
        wsRef.current.send(JSON.stringify(heartbeat(rmsRef.current())));
      }
    }, 2000);

    return () => {
      closedByUs = true;
      window.clearTimeout(retryTimer);
      window.clearInterval(hb);
      wsRef.current?.close();
    };
  }, []);

  return {
    send: (frame) => {
      if (wsRef.current?.readyState === WebSocket.OPEN) {
        wsRef.current.send(JSON.stringify(frame));
      } else {
        // §8: never a silent drop — the user hears why it didn't go.
        dispatchRef.current({
          type: "error",
          where: "connection",
          message: "not connected — message not sent",
        });
      }
    },
    sendAudio: (chunk) => {
      // Live mic audio is transient: dropping chunks during a blip is
      // the honest behavior, queueing stale PCM would be worse.
      if (wsRef.current?.readyState === WebSocket.OPEN) {
        wsRef.current.send(chunk);
      }
    },
  };
}
