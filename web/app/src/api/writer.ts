// The background writer (docs/background-writer.md): the door answers
// with liveness truth — running is the server's task, not a remembered
// flag, so a crashed pass can never leave the chip saying "writing".
import { json } from "./http";

export interface WriterStatus {
  running: boolean;
  section: string;
  brief: string;
}

export const getWriter = (): Promise<WriterStatus> =>
  fetch("/project/writer").then(json<WriterStatus>);
