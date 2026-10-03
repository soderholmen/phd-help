// The read-view door: the active paper as readable blocks, converted
// server-side by the pure readable.py kernel (SPEC §Section view,
// simple variant).
import { json } from "./http";
import type { ReadDocument } from "../types";

export const fetchDocument = (): Promise<ReadDocument> =>
  fetch("/document").then(json<ReadDocument>);
