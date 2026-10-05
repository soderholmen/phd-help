// The related-work door (SPEC §6, issue #29): read-only in slice 1 — the
// agent's searches record themselves; the UI still runs no search of
// its own. Find (slice 2) triggers the agent's pass, never a query box.
import type { RelatedList } from "../types";
import { json } from "./http";

export const getRelated = (): Promise<RelatedList> =>
  fetch("/project/related").then(json<RelatedList>);
