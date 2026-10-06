// The related-work door (SPEC §6, issue #29): the panel reads what the
// agent's searches recorded; Find triggers the agent's own librarian
// pass. The steer box is not a search box — its keywords and focus
// seed the agent's planning, and the agent still runs every search.
import type { RelatedList } from "../types";
import { json } from "./http";

export const getRelated = (): Promise<RelatedList> =>
  fetch("/project/related").then(json<RelatedList>);

// The door answers immediately; the pass runs in the background and
// reports through the notice event. A second click while one runs is a
// server-side no-op, so the button needs no client-side guard. The
// steer may be empty — the pass then plans from the paper alone.
export const findRelated = (
  keywords: string[],
  focus: string,
): Promise<{ searching: boolean }> =>
  fetch("/project/related/find", {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify({ keywords, focus }),
  }).then(json<{ searching: boolean }>);
