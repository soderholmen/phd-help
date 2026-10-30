# Background writer

One conversation, two sessions: the talker keeps the voice and the
keyboard, the writer drafts paper text beside it. Ask out loud —
"draft the related-work paragraph about retrieval in the background" —
and the conversation stays live while the writer works; its patches
show up as ordinary approval cards when they land.

## The shape

- **Trigger.** `delegate_write(section, brief)` is offered to the main
  agent (SYSTEM_PROMPT rule 7): the talker delegates, answers the user,
  and its turn returns immediately. One writer at a time — a second
  delegation bounces to the model, which tells the user to wait.
- **The writer is a separate LLM session.** Its own system prompt
  (`WRITER_PROMPT`), its own message list, its own tool loop
  (`run_writer` in `server/app.py`). Nothing of it enters the
  conversation's chat JSONL (§7 verbatim log): the talk owns one log,
  the writer is a hired hand with a note, not a party to the talk.
- **Silent by design.** No voice (the audio episode belongs to the
  talker's turn), no notices narrating content — just two bookend
  notices ("drafting" / "done: N patches") and the cards.
- **Cards are the only output.** The writer gets
  `section_read / section_write / section_create / cite_add /
  web_search / corpus_search / corpus_doc` and nothing else. No
  `pending_decide` (it would rubber-stamp its own work), no
  `memory_write / git_commit / undo_last` (the user's hands), no
  `delegate_write` (no recursion). Its `window` is empty even as a
  second belt. Every proposal rides the §5 approval gate exactly like
  the talker's.
- **The approval window is shared.** A card the writer lands joins the
  sitting's `pending_diffs`, so the *next thing you say to the talker*
  interprets "apply it" against the writer's proposal too — voice
  approval works across the two sessions for free (the note reads disk
  truth every turn). With no live sitting (a switch, a between-sittings
  landing), the diff stays disk truth: the next attach's reconcile
  re-presents it, same as any other pending diff (§7/§8).
- **Status.** `GET /project/writer` answers `{running, section, brief}`
  — running truth is the task object, never a stored flag, so a crashed
  writer cannot leave the header chip pulsing. The chip rides the 3 s
  tick with its title = the brief (a delegation spoken on the phone
  lights the chip on the laptop).

## Concurrency honesty

- The writer is a plain background task (`spawn`, like gists, ingest,
  and the librarian pass), guarded one-at-a-time like the librarian.
  It is not a §7 sitting: a sitting switch or idle-end does not kill a
  drafting pass, and a drafting pass does not keep a sitting alive.
- The project is captured at delegation (like the librarian's and the
  turn's): a switch mid-pass mutes the writer's notices and cards
  (project-identity guard) but its patch still lands on the paper it
  was drafted for.
- Write races go through the machinery that already exists: the writer
  proposes, never writes; apply-time re-anchor + hash checks bounce a
  card whose section moved under it — the same clobber protection that
  covers hand edits and the talker's own diffs.
- Both sessions share the one vLLM endpoint. A drafting pass and a
  spoken turn decode concurrently — on the shared production endpoint
  they share its throughput, so replies are a little slower while the
  chip pulses. That is the honest cost of the feature; there is no
  second GPU to pretend otherwise.

## Deliberate limits (slice 1)

- Existing sections only: the validator bounces a `delegate_write`
  whose section is missing (new files are the writer's own
  `section_create`, or the talker's).
- One writer per server, not per section; the queue door is a bounce,
  not a queue.
- The loop cap is 6 tool rounds — a rambling writer ends with the
  honest "done: N patches" count, not an endless pass.
- SPEC §5's approval rules are untouched; §5's and §4's single-agent
  wording now reads as the *talker*, with the writer as an extension
  this doc documents (SPEC amendment is the owner's ratification
  door, per the #24/#26 precedent).
