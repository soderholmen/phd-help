# Project & file doors (issue #28): what each door does, and where it bends

## The two kinds of write

The agent's write path is SPEC §5: propose -> lint -> pending diff ->
approve. The doors here are the user's own hand, not the agent's —
they write directly, with no approval card, because approving your own
file pick is theater. The one exception is `section_create`, which is
the agent speaking: it rides the §5 path exactly, with the new file's
bytes attached to the same approval the way a bib entry rides a cite
diff (the `bib_append` pattern). The diff card then shows "creates
sections/x.tex" plus the file's bytes above the `\input` wiring patch —
both halves land together on approve, or neither does.

## Voice: creating a part

"Start a new part on related work" -> `section_create(section, content,
after)` -> one card -> approve. `after` is the section to land behind
(its `\input` line is quoted from disk as the anchor); empty means
join the end of the document. The tool's `after` is schema-required
with empty-string-means-end semantics because OpenAI strict mode
forbids optional parameters.

## File doors (the active project)

- `GET /project/files` — every `.tex` except main.tex, marked
  `linked` (in the `\input` graph) or not, plus the trash listing.
- `POST /project/files?name=` — raw `.tex` bytes; lands at
  `sections/<name>.tex` and auto-`\input`s at the end (ratified).
- `POST /project/files/remove {path}` — strips the file's `\input`
  line from main.tex and moves the bytes to `.phd-helper/trash/`
  (mirroring the relative path; a second removal of the same path
  gets a timestamp suffix). The remove door IS the undo.
- `POST /project/files/restore {path}` — moves it back and re-wires.

Guards everywhere: `.tex` only, no traversal, never main.tex (the
spine) and refs.bib is not a listed target.

## Project doors

- `GET /projects` / `POST /projects/activate` — as before; switching
  ends the sitting (distillation runs against the OLD project) and the
  shell's `session_ended` fold clears the old project's cards.
- `POST /projects/new {name}` — scaffolds main.tex +
  sections/intro.tex + empty refs.bib, then activates.
- `POST /projects/import?name=` — a zip upload (a remote device cannot
  browse the server's disk): 25 MB cap, `.tex`/`.bib` members only,
  zip-slip guarded, main.tex required at the root or under one
  top-level folder (which is stripped). Validation runs before any
  extraction: a refused zip leaves no half-project.

## Git: every project a repo (commit + push when done)

Each project directory can become a git repo, driven by
`src/phd_helper/gitrepo.py` (pure parsers + one `run` seam, mineru's
shape). The `.gitignore` the kernel writes keeps `.phd-helper/` (agent
state) and LaTeX build dross (`*.aux/*.log/*.pdf`) out; a user's own
`.gitignore` survives — only the missing patterns are appended.

- `GET /project/git/status` — `{initialized, dirty, has_remote, last}`
  for the active project; **never 500s** (an uninitialized directory
  is a zero answer; a git fault arrives as an `error` string on a 200,
  so the chip rides the 3 s tick like health).
- `POST /project/git/commit {message?}` — lazy init (init on `main`,
  write `.gitignore`, initial commit) then commit everything the
  ignore allows; empty message auto-dates. `{"ok": true, "commit":
  sha|null}` — a clean tree is a success with `null`, not an error.
- `POST /project/git/push {}` — 400 `{"error": "no remote set"}`
  until a remote exists; **whatever is on disk is committed first**
  ("Update paper") — a push never silently ships stale state. Pushes
  `main` to the stored URL directly.
- `POST /project/git/remote {url}` — stores the push target in
  `.phd-helper/git.json` (not git config); 400 on anything that is
  not http(s)/ssh/scp-like/absolute-path.

Commit is also **automatic at sitting end** (before §7's no-turns
guard — uploads and hand edits dirty the tree without a turn), capped
at 20 s and swallowed: a git fault never blocks or fails the ending.
And by **voice**: `git_commit(message)` — the user's hand, #28
posture, no §5 card. **Push is never by voice**: it publishes
outward, so it stays an on-screen gesture (the ratified line).

## Honest deviations (documented, not hidden)

- **Direct writes by design.** The file doors skip §5's pending-diff
  gate — they are user instructions, not agent proposals.
- **Undo of a create orphans the file.** Undo rides the main.tex
  patch: the `\input` line goes, the file stays (invisible to the
  tree, removable via the file doors). Undo never deletes bytes the
  user may want.
- **Restore loses position.** A restored part rejoins at the end of
  main.tex, not its old slot; a timestamped trash entry restores
  under its timestamped name.
- **Import is zip-only.** No server-disk browsing, by design.
- **Uploads are not linted.** The §5 lint gates agent proposals;
  your own upload is your own bytes.
- **Git is lazy-init only.** `new`/`import` never touch git; the repo
  appears at the first commit or status. Existing project doors are
  byte-identical, and no door test spawns git.
- **A nested project shows dirt in an outer repo.** A project dir
  that is itself tracked inside another git repo (the shipped
  `sample_paper` inside phd-helper) shows `path (modified content)`
  in the outer `git status` — a nested repo's dirt is unfixable via
  `.gitignore`. Cosmetic; the outer repo is not phd-helper's problem.
- **Auto-commit needs a sitting.** No sitting ever opened (or a §8
  crash path) means no auto-commit — the button and the voice tool
  are always there.
- **Push needs a working credential path.** `GIT_TERMINAL_PROMPT=0`
  makes Git Credential Manager fail fast rather than hang a request
  on a GUI this server has; a box whose GCM cannot auth
  non-interactively (this one) sets `PHD_GIT_CREDENTIAL_HELPER` (the
  gh-helper knob) and push carries it as `-c`.
- **Identity is per-command.** `-c user.name/email` rides each
  commit; the server never writes global or system git config.
- **No branch/history UI.** Always `main`; no log view, no blame, no
  LFS. The chip shows the last commit's subject as a tooltip.

## Reopen: cards are disk truth

`revalidate_pending` runs on EVERY socket attach (not once per
sitting): it reconciles every on-disk pending diff through the
apply-time hash check / re-anchor path, rebuilds the sitting's live
list from the outcomes, and sends the cards to the attaching socket.
Bounces fan out to all tabs, so a phone reconnecting can neither miss
a card proposed while it was gone nor keep a ghost of one that
bounced.
