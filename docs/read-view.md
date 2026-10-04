# Read view (simple variant) + project download

## What shipped

SPEC §Section view specifies a rendered, read-only Read view with a
client-side LaTeX→HTML renderer and KaTeX. This slice is the **simple
server-side variant** the user chose instead: a pure-Python converter
(`readable.py`, unit-tested like the other pure kernels) and
`GET /document` returning JSON blocks. The door the variant left open
was later walked through: the client renders math with KaTeX over the
same block JSON (see the Math bullet).

Honest deviations from SPEC §Section view, stated not hidden:

- **Server-side conversion, not client-side** (SPEC:145 "one client-side
  renderer, no server render endpoint"). The blocks cross the wire as
  data; the client maps them to elements with no parsing logic.
- **Math renders with KaTeX; the raw source stays behind it.** Display
  math gets a rendered view before its monospace `pre` (CSS hides the
  pre only when a view precedes it); unparseable math has no view, so
  its source stays visible — never-drop again, locally. Inline `$…$`
  in paragraphs renders as KaTeX spans; the kernel keeps a literal
  dollar escaped as `\$` so the client can tell the two apart
  (paragraphs carrying `$` are prose-uneditable either way).
- **`\cite` → `[Surname, year]`** via the deterministic refs.bib lookup
  (SPEC's rule, kept); unknown keys show the key. `\ref`/`\eqref` and
  unknown macros stay raw (SPEC's raw list, kept).

The never-drop contract (SPEC:141) is the kernel's property under test:
anything unparseable — unknown environment, stray macro, unterminated
env — passes through as a `raw` block, visible. What is deliberately not
shown: the preamble (setup, not content; `\title` excepted) and a
figure's `\includegraphics` line (the image is not a project file; its
caption is the text). Tables stay raw: their rows are content.

## Doors

- `GET /document` — the active project as
  `{sections: [{path, title, blocks}]}`, root first, in **true document
  order**: depth-first over the section tree (`skeleton._graph`'s BFS
  order would place a nested child after its parent's next sibling —
  not where LaTeX puts it). Same posture as `/sections`: no params.
- `GET /projects/download?name=X` — the import door in reverse: the
  project's `.tex`/`.bib` files as a zip, exactly import's member
  allowlist, so **an exported project re-imports unchanged** (tested
  round-trip). `.phd-helper/` is agent state, not the paper; it stays
  out. `Content-Disposition` filename is sanitized to `[A-Za-z0-9._-]`.

## Shell

Talk/Read is a local view toggle in the center column (the transcript
keeps living underneath; Composer stays). `/document` rides the 3 s tick
only while reading, so the paper stays fresh as the agent writes.
Download is a `Download ▾` dropdown in the header listing every project
— grab any project without activating it (the project `select` can't
double as the picker: its onChange activates).

## Editing the paper

The read view is not write-only. Every block carries its source span in
the file (`start`/`end`), the hash of that region (`base`), and an
`editable` verdict — so the paper can be edited from the prose side
without ever round-tripping the lossy projection.

- **Click-to-edit prose.** A block the kernel marked `editable` (its
  source region is pure prose — no macro, cite, math, escape or comment)
  opens an inline textarea on click; Save patches exactly its span.
  Headings and captions span only their **brace argument**, so editing a
  heading replaces the title, not the `\section{…}`. Everything else —
  citations, lists, math, raw — shows an `edit source` link instead;
  inventing a safe prose patch for a `\cite` region would destroy
  markup, so the source editor is the honest surface there.
- **Source editor.** Per-section (and per-block) `Edit source` opens the
  raw `.tex` in a textarea; Save is the same patch door over the whole
  file. The editor owns its text — staleness surfaces as the 409 below,
  never as a silent overwrite.
- **Offset truth.** The kernel's preprocessing (comments, preamble,
  `\input` lines, `\end{document}` tail) **blanks in place** instead of
  deleting, so scan indices are file indices and spans need no mapping
  table.

The doors: `GET /project/files/content?path=X` → `{path, text, hash}`;
`POST /project/files/patch` `{path, start, end, base, text}` splices
`text` over `[start, end)` — one door for both editors. If the region no
longer hashes to `base` (the agent landed a diff since the fetch), the
save **bounces 409** — "changed since you opened it — reload" — and the
open draft stays open. That base-hash check is the whole conflict story;
there is no collaborative locking.

Known edge, stated: spans are Python **code-point** indices, and the
browser speaks UTF-16 — a non-BMP character (emoji, rare CJK) in a
`.tex` shifts every later span for the client. The failure mode is
honest (a 400 on bounds or a 409 on base), never a corrupted splice;
BMP text — including å/ä/ö — is unaffected.

Ratified posture (issue #28): the user's own hand writes **directly** —
no §5 approval card, no lint gate, same as uploads. Undo rides the
agent's history: every patch records a `record_apply` snapshot, so
`undo_last` reverts a hand edit exactly like an applied diff — and the
header's Undo button and the `undo_last` voice tool ride the same
history (see the undo door, `POST /project/undo`).

## Verify

```powershell
.venv/Scripts/python.exe -m pytest tests/test_readable.py tests/test_app.py -q
# live: curl -sk https://127.0.0.1:8777/document
#       curl -sk -o p.zip "https://127.0.0.1:8777/projects/download?name=<p>"
#       curl -sk "https://127.0.0.1:8777/project/files/content?path=main.tex"
#       curl -sk -X POST https://127.0.0.1:8777/project/files/patch \
#         -d '{"path":"main.tex","start":0,"end":7,"base":"<hash>","text":"\\title{X}"}'
```
