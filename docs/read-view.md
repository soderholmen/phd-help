# Read view (simple variant) + project download

## What shipped

SPEC §Section view specifies a rendered, read-only Read view with a
client-side LaTeX→HTML renderer and KaTeX. This slice is the **simple
server-side variant** the user chose instead: a pure-Python converter
(`readable.py`, unit-tested like the other pure kernels) and
`GET /document` returning JSON blocks — no new dependencies in either
package, and the door stays open for the full SPEC view later (a KaTeX
renderer would consume the same block JSON).

Honest deviations from SPEC §Section view, stated not hidden:

- **Server-side conversion, not client-side** (SPEC:145 "one client-side
  renderer, no server render endpoint"). The blocks cross the wire as
  data; the client maps them to elements with no parsing logic.
- **Math is raw, not KaTeX.** Display math lands in monospace `pre`
  blocks exactly as written; inline math stays inline and raw.
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

## Verify

```powershell
.venv/Scripts/python.exe -m pytest tests/test_readable.py tests/test_app.py -q
# live: curl -sk https://127.0.0.1:8777/document
#       curl -sk -o p.zip "https://127.0.0.1:8777/projects/download?name=<p>"
```
