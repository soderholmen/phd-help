"""The read view's kernel: LaTeX -> readable blocks (SPEC §Section view,
simple server-side variant).

Pure over a dict of project files, like sections.py: the caller supplies
path -> content, this module never touches the disk. The output is
JSON-ready blocks — heading/paragraph/list/math/caption/raw — assembled
in TRUE document order: depth-first over the section tree (a BFS walk
would place a nested child after its parent's next sibling, which is not
where LaTeX puts it).

Every block carries its source span in the file ("start"/"end"), the
hash of that region ("base") and an "editable" flag — the edit door's
seam. Spans are honest because preprocessing BLANKS IN PLACE (comments,
preamble, \\input lines become same-length whitespace, newlines kept):
scan indices stay file indices, no mapping table. "editable" means the
region is pure prose — its projection differs from its source only by
whitespace collapse — so writing edited text back over the span cannot
destroy a macro, cite or comment. Everything else is source-edit only.

The never-drop contract (SPEC:141): anything the kernel does not
understand — an unknown environment, \\ref, an unknown macro — passes
through as source, visible. What is deliberately NOT shown: the
preamble (setup, not content; \\title excepted) and a figure's
\\includegraphics line (the image is not a project file; its caption is
the text). Tables stay raw: their rows are content, and raw rows beat
dropped ones. Inline math stays inline in the paragraph text (the
client renders it with KaTeX); a literal dollar rides as \$ so the
client can tell the two apart.

for_speech (saytext.py) is the voice's lossy paraphrase and is NOT
reusable here: it replaces math and citations with placeholder words and
flattens every newline. The reading view keeps the content.
"""

import re

from phd_helper.bibtex import parse_bib
from phd_helper.patches import section_hash
from phd_helper.sections import parse_section_tree

_COMMENT = re.compile(r"(?<!\\)%.*")
_INPUT = re.compile(r"^[ \t]*\\(?:input|include)\{[^{}]*\}[ \t]*\n?",
                    re.M)
_TITLE = re.compile(r"\\title\s*\{")
_HEADING = (r"title|part|section|subsection|subsubsection|paragraph"
            r"|subparagraph")
_LEVEL = {"title": 0, "part": 0, "section": 1, "subsection": 2,
          "subsubsection": 3, "paragraph": 3, "subparagraph": 3}
_TOKEN = re.compile(
    r"\\begin\*?\{([^{}]*)\}"                      # 1: environment
    r"|((?<!\\)\\\[.+?(?<!\\)\\\])"                # 2: display \[..\]
    r"|((?<!\\)\$\$.+?(?<!\\)\$\$)"                # 3: display $$..$$
    r"|\\(" + _HEADING + r")\*?\s*\{",             # 4: heading
    re.DOTALL)
_ENV_MARK = re.compile(r"\\(begin|end)\*?\{([^{}]*)\}")
_CITE = re.compile(
    r"\\(?:cite|citep|citet|citealp|autocite|parencite|textcite)"
    r"\*?(?:\[[^\]]*\])*\s*\{([^{}]*)\}")  # nocite stays raw: invisible in
                                          # LaTeX, so no label to invent
_WRAPPER = re.compile(r"\\(?:textbf|emph|textit|texttt|textsf|underline"
                      r"|textsc|text)\*?\{([^{}]*)\}")
_LISTS = {"itemize": False, "enumerate": True}
_MATH_ENVS = ("equation", "align", "multline", "gather", "eqnarray")
# A region holding any of these is not pure prose: its projection
# changed something beyond whitespace, so prose cannot be written back
# over it without destroying markup.
_EDIT = re.compile(r"[\\$%&#_{}~]")


def read_document(files: dict[str, str], root: str,
                  bib: str = "") -> list[dict]:
    """[{path, title, blocks}] — the root first, then every linked file
    depth-first in document order. A file missing from `files` reads as
    no blocks (the caller's OSError degrade, mirrored)."""
    labels = _labels(bib)
    if root not in files:
        return []        # unreadable spine: nothing to read (the degrade)
    doc = [{"path": root, "title": None,
            "blocks": _to_blocks(files[root], labels)}]

    def walk(nodes):
        for n in nodes:
            doc.append({"path": n.path, "title": n.title,
                        "blocks": _to_blocks(files.get(n.path, ""),
                                             labels)})
            walk(n.children)

    walk(parse_section_tree(files, root))
    return doc


def to_blocks(text: str, bib: str = "") -> list[dict]:
    """One file's content as blocks (the root file's preamble rules
    apply when it carries \\begin{document})."""
    return _to_blocks(text, _labels(bib))


def _blank(m: re.Match) -> str:
    """Same-length whitespace, newlines kept: erasing content IN PLACE
    keeps the scan's indices equal to the file's, so every block's span
    is a real file offset."""
    return re.sub(r"[^\n]", " ", m.group(0))


def _span(orig: str, s: int, e: int, editable: bool = True) -> dict:
    """The span quadruple every block carries: the file region and the
    hash of it (the patch door's clobber check). `editable` is the
    pure-prose rule: a region with a macro, cite, math or escape is
    source-edit only, whatever the caller hoped."""
    return {"start": s, "end": e, "base": section_hash(orig[s:e]),
            "editable": editable and not _EDIT.search(orig, s, e)}


def _to_blocks(text: str, labels: dict[str, str]) -> list[dict]:
    code = _COMMENT.sub(_blank, text)
    out = []
    body, body_at = code, 0
    if "\\begin{document}" in code:
        at = code.index("\\begin{document}") + len("\\begin{document}")
        rest = code[at:]
        stop = rest.find("\\end{document}")
        body = rest if stop == -1 else rest[:stop]
        body_at = at
        m = _TITLE.search(code[:at])        # the preamble's one readable line
        if m:
            arg, end = _brace_arg(code, m.end())
            out.append({"kind": "heading", "level": 0,
                        "text": _inline(arg, labels),
                        **_span(text, m.end(), end - 1,
                                editable=False)})  # preamble: source-edit
    body = _INPUT.sub(_blank, body)         # children ride as their own sections
    body = re.sub(r"\\end\{document\}", _blank, body)
    out.extend(_scan(body, labels, body_at, text))
    return out


def _scan(body: str, labels: dict[str, str], at: int,
          orig: str) -> list[dict]:
    """Blocks over `body`, whose indices sit at `at` in `orig` (the real
    file text): spans, base hashes and the editable rule all speak of
    the original, never the blanked scan copy."""
    out: list[dict] = []

    def emit(chunk: str, chunk_at: int) -> None:
        t = _inline(chunk, labels)
        if t:
            # The span is the TRIMMED core: a patch replaces the text,
            # never the newline that separated the paragraphs.
            s = chunk_at + (len(chunk) - len(chunk.lstrip()))
            e = chunk_at + len(chunk) - (len(chunk) - len(chunk.rstrip()))
            out.append({"kind": "paragraph", "text": t,
                        **_span(orig, s, e)})

    def flush(chunk: str, chunk_at: int) -> None:
        start = 0
        for m in re.finditer(r"\n[ \t]*\n", chunk):
            emit(chunk[start:m.start()], chunk_at + start)
            start = m.end()
        emit(chunk[start:], chunk_at + start)

    pos = 0
    while True:
        m = _TOKEN.search(body, pos)
        if m is None:
            break
        flush(body[pos:m.start()], at + pos)
        if m.group(1) is not None:                     # environment
            name = m.group(1)
            end = _env_end(body, m.start(), name)
            out.extend(_env_block(name, body[m.start():end], labels,
                                  at + m.start(), orig))
            pos = end
        elif m.group(2) is not None or m.group(3) is not None:
            out.append({"kind": "math", "text": m.group(0),
                        **_span(orig, at + m.start(), at + m.end(),
                                editable=False)})     # raw, as written
            pos = m.end()
        else:                                          # heading
            arg, end = _brace_arg(body, m.end())
            s, e = at + m.end(), at + end - 1         # the argument only:
            out.append({"kind": "heading",             # edit replaces the
                        "level": _LEVEL[m.group(4)],   # title, not the
                        "text": _inline(arg, labels),  # \section wrapper
                        **_span(orig, s, e)})
            pos = end
    flush(body[pos:], at + pos)
    return out


def _env_end(body: str, begin_at: int, name: str) -> int:
    """Index just past the \\end matching the \\begin at begin_at,
    counting same-name marks (proper nesting). Unterminated: the rest
    of the file is the environment — never drop it."""
    depth = 0
    for m in _ENV_MARK.finditer(body, begin_at):
        if m.group(2) != name:
            continue
        depth += 1 if m.group(1) == "begin" else -1
        if depth == 0:
            return m.end()
    return len(body)


def _env_block(name: str, src: str, labels: dict[str, str], at: int,
               orig: str) -> list[dict]:
    m0 = re.match(r"\\begin\*?\{" + re.escape(name) + r"\}", src)
    inner_start = m0.end()
    stop = src.rfind("\\end{" + name + "}")
    inner = src[inner_start:] if stop == -1 else src[inner_start:stop]
                                        # unterminated: keep every char
    span = _span(orig, at, at + len(src),
                 editable=False)        # whole-env spans: source-edit
    if name in _LISTS:
        items = [t for t in (_inline(s, labels)
                             for s in re.split(r"\\item\b", inner)[1:]) if t]
        return [{"kind": "list", "ordered": _LISTS[name], "items": items,
                 **span}]
    if name == "abstract":
        return _scan(inner, labels, at + inner_start, orig)
    if name.rstrip("*") in _MATH_ENVS:
        return [{"kind": "math", "text": inner.strip(), **span}]
    if name.rstrip("*") == "figure":     # figure* (two-column floats) too
        m = re.search(r"\\caption\s*\{", src)
        if m:
            arg, end = _brace_arg(src, m.end())
            s, e = at + m.end(), at + end - 1
            return [{"kind": "caption", "text": _inline(arg, labels),
                     **_span(orig, s, e)}]
    return [{"kind": "raw", "text": src, **span}]


def _brace_arg(text: str, open_at: int) -> tuple[str, int]:
    """The braced argument whose '{' ends at open_at: (content, index
    past the '}'). Unbalanced: take the remainder (never crash)."""
    depth = 0
    for i in range(open_at - 1, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                return text[open_at:i], i + 1
    return text[open_at:], len(text)


def _surname(author: str) -> str:
    """First author's surname in either bibtex form: 'Vaswani, Ashish
    and others' and 'Ashish Vaswani and others' both give 'Vaswani'."""
    first = author.split(" and ")[0].strip()
    if not first:
        return ""
    return first.split(",")[0].strip() if "," in first else first.split()[-1]


def _labels(bib: str) -> dict[str, str]:
    """cite key -> "Surname, year" (SPEC's deterministic refs.bib
    lookup); an entry with neither author nor year labels as its key."""
    out = {}
    for e in parse_bib(bib):
        surname = _surname(e.fields.get("author", ""))
        year = e.fields.get("year", "").strip()
        out[e.key] = ", ".join(p for p in (surname, year) if p) or e.key
    return out


def _unescape(m) -> str:
    c = m.group(1)
    if c == ".":
        return ". "        # \cite{X}. — the abbreviation-space period
    if c in ",;: ":
        return " "         # LaTeX spacing commands
    if c == "$":
        return "\\$"       # stays escaped: the client splits unescaped
                           # $…$ into KaTeX spans, so a literal dollar
                           # must stay distinguishable from a delimiter
    return c               # \% \_ \& \# \{ \} -> the literal


def _inline(text: str, labels: dict[str, str]) -> str:
    """The prose pass. Order is load-bearing: the line break before the
    spacing-command unescape (else "\\\\ " loses a backslash), cites
    before wrappers (a wrapper inside a cite label is already gone),
    wrappers last-but-one. \\ref/\\eqref/\\label and unknown macros are
    deliberately untouched — SPEC's raw list."""
    t = re.sub(r"\\\\", " ", text)          # LaTeX line break
    t = re.sub(r"\\LaTeX\b", "LaTeX", t)
    t = re.sub(r"\\TeX\b", "TeX", t)
    t = re.sub(r"\\([%$_&#{}.,;: ])", _unescape, t)
    t = _CITE.sub(lambda m: "[" + "; ".join(
        labels.get(k.strip(), k.strip())
        for k in m.group(1).split(",") if k.strip()) + "]", t)
    while True:                              # innermost braces first
        t, n = _WRAPPER.subn(r"\1", t)
        if not n:
            break
    t = t.replace("~", " ")
    return re.sub(r"\s+", " ", t).strip()
