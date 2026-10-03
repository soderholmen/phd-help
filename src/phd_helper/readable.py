"""The read view's kernel: LaTeX -> readable blocks (SPEC §Section view,
simple server-side variant).

Pure over a dict of project files, like sections.py: the caller supplies
path -> content, this module never touches the disk. The output is
JSON-ready blocks — heading/paragraph/list/math/caption/raw — assembled
in TRUE document order: depth-first over the section tree (a BFS walk
would place a nested child after its parent's next sibling, which is not
where LaTeX puts it).

The never-drop contract (SPEC:141): anything the kernel does not
understand — an unknown environment, \\ref, an unknown macro — passes
through as source, visible. What is deliberately NOT shown: the
preamble (setup, not content; \\title excepted) and a figure's
\\includegraphics line (the image is not a project file; its caption is
the text). Tables stay raw: their rows are content, and raw rows beat
dropped ones. Inline math stays inline and raw — the KaTeX door is open,
this slice ships no renderer.

for_speech (saytext.py) is the voice's lossy paraphrase and is NOT
reusable here: it replaces math and citations with placeholder words and
flattens every newline. The reading view keeps the content.
"""

import re

from phd_helper.bibtex import parse_bib
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


def _to_blocks(text: str, labels: dict[str, str]) -> list[dict]:
    code = _COMMENT.sub("", text)
    out = []
    body = code
    if "\\begin{document}" in code:
        pre, _, rest = code.partition("\\begin{document}")
        body = rest.split("\\end{document}")[0]
        m = _TITLE.search(pre)          # the preamble's one readable line
        if m:
            arg, _ = _brace_arg(pre, m.end())
            out.append({"kind": "heading", "level": 0,
                        "text": _inline(arg, labels)})
    body = body.replace("\\end{document}", "")
    body = _INPUT.sub("", body)         # children ride as their own sections
    out.extend(_scan(body, labels))
    return out


def _scan(body: str, labels: dict[str, str]) -> list[dict]:
    out: list[dict] = []

    def flush(chunk: str) -> None:
        for para in re.split(r"\n[ \t]*\n", chunk):
            t = _inline(para, labels)
            if t:
                out.append({"kind": "paragraph", "text": t})

    pos = 0
    while True:
        m = _TOKEN.search(body, pos)
        if m is None:
            break
        flush(body[pos:m.start()])
        if m.group(1) is not None:                     # environment
            name = m.group(1)
            end = _env_end(body, m.start(), name)
            out.extend(_env_block(name, body[m.start():end], labels))
            pos = end
        elif m.group(2) is not None or m.group(3) is not None:
            out.append({"kind": "math",
                        "text": m.group(0)})           # raw, as written
            pos = m.end()
        else:                                          # heading
            arg, end = _brace_arg(body, m.end())
            out.append({"kind": "heading",
                        "level": _LEVEL[m.group(4)],
                        "text": _inline(arg, labels)})
            pos = end
    flush(body[pos:])
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


def _env_block(name: str, src: str,
               labels: dict[str, str]) -> list[dict]:
    inner = _inner(src, name)
    if name in _LISTS:
        items = [t for t in (_inline(s, labels)
                             for s in re.split(r"\\item\b", inner)[1:]) if t]
        return [{"kind": "list", "ordered": _LISTS[name], "items": items}]
    if name == "abstract":
        return _scan(inner, labels)
    if name.rstrip("*") in _MATH_ENVS:
        return [{"kind": "math", "text": inner.strip()}]
    if name.rstrip("*") == "figure":     # figure* (two-column floats) too
        m = re.search(r"\\caption\s*\{", src)
        if m:
            arg, _ = _brace_arg(src, m.end())
            return [{"kind": "caption", "text": _inline(arg, labels)}]
    return [{"kind": "raw", "text": src}]


def _inner(src: str, name: str) -> str:
    start = re.match(r"\\begin\*?\{" + re.escape(name) + r"\}", src).end()
    end = src.rfind("\\end{" + name + "}")
    return src[start:] if end == -1 else src[start:end]
                                         # unterminated: keep every char


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
    return c               # \% \$ \_ \& \# \{ \} -> the literal


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
