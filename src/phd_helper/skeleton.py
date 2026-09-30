"""The paper skeleton (SPEC §4): title, abstract, all headings — the
never-drop map that anchors every turn. Pure over the project files dict,
same seam style as the section-tree parser; per-section gists land with
the memory slice and append here later.

Title and abstract are searched through the \\input graph, not just the
root file: template layouts routinely \\input a preamble that carries
\\title and a separate abstract file.
"""

import re

from phd_helper.sections import parse_section_tree

_INPUT = re.compile(r"\\(?:input|include)\{([^}]*)\}")
_ABSTRACT = re.compile(r"\\begin\{abstract\}(.*?)\\end\{abstract\}", re.S)


def build_skeleton(files: dict[str, str], root: str) -> str:
    graph = _graph(files, root)  # document order; [] when root is missing
    title = next((t for text in graph if (t := _title_of(text))), "")
    abstract = ""
    for text in graph:
        m = _ABSTRACT.search(_strip(text))
        if m and (a := " ".join(m.group(1).split())):
            abstract = a
            break

    lines = ["Paper skeleton:"]
    lines.append(f"Title: {title}" if title else "Title: (untitled)")
    if abstract:
        lines.append(f"Abstract: {abstract}")
    lines.append("Headings:")
    if root in files:
        _render(parse_section_tree(files, root), lines, 0)
    return "\n".join(lines)


def _strip(text: str) -> str:
    return re.sub(r"(?<!\\)%.*", "", text)  # LaTeX comments


def _graph(files: dict[str, str], root: str) -> list[str]:
    """Root plus everything it \\inputs/\\includes, transitively, in
    document order — the paper's own files, nothing else."""
    out: list[str] = []
    seen: set[str] = set()
    stack = [root]
    while stack:
        path = stack.pop(0)
        if path in seen or path not in files:
            continue
        seen.add(path)
        out.append(files[path])
        for name in _INPUT.findall(_strip(files[path])):
            stack.append(name if name.endswith(".tex") else f"{name}.tex")
    return out


def _title_of(text: str) -> str:
    """Brace-match \\title{...} on the raw text (a "%" note inside the
    title must not eat the closing brace), then drop comments. A
    commented-out \\title line does not count."""
    for m in re.finditer(r"\\title\{", text):
        line_start = text.rfind("\n", 0, m.start()) + 1
        if re.search(r"(?<!\\)%", text[line_start:m.start()]):
            continue
        j, depth = m.end(), 1
        while j < len(text):
            c = text[j]
            if c == "\\":
                j += 2
                continue
            if c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
                if depth == 0:
                    break
            j += 1
        if depth:  # unbalanced: not a title we can trust
            continue
        body = _strip(text[m.end():j])
        if body := " ".join(body.split()):
            return body
    return ""


def _render(nodes, lines: list[str], depth: int) -> None:
    for n in nodes:
        lines.append(f"{'  ' * depth}- {n.title or n.path}")
        _render(n.children, lines, depth + 1)
