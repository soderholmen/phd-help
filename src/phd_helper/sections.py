"""The section tree, parsed from the root file's \\input/\\include graph
(SPEC §1).

Pure over a dict of project files so the parser has no I/O: the caller
supplies path -> content.
"""

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class SectionNode:
    path: str
    title: str | None = None
    children: tuple[SectionNode, ...] = ()


def parse_section_tree(files: dict[str, str], root: str) -> list[SectionNode]:
    return _children_of(files, files[root])


def _children_of(files: dict[str, str], content: str) -> list[SectionNode]:
    nodes: list[SectionNode] = []
    code = re.sub(r"(?<!\\)%.*", "", content)  # strip LaTeX comments
    for name in re.findall(r"\\(?:input|include)\{([^}]*)\}", code):
        path = name if name.endswith(".tex") else f"{name}.tex"
        child_content = files.get(path, "")
        heading = re.search(r"\\section\{([^}]*)\}", child_content)
        nodes.append(
            SectionNode(
                path=path,
                title=heading.group(1) if heading else None,
                children=tuple(_children_of(files, child_content)),
            )
        )
    return nodes
