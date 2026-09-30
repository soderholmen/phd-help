"""Per-section gists (SPEC §4): one line per section, auto-regenerated
whenever a section file changes, and the first thing to drop over budget
("far-away section gists").

Pure over the section tree, the project's files dict and a
``{path: {"sha": …, "gist": …}}`` cache — the LLM that writes the lines
rides the server's seam, same style as everything else that needs the
model. Staleness is content-hash truth: a section whose bytes changed
(or that was never gisted) is stale; a section that left the tree is
simply gone. The selected section never gets a gist line — its full
body already rides the turn, a summary of what's already there is noise.
"""

import hashlib

from phd_helper.sections import SectionNode


def body_sha(text: str) -> str:
    """The content-hash truth staleness is judged against; the server
    writes it into the cache entry beside the gist line."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def flatten(tree: list[SectionNode]) -> list[str]:
    """Section paths in document order (depth-first)."""
    out: list[str] = []
    for n in tree:
        out.append(n.path)
        out.extend(flatten(n.children))
    return out


def stale_sections(tree: list[SectionNode], files: dict[str, str],
                   cache: dict) -> list[str]:
    """Paths whose gist is missing or was built from different bytes."""
    return [p for p in flatten(tree)
            if (entry := cache.get(p)) is None
            or entry.get("sha") != hashlib.sha256(
                files.get(p, "").encode("utf-8")).hexdigest()]


def render_gists(tree: list[SectionNode], cache: dict, skip: str = "") -> str:
    """The far-gists block for context assembly; sections without a
    cached gist (never generated, or regeneration pending) are absent —
    the skeleton's heading line still names them."""
    lines = [f"- {p}: {cache[p]['gist']}"
             for p in flatten(tree)
             if p != skip and p in cache and cache[p].get("gist")]
    return "Section gists:\n" + "\n".join(lines) if lines else ""
