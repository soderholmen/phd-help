"""Per-section gists (SPEC §4): staleness is content-hash truth, the
rendered block is document order minus the selected section, and a
section that left the tree is simply gone. Pure over the tree, the
files dict and a {path: {"sha":…, "gist":…}} cache — the LLM that
writes the lines rides the server's seam, not this module.
"""

import hashlib

from phd_helper.gists import flatten, render_gists, stale_sections
from phd_helper.sections import SectionNode

TREE = [
    SectionNode("intro.tex", "Introduction",
                (SectionNode("intro/background.tex"),)),
    SectionNode("method.tex", "Method"),
]

FILES = {"intro.tex": "we present X",
         "intro/background.tex": "prior work",
         "method.tex": "transformers"}


def sha(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def cache_of(**gists):
    return {p: {"sha": sha(FILES[p]), "gist": g} for p, g in gists.items()}


def test_flatten_is_document_order():
    assert flatten(TREE) == ["intro.tex", "intro/background.tex",
                             "method.tex"]


def test_never_gisted_sections_are_stale():
    assert stale_sections(TREE, FILES, {}) == flatten(TREE)


def test_fresh_cache_is_not_stale():
    cache = cache_of(**{p: "one-liner" for p in FILES})
    assert stale_sections(TREE, FILES, cache) == []


def test_changed_bytes_make_only_that_section_stale():
    cache = cache_of(**{p: "one-liner" for p in FILES})
    changed = dict(FILES, **{"method.tex": "transformers, revised"})
    assert stale_sections(TREE, changed, cache) == ["method.tex"]


def test_section_removed_from_tree_is_neither_stale_nor_rendered():
    # The cache entry for a deleted file is dead weight, not a stale
    # section: it no longer exists to re-gist, and must not haunt the
    # rendered block.
    tree = [SectionNode("intro.tex", "Introduction")]
    cache = cache_of(**{p: "one-liner" for p in FILES})
    assert stale_sections(tree, FILES, cache) == []
    assert render_gists(tree, cache) == "Section gists:\n- intro.tex: one-liner"


def test_render_skips_the_selected_section():
    # Its full body already rides the turn; a summary of what's already
    # there is noise (§4).
    cache = cache_of(**{p: "one-liner" for p in FILES})
    out = render_gists(TREE, cache, skip="intro/background.tex")
    assert "intro/background.tex" not in out
    assert out == "Section gists:\n- intro.tex: one-liner\n- method.tex: one-liner"


def test_sections_without_a_cached_line_are_absent_not_placeholdered():
    # Regeneration pending: the skeleton's heading line still names the
    # section, so a "…" row would be noise.
    cache = cache_of(**{"method.tex": "one-liner"})
    assert render_gists(TREE, cache) == "Section gists:\n- method.tex: one-liner"


def test_empty_gist_string_is_absent_too():
    cache = {"method.tex": {"sha": sha(FILES["method.tex"]), "gist": ""}}
    assert render_gists(TREE, cache) == ""


def test_nothing_to_render_is_empty_string():
    assert render_gists([], {}) == ""
