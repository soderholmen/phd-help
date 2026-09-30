"""MinerU 4.0's content format is markdown (verified against the real
CLI: the --json envelope carries a markdown string, with page markers
and figure locators embedded). This pure parser maps it onto the Block
vocabulary the chunker already speaks — the adapter half (subprocess)
stays thin.
"""

import pytest

from phd_helper.mdblocks import markdown_to_blocks

DOC = """<!-- page 1 of 15 -->

# Attention Is All You Need

The dominant sequence transduction models are based on complex
recurrent networks.

## Abstract

We propose the transformer.

<!-- page 2 of 15 -->

Attention weights are scaled.
"""


def test_page_markers_track_pages():
    blocks = markdown_to_blocks(DOC)
    by_text = {b.text.split()[0]: b for b in blocks}
    assert by_text["The"].page == 1
    assert by_text["Attention"].page == 2  # the scaled-weights paragraph


def test_headings_carry_levels():
    blocks = markdown_to_blocks(DOC)
    heads = [b for b in blocks if b.kind == "heading"]
    assert [(b.text, b.level) for b in heads] == [
        ("Attention Is All You Need", 1), ("Abstract", 2)]
    assert all(b.kind == "heading" for b in heads)


def test_block_ids_are_page_local_and_reset_per_page():
    # §6: locators must be stable against the PDF, not against this
    # exact extraction — a document-wide counter lets a reparse of
    # page 2 shift every block id after it. Page-local ids (1-based,
    # matching MinerU's own convention) keep the blast radius one page.
    blocks = markdown_to_blocks(DOC)
    p1 = [b for b in blocks if b.page == 1]
    p2 = [b for b in blocks if b.page == 2]
    assert [b.block for b in p1] == [1, 2, 3, 4]
    assert [b.block for b in p2] == [1]


def test_multiline_paragraph_joins_into_one_block():
    blocks = markdown_to_blocks(DOC)
    para = [b for b in blocks if b.text.startswith("The dominant")]
    assert len(para) == 1 and para[0].kind == "text"


FIG = """<!-- page 3 of 15 -->

![Image block](doc:bdfaa68/tier:standard/page:3/block:1)

Figure 1: The transformer architecture.
"""


def test_figure_marker_becomes_figure_block():
    blocks = markdown_to_blocks(FIG)
    fig = [b for b in blocks if b.kind == "figure"]
    assert len(fig) == 1
    assert "Image block" in fig[0].text
    assert "doc:bdfaa68" not in fig[0].text  # locator noise stripped


def test_figure_keeps_minerus_real_page_and_block_locator():
    # The URL carries MinerU's real page:N/block:M — the PDF-stable
    # locator §6:161 makes load-bearing. It must survive the parse,
    # not be replaced by a synthesized id.
    blocks = markdown_to_blocks(FIG)
    fig = [b for b in blocks if b.kind == "figure"][0]
    assert (fig.page, fig.block) == (3, 1)


def test_figure_real_id_reserves_the_page_local_counter():
    # No two blocks on a page may share an id: the counter skips past
    # whatever real id the figure claimed.
    md = ("<!-- page 5 of 9 -->\n\n"
          "Intro text.\n\n"
          "![Image block](doc:x/tier:standard/page:5/block:2)\n\n"
          "Figure 2: A figure.\n\n"
          "More text.\n")
    blocks = markdown_to_blocks(md)
    assert [(b.kind, b.block) for b in blocks] == [
        ("text", 1), ("figure", 2), ("text", 3)]


def test_figure_url_page_wins_over_the_marker():
    # The URL is MinerU's authoritative locator for the asset.
    md = ("<!-- page 3 of 9 -->\n\n"
          "![Image block](doc:x/tier:standard/page:4/block:1)\n")
    blocks = markdown_to_blocks(md)
    assert (blocks[0].page, blocks[0].block) == (4, 1)


def test_figure_without_a_locator_falls_back_to_page_and_counter():
    md = ("<!-- page 2 of 9 -->\n\n![Image block](images/x.png)\n")
    blocks = markdown_to_blocks(md)
    assert (blocks[0].page, blocks[0].block) == (2, 1)


def test_caption_merges_into_the_figure_chunk():
    # A bare "Image block" is a contentless chunk; the caption paragraph
    # below it is the figure's searchable text (§6: useful standalone
    # figure chunks).
    blocks = markdown_to_blocks(FIG)
    assert len(blocks) == 1  # caption absorbed, not a separate text block
    assert "The transformer architecture" in blocks[0].text


def test_caption_like_prose_stays_its_own_text_block():
    md = ("<!-- page 1 of 1 -->\n\n"
          "![Image block](doc:x/tier:standard/page:1/block:1)\n\n"
          "Figure 1 shows that scaling works.\n")
    blocks = markdown_to_blocks(md)
    assert [b.kind for b in blocks] == ["figure", "text"]


def test_caption_like_prose_with_a_comma_stays_separate():
    md = ("<!-- page 1 of 1 -->\n\n"
          "![Image block](doc:x/tier:standard/page:1/block:1)\n\n"
          "Figure 1, which shows scaling, is famous.\n")
    blocks = markdown_to_blocks(md)
    assert [b.kind for b in blocks] == ["figure", "text"]


@pytest.mark.parametrize("cap", [
    "Fig. 1: A chart.",            # the abbreviation real papers use
    "Figure 1. Caption.",          # period separator
    "Figure 1(a): Detail.",        # subfigure label
    "Figure 1 (a): Detail.",       # ... with a space
    "TABLE 2. Results.",           # caps + period
    "Algorithm 1: Procedure.",     # algorithm floats
])
def test_wider_caption_shapes_merge_into_the_figure(cap):
    md = ("<!-- page 1 of 1 -->\n\n"
          "![Image block](doc:x/tier:standard/page:1/block:1)\n\n"
          f"{cap}\n")
    blocks = markdown_to_blocks(md)
    assert len(blocks) == 1, f"{cap!r} left a bare figure + separate prose"
    assert cap in blocks[0].text


EQ = """<!-- page 4 of 15 -->

$$
\\mathrm{Attention}(Q, K, V) = \\mathrm{softmax}\\left(\\frac{QK^T}{\\sqrt{d_k}}\\right)V
$$

We use h parallel layers.
"""


def test_display_math_is_one_equation_block():
    blocks = markdown_to_blocks(EQ)
    eqs = [b for b in blocks if b.kind == "equation"]
    assert len(eqs) == 1
    assert "softmax" in eqs[0].text
    assert eqs[0].page == 4


PIPE = """<!-- page 6 of 15 -->

| Layer Type | Complexity |
| --- | --- |
| Self-Attention | O(n2) |
| Recurrent | O(n) |

Some following text.
"""


def test_pipe_table_is_one_table_block():
    blocks = markdown_to_blocks(PIPE)
    tables = [b for b in blocks if b.kind == "table"]
    assert len(tables) == 1
    assert "Self-Attention" in tables[0].text and "Recurrent" in tables[0].text
    assert tables[0].page == 6


HTML = """<!-- page 8 of 15 -->

<table><tbody><tr><td>Model</td><td>BLEU</td></tr><tr><td>big</td><td>28.4</td></tr></tbody></table>

After the table.
"""


def test_html_table_is_one_table_block():
    blocks = markdown_to_blocks(HTML)
    tables = [b for b in blocks if b.kind == "table"]
    assert len(tables) == 1
    assert "28.4" in tables[0].text


def test_empty_input_yields_no_blocks():
    assert markdown_to_blocks("") == []
    assert markdown_to_blocks("\n\n  \n") == []
