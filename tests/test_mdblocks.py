"""MinerU 4.0's content format is markdown (verified against the real
CLI: the --json envelope carries a markdown string, with page markers
and figure locators embedded). This pure parser maps it onto the Block
vocabulary the chunker already speaks — the adapter half (subprocess)
stays thin.
"""

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


def test_block_ids_are_sequential_across_kinds():
    blocks = markdown_to_blocks(DOC)
    assert [b.block for b in blocks] == list(range(len(blocks)))


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
    assert fig[0].page == 3
    assert "Image block" in fig[0].text
    assert "doc:bdfaa68" not in fig[0].text  # locator noise stripped


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
