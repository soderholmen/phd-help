"""Section-aware chunking (SPEC §6) on MinerU's block tree: heading-path
prefix on the embedded text, tables/figures as standalone chunks, a token
cap, and stable page/block locators. Pure function — tests feed literal
block trees shaped from MinerU's model.json, no extraction involved.
"""

from phd_helper.chunking import Block, chunk_document


def B(kind, text, page=1, block=0, level=0):
    return Block(kind=kind, text=text, page=page, block=block, level=level)


def test_empty_document_yields_no_chunks():
    assert chunk_document("T", []) == []


def test_text_blocks_merge_into_one_chunk_with_title_prefix():
    blocks = [B("text", "First paragraph.", block=0),
              B("text", "Second paragraph.", block=1)]
    chunks = chunk_document("Attention Is All You Need", blocks)
    assert len(chunks) == 1
    c = chunks[0]
    assert c.text == "First paragraph. Second paragraph."
    # the embedded text carries the heading-path prefix (§6), raw stays raw
    assert c.embed_text == "Attention Is All You Need\n" + c.text
    assert (c.page_start, c.page_end) == (1, 1)
    assert (c.block_start, c.block_end) == (0, 1)
    assert c.kind == "text"


def test_heading_path_prefixes_embedded_text():
    blocks = [B("heading", "Method", block=0, level=1),
              B("heading", "Training", block=1, level=2),
              B("text", "We train for three days.", block=2)]
    chunks = chunk_document("Paper", blocks)
    assert len(chunks) == 1
    c = chunks[0]
    assert c.section_path == "Method » Training"
    assert c.embed_text == "Paper » Method » Training\nWe train for three days."
    # the heading itself is structure, not content: it is not chunk text
    assert c.text == "We train for three days."


def test_sibling_heading_replaces_path_and_starts_new_chunk():
    blocks = [B("heading", "Intro", block=0, level=1),
              B("text", "Intro text.", block=1),
              B("heading", "Method", block=2, level=1),
              B("text", "Method text.", block=3)]
    chunks = chunk_document("Paper", blocks)
    assert [c.section_path for c in chunks] == ["Intro", "Method"]
    assert [c.text for c in chunks] == ["Intro text.", "Method text."]
    assert [ (c.block_start, c.block_end) for c in chunks] == [(1, 1), (3, 3)]


def test_deeper_heading_then_sibling_pops_the_deeper_level():
    blocks = [B("heading", "Method", block=0, level=1),
              B("heading", "Training", block=1, level=2),
              B("text", "Train.", block=2),
              B("heading", "Eval", block=3, level=2),
              B("text", "Eval.", block=4),
              B("heading", "Results", block=5, level=1),
              B("text", "Results.", block=6)]
    chunks = chunk_document("Paper", blocks)
    assert [c.section_path for c in chunks] == [
        "Method » Training", "Method » Eval", "Results"]


def test_tables_and_figures_are_standalone_chunks():
    blocks = [B("heading", "Method", block=0, level=1),
              B("text", "Before the table.", block=1),
              B("table", "<table>BLEU 42</table>", block=2),
              B("text", "After the table.", block=3)]
    chunks = chunk_document("Paper", blocks)
    assert [c.kind for c in chunks] == ["text", "table", "text"]
    table = chunks[1]
    assert table.text == "<table>BLEU 42</table>"
    assert table.section_path == "Method"
    assert table.embed_text == "Paper » Method\n<table>BLEU 42</table>"
    # the surrounding text does not merge across the table
    assert chunks[0].text == "Before the table."
    assert chunks[2].text == "After the table."


def test_figure_is_a_standalone_chunk():
    blocks = [B("figure", "Figure 1: architecture diagram", block=0)]
    chunks = chunk_document("Paper", blocks)
    assert len(chunks) == 1
    assert chunks[0].kind == "figure"


def test_equations_stay_with_the_prose():
    blocks = [B("text", "where", block=0),
              B("equation", "E = mc^2", block=1),
              B("text", "and c is the speed of light.", block=2)]
    chunks = chunk_document("Paper", blocks)
    assert len(chunks) == 1
    assert chunks[0].kind == "text"
    assert chunks[0].text == ("where E = mc^2 and c is the speed of light.")


def test_chunk_flushes_at_the_token_cap():
    # ~4 chars per token heuristic; cap 64 tokens ≈ 256 chars per chunk.
    # Each paragraph is ~30 tokens: two fit, three do not.
    para = ("word " * 24).strip()  # 119 chars ≈ 30 tokens
    blocks = [B("text", para, block=i) for i in range(4)]
    chunks = chunk_document("Paper", blocks, token_cap=64)
    assert len(chunks) == 2
    assert chunks[0].text == f"{para} {para}"
    assert chunks[1].text == f"{para} {para}"
    assert (chunks[0].block_start, chunks[0].block_end) == (0, 1)
    assert (chunks[1].block_start, chunks[1].block_end) == (2, 3)


def test_oversized_paragraph_splits_on_sentences():
    sentence = "This sentence runs quite long on purpose."  # 41 chars
    para = " ".join([sentence] * 20)  # 839 chars ≈ 210 tokens
    blocks = [B("text", para, block=0)]
    chunks = chunk_document("Paper", blocks, token_cap=64)
    assert len(chunks) >= 3
    for c in chunks:
        assert c.kind == "text"
        assert c.section_path == ""
    # every sentence lands exactly once, in order
    assert " ".join(c.text for c in chunks) == para
    # locators: one source block, so every sub-chunk points at it
    assert all((c.block_start, c.block_end) == (0, 0) for c in chunks)


def test_locators_span_pages():
    blocks = [B("text", "Page one text.", page=3, block=0),
              B("text", "Page two text.", page=5, block=1)]
    chunks = chunk_document("Paper", blocks)
    assert len(chunks) == 1
    assert (chunks[0].page_start, chunks[0].page_end) == (3, 5)


def test_abstract_heading_flags_its_chunks():
    blocks = [B("heading", "Abstract", block=0, level=1),
              B("text", "We propose a thing.", block=1),
              B("heading", "Intro", block=2, level=1),
              B("text", "Intro.", block=3)]
    chunks = chunk_document("Paper", blocks)
    assert [c.is_abstract for c in chunks] == [True, False]
    assert chunks[0].section_path == "Abstract"


def test_heading_only_section_yields_no_chunk():
    blocks = [B("heading", "Empty Section", block=0, level=1),
              B("heading", "Next", block=1, level=1),
              B("text", "Body.", block=2)]
    chunks = chunk_document("Paper", blocks)
    assert [c.section_path for c in chunks] == ["Next"]


def test_blank_blocks_are_skipped():
    blocks = [B("text", "   ", block=0),
              B("text", "Real text.", block=1),
              B("table", "", block=2)]
    chunks = chunk_document("Paper", blocks)
    assert len(chunks) == 1
    assert chunks[0].text == "Real text."
