"""The read view's kernel: LaTeX -> readable blocks (SPEC §Section view,
simple server-side variant). Pure over a dict of project files, like
sections.py; the never-drop contract is the property under test."""

from phd_helper.patches import section_hash
from phd_helper.readable import read_document, to_blocks

BIB = (
    "@inproceedings{vaswani2017attention,\n"
    "  author={Ashish Vaswani and Noam Shazeer},\n"
    "  year={2017},\n  title={Attention Is All You Need},\n}\n"
    "@article{he2016resnet,\n"
    "  author={Kaiming He and Xiangyu Zhang},\n  year={2016},\n}\n"
)


def kinds(blocks):
    return [b["kind"] for b in blocks]


# -- document order ---------------------------------------------------------


def test_document_order_is_depth_first_not_breadth_first():
    # main -> [a, b] with a -> c: LaTeX puts c between a and b; a BFS
    # walk (skeleton._graph) would put it after b.
    files = {
        "main.tex": "\\begin{document}\n\\input{sections/a}\n"
                    "\\input{sections/b}\n\\end{document}\n",
        "sections/a.tex": "\\section{A}\nA prose.\n\\input{sections/c}\n",
        "sections/b.tex": "\\section{B}\nB prose.\n",
        "sections/c.tex": "\\subsection{C}\nC prose.\n",
    }
    doc = read_document(files, "main.tex")
    assert [s["path"] for s in doc] == [
        "main.tex", "sections/a.tex", "sections/c.tex", "sections/b.tex"]


def test_sections_carry_their_titles():
    files = {
        "main.tex": "\\input{sections/intro}\n",
        "sections/intro.tex": "\\section{Introduction}\nProse.\n",
    }
    doc = read_document(files, "main.tex")
    assert doc[1]["title"] == "Introduction"


def test_input_lines_do_not_appear_as_raw_blocks():
    files = {
        "main.tex": "\\begin{document}\n\\input{sections/a}\n"
                    "\\end{document}\n",
        "sections/a.tex": "\\section{A}\nA.\n",
    }
    doc = read_document(files, "main.tex")
    assert doc[0]["blocks"] == []  # the spine contributes no prose


# -- block kinds ------------------------------------------------------------


def test_heading_levels():
    blocks = to_blocks(
        "\\title{The Paper}\n\\section{S}\n\\subsection{SS}\n"
        "\\subsubsection{SSS}\n")
    assert [(b["kind"], b.get("level")) for b in blocks] == [
        ("heading", 0), ("heading", 1), ("heading", 2), ("heading", 3)]
    assert blocks[0]["text"] == "The Paper"


def test_paragraphs_split_on_blank_lines_and_join_soft_wraps():
    blocks = to_blocks("First line\nwrapped.\n\nSecond paragraph.\n")
    assert kinds(blocks) == ["paragraph", "paragraph"]
    assert blocks[0]["text"] == "First line wrapped."


def test_comments_are_stripped():
    blocks = to_blocks("Real prose. % TODO rewrite\n\\section{Keep me} % x\n")
    assert kinds(blocks) == ["paragraph", "heading"]
    assert blocks[0]["text"] == "Real prose."


def test_lists_split_on_items():
    blocks = to_blocks(
        "\\begin{itemize}\n\\item one\n\\item two\n\\end{itemize}\n"
        "\\begin{enumerate}\n\\item first\n\\end{enumerate}\n")
    assert kinds(blocks) == ["list", "list"]
    assert blocks[0]["ordered"] is False
    assert blocks[0]["items"] == ["one", "two"]
    assert blocks[1]["ordered"] is True
    assert blocks[1]["items"] == ["first"]


def test_display_math_stays_raw():
    blocks = to_blocks("Before.\n\\[E=mc^2\\]\nAfter.\n")
    assert kinds(blocks) == ["paragraph", "math", "paragraph"]
    assert blocks[1]["text"] == "\\[E=mc^2\\]"


def test_dollar_math_stays_raw_and_splits_the_prose():
    blocks = to_blocks("We use\n$$a_n = n^2$$\nhere.\n")
    assert kinds(blocks) == ["paragraph", "math", "paragraph"]
    assert blocks[1]["text"] == "$$a_n = n^2$$"


def test_abstract_becomes_paragraphs():
    blocks = to_blocks(
        "\\begin{abstract}\nWe study X.\n\nMore abstract.\n\\end{abstract}\n")
    assert kinds(blocks) == ["paragraph", "paragraph"]
    assert blocks[0]["text"] == "We study X."


def test_figure_becomes_its_caption():
    blocks = to_blocks(
        "\\begin{figure}\n\\includegraphics{plot}\n"
        "\\caption{The curve.}\n\\end{figure}\n")
    assert kinds(blocks) == ["caption"]
    assert blocks[0]["text"] == "The curve."


def test_figure_without_caption_stays_raw():
    src = "\\begin{figure}\n\\includegraphics{plot}\n\\end{figure}\n"
    blocks = to_blocks(src)
    assert kinds(blocks) == ["raw"]
    assert "\\includegraphics" in blocks[0]["text"]


def test_table_stays_raw_its_rows_are_content():
    blocks = to_blocks(
        "\\begin{table}\n\\caption{Results.}\n\\begin{tabular}{cc}\n"
        "a & b \\\\\n\\end{tabular}\n\\end{table}\n")
    assert kinds(blocks) == ["raw"]
    assert "a & b" in blocks[0]["text"]


def test_equation_environment_becomes_math():
    blocks = to_blocks(
        "\\begin{equation}\nL = \\sum_i \\ell_i\n\\end{equation}\n")
    assert kinds(blocks) == ["math"]
    assert "L = \\sum_i" in blocks[0]["text"]


def test_unknown_environment_stays_raw_never_dropped():
    blocks = to_blocks("\\begin{myenv}\nsome content\n\\end{myenv}\n")
    assert kinds(blocks) == ["raw"]
    assert "some content" in blocks[0]["text"]


def test_unterminated_environment_keeps_every_character():
    # A known env cut off at EOF (a torn write mid-tick): the never-drop
    # contract says every char survives — the rfind miss must not read
    # as src[start:-1] and silently eat the last one.
    blocks = to_blocks("\\begin{equation}\nx = 1")
    assert kinds(blocks) == ["math"]
    assert blocks[0]["text"] == "x = 1"
    blocks = to_blocks("\\begin{itemize}\n\\item alpha")
    assert blocks[0]["items"] == ["alpha"]


def test_starred_float_caption_is_extracted_too():
    # figure* is the two-column float: same content, starred name.
    blocks = to_blocks(
        "\\begin{figure*}\n\\includegraphics{wide}\n"
        "\\caption{Panorama.}\n\\end{figure*}\n")
    assert kinds(blocks) == ["caption"]
    assert blocks[0]["text"] == "Panorama."


# -- inline pass ------------------------------------------------------------


def test_cites_become_author_year_from_the_bib():
    blocks = to_blocks("As shown in \\cite{vaswani2017attention} and\n"
                       "\\citep[p.~1]{he2016resnet}.\n", bib=BIB)
    assert blocks[0]["text"] == (
        "As shown in [Vaswani, 2017] and [He, 2016].")


def test_inverted_author_form_labels_the_same():
    bib = ("@article{smith2019x,\n  author={Smith, Jane and Lee, Bo},\n"
           "  year={2019},\n}\n")
    blocks = to_blocks("Per \\cite{smith2019x}.\n", bib=bib)
    assert blocks[0]["text"] == "Per [Smith, 2019]."


def test_unknown_cite_key_shows_the_key():
    blocks = to_blocks("See \\cite{ghost2020}.\n", bib=BIB)
    assert blocks[0]["text"] == "See [ghost2020]."


def test_multi_key_cite_lists_both_labels():
    blocks = to_blocks("Both \\cite{vaswani2017attention,he2016resnet}.\n",
                       bib=BIB)
    assert blocks[0]["text"] == "Both [Vaswani, 2017; He, 2016]."


def test_formatting_wrappers_keep_their_argument():
    blocks = to_blocks("This is \\textbf{very} \\emph{important}.\n")
    assert blocks[0]["text"] == "This is very important."


def test_refs_stay_raw_per_spec():
    blocks = to_blocks("See \\ref{sec:intro} and \\eqref{eq:loss}.\n")
    assert "\\ref{sec:intro}" in blocks[0]["text"]
    assert "\\eqref{eq:loss}" in blocks[0]["text"]


def test_escapes_become_their_literal_except_dollars():
    # Deliberate carve-out (the KaTeX slice): \$ stays escaped so the
    # read view's client can tell a literal dollar from an inline-math
    # delimiter — it splits unescaped $…$ into rendered spans and
    # unescapes \$ back in the prose between them. Every other escape
    # becomes its literal.
    blocks = to_blocks("We hit 50\\% of \\_tasks\\.\n")
    assert blocks[0]["text"] == "We hit 50% of _tasks."
    blocks = to_blocks("It costs \\$5, and $x^2$ holds.\n")
    assert blocks[0]["text"] == "It costs \\$5, and $x^2$ holds."


def test_unknown_macros_stay_raw_never_dropped():
    blocks = to_blocks("Prose with \\foo{bar} and \\baz inside.\n")
    assert "\\foo{bar}" in blocks[0]["text"]
    assert "\\baz" in blocks[0]["text"]


def test_nocite_stays_raw_it_is_invisible_in_latex():
    # \nocite adds a bib entry without printing anything: labelling it
    # would invent visible text LaTeX never shows.
    blocks = to_blocks("Prose. \\nocite{vaswani2017attention}\n", bib=BIB)
    assert "\\nocite{vaswani2017attention}" in blocks[0]["text"]


# -- the root file ----------------------------------------------------------


def test_preamble_dropped_but_title_kept():
    blocks = to_blocks(
        "\\documentclass{article}\n\\usepackage{amsmath}\n"
        "\\title{Deep Nets}\n\\begin{document}\nProse.\n\\end{document}\n")
    assert kinds(blocks) == ["heading", "paragraph"]
    assert blocks[0]["text"] == "Deep Nets"
    assert blocks[1]["text"] == "Prose."


def test_heading_argument_with_nested_braces():
    blocks = to_blocks("\\section{A {B} C}\n")
    assert blocks[0]["text"] == "A {B} C"


def test_missing_root_reads_as_nothing_to_read():
    # The caller's OSError degrade mirrored: an unreadable spine is no
    # blocks, not a KeyError from the tree walk.
    files = {"sections/a.tex": "\\section{A}\nA.\n"}
    assert read_document(files, "main.tex") == []


# -- source spans (the edit door's seam) --------------------------------------


def test_paragraph_spans_point_into_the_original_file():
    # Blanking comments/preamble/\\input IN PLACE (never deleting) is
    # what makes span == file offset true even after preprocessing.
    src = ("\\begin{document}\n% a comment\nPlain prose here.\n"
           "\\input{sections/a}\nAfter.\n\\end{document}\n")
    blocks = to_blocks(src)
    assert kinds(blocks) == ["paragraph", "paragraph"]
    assert src[blocks[0]["start"]:blocks[0]["end"]] == "Plain prose here."
    assert src[blocks[1]["start"]:blocks[1]["end"]] == "After."
    assert blocks[0]["editable"] is True
    assert blocks[0]["base"] == section_hash("Plain prose here.")


def test_heading_span_is_the_argument_only():
    # Editing a heading replaces its title, not the \\section wrapper.
    src = "\\section{Deep Nets}\nProse.\n"
    h = to_blocks(src)[0]
    assert src[h["start"]:h["end"]] == "Deep Nets"
    assert h["editable"] is True


def test_heading_with_a_macro_is_source_edit_only():
    h = to_blocks("\\section{The \\textbf{real} title}\n")[0]
    assert h["editable"] is False


def test_caption_span_is_the_argument_and_editable():
    src = ("\\begin{figure}\n\\includegraphics{x}\n"
           "\\caption{A nice plot.}\n\\end{figure}\n")
    c = to_blocks(src)[0]
    assert src[c["start"]:c["end"]] == "A nice plot."
    assert c["editable"] is True


def test_cite_and_math_blocks_are_not_editable():
    # The projection is lossy on these — writing prose back would
    # destroy the \\cite, so the source editor is their only surface.
    blocks = to_blocks("Plain one.\n\nSee \\cite{x} here.\n")
    assert blocks[0]["editable"] is True
    assert blocks[1]["editable"] is False


def test_env_blocks_span_their_source():
    src = "\\begin{equation}\nx=1\n\\end{equation}\n"
    b = to_blocks(src)[0]
    assert src[b["start"]:b["end"]] == src.rstrip("\n")  # to the \end's }
    assert b["editable"] is False


def test_abstract_paragraph_spans_point_into_the_file():
    # The sub-scan threads its base: spans stay file-relative inside
    # nested scans, not relative to the environment slice.
    src = "\\begin{abstract}\nWe study X.\n\\end{abstract}\n"
    b = to_blocks(src)[0]
    assert src[b["start"]:b["end"]] == "We study X."
    assert b["editable"] is True
