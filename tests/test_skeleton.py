"""The paper skeleton (SPEC §4): title, abstract, all headings — the
never-drop map of the paper. Pure over the project files dict, same
seam style as the section-tree parser."""

from phd_helper.skeleton import build_skeleton

MAIN = ("\\documentclass{article}\n\\title{Attention Is All You Need}\n"
        "\\begin{document}\n"
        "\\begin{abstract}\nWe propose the Transformer, a architecture\n"
        "based on attention.\n\\end{abstract}\n"
        "\\input{sections/intro}\n\\input{sections/body}\n"
        "\\end{document}\n")

FILES = {
    "main.tex": MAIN,
    "sections/intro.tex": "\\section{Introduction}\nProse.\n",
    "sections/body.tex": ("\\section{Architecture}\n"
                          "\\input{sections/attention}\n"),
    "sections/attention.tex": "\\section{Attention}\nMore.\n",
}


def test_skeleton_carries_title_abstract_and_headings():
    skel = build_skeleton(FILES, "main.tex")
    assert "Attention Is All You Need" in skel
    # abstract whitespace collapses to one line
    assert "We propose the Transformer, a architecture based on attention." in skel
    assert "Introduction" in skel
    assert "Architecture" in skel


def test_nested_section_files_indent_under_their_parent():
    skel = build_skeleton(FILES, "main.tex")
    lines = skel.splitlines()
    arch = lines.index("- Architecture")
    assert lines[arch + 1].startswith("  - Attention")


def test_untitled_paper_and_missing_abstract_stay_honest():
    files = {"main.tex": "\\documentclass{article}\n"
                         "\\input{sections/intro}\n",
             "sections/intro.tex": "\\section{Intro}\n"}
    skel = build_skeleton(files, "main.tex")
    assert "untitled" in skel.lower()
    assert "Abstract" not in skel
    assert "Intro" in skel


def test_commented_out_title_does_not_count():
    files = {"main.tex": "% \\title{Draft Name}\n\\title{Real Name}\n"
                         "\\input{sections/a}\n",
             "sections/a.tex": "\\section{A}\n"}
    skel = build_skeleton(files, "main.tex")
    assert "Real Name" in skel
    assert "Draft Name" not in skel


def test_headingless_section_files_fall_back_to_their_path():
    files = {"main.tex": "\\input{sections/appendix}\n",
             "sections/appendix.tex": "no heading here\n"}
    skel = build_skeleton(files, "main.tex")
    assert "sections/appendix.tex" in skel


def test_missing_root_file_degrades_to_an_empty_skeleton():
    skel = build_skeleton({}, "main.tex")
    assert "untitled" in skel.lower()  # never raises into the turn


def test_title_and_abstract_are_found_through_the_input_graph():
    # Common template layout: main.tex \inputs the abstract (and a
    # preamble file that carries \title). The §4 never-drop map must not
    # silently ship "(untitled)" for a paper that has both.
    files = {
        "main.tex": ("\\input{preamble}\n\\begin{document}\n"
                     "\\input{abstract}\n\\input{sections/intro}\n"
                     "\\end{document}\n"),
        "preamble.tex": "\\usepackage{amsmath}\n"
                        "\\title{The Real Title}\n",
        "abstract.tex": "\\begin{abstract}\nA hidden abstract.\n"
                        "\\end{abstract}\n",
        "sections/intro.tex": "\\section{Intro}\n",
    }
    skel = build_skeleton(files, "main.tex")
    assert "The Real Title" in skel
    assert "A hidden abstract." in skel


def test_percent_note_inside_the_title_does_not_eat_it():
    # LaTeX treats "%" as a comment start inside \title too — the title
    # survives, the note does not.
    files = {"main.tex": "\\title{Scaling Laws % TODO verify}\n"
                         "\\input{sections/a}\n",
             "sections/a.tex": "\\section{A}\n"}
    skel = build_skeleton(files, "main.tex")
    assert "Scaling Laws" in skel
    assert "TODO" not in skel
    assert "untitled" not in skel.lower()
