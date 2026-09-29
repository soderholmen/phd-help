"""The section tree is parsed from the root file's \\input/\\include graph
(SPEC §1); clicking a node anchors the section-scoped discussion."""

from phd_helper.sections import parse_section_tree

MAIN = (
    "\\documentclass{article}\n"
    "\\begin{document}\n"
    "\\input{sections/intro}\n"
    "\\input{sections/method}\n"
    "\\end{document}\n"
)


def test_root_inputs_become_section_nodes_in_order():
    files = {
        "main.tex": MAIN,
        "sections/intro.tex": "The problem and our claim.\n",
        "sections/method.tex": "How we train.\n",
    }

    tree = parse_section_tree(files, root="main.tex")

    assert [node.path for node in tree] == [
        "sections/intro.tex",
        "sections/method.tex",
    ]


def test_node_title_comes_from_the_sections_section_command():
    files = {
        "main.tex": "\\input{sections/intro}\n",
        "sections/intro.tex": "\\section{Introduction}\n\nThe problem and our claim.\n",
    }

    tree = parse_section_tree(files, root="main.tex")

    assert tree[0].title == "Introduction"


def test_input_inside_a_section_file_becomes_a_child_node():
    files = {
        "main.tex": "\\input{sections/method}\n",
        "sections/method.tex": (
            "\\section{Method}\n"
            "Overview prose.\n"
            "\\input{sections/method/model}\n"
        ),
        "sections/method/model.tex": "\\subsection{Model}\nArchitecture prose.\n",
    }

    tree = parse_section_tree(files, root="main.tex")

    assert tree[0].path == "sections/method.tex"
    assert [child.path for child in tree[0].children] == ["sections/method/model.tex"]


def test_commented_out_input_does_not_create_a_node():
    files = {
        "main.tex": (
            "\\input{sections/intro}\n"
            "% \\input{sections/old-draft}\n"
            "\\input{sections/related}\n"
        ),
        "sections/intro.tex": "\\section{Introduction}\n",
        "sections/old-draft.tex": "\\section{Old Draft}\n",
        "sections/related.tex": "\\section{Related Work}\n",
    }

    tree = parse_section_tree(files, root="main.tex")

    assert [node.path for node in tree] == [
        "sections/intro.tex",
        "sections/related.tex",
    ]


def test_declared_but_missing_file_still_shows_as_a_node():
    files = {"main.tex": "\\input{sections/ghost}\n"}  # not synced yet

    tree = parse_section_tree(files, root="main.tex")

    assert [node.path for node in tree] == ["sections/ghost.tex"]
    assert tree[0].title is None


def test_include_declares_sections_like_input():
    files = {
        "main.tex": (
            "\\begin{document}\n"
            "\\include{sections/intro}\n"
            "\\input{sections/method}\n"
            "\\end{document}\n"
        ),
        "sections/intro.tex": "Intro.\n",
        "sections/method.tex": "Method.\n",
    }

    tree = parse_section_tree(files, root="main.tex")

    assert [node.path for node in tree] == [
        "sections/intro.tex",
        "sections/method.tex",
    ]
