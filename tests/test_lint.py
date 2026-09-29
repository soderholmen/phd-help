"""Deterministic LaTeX lint (SPEC §5): runs on the resulting file before the
diff is shown; failures bounce back to the agent, never to the user."""

from phd_helper.lint import lint_latex


def test_clean_section_passes_lint():
    assert lint_latex("We show that $f(x)$ is smooth.\n") == []


def test_lint_reports_unbalanced_braces():
    errors = lint_latex("We show that f(x} is smooth.\n")

    assert [e.rule for e in errors] == ["brace-balance"]


def test_lint_reports_unmatched_begin_end():
    errors = lint_latex("\\begin{equation}\n  y = Wx + b\n")

    assert [e.rule for e in errors] == ["begin-end"]


def test_lint_accepts_matched_environments():
    assert lint_latex(
        "\\begin{equation}\n  y = Wx + b\n\\end{equation}\n"
        "\\begin{align}\n  a &= b\n\\end{align}\n"
    ) == []


def test_lint_reports_citation_to_unknown_bib_key():
    errors = lint_latex(
        "As shown by \\citep{vaswani2023attention}.\n",
        bib_keys={"kothari2024something"},
    )

    assert [e.rule for e in errors] == ["unknown-citation"]


def test_lint_accepts_citations_to_known_bib_keys():
    assert lint_latex(
        "\\citep{vaswani2023attention} and \\citet[see][p.~1]{devlin2019bert}\n"
        "and \\cite{vaswani2023attention,devlin2019bert}\n",
        bib_keys={"vaswani2023attention", "devlin2019bert"},
    ) == []
