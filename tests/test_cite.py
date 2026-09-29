"""The cite loop (SPEC §6): bib entry + prose + \\cite land in ONE anchored
patch — one voice approval covers all three. Pre-approval lint rejects
unresolved/duplicate keys; re-citing an owned paper reuses its key.
"""

import pytest

from phd_helper.bibtex import BibEntry
from phd_helper.cascade import Lookup
from phd_helper.project import Project, ProposeError

MAIN = ("\\documentclass{article}\n\\begin{document}\n"
        "\\input{sections/intro}\n\\end{document}\n")
INTRO = "We use a transformer. It works well.\n"
BIB = """@inproceedings{vaswani2017,
  title={Attention Is All You Need},
  author={Vaswani, Ashish and others},
  booktitle={NeurIPS},
  year={2017}
}
@article{oldkey2019owned,
  title={An Owned Paper},
  author={Owner, O.},
  year={2019},
  doi={10.5555/owned}
}
"""


@pytest.fixture
def paper(tmp_path):
    (tmp_path / "sections").mkdir()
    (tmp_path / "main.tex").write_text(MAIN, encoding="utf-8")
    (tmp_path / "sections" / "intro.tex").write_text(INTRO, encoding="utf-8")
    (tmp_path / "refs.bib").write_text(BIB, encoding="utf-8")
    return Project(tmp_path)


def new_entry():
    return BibEntry(key="shazeer2024mesh", type="article", fields={
        "title": "Mesh Anything", "author": "Shazeer, Noam", "year": "2024",
        "arxiv": "2401.00002"})


def test_propose_cite_writes_nothing_until_approved(paper):
    diff = paper.propose_cite(
        "sections/intro.tex", find="We use a transformer.",
        replace="We use a transformer \\cite{2401.00002}.",
        lookup=Lookup(arxiv="2401.00002"), entry=new_entry())
    assert paper.read_section("sections/intro.tex") == INTRO
    assert "shazeer" not in paper.read_bib()
    assert diff.id


def test_one_approval_lands_section_and_bib(paper):
    diff = paper.propose_cite(
        "sections/intro.tex", find="We use a transformer.",
        replace="We use a transformer \\cite{2401.00002}.",
        lookup=Lookup(arxiv="2401.00002"), entry=new_entry())
    result = paper.apply_pending("sections/intro.tex", diff.id)
    assert result.applied
    text = paper.read_section("sections/intro.tex")
    assert "\\cite{shazeer2024mesh}" in text  # lookup id -> final key
    assert "@article{shazeer2024mesh" in paper.read_bib()


def test_reject_writes_neither(paper):
    diff = paper.propose_cite(
        "sections/intro.tex", find="We use a transformer.",
        replace="We use a transformer \\cite{2401.00002}.",
        lookup=Lookup(arxiv="2401.00002"), entry=new_entry())
    paper.reject_pending("sections/intro.tex", diff.id)
    assert paper.read_section("sections/intro.tex") == INTRO
    assert "shazeer" not in paper.read_bib()


def test_reciting_an_owned_paper_reuses_its_key(paper):
    owned = BibEntry(key="owner2019owned", type="article", fields={
        "title": "An Owned Paper", "author": "Owner, O.", "year": "2019",
        "doi": "10.5555/owned"})
    diff = paper.propose_cite(
        "sections/intro.tex", find="We use a transformer.",
        replace="We cite \\cite{10.5555/owned}.",
        lookup=Lookup(doi="10.5555/owned"), entry=owned)
    assert "\\cite{oldkey2019owned}" in diff.proposed_text
    assert diff.bib_append is None  # cascade only runs for new entries
    paper.apply_pending("sections/intro.tex", diff.id)
    assert paper.read_bib().count("@") == 2  # no second entry for one paper


def test_hallucinated_key_bounces_before_approval(paper):
    with pytest.raises(ProposeError) as exc:
        paper.propose_cite(
            "sections/intro.tex", find="We use a transformer.",
            replace="We cite \\cite{2401.00002} and \\cite{hallucinated2099}.",
            lookup=Lookup(arxiv="2401.00002"), entry=new_entry())
    assert "hallucinated2099" in str(exc.value)


def test_key_collision_with_a_different_paper_renumbers(paper):
    twin = BibEntry(key="vaswani2017", type="article", fields={
        "title": "A Different Paper", "author": "Vaswani, Ashish",
        "year": "2017", "arxiv": "9999.00001"})
    diff = paper.propose_cite(
        "sections/intro.tex", find="We use a transformer.",
        replace="We cite \\cite{9999.00001}.",
        lookup=Lookup(arxiv="9999.00001"), entry=twin)
    assert "\\cite{vaswani2017b}" in diff.proposed_text
    paper.apply_pending("sections/intro.tex", diff.id)
    assert "@article{vaswani2017b" in paper.read_bib()


# -- approval-time bib discipline (review: propose-time checks go stale) --

def test_second_approval_of_same_paper_skips_the_append(paper):
    # Both diffs proposed while the bib lacked the entry; the second
    # approval must reuse the landed entry, not duplicate the key.
    d1 = paper.propose_cite(
        "sections/intro.tex", find="We use a transformer.",
        replace="We use a transformer \\cite{2401.00002}.",
        lookup=Lookup(arxiv="2401.00002"), entry=new_entry())
    d2 = paper.propose_cite(
        "sections/intro.tex", find="It works well.",
        replace="It works well \\cite{2401.00002}.",
        lookup=Lookup(arxiv="2401.00002"), entry=new_entry())
    assert paper.apply_pending("sections/intro.tex", d1.id).applied
    assert paper.apply_pending("sections/intro.tex", d2.id).applied
    assert paper.read_bib().count("@article{shazeer2024mesh") == 1


def test_approval_bounces_when_key_taken_by_another_paper(paper):
    diff = paper.propose_cite(
        "sections/intro.tex", find="We use a transformer.",
        replace="We use a transformer \\cite{2401.00002}.",
        lookup=Lookup(arxiv="2401.00002"), entry=new_entry())
    # Another session lands a different paper under the same key meanwhile.
    (paper.root / "refs.bib").write_text(
        paper.read_bib() + "@article{shazeer2024mesh,\n"
        " title={Other Work},\n author={Other, O.},\n year={2024},\n"
        " arxiv={9999.99999},\n}\n", encoding="utf-8")
    result = paper.apply_pending("sections/intro.tex", diff.id)
    assert not result.applied
    assert "shazeer2024mesh" in result.reason
    assert paper.read_section("sections/intro.tex") == INTRO  # wrote nothing
    # stays pending so the user sees why
    assert any(d.id == diff.id for d in paper.list_pending("sections/intro.tex"))


def test_bib_append_starts_on_its_own_line(paper):
    (paper.root / "refs.bib").write_text(BIB.rstrip("\n"), encoding="utf-8")
    diff = paper.propose_cite(
        "sections/intro.tex", find="We use a transformer.",
        replace="We use a transformer \\cite{2401.00002}.",
        lookup=Lookup(arxiv="2401.00002"), entry=new_entry())
    paper.apply_pending("sections/intro.tex", diff.id)
    assert "}\n@article{shazeer2024mesh" in paper.read_bib()  # not glued on
