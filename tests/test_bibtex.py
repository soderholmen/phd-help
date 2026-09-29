"""BibTeX entry handling (SPEC §6): keys, parse/format, dedup.

Keys look like vaswani2023attention — first author surname + year + first
significant title word, the SPEC's worked example.
"""

from phd_helper.bibtex import (BibEntry, dedupe_key, format_entry, make_key,
                               parse_entry, same_paper)


def entry(**fields):
    return BibEntry(key="", type="inproceedings", fields=fields)


def test_make_key_follows_the_spec_example():
    e = entry(author="Vaswani, Ashish and others", year="2023",
              title="Attention Is All You Need")
    assert make_key(e) == "vaswani2023attention"


SAMPLE = """@inproceedings{vaswani2017,
  title={Attention Is All You Need},
  author={Vaswani, Ashish and others},
  booktitle={NeurIPS},
  year={2017}
}
"""


def test_parse_entry_reads_key_type_and_fields():
    e = parse_entry(SAMPLE)
    assert e.key == "vaswani2017"
    assert e.type == "inproceedings"
    assert e.fields["title"] == "Attention Is All You Need"
    assert e.fields["author"] == "Vaswani, Ashish and others"
    assert e.fields["year"] == "2017"


def test_format_round_trips_through_parse():
    assert parse_entry(format_entry(parse_entry(SAMPLE))) == parse_entry(SAMPLE)


def test_dedupe_key_appends_letter_suffix():
    taken = {"vaswani2023attention", "vaswani2023attentionb"}
    assert dedupe_key("vaswani2023attention", taken) == "vaswani2023attentionc"


def test_dedupe_key_passes_free_key_through():
    assert dedupe_key("free2024key", {"other2023x"}) == "free2024key"


def test_same_paper_matches_on_arxiv_id_or_doi():
    a = entry(arxiv="2301.00001", title="A")
    b = entry(arxiv="2301.00001", title="Different title casing A")
    c = entry(doi="10.1/xyz", title="C")
    d = entry(doi="10.1/xyz", title="D")
    e = entry(title="No identifiers")
    assert same_paper(a, b)
    assert same_paper(c, d)
    assert not same_paper(a, c)
    assert not same_paper(a, e)
