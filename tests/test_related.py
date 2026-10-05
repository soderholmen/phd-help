"""The related-work store (docs/related-work.md): the panel's whole
pipeline — record, merge, cited-mark, corpus-join, and the librarian's
JSON intake — as pure functions, so it is testable without FastAPI, the
LLM or the network."""

import json

from phd_helper.related import (RelatedEntry, corpus_join, from_dict,
                                mark_cited, merge, parse_json_array,
                                to_dict, to_entry)
from phd_helper.search import PaperHit


def hit(title, arxiv="", doi="", year="2024", authors=("Doe, Jane",),
        abstract="", source="arxiv"):
    return PaperHit(title=title, authors=authors, year=year, arxiv=arxiv,
                    doi=doi, venue="", source=source, abstract=abstract)


def test_a_search_entry_is_the_hit_plus_a_found_by_mark():
    e = to_entry(hit("Attention Is All You Need", arxiv="1706.03762"))
    assert e.title == "Attention Is All You Need"
    assert e.arxiv == "1706.03762"
    assert e.found_by == "search"
    assert e.why == "" and not e.cited


def test_a_search_records_its_hits_newest_first_and_deduped():
    old = to_entry(hit("Old Find", arxiv="1"))
    fresh = to_entry(hit("Fresh Find", arxiv="2"))
    again = to_entry(hit("Old Find", arxiv="1"))
    out = merge([old], [fresh, again])
    assert [e.title for e in out] == ["Fresh Find", "Old Find"]


def test_merging_keeps_the_why_the_librarian_wrote():
    curated = RelatedEntry(title="Attention Is All You Need",
                           authors=("Vaswani, Ashish",), year="2017",
                           arxiv="1706.03762", doi="", venue="",
                           abstract="", why="the transformer baseline",
                           found_by="librarian", cited=True)
    plain = to_entry(hit("Attention Is All You Need", arxiv="1706.03762",
                         year="2017", authors=("Vaswani, Ashish",)))
    out = merge([curated], [plain])
    assert len(out) == 1
    assert out[0].why == "the transformer baseline"
    assert out[0].cited  # the cited mark survives a rediscovery
    assert out[0].found_by == "librarian"  # ... and so does its provenance
    # but a fresh abstract the curated entry never had is not thrown away
    richer = to_entry(hit("Attention Is All You Need", arxiv="1706.03762",
                          year="2017", authors=("Vaswani, Ashish",),
                          abstract="The dominant sequence models..."))
    out = merge([curated], [richer])
    assert out[0].abstract == "The dominant sequence models..."


def test_merge_caps_the_list_at_the_newest_entries():
    existing = [to_entry(hit(f"E{i}", arxiv=str(i))) for i in range(30)]
    out = merge(existing, [to_entry(hit("Fresh", arxiv="99"))], cap=30)
    assert len(out) == 30
    assert out[0].title == "Fresh"
    assert all(e.title != "E29" for e in out)  # the oldest fell off


def test_cited_marking_matches_refs_bib_by_arxiv_id_or_doi():
    bib = """
@inproceedings{vaswani2017attention,
  title={Attention Is All You Need},
  author={Ashish Vaswani and others},
  year={2017},
  eprint={1706.03762}
}
@article{doe2019second,
  title={A Second Paper},
  doi={10.1000/second}
}
"""
    by_arxiv = to_entry(hit("Attention", arxiv="1706.03762v7"))
    by_doi = to_entry(hit("A Second Paper", doi="10.1000/second"))
    uncited = to_entry(hit("Unrelated", arxiv="9999.00001"))
    out = mark_cited([by_arxiv, by_doi, uncited], bib)
    assert [e.cited for e in out] == [True, True, False]


def test_cited_marking_survives_an_empty_bib():
    out = mark_cited([to_entry(hit("T"))], "")
    assert not out[0].cited


class Doc:
    """The registry's read shape, duck-typed (the real DocRecord carries
    more; the join only ever reads identity and pin state)."""

    def __init__(self, doc_id, status="indexed", arxiv="", doi="",
                 pinned_in=()):
        self.doc_id, self.status = doc_id, status
        self.arxiv, self.doi, self.pinned_in = arxiv, doi, pinned_in


def test_the_corpus_join_finds_a_doc_by_version_insensitive_arxiv_id():
    docs = [Doc("abc123", status="indexed", arxiv="arXiv:1706.03762v7",
                pinned_in=("paper",))]
    entry = to_entry(hit("Attention", arxiv="1706.03762"))
    rows = corpus_join([entry], docs, project="paper")
    assert rows[0]["in_corpus"] == {"doc_id": "abc123", "status": "indexed",
                                    "pinned_here": True}
    rows = corpus_join([to_entry(hit("Elsewhere", arxiv="9999.00001"))],
                       docs, project="paper")
    assert rows[0]["in_corpus"] is None


def test_the_corpus_join_falls_back_to_the_doi():
    docs = [Doc("d1", status="queued", doi="10.1000/second")]
    entry = to_entry(hit("A Second Paper", doi="10.1000/second"))
    rows = corpus_join([entry], docs, project="paper")
    assert rows[0]["in_corpus"]["doc_id"] == "d1"
    assert rows[0]["in_corpus"]["pinned_here"] is False


def test_entries_round_trip_through_the_store_shape():
    e = to_entry(hit("T", arxiv="1", abstract="abs"), why="w",
                 found_by="librarian")
    d = to_dict(e)
    assert json.loads(json.dumps(d)) == d  # plain JSON: no tuples on disk
    assert from_dict(d) == e


def test_a_torn_store_row_reads_as_nothing():
    # A hand-edited or half-written row is dropped by the next save,
    # not rendered as a card with an empty title.
    assert from_dict({"authors": []}) is None
    assert from_dict("not a dict") is None


def test_the_array_parser_reads_a_bare_json_array():
    assert parse_json_array('["a", "b"]') == ["a", "b"]


def test_the_array_parser_survives_fenced_prose_and_chatter():
    # The librarian's replies are model output: fences and a sentence
    # before/after the JSON are the normal shape, not the error case.
    assert parse_json_array(
        'Sure!\n```json\n["q1", "q2"]\n```\nDone.') == ["q1", "q2"]
    assert parse_json_array("Here: [1, 2, 3] - hope that helps") == [1, 2, 3]


def test_an_unparseable_or_non_array_reply_reads_as_empty():
    assert parse_json_array("") == []
    assert parse_json_array("no json here") == []
    assert parse_json_array('{"not": "an array"}') == []
    assert parse_json_array("[1, 2") == []
