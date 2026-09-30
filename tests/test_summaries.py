"""Rolling summaries (SPEC §4/§7): grouping is by the section an
exchange was anchored to, the render is the newest entry per section,
and undated discussion (no selection) rolls into nothing.
"""

from phd_helper.summaries import by_section, render


def ex(section, user, assistant="ok"):
    return [{"role": "user", "content": user, "section": section},
            {"role": "assistant", "content": assistant}]


def test_by_section_groups_exchanges_under_their_anchor():
    convo = ex("intro.tex", "tighten it") + ex("method.tex", "add ablation") \
        + ex("intro.tex", "and the hook")
    groups = by_section(convo)
    assert set(groups) == {"intro.tex", "method.tex"}
    assert "tighten it" in groups["intro.tex"] and "and the hook" in \
        groups["intro.tex"]
    assert "add ablation" in groups["method.tex"]


def test_unanchored_exchanges_roll_into_nothing():
    # Selected section cleared: nobody asked about a section, so there
    # is no section to roll the discussion into.
    convo = ex("", "what's the paper about?")
    assert by_section(convo) == {}


def test_tool_turns_stay_with_their_exchange():
    convo = [{"role": "user", "content": "cite vaswani",
              "section": "intro.tex"},
             {"role": "assistant", "content": None,
              "tool_calls": [{"id": "t1"}]},
             {"role": "tool", "tool_call_id": "t1", "content": "{}"},
             {"role": "assistant", "content": "cited"}]
    group = by_section(convo)["intro.tex"]
    assert "cite vaswani" in group and "cited" in group


def test_render_uses_the_newest_entry_per_section():
    summaries = {"intro.tex": [{"date": "2026-09-01", "text": "old"},
                               {"date": "2026-09-30", "text": "new"}],
                 "method.tex": [{"date": "2026-09-30", "text": "built"}]}
    out = render(summaries)
    assert "new" in out and "built" in out and "old" not in out
    assert out.startswith("Rolling summaries of past sessions:")


def test_empty_entries_render_empty_string():
    assert render({}) == ""
    assert render({"intro.tex": []}) == ""
    assert render({"intro.tex": [{"date": "x", "text": ""}]}) == ""
