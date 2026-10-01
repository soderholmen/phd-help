"""The §7 verbatim log at its pure seam: append, resume-tail, dividers,
crash shapes (torn lines, cut tool round-trips) and the recap line."""

import json
from datetime import date

from phd_helper import sessionlog


def msg(role, content, **extra):
    return {"role": role, "content": content, **extra}


def test_append_then_read_roundtrips_verbatim(tmp_path):
    p = tmp_path / "chat.jsonl"
    records = [msg("user", "tighten it", section="sections/intro.tex"),
               msg("assistant", "did")]
    sessionlog.append(p, records)
    assert sessionlog.read_tail(p) == records


def test_missing_file_reads_empty(tmp_path):
    p = tmp_path / "nope.jsonl"
    assert sessionlog.read_tail(p) == []
    assert sessionlog.last_divider(p) is None
    assert sessionlog.recap_line(None, 0) == ""


def test_divider_splits_the_log(tmp_path):
    p = tmp_path / "chat.jsonl"
    sessionlog.append(p, [msg("user", "old sitting")])
    sessionlog.append(p, [sessionlog.divider_record(
        "idle", 1, ["sections/intro.tex"], True, today=date(2026, 9, 30))])
    sessionlog.append(p, [msg("user", "new sitting")])
    assert sessionlog.read_tail(p) == [msg("user", "new sitting")]
    assert sessionlog.last_divider(p) == {
        "date": "2026-09-30", "reason": "idle", "turns": 1,
        "sections": ["sections/intro.tex"], "distilled": True}


def test_torn_line_is_invisible_not_fatal(tmp_path):
    # A crash mid-append leaves a cut line: it reads as absent, and — the
    # real hazard — must not glue onto the next append and poison that too.
    p = tmp_path / "chat.jsonl"
    sessionlog.append(p, [msg("user", "one")])
    p.open("a", encoding="utf-8").write('{"role": "user", "con')  # crash cut it
    sessionlog.append(p, [msg("assistant", "two")])
    assert sessionlog.read_tail(p) == [msg("user", "one"),
                                       msg("assistant", "two")]


def test_repair_drops_a_cut_tool_roundtrip(tmp_path):
    # Crash after the assistant asked for a tool call, before the result:
    # sending the dangling half would make vLLM 400 the resumed session.
    p = tmp_path / "chat.jsonl"
    sessionlog.append(p, [
        msg("user", "read it"),
        msg("assistant", None, tool_calls=[{"id": "t1", "type": "function",
                                            "function": {"name": "x",
                                                         "arguments": "{}"}}]),
    ])
    assert sessionlog.read_tail(p) == [msg("user", "read it")]


def test_repair_drops_orphan_tool_results(tmp_path):
    p = tmp_path / "chat.jsonl"
    sessionlog.append(p, [
        {"role": "tool", "tool_call_id": "gone", "content": "{}"},
        msg("user", "hi"), msg("assistant", "hello"),
    ])
    assert sessionlog.read_tail(p) == [msg("user", "hi"),
                                       msg("assistant", "hello")]


def test_repair_keeps_a_complete_roundtrip(tmp_path):
    p = tmp_path / "chat.jsonl"
    records = [
        msg("user", "read it"),
        msg("assistant", None, tool_calls=[{"id": "t1", "type": "function",
                                            "function": {"name": "x",
                                                         "arguments": "{}"}}]),
        {"role": "tool", "tool_call_id": "t1", "content": '{"ok": 1}'},
        msg("assistant", "it says ok"),
    ]
    sessionlog.append(p, records)
    assert sessionlog.read_tail(p) == records


def test_recap_line_variants():
    assert sessionlog.recap_line(None, 3) == (
        "Previous session ended abruptly — resumed 3 verbatim turns.")
    clean = {"date": "2026-09-30", "reason": "idle", "turns": 4,
             "sections": [], "distilled": True}
    assert sessionlog.recap_line(clean, 0) == (
        "Last sitting ended (idle) on 2026-09-30: 4 turns — distilled.")
    skipped = dict(clean, turns=1, distilled=False)
    assert sessionlog.recap_line(skipped, 0) == (
        "Last sitting ended (idle) on 2026-09-30: 1 turn — "
        "distillation was skipped.")
    assert sessionlog.recap_line(None, 0) == ""  # fresh project


def test_records_are_utf8_verbatim(tmp_path):
    p = tmp_path / "chat.jsonl"
    sessionlog.append(p, [msg("user", "E = mc² — naïve ✓")])
    assert sessionlog.read_tail(p) == [msg("user", "E = mc² — naïve ✓")]
