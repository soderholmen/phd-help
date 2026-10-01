"""Verbatim conversation history (SPEC §7): one JSONL per project under
``.phd-helper/``, the durable spine of a paper's sessions. Messages ride
as the exact dicts the LLM saw (plus a ``ts``); a dated divider marks a
sitting that ended and was distilled. Reopening resumes everything after
the last divider verbatim — older talk rides on as rolling summaries.

Pure over the file; the server owns when lines are written. A torn line
(a crash mid-append) reads as absent, never fatal (§8).
"""

import json
from datetime import date
from pathlib import Path


def append(path: Path, records) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    prefix = ""
    try:
        with path.open("rb") as f:
            f.seek(-1, 2)  # last byte
            if f.read(1) != b"\n":
                prefix = "\n"  # torn tail (crash mid-append): close it off,
    except OSError:           # else the next line glues on and poisons it too
        pass                  # missing or empty file: nothing to close
    with path.open("a", encoding="utf-8") as f:
        if prefix:
            f.write(prefix)
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def _records(path: Path) -> list[dict]:
    try:
        raw = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return []
    out = []
    for line in raw:
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue  # torn line: invisible, recoverable by hand (§8)
        if isinstance(rec, dict):
            out.append(rec)
    return out


def is_divider(record: dict) -> bool:
    return "divider" in record


def divider_record(reason: str, turns: int, sections, distilled: bool,
                   today: date | None = None) -> dict:
    return {"divider": {
        "date": (today or date.today()).isoformat(),
        "reason": reason, "turns": turns,
        "sections": sorted(sections), "distilled": distilled}}


def last_divider(path: Path) -> dict | None:
    for rec in reversed(_records(path)):
        if is_divider(rec):
            return rec["divider"]
    return None


def repair_tail(records: list[dict]) -> list[dict]:
    """A crash can cut an exchange mid tool round-trip: an assistant
    tool_calls message whose results never landed, or a tool result whose
    call fell out of the window. vLLM 400s on either half, so the resume
    drops dangling halves — an assistant message is one record, so a
    half-cut one loses its prose with its calls; the exchange before and
    after survives."""
    results = {r.get("tool_call_id") for r in records
               if r.get("role") == "tool"}
    drop = {i for i, r in enumerate(records)
            if r.get("role") == "assistant" and r.get("tool_calls")
            and not all(c.get("id") in results for c in r["tool_calls"])}
    kept_calls = {c.get("id")
                  for i, r in enumerate(records) if i not in drop
                  and r.get("role") == "assistant" and r.get("tool_calls")
                  for c in r["tool_calls"]}
    return [r for i, r in enumerate(records)
            if i not in drop
            and not (r.get("role") == "tool"
                     and r.get("tool_call_id") not in kept_calls)]


def read_tail(path: Path) -> list[dict]:
    """The verbatim messages after the last *distilled* divider, tool
    pairing repaired — what a reopened sitting resumes with. A divider
    whose distillation was skipped does NOT close the window: the talk
    stays verbatim so the next sitting's end retries the distillation
    (the promise the distill-fault comments make)."""
    records = _records(path)
    start = 0
    for i, rec in enumerate(records):
        if is_divider(rec) and rec["divider"].get("distilled"):
            start = i + 1
    return repair_tail([r for r in records[start:] if not is_divider(r)])


def recap_line(divider: dict | None, resumed: int) -> str:
    """The one-line on-screen recap of the next open (§7). Messages left
    after the last distilled divider mean either the §8 abrupt end
    (resumed verbatim) or a clean end whose distillation was skipped —
    the divider says which. A fresh project recaps nothing."""
    if resumed:
        if divider is not None and not divider.get("distilled"):
            return (f"Last sitting's distillation was skipped — resumed "
                    f"{resumed} verbatim turns to retry it.")
        return (f"Previous session ended abruptly — resumed {resumed} "
                "verbatim turns.")
    if divider is None:
        return ""
    turns = divider.get("turns", 0)
    line = (f"Last sitting ended ({divider.get('reason', '?')}) on "
            f"{divider.get('date', '?')}: {turns} "
            f"turn{'s' if turns != 1 else ''}")
    return line + (" — distilled." if divider.get("distilled")
                   else " — distillation was skipped.")
