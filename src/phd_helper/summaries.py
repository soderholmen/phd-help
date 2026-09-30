"""Per-section rolling summaries (SPEC §4/§7): the residue of
conversation that dropped out of the verbatim window, grouped by the
section the discussion was anchored to and persisted with dated session
dividers. Pure over the conversation log and a
``{path: [{"date": …, "text": …}, …]}`` store — the LLM that writes the
lines rides the server's seam. The latest entry is the live summary;
older entries are the dated dividers the user can inspect.
"""

from phd_helper.context import group_exchanges


def by_section(conversation: list[dict]) -> dict[str, str]:
    """Transcript lines per anchored section. Exchanges taken with no
    section selected carry no summary: nobody asked about a section, so
    there is no section to roll the discussion into."""
    groups: dict[str, list[str]] = {}
    for _, msgs in group_exchanges(conversation):
        section = next((m.get("section") or "" for m in msgs
                        if m.get("role") == "user"), "")
        lines = "\n".join(f"{m['role']}: {m['content']}" for m in msgs
                          if m.get("content"))
        if section and lines.strip():
            groups.setdefault(section, []).append(lines)
    return {sec: "\n".join(parts) for sec, parts in groups.items()}


def render(summaries: dict) -> str:
    """The rolling-summary block for context assembly: one line per
    section that has one, newest entry wins."""
    lines = [f"- {p}: {entries[-1]['text']}"
             for p, entries in summaries.items()
             if entries and entries[-1].get("text")]
    return ("Rolling summaries of past sessions:\n" + "\n".join(lines)
            if lines else "")
