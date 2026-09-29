"""Agent-facing tools (SPEC §2/§4): strict schemas, the validators that feed
client-side tool-call validation, and dispatch against a Project.

Every schema is OpenAI strict-style (additionalProperties false, all fields
required). section_write never writes — it proposes a lint-checked pending
diff that the user approves. Errors return as tool-result text so the model
sees them (the bounce), never the user.
"""

from phd_helper.cascade import Lookup, resolve_bibtex
from phd_helper.project import Project, ProposeError
from phd_helper.search import search_papers

TOOL_SCHEMAS = [
    {"type": "function", "function": {
        "name": "section_read",
        "description": "Read the current content of a paper section file.",
        "parameters": {
            "type": "object",
            "properties": {
                "section": {"type": "string",
                            "description": "Path relative to the project "
                                           "root, e.g. sections/intro.tex"}},
            "required": ["section"],
            "additionalProperties": False}}},
    {"type": "function", "function": {
        "name": "web_search",
        "description": "Search for papers (arXiv title search + OpenAlex "
                       "topics). Returns candidate papers with their ids; "
                       "pick the right one and call cite_add with its "
                       "arxiv or doi.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string",
                          "description": "Words that identify the paper: "
                                         "title, topic, or author and "
                                         "topic"}},
            "required": ["query"],
            "additionalProperties": False}}},
    {"type": "function", "function": {
        "name": "cite_add",
        "description": "Cite a paper in a section: resolves the BibTeX entry "
                       "(arXiv id, DOI, or title), and proposes ONE pending "
                       "diff covering the prose, the \\cite command, and the "
                       "refs.bib entry. Cite the paper in replace with "
                       "\\cite{<the id you passed>}.",
        "parameters": {
            "type": "object",
            "properties": {
                "section": {"type": "string"},
                "find": {"type": "string",
                         "description": "Exact text to replace, quoted from "
                                        "the section"},
                "replace": {"type": "string",
                            "description": "Replacement text containing "
                                           "\\cite{<paper id>}"},
                "arxiv": {"type": "string",
                          "description": "arXiv id, empty if unknown"},
                "doi": {"type": "string", "description": "DOI, empty if "
                                                         "unknown"},
                "title": {"type": "string",
                          "description": "Paper title, empty if an id is "
                                         "given"}},
            "required": ["section", "find", "replace", "arxiv", "doi",
                         "title"],
            "additionalProperties": False}}},
    {"type": "function", "function": {
        "name": "section_write",
        "description": "Propose an anchored find/replace patch to a section. "
                       "The patch becomes a pending diff the user approves; "
                       "find must be an exact quote from the section.",
        "parameters": {
            "type": "object",
            "properties": {
                "section": {"type": "string"},
                "find": {"type": "string",
                         "description": "Exact text to replace, quoted from "
                                        "the section"},
                "replace": {"type": "string",
                            "description": "Replacement text"}},
            "required": ["section", "find", "replace"],
            "additionalProperties": False}}},
]

OFFERED = {s["function"]["name"]: s["function"]["parameters"]["required"]
           for s in TOOL_SCHEMAS}


def make_validators(project: Project) -> dict:
    """Per-tool validators for validate_tool_calls (SPEC §2): the patch's
    find anchor must actually exist in the section."""
    def find_exists(args):
        try:
            text = project.read_section(args["section"])
        except (OSError, KeyError):
            return f"section '{args['section']}' does not exist"
        if args["find"] not in text:
            return (f"find anchor not present in '{args['section']}' — "
                    "read the section and quote it exactly")
        return None
    return {"section_write": find_exists, "cite_add": find_exists}


async def execute_async(call, project: Project, resolve=resolve_bibtex,
                        search=search_papers, fetch=None, mailto: str = "",
                        openalex_mailto: str = "") -> dict:
    """Async dispatch: web_search and cite_add hit the network; the rest
    is sync."""
    if call.name == "web_search":
        if fetch is None and search is search_papers:
            # The real search needs an HTTP fetcher; mis-wiring bounces.
            return {"error": "search unavailable (no HTTP fetcher)"}
        hits = await search(call.args["query"], fetch,
                            mailto=openalex_mailto)
        return {"results": [{"n": i, "title": h.title,
                             "authors": " and ".join(h.authors),
                             "year": h.year, "arxiv": h.arxiv,
                             "doi": h.doi, "venue": h.venue}
                            for i, h in enumerate(hits, 1)],
                "note": ("pass the chosen paper's arxiv or doi to cite_add"
                         if hits else
                         "no candidates — rephrase the query or ask the "
                         "user")}
    if call.name != "cite_add":
        return execute(call, project)
    if fetch is None and resolve is resolve_bibtex:
        # The real cascade needs an HTTP fetcher; a mis-wired caller gets a
        # tool-error bounce, never a TypeError from inside the cascade.
        # (An injected resolver owns its own fetcher, if any.)
        return {"error": "citation lookup unavailable (no HTTP fetcher)"}
    a = call.args
    lookup = Lookup(arxiv=a.get("arxiv", ""), doi=a.get("doi", ""),
                    title=a.get("title", ""))
    result = await resolve(lookup, fetch, mailto)
    if result.entry is None:
        return {"error": f"could not resolve the paper (tried: "
                         f"{', '.join(result.tried)})"}
    try:
        diff = project.propose_cite(a["section"], a["find"], a["replace"],
                                    lookup, result.entry)
    except ProposeError as e:
        return {"error": str(e)}
    except OSError:
        return {"error": f"section '{a['section']}' does not exist"}
    return {"status": "pending", "diff_id": diff.id,
            "section": diff.section_path, "key": diff.cite_key,
            "find": diff.patch.find, "replace": diff.patch.replace,
            "note": "diff (prose + \\cite + bib entry) shown to the user; "
                    "awaiting approval"}


def execute(call, project: Project) -> dict:
    """Run a validated call; return a JSON-serializable tool result."""
    if call.name == "section_read":
        try:
            return {"content": project.read_section(call.args["section"])}
        except OSError:
            return {"error": f"section '{call.args['section']}' "
                             "does not exist"}
    if call.name == "section_write":
        try:
            diff = project.propose_patch(call.args["section"],
                                         call.args["find"],
                                         call.args["replace"])
        except ProposeError as e:
            return {"error": str(e)}
        except OSError:
            return {"error": f"section '{call.args['section']}' "
                             "does not exist"}
        return {"status": "pending", "diff_id": diff.id,
                "section": diff.section_path,
                "note": "diff shown to the user; awaiting approval"}
    return {"error": f"unknown tool '{call.name}'"}
