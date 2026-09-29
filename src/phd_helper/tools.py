"""Agent-facing tools (SPEC §2/§4): strict schemas, the validators that feed
client-side tool-call validation, and dispatch against a Project.

Every schema is OpenAI strict-style (additionalProperties false, all fields
required). section_write never writes — it proposes a lint-checked pending
diff that the user approves. Errors return as tool-result text so the model
sees them (the bounce), never the user.
"""

from phd_helper.project import Project, ProposeError

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
    return {"section_write": find_exists}


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
