"""Agent-facing tools (SPEC §2/§4): strict schemas, the validators that feed
client-side tool-call validation, and dispatch against a Project.

Every schema is OpenAI strict-style (additionalProperties false, all fields
required). section_write never writes — it proposes a lint-checked pending
diff that the user approves. Errors return as tool-result text so the model
sees them (the bounce), never the user.
"""

from phd_helper import gitrepo
from phd_helper.bibtex import BibEntry, parse_bib, same_paper
from phd_helper.cascade import Lookup, resolve_bibtex
from phd_helper.patches import ApplyResult
from phd_helper.project import Project, ProposeError
from phd_helper.search import search_papers

# SPEC §8: embeddings/LanceDB down -> tool errors, and the agent says one
# line and continues via web search — the error text carries that advice.
CORPUS_DOWN = "corpus unavailable — continue with web_search"

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
        "name": "corpus_search",
        "description": "Search the reference corpus (the user's PDF "
                       "library) for passages relevant to a topic. "
                       "Returns ranked chunks with page/block locators and "
                       "doc ids; call corpus_doc with a doc id for its "
                       "abstract and headings.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "k": {"type": "integer",
                      "description": "Number of chunks to return, 1-20"},
                "boost_pinned": {
                    "type": "boolean",
                    "description": "Rank sources pinned to this project "
                                   "first"}},
            "required": ["query", "k", "boost_pinned"],
            "additionalProperties": False}}},
    {"type": "function", "function": {
        "name": "corpus_doc",
        "description": "Look up a corpus document by doc id: title, "
                       "abstract, section headings, and whether it is "
                       "already in the project's refs.bib.",
        "parameters": {
            "type": "object",
            "properties": {
                "doc_id": {"type": "string"}},
            "required": ["doc_id"],
            "additionalProperties": False}}},
    {"type": "function", "function": {
        "name": "section_write",
        "description": "Propose an anchored find/replace patch to a section. "
                       "The patch becomes a pending diff the user approves; "
                       "find must be an exact quote from the section. To "
                       "apply a draft the user just approved, pass "
                       "from_draft true with replace \"\" — never retype the "
                       "draft.",
        "parameters": {
            "type": "object",
            "properties": {
                "section": {"type": "string"},
                "find": {"type": "string",
                         "description": "Exact text to replace, quoted from "
                                        "the section"},
                "replace": {"type": "string",
                            "description": "Replacement text; empty string "
                                           "when from_draft is true"},
                "from_draft": {"type": "boolean",
                               "description": "true: take the replacement "
                                              "from the user-approved "
                                              "draft; then replace must be "
                                              "the empty string"}},
            "required": ["section", "find", "replace", "from_draft"],
            "additionalProperties": False}}},
    {"type": "function", "function": {
        "name": "section_create",
        "description": "Propose a NEW section file: the file's content "
                       "plus its \\input wiring into the root file become "
                       "one pending diff the user approves. The section "
                       "must not exist yet. Use after to place it right "
                       "after an existing section; without after it joins "
                       "the end of the document.",
        "parameters": {
            "type": "object",
            "properties": {
                "section": {"type": "string",
                            "description": "New file path, e.g. "
                                           "sections/method.tex"},
                "content": {"type": "string",
                            "description": "The full file content, LaTeX"},
                "after": {"type": "string",
                          "description": "Existing section path to place "
                                         "it after; empty string to join "
                                         "the end of the document"}},
            "required": ["section", "content", "after"],
            "additionalProperties": False}}},
    {"type": "function", "function": {
        "name": "pending_decide",
        "description": "Resolve a pending diff by the user's spoken "
                       "decision, while the approval window is open: "
                       "decision 'apply' for yes/apply/do it, 'discard' for "
                       "no/discard/forget it. diff_id from the pending note; "
                       "diff_id 'all' resolves every pending diff (apply "
                       "all). Only call this while a diff is pending.",
        "parameters": {
            "type": "object",
            "properties": {
                "diff_id": {"type": "string",
                            "description": "The pending diff's id, or 'all' "
                                           "for every pending diff"},
                "decision": {"type": "string",
                             "description": "'apply' or 'discard'"}},
            "required": ["diff_id", "decision"],
            "additionalProperties": False}}},
    {"type": "function", "function": {
        "name": "memory_write",
        "description": "Replace the paper memory file: the persistent "
                       "decisions, claims, terminology and TODOs that "
                       "must survive across sessions. The current memory "
                       "rides your context — merge this session into it; "
                       "remove only what was explicitly decided away.",
        "parameters": {
            "type": "object",
            "properties": {
                "content": {"type": "string",
                            "description": "The full new memory file, "
                                           "markdown"}},
            "required": ["content"],
            "additionalProperties": False}}},
    {"type": "function", "function": {
        "name": "git_commit",
        "description": "Commit the project's current files to its git "
                       "repo, when the user says to commit or save the "
                       "paper's progress. A direct write — no approval "
                       "card. Pushing is NOT yours: it is an on-screen "
                       "gesture because it publishes outward.",
        "parameters": {
            "type": "object",
            "properties": {
                "message": {"type": "string",
                            "description": "Short commit message in the "
                                           "user's words; the empty "
                                           "string lets the app date "
                                           "it"}},
            "required": ["message"],
            "additionalProperties": False}}},
    {"type": "function", "function": {
        "name": "undo_last",
        "description": "Undo the last change to the paper, whichever "
                       "section it touched, when the user says to undo "
                       "or revert the last edit. A direct write — no "
                       "approval card. It restores section text only: "
                       "a citation entry added with the undone change "
                       "stays in refs.bib.",
        "parameters": {
            "type": "object", "properties": {},
            "required": [], "additionalProperties": False}}},
]

# name -> {required param: json type} — validate_tool_calls checks both
# presence and type, so a {"query": 5} bounces instead of killing the turn.
OFFERED = {s["function"]["name"]: {
    p: s["function"]["parameters"]["properties"][p]["type"]
    for p in s["function"]["parameters"]["required"]}
    for s in TOOL_SCHEMAS}


def make_validators(project: Project, draft: str | None = None) -> dict:
    """Per-tool validators for validate_tool_calls (SPEC §2): the patch's
    find anchor must actually exist in the section. ``draft`` is the
    session's captured draft (stepwise writing): from_draft may only
    reference a draft that exists, and the bounce text teaches the model
    how to get one."""
    def find_exists(args):
        try:
            text = project.read_section(args["section"])
        except (OSError, KeyError):
            return f"section '{args['section']}' does not exist"
        if args["find"] not in text:
            return (f"find anchor not present in '{args['section']}' — "
                    "read the section and quote it exactly")
        return None

    def write_ok(args):
        if args.get("from_draft"):
            if args.get("replace"):
                return ('from_draft and a non-empty replace are mutually '
                        'exclusive — pass replace "" with from_draft')
            if draft is None:
                return ('no captured draft to apply — emit the draft in '
                        'your reply as a fenced ```latex block and get '
                        'the user to approve it first')
        return find_exists(args)

    def not_exists(args):
        # The inverse of find_exists: a create targets a path that must
        # NOT be on disk yet — the propose-side guards do the rest.
        try:
            project.read_section(args["section"])
        except (OSError, KeyError):
            return None
        return (f"section '{args['section']}' already exists — patch it "
                "with section_write instead of creating it")

    def decidable(args):
        if args["decision"] not in ("apply", "discard"):
            return "decision must be 'apply' or 'discard'"
        if args["diff_id"] != "all" and not any(
                d.id == args["diff_id"] for d in project.pending.list_all()):
            return (f"no pending diff '{args['diff_id']}' — check the "
                    "pending note")
        return None
    return {"section_write": write_ok, "cite_add": find_exists,
            "section_create": not_exists, "pending_decide": decidable}


async def execute_async(call, project: Project, resolve=resolve_bibtex,
                        search=search_papers, fetch=None, mailto: str = "",
                        openalex_mailto: str = "", corpus=None,
                        store=None, autojoin=None, on_results=None,
                        window: set[str] | None = None,
                        git_run=None, git_name: str = "phd-helper",
                        git_email: str = "phd-helper@local") -> dict:
    """Async dispatch: web_search and cite_add hit the network, the corpus
    tools hit the index, git_commit spawns git; the rest is sync."""
    if call.name == "git_commit":
        # The user's hand, #28 posture: a direct write, no §5 card.
        # The subprocess makes this leg async like the network tools.
        message = call.args.get("message", "").strip() \
            or gitrepo.default_message("Voice commit")
        try:
            sha = await gitrepo.auto_commit(
                project.root, message, git_name, git_email, run=git_run)
        except RuntimeError as e:
            return {"error": str(e)}
        return {"commit": sha,
                "note": ("the paper is saved in the project's git repo; "
                         "pushing it is an on-screen gesture" if sha else
                         "nothing to commit — the tree is clean")}
    if call.name == "web_search":
        if fetch is None and search is search_papers:
            # The real search needs an HTTP fetcher; mis-wiring bounces.
            return {"error": "search unavailable (no HTTP fetcher)"}
        hits = await search(call.args["query"], fetch,
                            mailto=openalex_mailto)
        if on_results is not None:
            # The related-work panel records the search whole (abstracts
            # and all — the panel shows them; the tool result below stays
            # abstract-free, §4). Best-effort like autojoin: a store fault
            # never sinks the result the model is waiting on.
            try:
                on_results(hits)
            except Exception:
                pass
        if autojoin is not None:
            # §6: papers with an openly downloadable PDF auto-join the
            # corpus (arXiv ids are the open path; paywalled hits get a
            # bib entry only). Best-effort — a dead fetch never sinks the
            # search result the model is waiting on.
            open_hits = [h for h in hits if h.arxiv]
            if open_hits:
                # The hits ride along whole: auto-join needs the title
                # for §6's embed prefix, not just the id.
                try:
                    await autojoin(open_hits)
                except Exception:
                    pass
        return {"results": [{"n": i, "title": h.title,
                             "authors": " and ".join(h.authors),
                             "year": h.year, "arxiv": h.arxiv,
                             "doi": h.doi, "venue": h.venue}
                            for i, h in enumerate(hits, 1)],
                "note": ("pass the chosen paper's arxiv or doi to cite_add"
                         if hits else
                         "no candidates — rephrase the query or ask the "
                         "user")}
    if call.name == "corpus_search":
        return await _corpus_search(call, project, corpus, store)
    if call.name == "corpus_doc":
        return await _corpus_doc(call, project, corpus, store)
    if call.name != "cite_add":
        return execute(call, project, window=window)
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


def execute(call, project: Project, window: set[str] | None = None) -> dict:
    """Run a validated call; return a JSON-serializable tool result.

    ``window`` is the session's approval window (the diff ids the user has
    actually been shown); ``None`` means no session restriction. Decisions
    never reach past the window: applying a diff the user never saw would
    bypass the §5 approval gate."""
    if call.name == "section_read":
        try:
            return {"content": project.read_section(call.args["section"])}
        except OSError:
            return {"error": f"section '{call.args['section']}' "
                             "does not exist"}
    if call.name == "memory_write":
        # Agent state under .phd-helper/, not a paper file: no §5
        # approval gate — the §4 side panel is the user's edit surface.
        project.save_memory(call.args["content"])
        return {"status": "written", "chars": len(call.args["content"])}
    if call.name == "undo_last":
        # SPEC:137: undo lands directly, no approval window. The user's
        # hand, #28 posture. A second undo of the same entry bounces
        # honestly (the inverse patch can't re-anchor what is already
        # undone) — the model says so; there is no redo.
        r = project.undo_last_change()
        if "undone" not in r:
            return {"error": r["reason"]}
        u = r["undone"]
        return {"status": "undone", "section": u["section"],
                "undid": f"{u['find']} → {u['replace']}",
                "note": "the section is back to how it was before that "
                        "change; refs.bib was not touched"}
    if call.name == "pending_decide":
        # §3 approval window: the agent interpreted the utterance against
        # the pending approval; this lands that interpretation through the
        # same apply/reject machinery the on-screen buttons use (§5).
        a = call.args
        if a["decision"] not in ("apply", "discard"):
            return {"error": "decision must be 'apply' or 'discard'"}
        pend = project.pending.list_all()
        if window is not None:
            pend = [d for d in pend if d.id in window]
        if a["diff_id"] == "all":
            targets = pend
        else:
            targets = [d for d in pend if d.id == a["diff_id"]]
            if len(targets) > 1:
                # Pre-global-ids, two sections could share an id; refuse
                # to guess which one the user meant.
                return {"error": f"diff id '{a['diff_id']}' is ambiguous — "
                                 "several pending diffs share it"}
        if not targets:
            return {"error": "no pending diff with that id in the approval "
                             "window — check the pending note"}
        resolutions = []
        for d in targets:
            if a["decision"] == "apply":
                try:
                    r = project.apply_pending(d.section_path, d.id)
                except OSError:
                    # The section file vanished since the diff was
                    # proposed: bounce this one, keep the pass going.
                    r = ApplyResult(applied=False,
                                    reason="the section file is gone")
                resolutions.append(
                    {"diff_id": d.id, "section": d.section_path,
                     "applied": r.applied, "reason": r.reason,
                     "text": r.text})
            else:
                project.reject_pending(d.section_path, d.id)
                resolutions.append(
                    {"diff_id": d.id, "section": d.section_path,
                     "applied": False, "reason": "discarded", "text": None})
        return {"status": "resolved", "resolutions": resolutions}
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
    if call.name == "section_create":
        try:
            diff = project.propose_create(call.args["section"],
                                          call.args["content"],
                                          call.args.get("after") or None)
        except ProposeError as e:
            return {"error": str(e)}
        except OSError:
            return {"error": f"cannot read the root file for "
                             f"section '{call.args['section']}'"}
        return {"status": "pending", "diff_id": diff.id,
                "section": diff.section_path,
                "find": diff.patch.find, "replace": diff.patch.replace,
                "created": {"path": diff.create_path,
                            "content": diff.create_content},
                "note": "new file + wiring shown to the user; awaiting "
                        "approval"}
    return {"error": f"unknown tool '{call.name}'"}


# -- corpus tools (SPEC §6): the store behind the CorpusStore protocol -----

async def _corpus_search(call, project: Project, corpus, store) -> dict:
    if store is None:
        return {"error": CORPUS_DOWN}
    a = call.args
    k = max(1, min(int(a["k"]), 20))
    boost = (set(corpus.pinned_ids(project.root.name))
             if a["boost_pinned"] and corpus is not None else set())
    try:
        hits = await store.search(a["query"], k, boost)
    except Exception:
        return {"error": CORPUS_DOWN}  # §8: faulting index == no index
    # One registry read for the whole page of hits (not one get per hit),
    # and the registry is the source of truth: a chunk whose doc is gone
    # (superseded mid-flight, failed re-fetch) must not surface with a
    # blank title the agent would read as a real paper.
    records = ({d.doc_id: d for d in corpus.list()}
               if corpus is not None else {})
    results = []
    for h in hits:
        rec = records.get(h.doc_id)
        if rec is None:
            continue
        results.append({"n": len(results) + 1, "doc_id": h.doc_id,
                        "title": rec.title, "arxiv": rec.arxiv,
                        "doi": rec.doi,
                        "section": h.section_path,
                        "locator": h.locator, "kind": h.kind,
                        "text": h.text})
    return {"results": results,
            "note": ("corpus_doc(doc_id) for abstract + headings; cite_add "
                     "with the hit's arxiv or doi" if results else
                     "nothing in the corpus — try web_search")}


async def _corpus_doc(call, project: Project, corpus, store) -> dict:
    if store is None:
        return {"error": CORPUS_DOWN}
    doc_id = call.args["doc_id"]
    rec = corpus.get(doc_id) if corpus is not None else None
    if rec is None:
        return {"error": "no such doc_id — run corpus_search first"}
    try:
        info = await store.doc(doc_id)
    except Exception:
        return {"error": CORPUS_DOWN}
    return {"doc_id": doc_id, "title": rec.title, "status": rec.status,
            "abstract": info.abstract if info else "",
            "headings": list(info.headings) if info else [],
            "arxiv": rec.arxiv, "doi": rec.doi, "year": rec.year,
            "bib": _bib_status(project, rec)}


def _bib_status(project: Project, rec) -> str:
    """Is this corpus paper already cited? Identifier match against the
    project's refs.bib — same rule the cite loop uses for key reuse (§6)."""
    if not (rec.arxiv or rec.doi):
        return "unknown (no arXiv/DOI id)"
    probe = BibEntry(key="", type="", fields={"arxiv": rec.arxiv,
                                              "doi": rec.doi})
    for e in parse_bib(project.read_bib()):
        if same_paper(e, probe):
            return f"already in refs.bib as {e.key}"
    return "not in refs.bib"
