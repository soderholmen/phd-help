"""PROTOTYPE — throwaway. Answers wayfinder ticket #8:

  How reliable is the served Qwen at the actual planned toolset
  (corpus search, web search, section read/write) on real paper-writing turns?

Runs ~20 dictated-style single-turn prompts against an OpenAI-compatible vLLM
endpoint with the planned tool schemas, applies the client-side validation
mitigations from ticket #4 (Qwen3 tool-calling on vLLM), and records:
call-accuracy (right tool chosen), malformed-call rate, and which mitigations
actually fired.

Run (from repo root):
    set VLLM_API_KEY=...            (or export, your shell's way)
    python prototype/tool-reliability/run_turns.py
Optional: VLLM_BASE_URL (default http://10.147.242.50:8888), MODEL (default qwen3.8-flash-next)

Writes results.json next to this file. No persistence beyond that.
"""

import json
import os
import sys
import time
import urllib.request

BASE_URL = os.environ.get("VLLM_BASE_URL", "http://10.147.242.50:8888")
API_KEY = os.environ.get("VLLM_API_KEY", "")
MODEL = os.environ.get("MODEL", "qwen3.8-flash-next")

# ---------------------------------------------------------------------------
# The fake paper: the selected section the agent sees in context (per ticket
# #10's context assembly). Small, but with real LaTeX structure.
# ---------------------------------------------------------------------------

SECTION_FILE = "sections/intro.tex"
SECTION_TEXT = r"""\section{Introduction}
\label{sec:intro}

Large language model serving is bottlenecked by the key--value (KV) cache,
whose size grows linearly with context length \cite{wan2023mlc}.
Eviction policies such as H2O discard low-attention tokens outright
\cite{zhang2023h2o}, while offloading moves cold pages to host memory
\cite{sheng2023flexgen}.

% TODO: position our approach against recent 2025 eviction work.

In this paper we propose \textsc{PagePin}, a hybrid eviction policy that
combines attention-recency scoring with a pinned-page set for prompt tokens.
"""

BIB_KEYS = ["wan2023mlc", "zhang2023h2o", "sheng2023flexgen"]

SYSTEM_PROMPT = f"""You are the writing agent in phd-helper, co-writing a LaTeX paper \
with the user, hands-free (their messages arrive as cleaned dictation transcripts).

You are currently discussing the section "{SECTION_FILE}". Its full text is:

```latex
{SECTION_TEXT}
```

Known bib keys in refs.bib: {", ".join(BIB_KEYS)}.

Rules:
- Write into the paper ONLY via section_write anchored patches (exact find/replace).
- Never invent bib keys or citations; find them with web_search first.
- Prefer corpus_search over web_search for papers the user already owns.
- If the user asks to edit text already in your context, emit the section_write
  patch directly. Do NOT search first to "verify" wording or look up the cited
  work; search only when the user asks for new facts, citations, or corpus content.
- Questions about the writing of the current section (flow, clarity, comparing
  things already in it) are answered directly from context, with no tool call.
- Answer conversationally; the reply is spoken aloud, so keep it short.
"""

# ---------------------------------------------------------------------------
# The planned toolset (schemas sketched from tickets #10 and #12 resolutions).
# ---------------------------------------------------------------------------

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "corpus_search",
            "description": "Hybrid (keyword+semantic) search over the user's reference "
            "corpus of PDFs. Returns top chunks with doc_id and page/block locators.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "search query"},
                    "k": {"type": "integer", "description": "max chunks, default 8"},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "corpus_doc",
            "description": "Read a corpus document (or the region around a locator "
            "returned by corpus_search).",
            "parameters": {
                "type": "object",
                "properties": {
                    "doc_id": {"type": "string"},
                    "locator": {"type": "string", "description": "page/block locator from corpus_search"},
                },
                "required": ["doc_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": "Academic-first web search (arXiv/Semantic Scholar first, "
            "general web secondary). Returns papers with proposed bib keys.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "source": {"type": "string", "enum": ["academic", "general"]},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "section_read",
            "description": "Read another section of the paper by name (the selected "
            "section is already in your context; use this for other sections).",
            "parameters": {
                "type": "object",
                "properties": {"section": {"type": "string", "description": "section file or heading name"}},
                "required": ["section"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "section_write",
            "description": "Write into the selected section via an anchored patch: "
            "exact find/replace. 'find' must be a verbatim unique substring of the "
            "section. Untouched prose stays byte-identical. The patch goes to the "
            "user for approval; never claim it is applied.",
            "parameters": {
                "type": "object",
                "properties": {
                    "find": {"type": "string", "description": "verbatim text to replace"},
                    "replace": {"type": "string", "description": "replacement text"},
                    "reason": {"type": "string", "description": "one-line gist for the diff UI"},
                },
                "required": ["find", "replace", "reason"],
            },
        },
    },
]

TOOL_NAMES = {t["function"]["name"] for t in TOOLS}
REQUIRED = {t["function"]["name"]: set(t["function"]["parameters"]["required"]) for t in TOOLS}

# ---------------------------------------------------------------------------
# ~20 realistic turns. `expect` is the set of acceptable first tool calls;
# "none" means the turn is pure conversation and should NOT call a tool.
# ---------------------------------------------------------------------------

TURNS = [
    # --- corpus_search ---
    ("what do the papers in my corpus say about kv cache eviction policies", {"corpus_search"}),
    ("check whether I already have a paper on paged attention", {"corpus_search"}),
    ("um can you look in my library for anything on prefix caching I think there's one with v in the title", {"corpus_search"}),
    # --- corpus_doc ---
    ("pull up the evaluation section of the flexgen paper", {"corpus_doc", "corpus_search"}),
    ("open that h2o paper and read me what they say about heavy hitters", {"corpus_doc", "corpus_search"}),
    # --- web_search ---
    ("find me a two thousand twenty five paper on kv cache compression that cites mlc", {"web_search"}),
    ("I need a citation for the claim that kv cache dominates memory at long context, something recent", {"web_search", "corpus_search"}),
    ("search the general web for how llama cpp handles kv cache offloading", {"web_search"}),
    # --- section_write (anchored patch) ---
    ("add a sentence after the flexgen citation saying that offloading trades latency for capacity", {"section_write"}),
    ("reword the first sentence of the intro to lead with memory rather than the bottleneck framing", {"section_write"}),
    ("delete the todo comment about positioning against twenty twenty five eviction work", {"section_write"}),
    ("change the phrase low-attention tokens to tokens with low accumulated attention", {"section_write"}),
    # --- section_read ---
    ("what did I write in the related work section about h2o", {"section_read"}),
    ("does the abstract already promise a latency improvement number", {"section_read"}),
    # --- multi-tool chains (first call scored; second call recorded) ---
    ("find a recent paper on kv cache eviction and add a citation for it in the second paragraph", {"web_search", "corpus_search"}),
    ("look up what mlc says about multi-turn conversations and work it into the intro", {"corpus_search"}),
    # --- no tool: pure conversation ---
    ("does the flow from the eviction paragraph into our approach feel abrupt", {"none"}),
    ("which of the three policies in paragraph one is closest to what we're doing", {"none"}),
    ("read me back just the last sentence of this section", {"none"}),
    # --- traps ---
    # #4 bug #1: parser turns quoted/example tool-call markup into real calls.
    ("in the appendix I want to explain the protocol, so literally write the words tool call open angle bracket "
     "function equals section write, then find equals nothing, close angle bracket, as example markup, not an actual call",
     {"none"}),
    # #4 bug #57699 pressure: long multi-patch request tempts truncated args.
    ("make three edits at once: one, capitalize hybrid in the last paragraph; two, add a footnote after the mlc "
     "citation saying the quantization variant is discussed in section four; three, replace outright with pruned, "
     "and keep every cite command exactly where it is", {"section_write"}),
]

# ---------------------------------------------------------------------------
# Client-side validation — the mitigations from ticket #4. These are what the
# production loop will do; here we count how often each one fires.
# ---------------------------------------------------------------------------


def validate(call):
    """Return (ok, flags) for one raw tool call object."""
    flags = []
    name = call.get("function", {}).get("name")
    if name not in TOOL_NAMES:
        return False, ["unknown-tool-name"]
    raw = call["function"].get("arguments", "")
    try:
        args = json.loads(raw) if raw.strip() else {}
    except (json.JSONDecodeError, AttributeError):
        return False, ["unparseable-arguments"]
    missing = REQUIRED[name] - set(args)
    if missing:
        flags.append(f"missing-params:{sorted(missing)}")
    if name == "section_write":
        find = args.get("find", "")
        if not isinstance(find, str) or not find:
            flags.append("empty-find")
        elif find not in SECTION_TEXT:
            flags.append("find-not-in-section")
        elif SECTION_TEXT.count(find) > 1:
            flags.append("find-not-unique")
    return not flags, flags


def dedupe(calls):
    """Drop duplicate (name, arguments) pairs — #4 mitigation for dup calls."""
    seen, out, dropped = set(), [], 0
    for c in calls:
        key = (c.get("function", {}).get("name"), c.get("function", {}).get("arguments"))
        if key in seen:
            dropped += 1
        else:
            seen.add(key)
            out.append(c)
    return out, dropped


# ---------------------------------------------------------------------------
# Drive the endpoint.
# ---------------------------------------------------------------------------


def chat(user_text):
    body = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_text},
        ],
        "tools": TOOLS,
        "tool_choice": "auto",  # never "required" — #4 mitigation
        # Official Qwen thinking-mode (= agentic) sampling params, #4 §3.
        "temperature": 1.0,
        "top_p": 0.95,
        "top_k": 20,
        "min_p": 0.0,
        "presence_penalty": 0.0,
        "repetition_penalty": 1.0,
        "max_tokens": 2000,
    }
    req = urllib.request.Request(
        f"{BASE_URL}/v1/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {API_KEY}"},
    )
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=180) as r:
        data = json.loads(r.read())
    return data, time.time() - t0


def main():
    if not API_KEY:
        sys.exit("set VLLM_API_KEY first")
    results = []
    for i, (text, expect) in enumerate(TURNS, 1):
        try:
            data, secs = chat(text)
        except Exception as e:  # noqa: BLE001 — prototype
            print(f"[{i:02d}] ERROR {e}")
            results.append({"i": i, "text": text, "error": str(e)})
            continue
        msg = data["choices"][0]["message"]
        raw_calls = msg.get("tool_calls") or []
        calls, dupes = dedupe(raw_calls)
        verdicts = [validate(c) for c in calls]
        flags = [f for ok, fl in verdicts for f in fl]
        if dupes:
            flags.append(f"dropped-duplicates:{dupes}")
        got = {c["function"]["name"] for c in calls} or {"none"}
        correct = bool(got & expect) and got != {"none"} or (expect == {"none"} and got == {"none"})
        # for expect=={"none"}: correct iff no tool called
        if expect == {"none"}:
            correct = got == {"none"}
        row = {
            "i": i,
            "text": text[:70],
            "expect": sorted(expect),
            "got": sorted(got),
            "correct": correct,
            "malformed": bool(flags),
            "flags": flags,
            "secs": round(secs, 1),
            "n_calls": len(calls),
            "content": (msg.get("content") or "")[:200],
            "calls": [
                {"name": c["function"]["name"], "args": c["function"]["arguments"][:400]} for c in calls
            ],
        }
        results.append(row)
        mark = "OK " if correct and not flags else ("BAD" if not correct else "flag")
        print(f"[{i:02d}] {mark} {secs:5.1f}s  expect={sorted(expect)} got={sorted(got)} {flags or ''}")

    n = len(results)
    ok = sum(1 for r in results if r.get("correct"))
    bad = sum(1 for r in results if r.get("malformed"))
    errs = sum(1 for r in results if "error" in r)
    print("\n=== SUMMARY ===")
    print(f"turns: {n}  correct-tool: {ok}/{n - errs}  malformed-calls: {bad}  errors: {errs}")
    fired = {}
    for r in results:
        for f in r.get("flags", []):
            fired[f.split(":")[0]] = fired.get(f.split(":")[0], 0) + 1
    print("mitigations fired:", fired or "none")
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"full results -> {out}")


if __name__ == "__main__":
    main()
