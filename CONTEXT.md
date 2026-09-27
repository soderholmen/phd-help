# CONTEXT

Glossary for phd-helper: a personal tool for co-writing research papers with a local-model AI agent.

## Terms

- **Paper project** — a LaTeX paper as a multi-file project (root file + one `.tex` per section). The unit the tool operates on. Not yet started; the tool is general, built for papers that don't exist yet.
- **Section-scoped discussion** — a conversation anchored to a selected section (or paragraph) of the paper. The agent sees that section and its sources, and writes back only into that section's file. The user's manual edits are never clobbered: the agent re-reads before writing.
- **Reference corpus** — the persistent library of reference PDFs the user accumulates across their PhD. Auto-indexed from a drop folder; searchable by the agent (hybrid: keyword + semantic).
- **Dictation** — voice input of prose the agent should write, cleaned before it reaches the agent (disfluencies removed, spoken LaTeX normalized to markup).
- **Conversation mode** — hands-free voice interaction, armed only by an explicit user toggle (never push-to-talk, never auto-armed): once on, VAD-driven capture, and the agent replies by voice (TTS) as well as on screen.
- **Barge-in** — user speech that interrupts the agent while it speaks; it stops the agent's voice output immediately.
- **Voice endpoint** — the device that currently holds conversation mode: the only client allowed to capture and play audio. All other clients are view-only; arming the toggle elsewhere hands the endpoint over explicitly.
- **Backchannel** — a short spoken filler the agent emits when an operation will take seconds ("let me check your corpus"); signals liveness without narrating content.
- **Approval window** — the span while a proposed diff is pending, during which the user's next utterance is interpreted against that approval (approve / reject / amend) rather than as ordinary conversation.
- **Academic-first search** — web search oriented at scholarly sources (arXiv, Semantic Scholar) with a find-paper → BibTeX → `.bib` loop; general web as secondary.
- **Paper skeleton** — the whole-paper view the agent sees alongside the selected section: title, abstract, all section headings, and a one-line gist per section. Gists regenerate whenever a section file changes.
- **Paper memory** — the small persistent list that survives across sections: decisions made, claims to keep consistent, terminology, open TODOs. Distilled at session end plus on voice command; user-editable.
- **Pinned source** — a reference PDF attached to a section; always contributes its abstract and headings to that section's context, and is chunk-searched for detail.
- **Anchored patch** — the default write into a section: an exact find/replace whose untouched prose stays byte-identical. Full-section rewrite happens only on explicit user command.
- **Cleanup pass** — the dedicated fast transformation of raw dictation transcript into clean instruction/prose before the agent's turn; the cleaned text is what enters conversation history.
- **Local model** — the LLM served by the user's own vLLM instance (Qwen3-class, OpenAI-compatible API). No cloud model APIs in the writing path.
