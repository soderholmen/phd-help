# CONTEXT

Glossary for phd-helper: a personal tool for co-writing research papers with a local-model AI agent.

## Terms

- **Paper project** — a LaTeX paper as a multi-file project (root file + one `.tex` per section). The unit the tool operates on. Not yet started; the tool is general, built for papers that don't exist yet.
- **Section-scoped discussion** — a conversation anchored to a selected section (or paragraph) of the paper. The agent sees that section and its sources, and writes back only into that section's file. The user's manual edits are never clobbered: the agent re-reads before writing.
- **Reference corpus** — the persistent library of reference PDFs the user accumulates across their PhD. Auto-indexed from a drop folder; searchable by the agent (hybrid: keyword + semantic).
- **Dictation** — voice input of prose the agent should write, cleaned before it reaches the agent (disfluencies removed, spoken LaTeX normalized to markup).
- **Conversation mode** — hands-free voice interaction with no push-to-talk: VAD-driven capture, and the agent replies by voice (TTS) as well as on screen.
- **Academic-first search** — web search oriented at scholarly sources (arXiv, Semantic Scholar) with a find-paper → BibTeX → `.bib` loop; general web as secondary.
- **Local model** — the LLM served by the user's own vLLM instance (Qwen3-class, OpenAI-compatible API). No cloud model APIs in the writing path.
