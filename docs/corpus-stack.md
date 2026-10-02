# Corpus stack: sidecar topology and ops runbook

## Why a sidecar

SPEC §6 puts the harrier embedder and the Qwen reranker in the backend
process. That is not possible on this box: Smart App Control blocks
`torch._C` in the main venv (Python 3.14), and sentence-transformers
wants 3.12 anyway. So both models live in one py3.12 sidecar process as
plain HTTP — the same MinerU/audio convention: ops starts the sidecar,
the backend holds thin sync clients (`server/corpus_sidecar.py`), and a
dead sidecar degrades per §8. Sync is honest here, not a shortcut:
LanceStore calls both seams exclusively through `asyncio.to_thread`, so
the clients are true drop-ins for `HarrierEmbedder`/`QwenReranker` —
which stay in-tree for the 3090 prod slice, where torch loads again.

The wire carries raw rerank margins; the client sigmoids them (the
probability contract, and the §6 boost's positive-scale requirement,
live with the tested adapter). `normalize_embeddings` stays
server-side — LanceDB's L2-as-cosine depends on it, and no client may
opt out.

## One-time setup

```powershell
# torch first, from the cu128 index (same build the ASR venv uses), so
# sentence-transformers never pulls a mismatched default wheel
py -3.12 -m venv .venv-embed
.venv-embed/Scripts/pip.exe install "torch==2.9.1+cu128" `
    --index-url https://download.pytorch.org/whl/cu128
.venv-embed/Scripts/pip.exe install sentence-transformers fastapi uvicorn
```

No model download: the HF cache is already populated (both models were
pulled by the earlier main-venv probe runs).

## Start / stop

```powershell
.venv-embed/Scripts/python.exe scripts/corpus_server.py --port 8091
# backend
set PHD_CORPUS_STACK=local
```

Both models load lazily and independently — the sidecar answers
`/health` immediately with `loaded: {embed: false, rerank: false}`, and
the first search/index pays the load inside the client's 120 s embed
timeout (60 s rerank).

**Honest `/health` semantics.** The backend's `corpus` chip is the
store probe AND the embedder probe (counter first, then a live probe
offloaded via `to_thread`): the store's own probe deliberately never
encodes, so without the embedder half a dead model would hide behind an
ok chip while every search CORPUS_DOWNs. A dead *reranker* does not
flip the chip — that is by design: rerank-down degrades to RRF order,
which is a working search, and the shell renders only `vllm` +
`corpus`. Rerank status is visible at the sidecar's own `/health`.

**VRAM budget (16.3 GB, approximate):** MOSS ~11.5 + parakeet bf16
~2.5 + corpus (both models fp16) ~2.4, plus a per-process CUDA context
each — all of it does not fit. The lazy loads are what make this
workable: the corpus models are resident only after the first
search/index of the process's life. Voice session + fresh search at the
same moment can OOM the sidecar; that surfaces as a 500 → faulted
counter → `CORPUS_DOWN`, never as a silent lie. `PHD_CORPUS_DTYPE=fp32`
exists as a drift escape hatch, not a coexistence knob.

**Ingest while down:** docs are marked `failed` and re-driven on
backend restart (ingest.py); the registry shows them visibly the whole
time (§6, "failures are visible, not rot").

**Reindex caveat:** the existing LanceDB table is 1024-d (harrier-
0.6b). Changing `PHD_EMBED_MODEL` changes the vector width — the table
must be rebuilt, not just repointed.

## Verify

```powershell
.venv/Scripts/python.exe scripts/smoke_corpus.py
```

Checks `/health` loaded flags, one embed (dim 1024, ‖v‖≈1.0), and one
rerank (three docs, the answer must rank first). Exit 0 only if all
three legs worked.

## Measured on the 5070 Ti (live smoke, 2026-10-01)

| component | number |
| --- | --- |
| embed, first call (lazy load) | 28.6 s (one sentence) |
| embed, warm | 0.1 s, dim 1024, ‖v‖ 1.0001 |
| rerank, first call (lazy load, torch warm) | 3.2 s (3 docs) |
| rerank, warm | 0.1 s (3 docs; margins +8.2 / −9.0 / −8.4) |
| real search, existing 1024-d table | hybrid + rerank, top hit P=0.999, no reindex |

## Deferred (seams intact)

3090 prod slice (in-process classes, or the sidecar serving the 3090
box); a shared sidecar base class once a third family exists; shell-
side rerank-status display; float32-binary `/embed` bodies if ingest
JSON ever proves slow.
