"""Live proof for the corpus sidecar — stdlib + httpx only, so it runs
in the main venv without touching torch (docs/corpus-stack.md).

With the sidecar up:
  .venv/Scripts/python.exe scripts/smoke_corpus.py
First calls pay the lazy model loads (tens of seconds); the loaded
flags in /health tell cold from warm. Nonzero exit on any failure.
"""

import math
import os
import sys
import time

import httpx

URL = os.environ.get("PHD_CORPUS_URL", "http://127.0.0.1:8091")


def main() -> bool:
    with httpx.Client(base_url=URL, timeout=180.0) as c:
        h = c.get("/health")
        h.raise_for_status()
        body = h.json()
        print(f"health: {body['status']}, dtype {body['dtype']}, "
              f"loaded {body['loaded']}")

        t0 = time.time()
        r = c.post("/embed",
                   json={"texts": ["Scaled dot-product attention."]})
        r.raise_for_status()
        v = r.json()["vectors"][0]
        norm = math.sqrt(sum(x * x for x in v))
        print(f"embed: dim {len(v)}, |v| {norm:.4f}, "
              f"{time.time() - t0:.1f} s (first call pays the load)")
        # 1024 is the contract with the existing LanceDB table: a
        # different-dim model means a reindex, not a smoke pass.
        if len(v) != 1024 or abs(norm - 1.0) > 0.01:
            print("EMBED: wrong dim (table is 1024) or unnormalized")
            return False

        query = "how is attention computed?"
        docs = ["Attention weights are computed by scaled dot product.",
                "Parakeets mimic human speech with their syrinx.",
                "The transformer stacks identical decoder layers."]
        t0 = time.time()
        r = c.post("/rerank", json={"query": query, "docs": docs})
        r.raise_for_status()
        margins = r.json()["scores"]
        if len(margins) != len(docs):
            print(f"RERANK: {len(margins)} scores for {len(docs)} docs")
            return False
        probs = [1.0 / (1.0 + math.exp(-m)) for m in margins]
        best = max(range(len(probs)), key=lambda i: probs[i])
        print(f"rerank: margins {[round(m, 2) for m in margins]} in "
              f"{time.time() - t0:.1f} s, best doc {best}")
        if best != 0:
            print("RERANK: the query did not rank its own answer first")
            return False
    return True


if __name__ == "__main__":
    try:
        ok = main()
    except httpx.HTTPError as e:
        print(f"sidecar unreachable: {e}")
        ok = False
    sys.exit(0 if ok else 1)
