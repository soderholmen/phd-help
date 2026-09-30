"""Environment-driven config. Keys come from the environment first, then the
gitignored local key store (SPEC §6: keys never in the repo).
"""

import os
from dataclasses import dataclass, field
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
KEY_STORE = REPO_ROOT / "keys.txt"

# SPEC §2 agentic sampling (thinking mode) — never starve thinking.
THINKING_SAMPLING = {"temperature": 1.0, "top_p": 0.95, "top_k": 20,
                     "min_p": 0.0, "presence_penalty": 0.0,
                     "repetition_penalty": 1.0}
# SPEC §2 non-thinking mode.
PLAIN_SAMPLING = {"temperature": 0.7, "top_p": 0.8, "presence_penalty": 1.5}


def _read_key_store(name):
    try:
        for line in KEY_STORE.read_text(encoding="utf-8").splitlines():
            key, _, value = line.partition(":")
            if key.strip().lower() == name.lower() and value.strip():
                return value.strip()
    except OSError:
        pass
    return None


@dataclass
class Config:
    llm_base_url: str = os.environ.get(
        "PHD_LLM_BASE_URL", "http://10.147.242.50:8888")
    llm_model: str = os.environ.get("PHD_LLM_MODEL", "qwen3.8-flash-next")
    llm_api_key: str | None = field(
        default_factory=lambda: os.environ.get("PHD_LLM_API_KEY")
        or _read_key_store("vLLM"))
    # SPEC §3 endpointing hangovers (config settings).
    conversation_hangover_s: float = 0.6
    dictation_hangover_s: float = 1.0
    # SPEC §8 heartbeat cadence.
    ping_interval_s: float = 2.0
    lease_timeout_s: float = 60.0
    # SPEC §6 polite pools (gitignored env, never committed).
    crossref_mailto: str = os.environ.get("PHD_CROSSREF_MAILTO", "")
    openalex_mailto: str = os.environ.get("PHD_OPENALEX_MAILTO", "")
    # Where the heavy corpus stack runs: "off" keeps the §8 degraded path
    # (store down, docs visibly queued); "local" wires MinerU + harrier +
    # LanceDB on this machine (dev box today, the 3090 later — the server
    # stays prod, this only picks where the adapters live).
    corpus_stack: str = os.environ.get("PHD_CORPUS_STACK", "off")

    def sampling(self, thinking: bool) -> dict:
        return THINKING_SAMPLING if thinking else PLAIN_SAMPLING


def load() -> Config:
    return Config()
