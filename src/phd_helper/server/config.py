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


def _context_budget() -> int:
    # Parsed at construction, not import: a typo must fail Config() with
    # an actionable message, never crash every importer of this module.
    raw = os.environ.get("PHD_CONTEXT_BUDGET", "8000")
    try:
        return int(raw)
    except ValueError:
        raise ValueError(
            f"PHD_CONTEXT_BUDGET must be an integer, got {raw!r}") from None


def _session_idle() -> float:
    # SPEC §7: ~30 min idle ends a sitting. Parsed at construction, like
    # the budget — a typo must fail Config(), not every importer.
    raw = os.environ.get("PHD_SESSION_IDLE_S", "1800")
    try:
        return float(raw)
    except ValueError:
        raise ValueError(
            f"PHD_SESSION_IDLE_S must be a number, got {raw!r}") from None


def _vad_threshold() -> float:
    # Energy-VAD speech floor (RMS of 16 kHz mono float). Parsed at
    # construction like the other knobs — a typo fails Config(), not the
    # import.
    raw = os.environ.get("PHD_VAD_THRESHOLD", "0.012")
    try:
        return float(raw)
    except ValueError:
        raise ValueError(
            f"PHD_VAD_THRESHOLD must be a number, got {raw!r}") from None


def _flag(name: str, default: str) -> bool:
    # Parsed at construction like the numeric knobs: a typo must fail
    # Config() with an actionable message, never silently flip a switch.
    raw = os.environ.get(name, default)
    if raw in ("0", "1"):
        return raw == "1"
    raise ValueError(f"{name} must be '0' or '1', got {raw!r}")


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
    # SPEC §3: blips under this never become finals (utterance floor).
    min_utterance_s: float = 0.25
    # SPEC §8 heartbeat cadence.
    ping_interval_s: float = 2.0
    lease_timeout_s: float = 60.0
    # SPEC §7: a sitting ends after this much idle (no user turn).
    session_idle_s: float = field(default_factory=_session_idle)
    # SPEC §4 per-turn assembly budget (approx tokens). A cost/focus knob
    # for the assembly — window-agnostic priority, not a window cap; the
    # current exchange's tool round-trips ride outside it (trimming them
    # mid-loop would orphan tool results from their calls).
    context_budget_tokens: int = field(default_factory=_context_budget)
    # SPEC §6 polite pools (gitignored env, never committed).
    crossref_mailto: str = os.environ.get("PHD_CROSSREF_MAILTO", "")
    openalex_mailto: str = os.environ.get("PHD_OPENALEX_MAILTO", "")
    # Where the heavy corpus stack runs: "off" keeps the §8 degraded path
    # (store down, docs visibly queued); "local" wires MinerU + LanceDB
    # here and the MODELS (harrier embed, Qwen rerank) over the corpus
    # sidecar — SAC blocks torch in this venv (docs/corpus-stack.md). The
    # in-process HarrierEmbedder/QwenReranker stay in-tree for the 3090
    # prod slice, where torch loads again.
    corpus_stack: str = os.environ.get("PHD_CORPUS_STACK", "off")
    corpus_url: str = field(
        default_factory=lambda: os.environ.get(
            "PHD_CORPUS_URL", "http://127.0.0.1:8091"))
    # Where the voice stack runs: "off" keeps the stubs (§8 honest
    # silence); "local" talks to the STT/TTS sidecars — separate py3.12
    # processes, because SAC blocks torch in this venv and NeMo needs 3.12
    # (docs/audio-stack.md). URLs are read at construction so a restarted
    # process (or a test) can repoint them.
    audio_stack: str = os.environ.get("PHD_AUDIO_STACK", "off")
    stt_url: str = field(
        default_factory=lambda: os.environ.get(
            "PHD_STT_URL", "http://127.0.0.1:8090"))
    tts_url: str = field(
        default_factory=lambda: os.environ.get(
            "PHD_TTS_URL", "http://127.0.0.1:8083"))
    tts_prompt_wav: str = field(
        default_factory=lambda: os.environ.get(
            "PHD_TTS_PROMPT_WAV",
            str(REPO_ROOT / ".probe" / "MOSS-TTS" / "assets" / "audio"
                / "reference_en_0.mp3")))
    vad_threshold: float = field(default_factory=_vad_threshold)
    # Sentence-level TTS off the token stream (docs/audio-stack.md). The
    # kill switch exists because vLLM SSE × thinking mode × tool-call
    # fragmentation is live-risk and only partially unit-testable: "0"
    # falls back to one-shot chat() + one push, same audio episode.
    stream_tts: bool = field(
        default_factory=lambda: _flag("PHD_STREAM_TTS", "1"))

    def __post_init__(self):
        # A typo'd stack must not silently degrade to "off" (§8: state is
        # known at startup, not discovered mid-turn).
        for name, value in (("PHD_CORPUS_STACK", self.corpus_stack),
                            ("PHD_AUDIO_STACK", self.audio_stack)):
            if value not in ("off", "local"):
                raise ValueError(
                    f"{name} must be 'off' or 'local', got {value!r}")

    def sampling(self, thinking: bool) -> dict:
        return THINKING_SAMPLING if thinking else PLAIN_SAMPLING


def load() -> Config:
    return Config()
