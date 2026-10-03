"""Config validation: a typo'd corpus stack must fail at startup, not
silently degrade to the §8 off path (state is known, not discovered)."""

import pytest

from phd_helper.server.config import Config


def test_valid_corpus_stack_values_pass():
    assert Config(corpus_stack="off").corpus_stack == "off"
    assert Config(corpus_stack="local").corpus_stack == "local"


def test_typo_corpus_stack_fails_fast():
    with pytest.raises(ValueError, match="PHD_CORPUS_STACK"):
        Config(corpus_stack="lokal")


def test_malformed_context_budget_fails_loudly_at_construction(monkeypatch):
    # Parsed lazily (default_factory): a typo must fail Config() with an
    # actionable message, never crash the module import for everyone.
    monkeypatch.setenv("PHD_CONTEXT_BUDGET", "8k")
    with pytest.raises(ValueError, match="PHD_CONTEXT_BUDGET"):
        Config()
    monkeypatch.setenv("PHD_CONTEXT_BUDGET", "12000")
    assert Config().context_budget_tokens == 12000


def test_explicit_context_budget_beats_the_env(monkeypatch):
    monkeypatch.setenv("PHD_CONTEXT_BUDGET", "12000")
    assert Config(context_budget_tokens=500).context_budget_tokens == 500


def test_valid_audio_stack_values_pass():
    assert Config(audio_stack="off").audio_stack == "off"
    assert Config(audio_stack="local").audio_stack == "local"


def test_typo_audio_stack_fails_fast():
    with pytest.raises(ValueError, match="PHD_AUDIO_STACK"):
        Config(audio_stack="lokal")


def test_audio_sidecar_urls_default_to_loopback():
    c = Config()
    assert c.stt_url == "http://127.0.0.1:8090"
    assert c.tts_url == "http://127.0.0.1:8083"


def test_audio_sidecar_urls_read_env_at_construction(monkeypatch):
    # default_factory, not a class-body os.environ.get: the env must be
    # read when Config() runs, so a test (or a restarted process) can
    # point the adapters at a different port.
    monkeypatch.setenv("PHD_STT_URL", "http://127.0.0.1:9999")
    assert Config().stt_url == "http://127.0.0.1:9999"


def test_malformed_vad_threshold_fails_loudly_at_construction(monkeypatch):
    monkeypatch.setenv("PHD_VAD_THRESHOLD", "0,o12")
    with pytest.raises(ValueError, match="PHD_VAD_THRESHOLD"):
        Config()
    monkeypatch.setenv("PHD_VAD_THRESHOLD", "0.02")
    assert Config().vad_threshold == 0.02


def test_corpus_url_defaults_to_the_sidecar_port():
    assert Config().corpus_url == "http://127.0.0.1:8091"


def test_corpus_url_reads_env_at_construction(monkeypatch):
    # default_factory like the audio URLs: a test (or a restarted
    # process) can repoint the corpus sidecar.
    monkeypatch.setenv("PHD_CORPUS_URL", "http://127.0.0.1:9991")
    assert Config().corpus_url == "http://127.0.0.1:9991"


def test_vad_engine_defaults_to_silero():
    assert Config().vad_engine == "silero"


def test_typo_vad_engine_fails_fast():
    # A typo'd endpointing engine must not silently run the energy gate
    # the user just complained about (§8: state is known at startup;
    # explicit-value call like the stack tests — the string switch reads
    # the env in the class body, same as corpus_stack).
    with pytest.raises(ValueError, match="PHD_VAD"):
        Config(vad_engine="sileroo")


def test_stream_tts_defaults_on(monkeypatch):
    monkeypatch.delenv("PHD_STREAM_TTS", raising=False)
    assert Config().stream_tts is True


def test_stream_tts_reads_env(monkeypatch):
    monkeypatch.setenv("PHD_STREAM_TTS", "0")
    assert Config().stream_tts is False


def test_malformed_stream_tts_fails_loudly_at_construction(monkeypatch):
    # A typo must not silently flip the kill switch (§8: state is known
    # at startup, not discovered mid-turn).
    monkeypatch.setenv("PHD_STREAM_TTS", "maybe")
    with pytest.raises(ValueError, match="PHD_STREAM_TTS"):
        Config()
    monkeypatch.setenv("PHD_STREAM_TTS", "1")
    assert Config().stream_tts is True
