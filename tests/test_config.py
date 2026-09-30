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
