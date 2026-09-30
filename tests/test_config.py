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
