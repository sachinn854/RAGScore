"""Shared test fixtures for LLM-based metric tests -- no real API calls."""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from ragscore.llm import ChatResult


class FakeLLM:
    """Stand-in for ragscore.llm.LLM that returns one canned response.

    Records every call (`self.calls`) so a test can assert on what was sent.
    Never touches the network.
    """

    def __init__(self, response_text: str):
        self.response_text = response_text
        self.cfg = SimpleNamespace(judge_model="fake-judge", embed_model="fake-embed", embed_dim=8)
        self.calls: list[list[dict]] = []

    def chat(self, messages, *, model=None, temperature=0.0, max_tokens=1024, json_mode=False):
        self.calls.append(messages)
        return ChatResult(
            text=self.response_text,
            model=model or self.cfg.judge_model,
            prompt_tokens=0,
            completion_tokens=0,
            cost=0.0,
            cached=False,
        )

    def embed(self, texts, *, model=None):
        # Deterministic fake vectors -- not used by M5 tests, useful for M6.
        return [[float(len(t))] * self.cfg.embed_dim for t in texts]


@pytest.fixture
def make_fake_llm():
    """Factory fixture: make_fake_llm('{"...": ...}') -> FakeLLM instance."""
    return FakeLLM
