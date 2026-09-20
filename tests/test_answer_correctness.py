"""Unit tests for ragscore.metrics.answer_correctness -- the judge LLM is mocked."""
import json

import pytest

from ragscore.metrics.answer_correctness import answer_correctness, cosine_similarity


def test_cosine_similarity_identical_vectors():
    assert cosine_similarity([1, 2, 3], [1, 2, 3]) == pytest.approx(1.0)


def test_cosine_similarity_orthogonal_vectors():
    assert cosine_similarity([1, 0], [0, 1]) == pytest.approx(0.0)


def test_cosine_similarity_opposite_vectors():
    assert cosine_similarity([1, 0], [-1, 0]) == pytest.approx(-1.0)


def test_cosine_similarity_zero_vector_is_zero():
    assert cosine_similarity([0, 0], [1, 1]) == 0.0


def test_answer_correctness_combines_embedding_and_judge(make_fake_llm):
    # FakeLLM.embed gives a constant-per-text vector, so cosine similarity
    # between any two non-empty texts is always 1.0 -- isolates the judge's
    # contribution below.
    llm = make_fake_llm(json.dumps({"score": 0.6, "reason": "mostly agrees"}))
    result = answer_correctness("q", "64", "64 dimensions", llm=llm)
    assert result.embedding_similarity == pytest.approx(1.0)
    assert result.factual_agreement == 0.6
    assert result.score == pytest.approx(0.5 * 1.0 + 0.5 * 0.6)


def test_full_agreement_gives_score_one(make_fake_llm):
    llm = make_fake_llm(json.dumps({"score": 1.0, "reason": "identical"}))
    result = answer_correctness("q", "a", "a", llm=llm)
    assert result.score == pytest.approx(1.0)


def test_zero_agreement_halves_the_score(make_fake_llm):
    llm = make_fake_llm(json.dumps({"score": 0.0, "reason": "contradicts"}))
    result = answer_correctness("q", "a", "b", llm=llm)
    assert result.score == pytest.approx(0.5)  # 0.5*1.0 (fake embed) + 0.5*0.0
