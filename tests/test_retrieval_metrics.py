"""Unit tests for ragscore.metrics.retrieval -- fake rankings, hand-computed values.

No network, no LLM: retrieval metrics are pure functions over plain lists.
"""
import pytest

from ragscore.metrics.retrieval import (
    hit_at_k,
    mean_reciprocal_rank,
    precision_at_k,
    reciprocal_rank,
    recall_at_k,
)

# Shared fixture: 5 retrieved docs, ranked, with 2 of them relevant.
RETRIEVED = ["A", "B", "C", "D", "E"]
RELEVANT = {"B", "D"}


def test_precision_at_k():
    assert precision_at_k(RETRIEVED, RELEVANT, k=1) == 0.0      # ["A"]              -> 0/1
    assert precision_at_k(RETRIEVED, RELEVANT, k=3) == pytest.approx(1 / 3)  # ["A","B","C"] -> 1/3
    assert precision_at_k(RETRIEVED, RELEVANT, k=5) == pytest.approx(2 / 5)  # all 5 -> 2/5


def test_recall_at_k():
    assert recall_at_k(RETRIEVED, RELEVANT, k=1) == 0.0                     # neither found yet
    assert recall_at_k(RETRIEVED, RELEVANT, k=3) == pytest.approx(0.5)      # found B, not D -> 1/2
    assert recall_at_k(RETRIEVED, RELEVANT, k=5) == pytest.approx(1.0)      # found both -> 2/2


def test_hit_at_k():
    assert hit_at_k(RETRIEVED, RELEVANT, k=1) == 0   # "A" is not relevant
    assert hit_at_k(RETRIEVED, RELEVANT, k=2) == 1   # "B" appears by rank 2
    assert hit_at_k(RETRIEVED, RELEVANT, k=5) == 1


def test_reciprocal_rank_finds_first_relevant():
    # First relevant doc "B" is at rank 2 -> 1/2
    assert reciprocal_rank(RETRIEVED, RELEVANT) == pytest.approx(0.5)


def test_reciprocal_rank_no_relevant_docs_is_zero():
    assert reciprocal_rank(["A", "B", "C"], {"Z"}) == 0.0


def test_duplicate_doc_ids_in_retrieved_list():
    # Same doc retrieved twice (two chunks from doc "X"), relevant doc "Y" is last.
    retrieved = ["X", "X", "Y"]
    relevant = {"Y"}
    assert precision_at_k(retrieved, relevant, k=3) == pytest.approx(1 / 3)
    assert recall_at_k(retrieved, relevant, k=3) == pytest.approx(1.0)
    assert reciprocal_rank(retrieved, relevant) == pytest.approx(1 / 3)


def test_unanswerable_question_has_no_relevant_docs():
    # relevant_doc_ids == [] for our unanswerable eval questions (q19, q20).
    retrieved = ["A", "B"]
    relevant: set[str] = set()
    assert precision_at_k(retrieved, relevant, k=2) == 0.0
    assert recall_at_k(retrieved, relevant, k=2) == 0.0
    assert hit_at_k(retrieved, relevant, k=2) == 0
    assert reciprocal_rank(retrieved, relevant) == 0.0


def test_k_larger_than_retrieved_list_still_divides_by_k():
    # Only 1 item retrieved but k=5 requested -> missing slots count as misses.
    assert precision_at_k(["A"], {"A"}, k=5) == pytest.approx(0.2)   # 1/5, not 1/1
    assert recall_at_k(["A"], {"A"}, k=5) == pytest.approx(1.0)      # the 1 relevant doc was found
    assert hit_at_k(["A"], {"A"}, k=5) == 1


def test_invalid_k_raises():
    with pytest.raises(ValueError):
        precision_at_k(["A"], {"A"}, k=0)
    with pytest.raises(ValueError):
        recall_at_k(["A"], {"A"}, k=-1)
    with pytest.raises(ValueError):
        hit_at_k(["A"], {"A"}, k=0)


def test_mean_reciprocal_rank():
    rankings = [
        (["A", "B"], {"B"}),   # B at rank 2 -> RR = 0.5
        (["C"], {"D"}),        # not found    -> RR = 0.0
    ]
    assert mean_reciprocal_rank(rankings) == pytest.approx(0.25)


def test_mean_reciprocal_rank_empty_input():
    assert mean_reciprocal_rank([]) == 0.0
