"""Retrieval metrics: precision@k, recall@k, hit@k, MRR.

These operate purely on ranked lists of doc ids -- no LLM, no network, no
Chroma. That keeps them fast, deterministic, and trivial to unit test with
hand-made fake rankings (see tests/test_retrieval_metrics.py).

All functions compare a *ranked* list of retrieved doc ids (rank 1 first)
against the ground-truth set of relevant doc ids for that question, taken
from `EvalQuestion.relevant_doc_ids`.
"""
from __future__ import annotations

from typing import Iterable, Sequence


def _validate_k(k: int) -> None:
    if k <= 0:
        raise ValueError("k must be positive")


def precision_at_k(
    retrieved_doc_ids: Sequence[str], relevant_doc_ids: Iterable[str], k: int
) -> float:
    """Fraction of the top-k retrieved items whose doc is relevant.

    precision@k = (relevant docs among the top k) / k

    If fewer than k items were retrieved, the missing slots count as
    non-relevant, so the score is still divided by k.
    """
    _validate_k(k)
    relevant = set(relevant_doc_ids)
    top_k = list(retrieved_doc_ids)[:k]
    hits = sum(1 for doc_id in top_k if doc_id in relevant)
    return hits / k


def recall_at_k(
    retrieved_doc_ids: Sequence[str], relevant_doc_ids: Iterable[str], k: int
) -> float:
    """Fraction of all relevant docs that appear in the top-k retrieved items.

    recall@k = (relevant docs found in top k) / (total relevant docs)

    An unanswerable question has no relevant docs; recall is defined as 0.0
    in that case (there is nothing to recall), matching the MRR convention.
    """
    _validate_k(k)
    relevant = set(relevant_doc_ids)
    if not relevant:
        return 0.0
    top_k = list(retrieved_doc_ids)[:k]
    found = {doc_id for doc_id in top_k if doc_id in relevant}
    return len(found) / len(relevant)


def hit_at_k(
    retrieved_doc_ids: Sequence[str], relevant_doc_ids: Iterable[str], k: int
) -> int:
    """1 if any of the top-k retrieved items is relevant, else 0."""
    _validate_k(k)
    relevant = set(relevant_doc_ids)
    top_k = list(retrieved_doc_ids)[:k]
    return 1 if any(doc_id in relevant for doc_id in top_k) else 0


def reciprocal_rank(
    retrieved_doc_ids: Sequence[str], relevant_doc_ids: Iterable[str]
) -> float:
    """1 / rank of the first relevant doc (rank starts at 1); 0.0 if none found.

    Unlike the @k metrics above, this looks through the whole ranking, not
    just a top-k slice -- MRR cares about how early the first relevant hit
    shows up, however far down it is.
    """
    relevant = set(relevant_doc_ids)
    for rank, doc_id in enumerate(retrieved_doc_ids, start=1):
        if doc_id in relevant:
            return 1.0 / rank
    return 0.0


def mean_reciprocal_rank(
    rankings: Sequence[tuple[Sequence[str], Iterable[str]]]
) -> float:
    """Average reciprocal_rank over many (retrieved, relevant) pairs, one per question."""
    if not rankings:
        return 0.0
    scores = [reciprocal_rank(retrieved, relevant) for retrieved, relevant in rankings]
    return sum(scores) / len(scores)
