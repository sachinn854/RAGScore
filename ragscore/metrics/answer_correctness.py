"""Answer correctness: how close is the generated answer to the ideal answer?

Concept: this is reference-based evaluation -- unlike faithfulness (checked
against retrieved context) and relevance (checked against the question
alone), this checks the answer against a human-written `ideal_answer`.

Two signals are combined, because each catches something the other misses:
- embedding cosine similarity: cheap, catches paraphrases, but can be fooled
  by text that is semantically close yet factually wrong (e.g. "d_k is 64"
  vs "d_k is 46" look similar as vectors).
- LLM judge factual agreement: reads for actual fact agreement, catching
  exactly that failure mode.

    answer_correctness = 0.5 * cosine_similarity + 0.5 * factual_agreement
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

from ._judge import call_judge_json

_SYSTEM_PROMPT = (
    "You are a judge scoring how factually consistent an ANSWER is with a "
    "REFERENCE answer, on a scale from 0.0 (contradicts or misses the "
    "reference's facts) to 1.0 (states the same facts; wording may differ). "
    "Give partial credit for partially correct or incomplete answers. If "
    "the reference declines to answer (e.g. \"I don't know\") and the "
    "answer also declines, that counts as full agreement.\n\n"
    'Respond as JSON: {"score": <float 0.0-1.0>, "reason": "<one sentence>"}'
)


def cosine_similarity(a: Sequence[float], b: Sequence[float]) -> float:
    """Cosine similarity between two equal-length vectors, in [-1, 1].

    Returns 0.0 if either vector has zero magnitude (undefined otherwise).
    """
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


@dataclass
class AnswerCorrectnessResult:
    score: float
    embedding_similarity: float
    factual_agreement: float
    reason: str


def answer_correctness(
    question: str, answer: str, ideal_answer: str, *, llm
) -> AnswerCorrectnessResult:
    """Score `answer` against `ideal_answer` via embedding similarity + judge."""
    vectors = llm.embed([answer, ideal_answer])
    similarity = max(0.0, cosine_similarity(vectors[0], vectors[1]))

    messages = [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {
            "role": "user",
            "content": f"Question: {question}\n\nReference:\n{ideal_answer}\n\nAnswer:\n{answer}",
        },
    ]
    data = call_judge_json(llm, messages)
    agreement = float(data["score"])
    reason = str(data.get("reason", ""))

    score = 0.5 * similarity + 0.5 * agreement
    return AnswerCorrectnessResult(
        score=score, embedding_similarity=similarity, factual_agreement=agreement, reason=reason
    )


if __name__ == "__main__":
    import argparse

    from ..config import build_arg_parser, load_config
    from ..dataset import load_eval_set
    from ..llm import LLM
    from ..rag import generate

    parser = argparse.ArgumentParser(
        parents=[build_arg_parser()], description="Score answer correctness for one eval question."
    )
    parser.add_argument("--id", default=None, help="eval_set question id (default: first)")
    args = parser.parse_args()

    cfg = load_config()
    llm = LLM(cfg)
    questions = load_eval_set(cfg.eval_set_path)
    q = next((x for x in questions if x.id == args.id), questions[0]) if args.id else questions[0]

    result = generate(q.question, cfg=cfg, llm=llm)
    cr = answer_correctness(q.question, result.answer, q.ideal_answer, llm=llm)

    print(f"question: {q.question}")
    print(f"answer:   {result.answer}")
    print(f"ideal:    {q.ideal_answer}")
    print(
        f"correctness: {cr.score:.2f}  "
        f"(embed_sim={cr.embedding_similarity:.2f}, judge={cr.factual_agreement:.2f})"
    )
    print(f"reason: {cr.reason}")
