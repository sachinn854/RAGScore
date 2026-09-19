"""Answer relevance: does the answer directly address the question asked?

Concept: independent of whether the answer is grounded in context or
factually correct, does it actually engage with what was asked? An answer
can be perfectly faithful to the context yet dodge the question. This looks
only at (question, answer) -- context and factual correctness are scored by
faithfulness.py and answer_correctness.py instead.
"""
from __future__ import annotations

from dataclasses import dataclass

from ._judge import call_judge_json

_SYSTEM_PROMPT = (
    "You are a judge scoring how directly an ANSWER addresses a QUESTION, "
    "on a scale from 0.0 (ignores the question or answers something else "
    "entirely) to 1.0 (directly and fully engages with exactly what was "
    "asked). Judge relevance only -- do not judge whether the answer is "
    "factually correct or complete. A direct, honest refusal such as "
    "\"I don't know\" counts as fully relevant (1.0) if it responds to this "
    "exact question; score low only if the answer is vague, generic, or "
    "clearly about a different question.\n\n"
    'Respond as JSON: {"score": <float 0.0-1.0>, "reason": "<one sentence>"}'
)


@dataclass
class RelevanceResult:
    score: float
    reason: str


def answer_relevance(question: str, answer: str, *, llm) -> RelevanceResult:
    """Score how directly `answer` addresses `question`, ignoring context/correctness."""
    messages = [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user", "content": f"Question: {question}\n\nAnswer:\n{answer}"},
    ]
    data = call_judge_json(llm, messages)
    return RelevanceResult(score=float(data["score"]), reason=str(data.get("reason", "")))


if __name__ == "__main__":
    import argparse

    from ..config import build_arg_parser, load_config
    from ..dataset import load_eval_set
    from ..llm import LLM
    from ..rag import generate

    parser = argparse.ArgumentParser(
        parents=[build_arg_parser()], description="Score answer relevance for one eval question."
    )
    parser.add_argument("--id", default=None, help="eval_set question id (default: first)")
    args = parser.parse_args()

    cfg = load_config()
    llm = LLM(cfg)
    questions = load_eval_set(cfg.eval_set_path)
    q = next((x for x in questions if x.id == args.id), questions[0]) if args.id else questions[0]

    result = generate(q.question, cfg=cfg, llm=llm)
    rr = answer_relevance(q.question, result.answer, llm=llm)

    print(f"question: {q.question}")
    print(f"answer:   {result.answer}")
    print(f"relevance: {rr.score:.2f}  ({rr.reason})")
