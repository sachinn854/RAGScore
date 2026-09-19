"""Faithfulness: are the answer's claims actually supported by the retrieved context?

Concept: this guards against hallucination. The judge LLM breaks the
generated answer into its atomic factual claims, then labels each claim as
supported or not by the retrieved context.

    faithfulness = (supported claims) / (total claims)

An answer with no factual claims at all (for example "I don't know") is
defined as fully faithful (1.0): there is nothing unsupported for it to
hallucinate.
"""
from __future__ import annotations

from dataclasses import dataclass

from ._judge import call_judge_json

_SYSTEM_PROMPT = (
    "You are a strict fact-checking judge. Break the ANSWER into its atomic "
    "factual claims -- each a minimal, standalone statement. For each claim, "
    "decide whether it is directly supported by the CONTEXT. A claim is "
    "supported only if the context states it or clearly implies it; do not "
    "use outside knowledge. If the answer makes no factual claims (for "
    "example it says \"I don't know\" or declines to answer), return an "
    "empty claims list.\n\n"
    'Respond as JSON: {"claims": [{"claim": "...", "supported": true|false}]}'
)


@dataclass
class Claim:
    text: str
    supported: bool


@dataclass
class FaithfulnessResult:
    score: float
    claims: list[Claim]


def faithfulness(question: str, answer: str, context: str, *, llm) -> FaithfulnessResult:
    """Score how well `answer`'s claims are supported by `context`."""
    messages = [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {
            "role": "user",
            "content": f"Question: {question}\n\nContext:\n{context}\n\nAnswer:\n{answer}",
        },
    ]
    data = call_judge_json(llm, messages)
    raw_claims = data.get("claims", [])
    claims = [Claim(text=c["claim"], supported=bool(c["supported"])) for c in raw_claims]

    if not claims:
        return FaithfulnessResult(score=1.0, claims=[])

    supported = sum(1 for c in claims if c.supported)
    return FaithfulnessResult(score=supported / len(claims), claims=claims)


if __name__ == "__main__":
    import argparse

    from ..config import build_arg_parser, load_config
    from ..dataset import load_eval_set
    from ..llm import LLM
    from ..rag import generate

    parser = argparse.ArgumentParser(
        parents=[build_arg_parser()], description="Score faithfulness for one eval question."
    )
    parser.add_argument("--id", default=None, help="eval_set question id (default: first)")
    args = parser.parse_args()

    cfg = load_config()
    llm = LLM(cfg)
    questions = load_eval_set(cfg.eval_set_path)
    q = next((x for x in questions if x.id == args.id), questions[0]) if args.id else questions[0]

    result = generate(q.question, cfg=cfg, llm=llm)
    fr = faithfulness(q.question, result.answer, result.context, llm=llm)

    print(f"question: {q.question}")
    print(f"answer:   {result.answer}")
    print(f"faithfulness: {fr.score:.2f}")
    for c in fr.claims:
        print(f"  [{'OK' if c.supported else 'X '}] {c.text}")
