"""Run the full evaluation: every question in eval_set.jsonl, every metric.

Concept: this is what turns individual metric functions into a repeatable,
CI-style evaluation. One command produces one timestamped results file that
can later be diffed against other runs (that's what M8 experiments do).
"""
from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

from .config import Config, build_arg_parser, load_config
from .dataset import EvalQuestion, load_eval_set
from .llm import LLM
from .metrics.answer_correctness import answer_correctness
from .metrics.answer_relevance import answer_relevance
from .metrics.faithfulness import faithfulness
from .metrics.retrieval import hit_at_k, precision_at_k, reciprocal_rank, recall_at_k
from .rag import generate

# Reported average of "reciprocal_rank" across questions is MRR by definition.
METRIC_FIELDS = [
    "precision_at_k",
    "recall_at_k",
    "hit_at_k",
    "reciprocal_rank",
    "faithfulness",
    "answer_relevance",
    "answer_correctness",
]


@dataclass
class QuestionResult:
    id: str
    question: str
    answer: str
    ideal_answer: str
    retrieved_doc_ids: list[str]
    relevant_doc_ids: list[str]
    precision_at_k: float
    recall_at_k: float
    hit_at_k: int
    reciprocal_rank: float
    faithfulness: float
    answer_relevance: float
    answer_correctness: float


def evaluate_question(q: EvalQuestion, *, cfg: Config, llm: LLM) -> QuestionResult:
    """Run the RAG pipeline plus every metric for one eval question."""
    result = generate(q.question, cfg=cfg, llm=llm)
    retrieved_doc_ids = [c.doc_id for c in result.retrieved]

    faith = faithfulness(q.question, result.answer, result.context, llm=llm)
    relevance = answer_relevance(q.question, result.answer, llm=llm)
    correctness = answer_correctness(q.question, result.answer, q.ideal_answer, llm=llm)

    return QuestionResult(
        id=q.id,
        question=q.question,
        answer=result.answer,
        ideal_answer=q.ideal_answer,
        retrieved_doc_ids=retrieved_doc_ids,
        relevant_doc_ids=q.relevant_doc_ids,
        precision_at_k=precision_at_k(retrieved_doc_ids, q.relevant_doc_ids, cfg.top_k),
        recall_at_k=recall_at_k(retrieved_doc_ids, q.relevant_doc_ids, cfg.top_k),
        hit_at_k=hit_at_k(retrieved_doc_ids, q.relevant_doc_ids, cfg.top_k),
        reciprocal_rank=reciprocal_rank(retrieved_doc_ids, q.relevant_doc_ids),
        faithfulness=faith.score,
        answer_relevance=relevance.score,
        answer_correctness=correctness.score,
    )


def run_eval(cfg: Config | None = None, llm: LLM | None = None) -> dict:
    """Evaluate every question in the eval set. Returns the full results dict."""
    cfg = cfg or load_config()
    llm = llm or LLM(cfg)

    questions = load_eval_set(cfg.eval_set_path)
    per_question = [evaluate_question(q, cfg=cfg, llm=llm) for q in questions]

    averages = {
        name: sum(getattr(r, name) for r in per_question) / len(per_question)
        for name in METRIC_FIELDS
    }

    return {
        "timestamp": datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"),
        "config": cfg.masked_dict(),
        "num_questions": len(per_question),
        "averages": averages,
        "usage": llm.usage.as_dict(),
        "questions": [asdict(r) for r in per_question],
    }


def save_results(results: dict, *, cfg: Config) -> Path:
    """Write the results dict to results/run_<timestamp>.json. Returns the path."""
    cfg.ensure_dirs()
    path = cfg.results_dir / f"run_{results['timestamp']}.json"
    path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    return path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        parents=[build_arg_parser()], description="Run the full eval set through every metric."
    )
    args = parser.parse_args()

    cfg = load_config()
    llm = LLM(cfg)

    start = time.monotonic()
    results = run_eval(cfg=cfg, llm=llm)
    elapsed = time.monotonic() - start

    path = save_results(results, cfg=cfg)
    print(f"{results['num_questions']} questions evaluated in {elapsed:.1f}s")
    print(f"saved: {path}")
    for name, val in results["averages"].items():
        print(f"  {name:20} {val:.3f}")
    print(f"usage: {results['usage']}")
