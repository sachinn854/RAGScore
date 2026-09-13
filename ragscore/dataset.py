"""Load the hand-built evaluation set (data/eval_set.jsonl).

Each line is one question paired with its ground truth: the doc ids a correct
retrieval should surface (`relevant_doc_ids`) and the answer a correct
generation should produce (`ideal_answer`). The M4-M6 metrics compare a
`RagResult` (from rag.py) against this ground truth.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

from .config import Config, build_arg_parser, load_config


@dataclass
class EvalQuestion:
    id: str
    question: str
    ideal_answer: str
    relevant_doc_ids: list[str]


def load_eval_set(path: Path) -> list[EvalQuestion]:
    """Parse a JSONL eval set file into a list of EvalQuestion."""
    if not path.exists():
        raise FileNotFoundError(f"eval set not found: {path}")

    questions: list[EvalQuestion] = []
    with path.open(encoding="utf-8") as f:
        for line_no, raw_line in enumerate(f, start=1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no}: invalid JSON") from exc
            questions.append(
                EvalQuestion(
                    id=row["id"],
                    question=row["question"],
                    ideal_answer=row["ideal_answer"],
                    relevant_doc_ids=row["relevant_doc_ids"],
                )
            )

    _check_unique_ids(questions, path)
    return questions


def _check_unique_ids(questions: list[EvalQuestion], path: Path) -> None:
    """Fail loudly on duplicate ids instead of silently overwriting scores later."""
    ids = [q.id for q in questions]
    dupes = {i for i in ids if ids.count(i) > 1}
    if dupes:
        raise ValueError(f"duplicate question ids in {path}: {sorted(dupes)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        parents=[build_arg_parser()], description="Load and summarize the eval set."
    )
    args = parser.parse_args()
    cfg: Config = load_config()

    questions = load_eval_set(cfg.eval_set_path)
    multi = sum(1 for q in questions if len(q.relevant_doc_ids) > 1)
    unanswerable = sum(1 for q in questions if len(q.relevant_doc_ids) == 0)
    single = len(questions) - multi - unanswerable

    print(f"{len(questions)} questions from {cfg.eval_set_path}")
    print(f"single-doc={single}  multi-hop={multi}  unanswerable={unanswerable}")
    print("\n--- first question ---")
    print(questions[0])
