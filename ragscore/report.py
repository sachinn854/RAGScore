"""Print a results JSON (from runner.py) as a readable terminal report.

Concept: aggregation. One averaged number per metric hides where the system
actually struggles -- this also surfaces the worst-scoring questions, so
weaknesses stay visible instead of getting buried in an average.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from rich.console import Console
from rich.table import Table

from .config import build_arg_parser, load_config

_METRIC_LABELS = {
    "precision_at_k": "precision@k",
    "recall_at_k": "recall@k",
    "hit_at_k": "hit@k",
    "reciprocal_rank": "MRR",
    "faithfulness": "faithfulness",
    "answer_relevance": "answer_relevance",
    "answer_correctness": "answer_correctness",
}


def load_results(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def latest_results_path(results_dir: Path) -> Path:
    """Most recently created run_*.json under results_dir."""
    runs = sorted(results_dir.glob("run_*.json"))
    if not runs:
        raise FileNotFoundError(
            f"no results found in {results_dir} -- run: python -m ragscore.runner"
        )
    return runs[-1]


def print_summary(results: dict, console: Console) -> None:
    table = Table(title=f"RAGScore  --  {results['num_questions']} questions  --  {results['timestamp']}")
    table.add_column("metric")
    table.add_column("average", justify="right")
    for name, val in results["averages"].items():
        table.add_row(_METRIC_LABELS.get(name, name), f"{val:.3f}")
    console.print(table)


def print_worst_cases(
    results: dict, console: Console, *, metric: str = "answer_correctness", n: int = 5
) -> None:
    worst = sorted(results["questions"], key=lambda q: q[metric])[:n]
    table = Table(title=f"Worst {n} by {_METRIC_LABELS.get(metric, metric)}")
    table.add_column("id")
    table.add_column("question")
    table.add_column(metric, justify="right")
    for q in worst:
        table.add_row(q["id"], q["question"][:60], f"{q[metric]:.2f}")
    console.print(table)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        parents=[build_arg_parser()], description="Print a report for a results JSON file."
    )
    parser.add_argument("--path", type=Path, default=None, help="results JSON (default: latest run)")
    parser.add_argument("--worst-metric", default="answer_correctness")
    parser.add_argument("--worst-n", type=int, default=5)
    args = parser.parse_args()

    cfg = load_config()
    path = args.path or latest_results_path(cfg.results_dir)
    results = load_results(path)

    console = Console()
    print_summary(results, console)
    print_worst_cases(results, console, metric=args.worst_metric, n=args.worst_n)
