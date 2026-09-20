"""Ablation experiments: vary chunk_size / top_k, re-evaluate, compare.

Concept: a single eval run tells you how the system performs under one
configuration. Ablations tell you *why* -- which knob actually moves the
numbers, so tuning choices are backed by evidence instead of guesses.

chunk_size changes require rebuilding the Chroma index (chunk boundaries
change). top_k does not -- it only changes how many already-embedded chunks
are pulled back at query time.
"""
from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from pathlib import Path

from rich.console import Console
from rich.table import Table

from .config import Config, build_arg_parser, load_config
from .ingest import ingest
from .llm import LLM, Usage
from .report import METRIC_LABELS
from .runner import METRIC_FIELDS, run_eval

CHUNK_SIZE_GRID = [128, 256, 512]
TOP_K_GRID = [3, 5, 8]


def _overlap_for(chunk_size: int) -> int:
    """Keep overlap proportional to chunk_size (~1/8 of it), floor 16."""
    return max(16, chunk_size // 8)


@dataclass
class ExperimentResult:
    name: str
    overrides: dict
    averages: dict
    usage: dict
    elapsed_s: float


def run_variant(
    name: str, base_cfg: Config, llm: LLM, *, overrides: dict, reingest: bool
) -> ExperimentResult:
    """Build a Config with `overrides` applied, (re)ingest if needed, evaluate it."""
    cfg = replace(base_cfg, **overrides) if overrides else base_cfg
    if reingest:
        ingest(cfg, rebuild=True)

    llm.usage = Usage()  # isolate this variant's cost/call count
    start = time.monotonic()
    results = run_eval(cfg=cfg, llm=llm)
    elapsed = time.monotonic() - start

    return ExperimentResult(
        name=name,
        overrides=overrides,
        averages=results["averages"],
        usage=results["usage"],
        elapsed_s=elapsed,
    )


def run_chunk_size_sweep(base_cfg: Config, llm: LLM) -> list[ExperimentResult]:
    """Fix top_k, vary chunk_size (and its proportional overlap)."""
    variants = []
    for size in CHUNK_SIZE_GRID:
        overrides = {"chunk_size": size, "chunk_overlap": _overlap_for(size)}
        variants.append(
            run_variant(f"chunk_size={size}", base_cfg, llm, overrides=overrides, reingest=True)
        )
    return variants


def run_top_k_sweep(base_cfg: Config, llm: LLM) -> list[ExperimentResult]:
    """Fix chunk_size, vary top_k. Re-ingest once so the index matches base_cfg."""
    ingest(base_cfg, rebuild=True)
    variants = []
    for k in TOP_K_GRID:
        variants.append(
            run_variant(f"top_k={k}", base_cfg, llm, overrides={"top_k": k}, reingest=False)
        )
    return variants


def print_comparison(title: str, variants: list[ExperimentResult], console: Console) -> None:
    table = Table(title=title)
    table.add_column("variant")
    for metric in METRIC_FIELDS:
        table.add_column(METRIC_LABELS.get(metric, metric), justify="right")
    table.add_column("cost ($)", justify="right")
    for v in variants:
        row = [v.name] + [f"{v.averages[m]:.3f}" for m in METRIC_FIELDS]
        row.append(f"{v.usage['cost']:.4f}")
        table.add_row(*row)
    console.print(table)


def save_experiment_results(sweeps: dict[str, list[ExperimentResult]], *, cfg: Config) -> Path:
    cfg.ensure_dirs()
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    payload = {
        "timestamp": timestamp,
        "sweeps": {
            sweep_name: [asdict(v) for v in variants] for sweep_name, variants in sweeps.items()
        },
    }
    path = cfg.results_dir / f"experiments_{timestamp}.json"
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        parents=[build_arg_parser()], description="Run chunk_size and top_k ablations."
    )
    args = parser.parse_args()

    base_cfg = load_config()
    llm = LLM(base_cfg)
    # Explicit width: the comparison table has 9 columns and can render blank
    # if Rich auto-detects a narrow/no terminal (e.g. piped or non-interactive output).
    console = Console(width=200)

    print(f"=== chunk_size sweep {CHUNK_SIZE_GRID} (top_k={base_cfg.top_k} fixed) ===")
    chunk_size_variants = run_chunk_size_sweep(base_cfg, llm)
    print_comparison("chunk_size ablation", chunk_size_variants, console)

    print(f"\n=== top_k sweep {TOP_K_GRID} (chunk_size={base_cfg.chunk_size} fixed) ===")
    top_k_variants = run_top_k_sweep(base_cfg, llm)
    print_comparison("top_k ablation", top_k_variants, console)

    # Leave the shared index rebuilt at the base config so a later
    # `python -m ragscore.runner` run isn't left on the last swept variant.
    ingest(base_cfg, rebuild=True)

    path = save_experiment_results(
        {"chunk_size": chunk_size_variants, "top_k": top_k_variants}, cfg=base_cfg
    )
    print(f"\nsaved: {path}")
