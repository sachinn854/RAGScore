"""Central configuration for RAGScore.

One place for every knob: API keys, models, chunking/retrieval params, paths.
Defaults come from `.env`, and CLI flags override them. This object is the
settings snapshot for a single eval run -- `masked_dict()` gets written into the
results JSON so runs can be compared later.
"""
from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass, fields
from pathlib import Path

from dotenv import load_dotenv

_ROOT = Path(__file__).resolve().parent.parent


def _env(key: str, default: str | None = None, *, required: bool = False) -> str:
    """Read one env var; raise a clear error if required and empty."""
    val = os.getenv(key, default)
    if required and not val:
        raise RuntimeError(f"Missing required env var: {key} (see .env.example)")
    return val or ""


@dataclass(frozen=True)
class Config:
    """Immutable settings snapshot for one eval run."""

    # secrets -- env only, never logged or exposed via CLI
    openrouter_api_key: str
    gemini_api_key: str
    # endpoints
    openrouter_base_url: str
    gemini_base_url: str
    # models
    gen_model: str
    judge_model: str
    embed_model: str
    # knobs -- experiments vary these
    embed_dim: int = 768
    chunk_size: int = 512
    chunk_overlap: int = 64
    top_k: int = 5
    # artifact paths
    corpus_dir: Path = _ROOT / "data" / "corpus"
    eval_set_path: Path = _ROOT / "data" / "eval_set.jsonl"
    chroma_dir: Path = _ROOT / "chroma"
    cache_dir: Path = _ROOT / ".cache"
    results_dir: Path = _ROOT / "results"

    @classmethod
    def from_env(cls, **overrides: object) -> "Config":
        """Build a Config from `.env`. Non-None `overrides` win (used by the CLI)."""
        load_dotenv(_ROOT / ".env")
        base: dict[str, object] = dict(
            openrouter_api_key=_env("OPENROUTER_API_KEY", required=True),
            gemini_api_key=_env("GEMINI_API_KEY", required=True),
            openrouter_base_url=_env(
                "OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"
            ),
            gemini_base_url=_env(
                "GEMINI_BASE_URL",
                "https://generativelanguage.googleapis.com/v1beta/openai/",
            ),
            gen_model=_env("GEN_MODEL", required=True),
            judge_model=_env("JUDGE_MODEL", required=True),
            embed_model=_env("EMBED_MODEL", required=True),
            embed_dim=int(_env("EMBED_DIM", "768")),
        )
        base.update({k: v for k, v in overrides.items() if v is not None})
        return cls(**base)  # type: ignore[arg-type]

    def ensure_dirs(self) -> None:
        """Create artifact directories. Explicit side-effect -- call before writing."""
        for d in (self.chroma_dir, self.cache_dir, self.results_dir):
            d.mkdir(parents=True, exist_ok=True)

    def masked_dict(self) -> dict[str, object]:
        """Serialisable view for logs / results JSON -- secrets masked, Path -> str."""
        out: dict[str, object] = {}
        for f in fields(self):
            val = getattr(self, f.name)
            if f.name.endswith("_api_key"):
                val = (str(val)[:6] + "...") if val else ""
            elif isinstance(val, Path):
                val = str(val)
            out[f.name] = val
        return out


def build_arg_parser() -> argparse.ArgumentParser:
    """Config-override flags. Other modules reuse this via `parents=[build_arg_parser()]`."""
    p = argparse.ArgumentParser(add_help=False)
    g = p.add_argument_group("config overrides")
    g.add_argument("--gen-model", dest="gen_model")
    g.add_argument("--judge-model", dest="judge_model")
    g.add_argument("--embed-model", dest="embed_model")
    g.add_argument("--embed-dim", dest="embed_dim", type=int)
    g.add_argument("--chunk-size", dest="chunk_size", type=int)
    g.add_argument("--chunk-overlap", dest="chunk_overlap", type=int)
    g.add_argument("--top-k", dest="top_k", type=int)
    return p


def load_config(argv: list[str] | None = None) -> Config:
    """Return a Config built from the environment merged with CLI overrides.

    Unknown flags are ignored so a module can add its own (e.g. ``--question``)
    without upsetting the shared config parsing.
    """
    parser = argparse.ArgumentParser(add_help=False, parents=[build_arg_parser()])
    args, _ = parser.parse_known_args(argv)
    return Config.from_env(**vars(args))


if __name__ == "__main__":
    print(json.dumps(load_config().masked_dict(), indent=2))
