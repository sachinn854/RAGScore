"""Central configuration for RAGScore.

Ek jagah saare knobs: API keys, models, chunking/retrieval params, paths.
`.env` se defaults, CLI se override. Har eval run ka settings-snapshot yahi hai --
results JSON me `masked_dict()` jaata hai taaki runs compare ho saken.
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
    """Ek env var padho; `required` ho aur khaali ho to saaf error."""
    val = os.getenv(key, default)
    if required and not val:
        raise RuntimeError(f"Missing required env var: {key} (see .env.example)")
    return val or ""


@dataclass(frozen=True)
class Config:
    """Immutable settings snapshot for one eval run."""

    # secrets -- env only, kabhi CLI/log me nahi
    openrouter_api_key: str
    gemini_api_key: str
    # endpoints
    openrouter_base_url: str
    gemini_base_url: str
    # models
    gen_model: str
    judge_model: str
    embed_model: str
    # knobs -- experiments inhe vary karte hain
    embed_dim: int = 768
    chunk_size: int = 512
    chunk_overlap: int = 64
    top_k: int = 5
    # artifact paths
    corpus_dir: Path = _ROOT / "data" / "corpus"
    chroma_dir: Path = _ROOT / "chroma"
    cache_dir: Path = _ROOT / ".cache"
    results_dir: Path = _ROOT / "results"

    @classmethod
    def from_env(cls, **overrides: object) -> "Config":
        """`.env` load karke Config banao. Non-None `overrides` sabse upar (CLI)."""
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
        """Artifact dirs bana do -- explicit side-effect, jab likhna ho tab call karo."""
        for d in (self.chroma_dir, self.cache_dir, self.results_dir):
            d.mkdir(parents=True, exist_ok=True)

    def masked_dict(self) -> dict[str, object]:
        """Logs / results JSON ke liye -- secrets masked, Path -> str."""
        out: dict[str, object] = {}
        for f in fields(self):
            val = getattr(self, f.name)
            if f.name.endswith("_api_key"):
                val = (str(val)[:6] + "…") if val else ""
            elif isinstance(val, Path):
                val = str(val)
            out[f.name] = val
        return out


def build_arg_parser() -> argparse.ArgumentParser:
    """Config-override flags. Modules isse `parents=[build_arg_parser()]` se reuse karein."""
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
    """env + CLI merge karke Config return karo."""
    parser = argparse.ArgumentParser(parents=[build_arg_parser()])
    args = parser.parse_args(argv)
    return Config.from_env(**vars(args))


if __name__ == "__main__":
    print(json.dumps(load_config().masked_dict(), indent=2))
