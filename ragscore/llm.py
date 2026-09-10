"""LLM access layer: chat (OpenRouter) + embeddings (Gemini), with a disk cache.

Every LLM / embedding call goes through here for two reasons:
1. **Cache** -- the same (model, params, payload) resolves to `.cache/<hash>.json`
   with no API call, so re-runs and experiments are cheap and repeatable.
2. **One place** for retry/backoff and a per-call token + approximate cost tally,
   so the runner can report the total spend for a run.
"""
from __future__ import annotations

import hashlib
import json
import logging
import random
import time
from dataclasses import dataclass
from pathlib import Path

import openai
from openai import OpenAI

from .config import Config, load_config

logger = logging.getLogger("ragscore.llm")

# Transient errors: retry these, raise everything else immediately.
_RETRY_ERRORS = (
    openai.RateLimitError,
    openai.APIConnectionError,
    openai.APITimeoutError,
    openai.InternalServerError,
)


def _hash(payload: dict) -> str:
    """SHA-256 of the canonical JSON form -- used as the cache filename."""
    blob = json.dumps(payload, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def _retry(fn, *, tries: int = 4, base: float = 1.0):
    """Call `fn()` with exponential backoff, up to `tries` attempts."""
    for attempt in range(1, tries + 1):
        try:
            return fn()
        except _RETRY_ERRORS as e:
            if attempt == tries:
                raise
            wait = base * 2 ** (attempt - 1) + random.uniform(0, 0.3)
            logger.warning(
                "llm retry %d/%d in %.1fs (%s)", attempt, tries, wait, type(e).__name__
            )
            time.sleep(wait)


@dataclass
class ChatResult:
    text: str
    model: str
    prompt_tokens: int
    completion_tokens: int
    cost: float          # USD; 0.0 when the provider does not report it
    cached: bool


@dataclass
class Usage:
    """Running tally over the lifetime of one LLM instance."""

    chat_calls: int = 0
    embed_calls: int = 0
    cache_hits: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    embed_tokens: int = 0
    cost: float = 0.0

    def as_dict(self) -> dict:
        return self.__dict__.copy()


class LLM:
    """Cached chat + embedding client for a single Config."""

    def __init__(self, cfg: Config | None = None) -> None:
        self.cfg = cfg or load_config()
        self.cfg.ensure_dirs()
        self._cache_dir: Path = self.cfg.cache_dir
        self.usage = Usage()
        self._chat_client = OpenAI(
            api_key=self.cfg.openrouter_api_key, base_url=self.cfg.openrouter_base_url
        )
        self._embed_client = OpenAI(
            api_key=self.cfg.gemini_api_key, base_url=self.cfg.gemini_base_url
        )

    # ---- cache helpers -------------------------------------------------
    def _cache_path(self, key: str) -> Path:
        return self._cache_dir / f"{key}.json"

    def _cache_get(self, key: str) -> dict | None:
        p = self._cache_path(key)
        if p.exists():
            return json.loads(p.read_text(encoding="utf-8"))
        return None

    def _cache_put(self, key: str, value: dict) -> None:
        self._cache_path(key).write_text(
            json.dumps(value, ensure_ascii=False), encoding="utf-8"
        )

    # ---- chat --------------------------------------------------------
    def chat(
        self,
        messages: list[dict],
        *,
        model: str | None = None,
        temperature: float = 0.0,
        max_tokens: int = 1024,
        json_mode: bool = False,
    ) -> ChatResult:
        """OpenRouter chat completion. `json_mode=True` sets response_format json_object."""
        model = model or self.cfg.gen_model
        req = {
            "kind": "chat",
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "json_mode": json_mode,
        }
        key = _hash(req)

        cached = self._cache_get(key)
        if cached is not None:
            self.usage.chat_calls += 1
            self.usage.cache_hits += 1
            return ChatResult(**cached["result"], cached=True)

        kwargs: dict = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}

        resp = _retry(lambda: self._chat_client.chat.completions.create(**kwargs))
        u = resp.usage
        result = dict(
            text=resp.choices[0].message.content or "",
            model=model,
            prompt_tokens=getattr(u, "prompt_tokens", 0),
            completion_tokens=getattr(u, "completion_tokens", 0),
            cost=float(getattr(u, "cost", 0.0) or 0.0),
        )
        self._cache_put(key, {"request": req, "result": result})

        self.usage.chat_calls += 1
        self.usage.prompt_tokens += result["prompt_tokens"]
        self.usage.completion_tokens += result["completion_tokens"]
        self.usage.cost += result["cost"]
        return ChatResult(**result, cached=False)

    # ---- embeddings ------------------------------------------------
    def embed(self, texts: list[str], *, model: str | None = None) -> list[list[float]]:
        """Gemini embeddings, cached per text so partial cache hits work."""
        model = model or self.cfg.embed_model
        dim = self.cfg.embed_dim

        keys = [_hash({"kind": "embed", "model": model, "dim": dim, "text": t}) for t in texts]
        vectors: list[list[float] | None] = [None] * len(texts)
        misses: list[int] = []
        for i, k in enumerate(keys):
            hit = self._cache_get(k)
            if hit is not None:
                vectors[i] = hit["vector"]
            else:
                misses.append(i)

        if misses:
            resp = _retry(
                lambda: self._embed_client.embeddings.create(
                    model=model, input=[texts[i] for i in misses], dimensions=dim
                )
            )
            for slot, item in zip(misses, resp.data):
                vectors[slot] = item.embedding
                self._cache_put(keys[slot], {"model": model, "dim": dim, "vector": item.embedding})
            self.usage.embed_tokens += getattr(resp.usage, "prompt_tokens", 0)
        else:
            self.usage.cache_hits += 1

        self.usage.embed_calls += 1
        return [v for v in vectors if v is not None]


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    llm = LLM()
    r1 = llm.chat([{"role": "user", "content": "Reply with exactly: pong"}])
    print("chat  :", repr(r1.text), "cached=", r1.cached, "cost=", r1.cost)
    r2 = llm.chat([{"role": "user", "content": "Reply with exactly: pong"}])
    print("chat  :", repr(r2.text), "cached=", r2.cached, "(should be True)")
    v = llm.embed(["hello world", "hello world", "a different sentence"])
    print("embed :", len(v), "vectors, dim", len(v[0]))
    print("usage :", json.dumps(llm.usage.as_dict(), indent=2))
