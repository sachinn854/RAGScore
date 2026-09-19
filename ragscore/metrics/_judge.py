"""Shared helper for the LLM-as-judge metrics: call the judge and parse JSON.

Private to the metrics package (leading underscore) -- faithfulness.py and
answer_relevance.py both use this instead of duplicating the call+parse logic.
"""
from __future__ import annotations

import json


def call_judge_json(llm, messages: list[dict], *, model: str | None = None) -> dict:
    """Ask the judge model for a JSON object and parse the response.

    Raises ValueError with the raw text on a parse failure, so a malformed
    judge response fails loudly instead of silently becoming a wrong score.
    """
    model = model or llm.cfg.judge_model
    result = llm.chat(messages, model=model, temperature=0.0, json_mode=True)
    try:
        return json.loads(result.text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"judge did not return valid JSON: {result.text!r}") from exc
