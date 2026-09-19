"""Unit tests for ragscore.metrics.answer_relevance -- the judge LLM is mocked."""
import json

from ragscore.metrics.answer_relevance import answer_relevance


def test_score_and_reason_are_parsed(make_fake_llm):
    llm = make_fake_llm(json.dumps({"score": 0.9, "reason": "directly answers the question"}))
    result = answer_relevance("What is d_k?", "64", llm=llm)
    assert result.score == 0.9
    assert result.reason == "directly answers the question"


def test_low_relevance_score(make_fake_llm):
    llm = make_fake_llm(json.dumps({"score": 0.1, "reason": "off topic"}))
    result = answer_relevance("What is d_k?", "The sky is blue.", llm=llm)
    assert result.score == 0.1


def test_context_is_never_sent_to_the_judge(make_fake_llm):
    # answer_relevance only looks at (question, answer) -- no context param exists.
    llm = make_fake_llm(json.dumps({"score": 1.0, "reason": "ok"}))
    answer_relevance("question text", "answer text", llm=llm)
    full_prompt = " ".join(m["content"] for m in llm.calls[0])
    assert "question text" in full_prompt
    assert "answer text" in full_prompt
