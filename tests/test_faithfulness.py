"""Unit tests for ragscore.metrics.faithfulness -- the judge LLM is mocked."""
import json

import pytest

from ragscore.metrics.faithfulness import faithfulness


def test_all_claims_supported(make_fake_llm):
    response = json.dumps(
        {
            "claims": [
                {"claim": "d_k is 64", "supported": True},
                {"claim": "the base model has 8 heads", "supported": True},
            ]
        }
    )
    llm = make_fake_llm(response)
    result = faithfulness("q", "a", "context", llm=llm)
    assert result.score == 1.0
    assert len(result.claims) == 2
    assert all(c.supported for c in result.claims)


def test_partial_support_scores_fraction(make_fake_llm):
    response = json.dumps(
        {
            "claims": [
                {"claim": "A", "supported": True},
                {"claim": "B", "supported": False},
                {"claim": "C", "supported": False},
            ]
        }
    )
    llm = make_fake_llm(response)
    result = faithfulness("q", "a", "ctx", llm=llm)
    assert result.score == pytest.approx(1 / 3)


def test_no_claims_is_fully_faithful(make_fake_llm):
    llm = make_fake_llm(json.dumps({"claims": []}))
    result = faithfulness("q", "I don't know", "ctx", llm=llm)
    assert result.score == 1.0
    assert result.claims == []


def test_sends_question_context_answer_to_judge(make_fake_llm):
    llm = make_fake_llm(json.dumps({"claims": []}))
    faithfulness("What is d_k?", "64", "d_k is 64 in the base model.", llm=llm)
    sent_content = llm.calls[0][-1]["content"]
    assert "What is d_k?" in sent_content
    assert "64 in the base model" in sent_content


def test_invalid_json_raises(make_fake_llm):
    llm = make_fake_llm("not valid json")
    with pytest.raises(ValueError):
        faithfulness("q", "a", "ctx", llm=llm)
