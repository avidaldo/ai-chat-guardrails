import pytest
from chatbot.guardrails import output_guard

def dummy_judge_safe(prompt: str) -> str:
    return "SAFE"

def dummy_judge_unsafe(prompt: str) -> str:
    return "UNSAFE: Leaks internal data"

def test_check_not_empty():
    ok, text = output_guard.validate("   ", llm_judge=dummy_judge_safe)
    assert not ok
    assert "did not generate a valid response" in text

    ok, text = output_guard.validate("abc", llm_judge=dummy_judge_safe)
    assert not ok

    ok, text = output_guard.validate("Valid response", llm_judge=dummy_judge_safe)
    assert ok
    assert text == "Valid response"

def test_sensitive_leak():
    ok, text = output_guard.validate("Here is the secret: api_key=123", llm_judge=dummy_judge_safe)
    assert not ok
    assert "sensitive information leak" in text

def test_llm_judge():
    ok, text = output_guard.validate("This seems normal", llm_judge=dummy_judge_unsafe)
    assert not ok
    assert "Leaks internal data" in text

def test_whitespace_stripped():
    ok, text = output_guard.validate("   Great answer   ", llm_judge=dummy_judge_safe)
    assert ok
    assert text == "Great answer"
