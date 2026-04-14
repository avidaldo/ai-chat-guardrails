"""
output_guard.py

Validates the LLM's response before showing it to the user.

Why do we need an output guardrail?
    Even with a well-crafted system prompt, LLMs can sometimes produce:
    - Empty or malformed responses (API errors, safety refusals that return nothing)
    - Accidental leaks of sensitive information from the system prompt
    - Responses that violate the application's content policy

This module is the last line of defense before text reaches the user.

In this example the validation is simple, but the pattern is extensible:
    - Add a toxicity classifier to block harmful content
    - Verify that the response is valid JSON when the app expects structured output
    - Cap response length to avoid unexpectedly long outputs
    - Detect and redact PII (Personally Identifiable Information) with presidio-analyzer
"""

# Phrases that the model should NEVER output.
# If any of these appear, it likely means the model has "leaked" information
# from its system prompt or its training data.
#
# This list must be adapted to your deployment context. For example, if your
# system prompt contains a custom persona name, add it here so the model
# can't accidentally reveal its instructions.
_SENSITIVE_PHRASES = [
    "my api key",
    "api_key",
    "password:",
    "secret token",
    "system prompt:",
]

# A response shorter than this is almost certainly an error (empty safety refusal,
# partial network response, etc.) rather than a meaningful answer.
_MIN_OUTPUT_CHARS = 5


def _check_not_empty(text: str) -> tuple[bool, str]:
    """
    Detects empty or too-short responses.

    Why this can happen:
        - The model triggered a safety filter and returned an empty string
        - A network error truncated the response
        - The model produced only whitespace (rare but possible)
    """
    if not text or len(text.strip()) < _MIN_OUTPUT_CHARS:
        return False, "[The model did not generate a valid response. Please try again.]"
    return True, ""


def _check_sensitive_leak(text: str) -> tuple[bool, str]:
    """
    Checks whether the response contains sensitive information.

    This is a last-resort guardrail. The system prompt should already instruct
    the model not to reveal configuration details, but models are not perfectly
    obedient — especially when users craft clever prompts. This filter catches
    cases that slip through.

    Note: lowercase comparison avoids trivial bypasses via capitalization.
    """
    text_lower = text.lower()
    for phrase in _SENSITIVE_PHRASES:
        if phrase.lower() in text_lower:
            return False, "[Response blocked: possible sensitive information leak.]"
    return True, ""


def validate(text: str) -> tuple[bool, str]:
    """
    Entry point for the output guardrail.

    Returns (True, cleaned_text) if all checks pass.
    Returns (False, error_message) if any check fails.

    Note the asymmetry with input_guard.validate():
        - Input guard returns (bool, reason_or_empty_string)
        - Output guard returns (bool, error_message_or_cleaned_text)

    This is intentional: the output guard needs to return the (stripped) text
    when it passes, so the engine receives the final clean string in one call.

    Usage in engine.py:
        ok, result = output_guard.validate(raw)
        if not ok:
            return f"⚠️  {result}"   # result is the error message
        # result is now the cleaned response text
    """
    for check in [_check_not_empty, _check_sensitive_leak]:
        ok, result = check(text)
        if not ok:
            return False, result
    # text.strip() removes any leading/trailing whitespace from the model's response
    return True, text.strip()
