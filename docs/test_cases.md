# Test Cases

This document describes the testing strategy and scenarios covered by our automated test suite in the `tests/` directory.

## 1. Input Guardrail (`test_input_guard.py`)

The input guardrail validates user input before it is sent to the main LLM. It acts as the first line of defense.

### Test Scenarios
- **Length Constraint**: Verifies that empty strings and excessively long strings (e.g., > 500 characters) are rejected immediately.
- **Blocked Fragments**: Verifies that known malicious strings (like `<script>`, `DROP TABLE`) are blocked via heuristic checks.
- **Regex Injection Patterns**: Verifies that common prompt injection attempts (e.g., "ignore all previous instructions") are caught.
- **LLM Judge**: Verifies that complex or obfuscated malicious intents that bypass heuristics are blocked by the LLM security evaluator.

## 2. Output Guardrail (`test_output_guard.py`)

The output guardrail ensures the bot's responses do not violate policies or leak sensitive data.

### Test Scenarios
- **Empty Responses**: Verifies that if the LLM backend returns an empty string or just whitespace, an appropriate error is substituted.
- **Sensitive Leaks**: Verifies that if the bot accidentally includes "api_key" or "system prompt", the response is blocked.
- **LLM Judge Evaluation**: Verifies that the LLM evaluator correctly identifies and blocks nuanced policy violations in the assistant's output.
- **Formatting**: Ensures the final approved response has leading and trailing whitespaces stripped.

## 3. Engine Orchestrator (`test_engine.py`)

The `ChatEngine` manages conversation state, backend routing, and guardrail integration.

### Test Scenarios
- **Successful Turn**: Verifies that a safe user message is correctly appended to history, passed to the backend, the safe response is returned, and history is updated.
- **Backend Failure Handling**: Verifies that if the backend throws an exception (e.g., network error), the orchestrator intercepts it, returns a graceful error to the user, and pops the user's message from history so the session isn't corrupted.
- **History Trimming (Context Window Limit)**: Verifies that when the conversation exceeds the `MAX_HISTORY_TURNS`, the oldest messages are evicted, keeping the history strictly within bounds.
