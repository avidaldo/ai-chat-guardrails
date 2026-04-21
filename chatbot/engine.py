"""
engine.py

The central orchestrator of the chatbot.

This module is the only one that imports from all the others. Everything else
is isolated: guardrails don't know about backends, backends don't know about
each other, and main.py only knows about this module.

Message flow for each turn:
    user input
        ↓
    [input_guard.validate()]  ← reject if unsafe (no tokens spent)
        ↓ (if ok)
    history.append(user message)
        ↓
    [backend.get_response()]  ← call the LLM (local or remote)
        ↓
    [output_guard.validate()] ← reject if response is problematic
        ↓ (if ok)
    history.append(assistant message)
        ↓
    return response to caller (main.py)

If the backend call raises an exception (network error, model not found…),
or if the output guardrail blocks the response, the user's message is removed
from history so the next turn starts from a clean state.
"""

from chatbot.config import BaseChatConfig
from chatbot.backends import create_backend, create_judge_backend
from chatbot.guardrails import input_guard, output_guard


class ChatEngine:
    """
    Manages a complete conversation session.

    This class is the heart of the application. It holds the conversation
    state (history) and runs every message through the full pipeline:
    guardrails → backend → guardrails.

    One ChatEngine instance = one conversation session.
    To start a new conversation, create a new ChatEngine (or call history.clear()).
    """

    def __init__(self, config: BaseChatConfig) -> None:
        self.config = config
        # History is a list of {"role": ..., "content": ...} dicts.
        # Roles: "user" for user messages; "model" (Gemini) or "assistant" (Ollama) for bot.
        self.history: list[dict] = []
        self.backend = create_backend(config)
        self.judge_backend = create_judge_backend(config)

    def _trim_history(self) -> None:
        """
        Discards old messages when history exceeds the configured turn limit.

        Why this matters:
            LLMs have a "context window" — a maximum number of tokens they can
            process in a single call. Sending the full history of a long conversation
            can exceed that limit (and increases API cost for remote models).

            1 turn = 1 user message + 1 assistant message = 2 list entries.
            We keep only the most recent max_history_turns turns.
        """
        max_messages = self.config.max_history_turns * 2
        if len(self.history) > max_messages:
            # Slice from the end: keep the most recent messages
            self.history = self.history[-max_messages:]

    def chat(self, user_input: str) -> str:
        """
        Processes one user message and returns the chatbot's response string.

        This method never raises exceptions — it catches all errors from the
        backend and returns a user-friendly error message instead. This keeps
        the calling code (main.py) simple.

        Returns:
            The bot's response, or an error/warning message prefixed with
            ⚠️ (guardrail rejection) or ❌ (backend/network error).
        """

        def llm_judge(prompt: str) -> str:
            judge_history = [{"role": "user", "content": prompt}]
            try:
                return self.judge_backend.get_response(judge_history)
            except Exception as exc:
                return "ERROR"

        # ── STEP 1: INPUT GUARDRAIL ─────────────────────────────────────────
        ok, reason = input_guard.validate(user_input, llm_judge)
        if not ok:
            return f"⚠️  {reason}"

        # ── STEP 2: ADD TO HISTORY ───────────────────────────────────────────
        self.history.append({"role": "user", "content": user_input})
        self._trim_history()

        # ── STEP 3: CALL THE BACKEND ─────────────────────────────────────────
        try:
            raw = self.backend.get_response(self.history)
        except Exception as exc:
            self.history.pop()
            return f"❌  Error connecting to the model: {exc}"

        # ── STEP 4: OUTPUT GUARDRAIL ─────────────────────────────────────────
        ok, result = output_guard.validate(raw, llm_judge)
        if not ok:
            self.history.pop()
            return f"⚠️  {result}"

        # ── STEP 5: SAVE RESPONSE AND RETURN ────────────────────────────────
        # Gemini uses "model" as the assistant role; Ollama uses "assistant".
        # We store whatever convention matches the current backend.
        assistant_role = "model" if self.config.mode == "remote" else "assistant"
        self.history.append({"role": assistant_role, "content": result})

        return result
