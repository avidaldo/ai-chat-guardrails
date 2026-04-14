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

from dataclasses import dataclass

from chatbot.guardrails import input_guard, output_guard

# Backends are imported lazily inside chat() so that — for example — running
# in remote mode does not require the `ollama` package to be installed, and
# vice versa. A top-level import would force both dependencies at startup.


@dataclass
class ChatConfig:
    """
    All configuration for one chatbot session.

    Using a dataclass instead of a plain dict or positional arguments gives:
    - Named fields: config.mode instead of config[0] or config["mode"]
    - IDE autocompletion for all fields
    - A readable __repr__ for debugging: ChatConfig(mode='remote', ...)
    - Easy extensibility: add a field here and it's available everywhere

    The fields with defaults (max_history_turns) must come after fields without.
    """
    mode: str               # "local" (Ollama) or "remote" (Gemini)
    system_prompt: str      # the personality/instructions for the model
    remote_model: str       # Gemini model name, e.g. "gemini-2.5-flash"
    local_model: str        # Ollama model name, e.g. "llama3.2"
    ollama_url: str         # Ollama server address, e.g. "http://localhost:11434"
    api_key: str            # Gemini API key (empty string if mode == "local")
    max_history_turns: int = 10  # number of question+answer pairs to keep


class ChatEngine:
    """
    Manages a complete conversation session.

    This class is the heart of the application. It holds the conversation
    state (history) and runs every message through the full pipeline:
    guardrails → backend → guardrails.

    One ChatEngine instance = one conversation session.
    To start a new conversation, create a new ChatEngine (or call history.clear()).
    """

    def __init__(self, config: ChatConfig) -> None:
        self.config = config
        # History is a list of {"role": ..., "content": ...} dicts.
        # Roles: "user" for user messages; "model" (Gemini) or "assistant" (Ollama) for bot.
        self.history: list[dict] = []

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

        # ── STEP 1: INPUT GUARDRAIL ─────────────────────────────────────────
        # Check the user's message before spending any tokens.
        # If the guardrail rejects it, we return immediately without touching
        # the history — this exchange never happened from the model's perspective.
        ok, reason = input_guard.validate(user_input)
        if not ok:
            return f"⚠️  {reason}"

        # ── STEP 2: ADD TO HISTORY ───────────────────────────────────────────
        # Add the user message now. If the backend fails, we'll remove it.
        self.history.append({"role": "user", "content": user_input})
        self._trim_history()

        # ── STEP 3: CALL THE BACKEND ─────────────────────────────────────────
        # Import only the backend we actually need. This keeps the remote mode
        # free of the `ollama` dependency and local mode free of `google-genai`.
        try:
            if self.config.mode == "local":
                from chatbot.backends import local
                raw = local.get_response(
                    history=self.history,
                    system_prompt=self.config.system_prompt,
                    model=self.config.local_model,
                )
            else:
                from chatbot.backends import remote
                raw = remote.get_response(
                    history=self.history,
                    system_prompt=self.config.system_prompt,
                    model=self.config.remote_model,
                    api_key=self.config.api_key,
                )
        except Exception as exc:
            # Something went wrong (no internet, Ollama not running, invalid key…).
            # Remove the user's message so the next turn starts cleanly.
            self.history.pop()
            return f"❌  Error connecting to the model: {exc}"

        # ── STEP 4: OUTPUT GUARDRAIL ─────────────────────────────────────────
        # Check the model's response before showing it.
        # If blocked, we also undo the user's message from history.
        ok, result = output_guard.validate(raw)
        if not ok:
            self.history.pop()
            return f"⚠️  {result}"

        # ── STEP 5: SAVE RESPONSE AND RETURN ────────────────────────────────
        # Gemini uses "model" as the assistant role; Ollama uses "assistant".
        # We store whatever convention matches the current backend.
        assistant_role = "model" if self.config.mode == "remote" else "assistant"
        self.history.append({"role": assistant_role, "content": result})

        return result
