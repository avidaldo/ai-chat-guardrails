"""
main.py

Entry point for the guardrailed chatbot.

Responsibilities of this file (and ONLY this file):
    - Load configuration from the .env file
    - Instantiate ChatEngine
    - Run the terminal conversation loop

All the interesting logic (guardrails, backends, history management) lives in
the chatbot/ package. Keeping main.py thin makes it easy to swap out the UI:
to build a web interface you would create a gradio_app.py that imports ChatEngine
and calls engine.chat() — no changes needed anywhere else.
"""

import os

from dotenv import load_dotenv

from chatbot.engine import ChatConfig, ChatEngine

# load_dotenv() reads the .env file in the current directory and sets each
# variable as an environment variable. It does nothing if .env doesn't exist,
# which is fine — the variables may already be set by the shell or CI system.
load_dotenv()


def load_config() -> ChatConfig:
    """
    Reads environment variables and builds the ChatConfig dataclass.

    Design decision — fail early:
        If CHAT_MODE=remote but GEMINI_API_KEY is not set, we raise a ValueError
        immediately with a clear, actionable message. The alternative — letting
        the program start and failing on the first API call — would be confusing.

    All os.getenv() calls have a sensible default so the chatbot can run with
    minimal configuration (just set GEMINI_API_KEY and you're done).
    """
    mode = os.getenv("CHAT_MODE", "remote").lower()
    api_key = os.getenv("GEMINI_API_KEY", "")

    if mode == "remote" and not api_key:
        raise ValueError(
            "CHAT_MODE=remote requires GEMINI_API_KEY to be set in the .env file.\n"
            "Get a free key at: https://aistudio.google.com/apikey"
        )

    return ChatConfig(
        mode=mode,
        remote_model=os.getenv("REMOTE_MODEL", "gemini-2.5-flash"),
        local_model=os.getenv("LOCAL_MODEL", "llama3.2"),
        ollama_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
        api_key=api_key,
        system_prompt=os.getenv(
            "SYSTEM_PROMPT",
            "You are a helpful and concise technical assistant.",
        ),
        max_history_turns=int(os.getenv("MAX_HISTORY_TURNS", 10)),
    )


def main() -> None:
    config = load_config()
    engine = ChatEngine(config)

    print(f"\n🤖  Guardrailed Chatbot  |  Mode: {config.mode.upper()}")
    print("    Type 'exit' to quit.\n")

    while True:
        try:
            user_input = input("You: ").strip()
        except (KeyboardInterrupt, EOFError):
            # Handle Ctrl+C and EOF (e.g., piped input) gracefully
            print("\nBot: Goodbye!")
            break

        if not user_input:
            continue  # ignore blank lines

        if user_input.lower() in {"exit", "quit", "bye"}:
            print("Bot: Goodbye!")
            break

        response = engine.chat(user_input)
        print(f"\nBot: {response}\n")


if __name__ == "__main__":
    main()
