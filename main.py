"""
main.py

Entry point for the guardrailed chatbot.

Responsibilities of this file (and ONLY this file):
    - Load configuration using Pydantic Settings
    - Setup logging
    - Instantiate ChatEngine
    - Run the terminal conversation loop
"""

from chatbot.config import load_config
from chatbot.engine import ChatEngine

def main() -> None:
    
    try:
        config = load_config()
    except Exception as e:
        print(f"❌ Configuration error: {e}")
        return

    engine = ChatEngine(config)

    print(f"\n🤖  Guardrailed Chatbot  |  Mode: {config.mode.upper()}")
    print("    Type 'exit' to quit.\n")

    while True:

        # ── Get user input ─────────────────────────────────────────
        try:
            user_input = input("You: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nBot: Goodbye!")
            break

        if not user_input:
            continue

        if user_input.lower() in {"exit", "quit", "bye"}:
            print("Bot: Goodbye!")
            break

        # ── Get bot response ─────────────────────────────────────────
        response = engine.chat(user_input)
        print(f"\nBot: {response}\n")

if __name__ == "__main__":
    main()
