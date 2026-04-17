"""
main.py

Entry point for the guardrailed chatbot.

Responsibilities of this file (and ONLY this file):
    - Load configuration using Pydantic Settings
    - Setup logging
    - Instantiate ChatEngine
    - Run the terminal conversation loop
"""

import logging
from chatbot.config import load_config
from chatbot.engine import ChatEngine

def setup_logging():
    logging.basicConfig(
        level=logging.WARNING, # Default to warning to keep CLI clean
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

def main() -> None:
    setup_logging()
    
    try:
        config = load_config()
    except Exception as e:
        print(f"❌ Configuration error: {e}")
        return

    engine = ChatEngine(config)

    print(f"\n🤖  Guardrailed Chatbot  |  Mode: {config.mode.upper()}")
    print("    Type 'exit' to quit.\n")

    while True:
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

        response = engine.chat(user_input)
        print(f"\nBot: {response}\n")

if __name__ == "__main__":
    main()
