"""
ollama.py

Sends the conversation history to an Ollama model and returns the response.

Ollama is a tool that runs open-weight LLMs (Llama, Mistral, Phi…) locally
or remotely. It exposes an HTTP server that this library talks to.

Library: ollama >= 0.4  (the official Python client)

References:
    https://github.com/ollama/ollama-python   ← Python client source and docs
    https://ollama.com/library                ← catalog of available models
    https://ollama.com/download               ← installation instructions
"""

import ollama


class OllamaBackend:
    assistant_role: str = "assistant"

    def __init__(self, model: str, system_prompt: str) -> None:
        self.model = model
        self.system_prompt = system_prompt

    def ping(self) -> None:
        ollama.list()

    def get_response(self, history: list[dict]) -> str:
        """
        Sends the conversation history to an Ollama model.
        """
        messages: list[dict] = [{"role": "system", "content": self.system_prompt}]

        for msg in history:
            messages.append({
                "role": msg["role"],
                "content": msg["content"],
            })

        response = ollama.chat(model=self.model, messages=messages)
        return response.message.content
