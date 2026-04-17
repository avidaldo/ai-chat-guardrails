"""
local.py

Sends the conversation history to a local Ollama model and returns the response.

Ollama is a tool that runs open-weight LLMs (Llama, Mistral, Phi…) locally
on your machine. It exposes a local HTTP server that this library talks to.

Library: ollama >= 0.4  (the official Python client)

References:
    https://github.com/ollama/ollama-python   ← Python client source and docs
    https://ollama.com/library                ← catalog of available models
    https://ollama.com/download               ← installation instructions
"""

import ollama

def get_response(
    history: list[dict],
    system_prompt: str,
    model: str,
    **kwargs,
) -> str:
    """
    Sends the conversation history to a locally running Ollama model.
    """
    messages: list[dict] = [{"role": "system", "content": system_prompt}]

    for msg in history:
        messages.append({
            "role": msg["role"].replace("model", "assistant"),
            "content": msg["content"],
        })

    response = ollama.chat(model=model, messages=messages)
    return response.message.content
