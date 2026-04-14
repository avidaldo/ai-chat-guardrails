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
) -> str:
    """
    Sends the conversation history to a locally running Ollama model.

    Args:
        history:       Accumulated messages. Each entry is a dict:
                           {"role": "user" | "assistant", "content": "<text>"}
                       Note: Ollama follows the OpenAI convention and uses
                       "assistant" (not "model") for the assistant role.
        system_prompt: Injected as the first message with role "system".
                       Ollama handles the system prompt as part of the message list,
                       unlike Gemini which has a dedicated field for it.
        model:         Name of the locally installed model. Examples:
                           "llama3.2"  ← good balance of quality and speed, ~2 GB
                           "mistral"   ← fast on CPU, good for low-resource machines
                           "phi4"      ← very lightweight, ideal without a GPU
                       Run `ollama list` to see what is installed on your machine.

    Returns:
        The text content of the model's response.

    Key difference from remote mode:
        No API key needed. No internet connection needed.
        The model runs 100% on your local machine.
        Performance depends heavily on your hardware — a GPU significantly
        accelerates inference, but CPU-only works fine for smaller models.

    Before using this backend:
        1. Install Ollama: curl -fsSL https://ollama.com/install.sh | sh
        2. Pull a model:   ollama pull llama3.2
        3. Start server:   ollama serve  (or it may start automatically)
    """

    # Ollama's message format: a list of {"role": ..., "content": ...} dicts.
    # The system prompt is the first message with role "system".
    messages: list[dict] = [{"role": "system", "content": system_prompt}]

    # Append the conversation history.
    # We translate "model" → "assistant" to handle cases where the history
    # was built while in remote mode (Gemini uses "model" as the role name).
    for msg in history:
        messages.append({
            "role": msg["role"].replace("model", "assistant"),
            "content": msg["content"],
        })

    # ollama.chat() blocks until the full response is generated and returns it.
    # For streaming (printing tokens as they arrive), pass stream=True — you
    # then iterate over the generator: for chunk in ollama.chat(..., stream=True)
    response = ollama.chat(model=model, messages=messages)

    return response.message.content
