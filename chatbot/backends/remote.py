"""
remote.py

Sends the conversation history to Google Gemini and returns the response.

Reference documentation:
    https://ai.google.dev/gemini-api/docs/quickstart
    https://googleapis.github.io/python-genai/
"""

from google import genai
from google.genai import types


def get_response(
    history: list[dict],
    system_prompt: str,
    model: str,
    api_key: str,
) -> str:
    """
    Sends the full conversation history to Gemini and returns the response text.

    Args:
        history:       Accumulated messages. Each entry is a dict:
                           {"role": "user" | "model", "content": "<text>"}
                       Gemini uses "model" (not "assistant") as the assistant role.
        system_prompt: Fixed instruction that defines the chatbot's personality.
                       Gemini receives this as a separate field — NOT as a message
                       in the history — which is why we pass it explicitly here.
        model:         Gemini model identifier. Current options (April 2026):
                           "gemini-2.5-flash"   ← fast, cheap, good for development
                           "gemini-2.5-pro"     ← more capable, higher cost per token
        api_key:       Key from https://aistudio.google.com/apikey
                       The free tier is sufficient for classroom use.

    Returns:
        The text content of the model's response.

    Design note — why we create a new client per call:
        This is slightly less efficient than reusing a client, but it makes the
        function self-contained and stateless, which is easier to understand and
        test. For production use you would create the client once and reuse it.
    """
    client = genai.Client(api_key=api_key)

    # Convert our simple dict history into the SDK's Content objects.
    # Each Content has:
    #   - role: "user" or "model"
    #   - parts: a list of Part objects (each Part can be text, image, audio, etc.)
    # In this example every message is plain text, so each Content has exactly one Part.
    contents = [
        types.Content(
            role=msg["role"],
            parts=[types.Part(text=msg["content"])],
        )
        for msg in history
    ]

    # GenerateContentConfig bundles all generation parameters together.
    # system_instruction goes here, outside the message history.
    config = types.GenerateContentConfig(
        system_instruction=system_prompt,
        max_output_tokens=1024,  # cap the response length to control cost
        temperature=0.7,         # 0.0 = fully deterministic, 1.0 = most creative
    )

    response = client.models.generate_content(
        model=model,
        contents=contents,
        config=config,
    )

    # response.text is the simplest accessor when the response has a single text part.
    # For more complex responses (multi-modal, function calls) you would inspect
    # response.candidates[0].content.parts instead.
    return response.text
