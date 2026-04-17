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
    **kwargs,
) -> str:
    """
    Sends the full conversation history to Gemini and returns the response text.
    """
    api_key = kwargs.get("api_key")
    if not api_key:
        raise ValueError("api_key is required for RemoteBackend")
        
    client = genai.Client(api_key=api_key)

    contents = [
        types.Content(
            role=msg["role"],
            parts=[types.Part(text=msg["content"])],
        )
        for msg in history
    ]

    config = types.GenerateContentConfig(
        system_instruction=system_prompt,
        max_output_tokens=1024,
        temperature=0.7,
    )

    response = client.models.generate_content(
        model=model,
        contents=contents,
        config=config,
    )

    return response.text
