"""
gemini.py

Sends the conversation history to Google Gemini and returns the response.

Reference documentation:
    https://ai.google.dev/gemini-api/docs/quickstart
    https://googleapis.github.io/python-genai/
"""

from google import genai
from google.genai import types

class GeminiBackend:
    assistant_role: str = "model"

    def __init__(self, model: str, api_key: str, system_prompt: str) -> None:
        self.model = model
        self.system_prompt = system_prompt
        self._client = genai.Client(api_key=api_key)

    def ping(self) -> None:
        self._client.models.get(model=self.model)

    def get_response(self, history: list[dict]) -> str:
        """
        Sends the full conversation history to Gemini and returns the response text.
        """
        contents = [
            types.Content(
                role=msg["role"],
                parts=[types.Part(text=msg["content"])],
            )
            for msg in history
        ]

        config = types.GenerateContentConfig(
            system_instruction=self.system_prompt,
            max_output_tokens=1024,
            temperature=0.7,
        )

        response = self._client.models.generate_content(
            model=self.model,
            contents=contents,
            config=config,
        )

        return response.text
