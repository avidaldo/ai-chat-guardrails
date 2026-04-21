from typing import Protocol, runtime_checkable

from chatbot.config import BaseChatConfig


@runtime_checkable
class BackendProtocol(Protocol):
    assistant_role: str

    def get_response(self, history: list[dict]) -> str:
        ...

    def ping(self) -> None:
        ...


def create_backend(config: BaseChatConfig) -> "BackendProtocol":
    return _build_backend(config, config.system_prompt)

def create_judge_backend(config: BaseChatConfig) -> "BackendProtocol":
    return _build_backend(config, config.judge_system_prompt)


def _build_backend(config: BaseChatConfig, system_prompt: str) -> "BackendProtocol":
    if config.mode == "gemini":
        from chatbot.backends.gemini import GeminiBackend
        return GeminiBackend(
            model=config.model_name,
            api_key=config.api_key,
            system_prompt=system_prompt,
        )
    else:
        from chatbot.backends.ollama import OllamaBackend
        return OllamaBackend(
            model=config.model_name,
            system_prompt=system_prompt,
        )
