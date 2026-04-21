from typing import Protocol, runtime_checkable

from chatbot.config import BaseChatConfig


@runtime_checkable
class BackendProtocol(Protocol):
    def get_response(self, history: list[dict]) -> str:
        ...


def create_backend(config: BaseChatConfig) -> "BackendProtocol":
    return _build_backend(config, config.system_prompt)


def create_judge_backend(config: BaseChatConfig) -> "BackendProtocol":
    return _build_backend(config, config.judge_system_prompt)


def _build_backend(config: BaseChatConfig, system_prompt: str) -> "BackendProtocol":
    if config.mode == "remote":
        from chatbot.backends.remote import RemoteBackend
        return RemoteBackend(
            model=config.model_name,
            api_key=config.api_key,
            system_prompt=system_prompt,
        )
    else:
        from chatbot.backends.local import LocalBackend
        return LocalBackend(
            model=config.model_name,
            system_prompt=system_prompt,
        )
