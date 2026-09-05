from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field, field_validator
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _read_prompt(filename: str) -> str:
    return (PROJECT_ROOT / "prompts" / filename).read_text(encoding="utf-8").strip()

class BaseChatConfig(BaseSettings):
    """
    Base configuration shared by all modes.
    """
    model_config = SettingsConfigDict(env_file=PROJECT_ROOT / '.env', env_file_encoding='utf-8', extra='ignore')

    chat_mode: str
    system_prompt: str = Field(default_factory=lambda: _read_prompt("system.txt"))
    judge_system_prompt: str = Field(default_factory=lambda: _read_prompt("judge.txt"))
    max_history_turns: int = Field(default=10)

    @field_validator("chat_mode", mode="before")
    @classmethod
    def normalize_chat_mode(cls, value: str) -> str:
        normalized = str(value).strip().lower()
        aliases = {
            "local": "ollama",
            "remote": "gemini",
        }
        normalized = aliases.get(normalized, normalized)
        if normalized not in {"gemini", "ollama"}:
            raise ValueError("CHAT_MODE must be one of: gemini, ollama, remote, local")
        return normalized

    @property
    def mode(self) -> str:
        return self.chat_mode.lower()

class GeminiConfig(BaseChatConfig):
    """Fields exclusive to the Gemini (remote) backend."""
    model_name: str = Field(default="gemini-2.5-flash")
    api_key: str

class OllamaConfig(BaseChatConfig):
    """Fields exclusive to the Ollama backend."""
    model_name: str = Field(default="llama3.2")
    base_url: str = Field(default="http://localhost:11434")

def load_config() -> BaseChatConfig:
    """Reads the CHAT_MODE first and returns the appropriate exclusive config."""
    base = BaseChatConfig()
    if base.mode == "gemini":
        return GeminiConfig()
    if base.mode == "ollama":
        return OllamaConfig()
    raise ValueError(f"Unsupported chat mode: {base.mode}")
