from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field
from pathlib import Path

class BaseChatConfig(BaseSettings):
    """
    Base configuration shared by all modes.
    """
    model_config = SettingsConfigDict(env_file='.env', env_file_encoding='utf-8', extra='ignore')

    chat_mode: str
    system_prompt: str = Field(default_factory=lambda: Path("prompts/system.txt").read_text(encoding="utf-8").strip())
    judge_system_prompt: str = Field(default_factory=lambda: Path("prompts/judge.txt").read_text(encoding="utf-8").strip())
    max_history_turns: int = Field(default=10)

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
    else:
        return OllamaConfig()
