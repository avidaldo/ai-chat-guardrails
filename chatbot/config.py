from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field
from pathlib import Path

class BaseChatConfig(BaseSettings):
    """
    Base configuration shared by all modes.
    """
    model_config = SettingsConfigDict(env_file='.env', env_file_encoding='utf-8', extra='ignore')

    chat_mode: str = Field(default="remote")
    system_prompt: str = Field(default_factory=lambda: Path("prompts/system.txt").read_text(encoding="utf-8").strip())
    max_history_turns: int = Field(default=10)

    @property
    def mode(self) -> str:
        return self.chat_mode.lower()

class RemoteConfig(BaseChatConfig):
    """Fields exclusive to the remote mode."""
    model_name: str = Field(default="gemini-2.5-flash")
    api_key: str

class LocalConfig(BaseChatConfig):
    """Fields exclusive to the local mode."""
    model_name: str = Field(default="llama3.2")
    base_url: str = Field(default="http://localhost:11434")

def load_config() -> BaseChatConfig:
    """Reads the CHAT_MODE first and returns the appropriate exclusive config."""
    base = BaseChatConfig()
    if base.mode == "remote":
        return RemoteConfig()
    else:
        return LocalConfig()
