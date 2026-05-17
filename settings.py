from functools import lru_cache
from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

class LLMSettings(BaseSettings):
    api_key: SecretStr | None = None
    use_local: bool = False
    base_url: str = "https://openrouter.ai/api/v1"
    model: str = "google/gemini-2.5-flash"

    model_config = SettingsConfigDict(
        env_file=".env", 
        env_prefix="LLM_", 
        extra="ignore"
    )

@lru_cache
def get_settings() -> LLMSettings:
    return LLMSettings()