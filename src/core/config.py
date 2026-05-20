from functools import lru_cache
from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

class LLMSettings(BaseSettings):
    api_key: SecretStr | None = None
    base_url: str = "https://openrouter.ai/api/v1"
    model: str = "google/gemini-2.5-flash"

    model_config = SettingsConfigDict(
        env_file=".env", 
        env_prefix="LLM_", 
        extra="ignore"
    )

class DatabaseSettings(BaseSettings):
    user: str = "postgres"
    password: str
    db: str = "postgres"
    host: str = "localhost"
    port: str = "5432"

    @property
    def database_url(self) -> str:
        return f"postgresql+asyncpg://{self.user}:{self.password}@{self.host}:{self.port}/{self.db}"

    model_config = SettingsConfigDict(
        env_file=".env", 
        env_prefix="POSTGRES_", 
        extra="ignore"
    )

@lru_cache
def get_settings() -> LLMSettings:
    return LLMSettings()

@lru_cache
def get_db_settings() -> DatabaseSettings:
    return DatabaseSettings()