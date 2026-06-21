# app/shared/config.py

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict
from functools import lru_cache

class AppSettings(BaseSettings):
    title: str = "Chat Application with PDF Knowledge Base"
    version: str = "1.0.0"
    host: str = "0.0.0.0"
    port: int = Field(default=8000)
    reload: bool = True

    model_config = SettingsConfigDict(env_file=".env", env_prefix="APP_", extra="ignore")

@lru_cache
def get_app_settings() -> AppSettings: return AppSettings()


class LoggerSettings(BaseSettings):
    vector_host: str = "localhost"
    vector_port: int = Field(default=9000)
    log_level: str = Field(default="INFO")

    model_config = SettingsConfigDict(env_file=".env", env_prefix="LOGGER_", extra="ignore")


@lru_cache
def get_logger_settings() -> LoggerSettings: return LoggerSettings()

class DatabaseSettings(BaseSettings):
    user: str = Field(default="postgres")
    password: str = Field(default="postgres")
    db: str = Field(default="postgres")
    port: int = Field(default=5432)
    host: str = Field(default="localhost")

    model_config = SettingsConfigDict(env_file=".env", env_prefix="POSTGRES_", extra="ignore")

    @property
    def async_database_url(self) -> str:
        return f"postgresql+asyncpg://{self.user}:{self.password}@{self.host}:{self.port}/{self.db}"


@lru_cache
def get_db_settings() -> DatabaseSettings: return DatabaseSettings()