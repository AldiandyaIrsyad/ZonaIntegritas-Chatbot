"""Centralised Pydantic-Settings configuration for the whole application.

This module is the single source of truth for environment-driven settings.
Each settings group is a ``BaseSettings`` subclass with its own env prefix so
that one ``.env`` file can configure every bounded context (chat, kb) and
the shared infrastructure (logging, database) without cross-talk.

Settings groups:
    - :class:`AppSettings`      — FastAPI app metadata (``APP_*`` env vars).
    - :class:`LoggerSettings`   — structlog + Vector TCP sink (``LOGGER_*``).
    - :class:`DatabaseSettings` — PostgreSQL connection (``POSTGRES_*``).

Each ``get_*`` factory is ``@lru_cache``-d so the same singleton instance is
returned for the lifetime of the process; tests that need to override
settings should call ``get_*_settings.cache_clear()``.

Cross-cutting infrastructure (``shared/``, see ``docs/02-arsitektur.md``
§2.1-2.2): this specific module is consumed directly by ``app/main.py``
(``get_app_settings``) and indirectly by ``app/shared/db.py`` and
``app/shared/logging.py`` (``get_db_settings`` / ``get_logger_settings``).
Bounded-context settings (``ChatConfig``, ``KBConfig``) live in their own
``chat/config.py`` / ``kb/config.py`` modules, not here. Per the
``thesis/`` purity rule, ``thesis/{chunking,ivm,ram,prompts}`` may not
import this module at all; ``thesis/vlm`` is httpx-only and doesn't import
it either — every concrete setting reaches the research core only as a
plain value passed in by the ``chat``/``kb`` composition roots.
"""

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict
from functools import lru_cache


class AppSettings(BaseSettings):
    """FastAPI application metadata (env prefix: ``APP_``).

    Attributes:
        title: Window/tab title and OpenAPI title.
        version: API version string surfaced by ``/docs``.
        host: Bind address for uvicorn.
        port: Bind port for uvicorn.
        reload: Enable uvicorn auto-reload (dev only).
    """

    title: str = "Chat Application with PDF Knowledge Base"
    version: str = "1.0.0"
    host: str = "0.0.0.0"
    port: int = Field(default=8000)
    reload: bool = True

    model_config = SettingsConfigDict(env_file=".env", env_prefix="APP_", extra="ignore")


@lru_cache
def get_app_settings() -> AppSettings:
    """Return the cached :class:`AppSettings` singleton."""
    return AppSettings()


class LoggerSettings(BaseSettings):
    """Logging sink configuration (env prefix: ``LOGGER_``).

    Attributes:
        vector_host: Host of the Vector TCP socket that ingests JSON logs.
        vector_port: Port of the Vector TCP socket.
        log_level: Root log level (e.g. ``"INFO"``, ``"DEBUG"``).
    """

    vector_host: str = "localhost"
    vector_port: int = Field(default=9000)
    log_level: str = Field(default="INFO")

    model_config = SettingsConfigDict(env_file=".env", env_prefix="LOGGER_", extra="ignore")


@lru_cache
def get_logger_settings() -> LoggerSettings:
    """Return the cached :class:`LoggerSettings` singleton."""
    return LoggerSettings()


class DatabaseSettings(BaseSettings):
    """PostgreSQL connection settings (env prefix: ``POSTGRES_``).

    Attributes:
        user: Database role.
        password: Database role password.
        db: Database name.
        port: PostgreSQL port.
        host: PostgreSQL host.
    """

    user: str = Field(default="postgres")
    password: str = Field(default="postgres")
    db: str = Field(default="postgres")
    port: int = Field(default=5432)
    host: str = Field(default="localhost")

    model_config = SettingsConfigDict(env_file=".env", env_prefix="POSTGRES_", extra="ignore")

    @property
    def async_database_url(self) -> str:
        """Build the asyncpg SQLAlchemy URL from the individual fields."""
        return f"postgresql+asyncpg://{self.user}:{self.password}@{self.host}:{self.port}/{self.db}"


@lru_cache
def get_db_settings() -> DatabaseSettings:
    """Return the cached :class:`DatabaseSettings` singleton."""
    return DatabaseSettings()