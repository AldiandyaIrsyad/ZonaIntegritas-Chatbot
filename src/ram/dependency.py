"""
Dependency injection for the RAM (Response Assessment Module) pipeline.

NLIProvider and RAMService are singletons — the model loads once at startup
and is shared across all requests.
"""
from functools import lru_cache

from src.core.config import get_ram_settings
from src.infra.nli import NLIProvider

from .service import RAMService


@lru_cache
def get_nli_provider() -> NLIProvider:
    """Singleton NLIProvider — model loads once at startup."""
    settings = get_ram_settings()
    return NLIProvider(
        model=settings.nli_model,
        device=settings.nli_device,
        max_length=settings.nli_max_length,
    )


@lru_cache
def get_ram_service() -> RAMService:
    """Singleton RAMService — reuses the singleton NLIProvider."""
    settings = get_ram_settings()
    return RAMService(
        nli=get_nli_provider(),
        enabled=settings.nli_enabled,
    )
