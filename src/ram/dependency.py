"""
Dependency injection for the RAM (Response Assessment Module) pipeline.

NLIProvider and RAMService are singletons — the HTTP client is created once
at startup and shared across all requests. The underlying model runs inside
the Infinity server; this factory only wires up the HTTP adapter.
"""
from functools import lru_cache

from src.core.config import get_infinity_settings, get_ram_settings
from src.infra.nli import NLIProvider

from .service import RAMService


@lru_cache
def get_nli_provider() -> NLIProvider:
    """Singleton NLIProvider — HTTP client initialised lazily on the first request.

    Returns:
        NLIProvider: The configured NLI provider singleton.
    """
    inf = get_infinity_settings()
    return NLIProvider(
        base_url=inf.base_url,
        model=inf.nli_model,
    )


@lru_cache
def get_ram_service() -> RAMService:
    """Singleton RAMService — reuses the singleton NLIProvider.

    Returns:
        RAMService: The configured RAM service singleton.
    """
    settings = get_ram_settings()
    return RAMService(
        nli=get_nli_provider(),
        enabled=settings.nli_enabled,
    )
