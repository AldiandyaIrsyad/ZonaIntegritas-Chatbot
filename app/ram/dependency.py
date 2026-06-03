"""
Dependency injection for the RAM (Response Assessment Module) pipeline.

NLIProvider and RAMService are singletons — the HTTP client is created once
at startup and shared across all requests. The underlying model runs inside
the Infinity server; this factory only wires up the HTTP adapter.
"""
from functools import lru_cache

import structlog

from app.core.config import get_infinity_settings, get_ram_settings
from app.core.interfaces.ai import INLIProvider
from app.core.interfaces.ram import IRAMService
from app.infra.nli import NLIProvider
from app.rag.dependency import get_embedding_provider

from .service import RAMService

logger = structlog.get_logger(__name__)

@lru_cache
def get_nli_provider() -> INLIProvider:
    """Singleton NLIProvider — HTTP client initialised lazily on the first request.

    Returns:
        INLIProvider: The configured NLI provider singleton.
    """
    inf = get_infinity_settings()
    logger.info("Initializing NLI provider")
    return NLIProvider(
        base_url=inf.base_url,
        model=inf.nli_model,
    )


@lru_cache
def get_ram_service() -> IRAMService:
    """Singleton RAMService — reuses the singleton NLIProvider.

    Returns:
        IRAMService: The configured RAM service singleton.
    """
    settings = get_ram_settings()
    logger.info("Initializing RAM service", enabled=settings.nli_enabled)
    return RAMService(
        nli=get_nli_provider(),
        embedding_provider=get_embedding_provider(),
        enabled=settings.nli_enabled,
    )
