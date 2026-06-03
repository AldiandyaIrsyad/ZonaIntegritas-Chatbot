"""
Dependency injection for the LLM module.

Provides a request-scoped singleton for the LLM application service.
"""
from functools import lru_cache

from app.core.config import get_llm_settings
from app.core.interfaces.llm import ILLMService
from app.infra.llm_connection import LLMConnection

from .service import LLMService


@lru_cache
def get_llm_service() -> ILLMService:
    """Factory function for Dependency Injection.
    
    Isolates the instantiation logic from the consuming vertical slices.

    Returns:
        ILLMService: The configured LLM service singleton.
    """
    settings = get_llm_settings()
    connection = LLMConnection(
        base_url=settings.base_url,
        api_key=settings.api_key,
        default_headers=settings.default_headers
    )
    return LLMService(
        connection=connection,
        model=settings.model,
        max_tokens=settings.max_tokens,
        max_completion_tokens=settings.max_completion_tokens
    )
