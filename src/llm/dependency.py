from functools import lru_cache

from src.core import LLMSettings, get_settings
from src.infra import LLMConnection

from .service import LLMService


@lru_cache
def get_llm_service() -> LLMService:
    """
    Factory function for Dependency Injection.
    Isolates the instantiation logic from the consuming vertical slices.
    """
    settings = get_settings()
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
