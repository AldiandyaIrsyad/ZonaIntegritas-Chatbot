from src.core.config import LLMSettings
from src.infra.llm_connection import LLMConnection
from src.llm.service import LLMService

from functools import lru_cache

@lru_cache
def get_llm_service(settings: LLMSettings) -> LLMService:
    """
    Factory function for Dependency Injection.
    Isolates the instantiation logic from the consuming vertical slices.
    """
    connection = LLMConnection(
        base_url=settings.base_url,
        api_key=settings.api_key
    )
    return LLMService(
        connection=connection,
        model=settings.model,
        max_tokens=settings.max_tokens,
        max_completion_tokens=settings.max_completion_tokens
    )
