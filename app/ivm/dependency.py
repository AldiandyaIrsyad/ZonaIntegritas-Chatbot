"""
Dependency injection for the Input Validation Module (IVM).

Provides request-scoped and cached singletons for input security checks.
"""
from functools import lru_cache

from app.core.config import get_infinity_settings, get_ivm_settings
from app.core.interfaces.ivm import IIVMService
from app.infra import PromptGuardProvider
from app.rag.dependency import get_embedding_provider, get_vector_store

from .service import IVMService


@lru_cache
def get_ivm_service() -> IIVMService:
    """Factory function for Dependency Injection.
    
    Isolates the instantiation logic from the consuming vertical slices.

    Returns:
        IIVMService: The configured IVM service singleton.
    """
    settings = get_ivm_settings()
    inf = get_infinity_settings()
    embedding_provider = get_embedding_provider()
    vector_store = get_vector_store()
    prompt_guard = PromptGuardProvider(
        base_url=inf.base_url,
        model=inf.prompt_guard_model,
        security_threshold=settings.security_threshold,
    )

    return IVMService(
        prompt_guard=prompt_guard,
        security_threshold=settings.security_threshold,
        similarity_threshold=settings.similarity_threshold,
        embedding_provider=embedding_provider,
        vector_store=vector_store,
    )
