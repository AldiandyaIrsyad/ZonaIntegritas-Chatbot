from functools import lru_cache

from src.core.config import get_ivm_settings
from src.infra import PromptGuardProvider
from src.rag.dependency import get_embedding_provider, get_vector_store

from .service import IVMService

@lru_cache
def get_ivm_service() -> IVMService:
    """
    Factory function for Dependency Injection.
    Isolates the instantiation logic from the consuming vertical slices.
    """
    settings = get_ivm_settings()
    embedding_provider = get_embedding_provider()
    vector_store = get_vector_store()
    prompt_guard = PromptGuardProvider()

    return IVMService(
        prompt_guard=prompt_guard,
        security_threshold=settings.security_threshold,
        similarity_threshold=settings.similarity_threshold,
        embedding_provider=embedding_provider,
        vector_store=vector_store,
    )

