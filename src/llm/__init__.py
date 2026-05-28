"""LLM Package."""

from .dependency import get_llm_service
from .service import LLMService

__all__ = [
    "LLMService",
    "get_llm_service",
]
