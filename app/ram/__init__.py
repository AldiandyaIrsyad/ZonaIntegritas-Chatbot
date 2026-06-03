"""
Response Assessment Module (RAM)

RAM validates LLM-generated output against the retrieved knowledge-base context
using Natural Language Inference (NLI). It operates sentence-by-sentence within
the streaming pipeline, annotating contradicted statements inline.
"""
from .api import router as ram_api_router
from .presentation import router as ram_presentation_router
from .service import RAMService

__all__ = ["RAMService", "ram_api_router", "ram_presentation_router"]
