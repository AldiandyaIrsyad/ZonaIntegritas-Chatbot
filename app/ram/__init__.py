"""
Response Assessment Module (RAM)

RAM validates LLM-generated output against the retrieved knowledge-base context
using Natural Language Inference (NLI). It operates sentence-by-sentence within
the streaming pipeline, annotating contradicted statements inline.
"""
from .service import RAMService

__all__ = ["RAMService"]
