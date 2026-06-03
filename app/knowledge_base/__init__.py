"""
Knowledge Base module.

Handles administration, ingestion, and management of global PDF documents
used for Retrieval-Augmented Generation (RAG).
"""
from .dependency import get_pdf_service
from .model import PDFDocument
from .repository import PDFRepository
from .api import router as kb_api_router
from .presentation import router as kb_presentation_router
from .service import KnowledgeBase

__all__ = [
    "kb_api_router",
    "kb_presentation_router",
    "get_pdf_service",
    "KnowledgeBase",
    "PDFRepository",
    "PDFDocument",
]