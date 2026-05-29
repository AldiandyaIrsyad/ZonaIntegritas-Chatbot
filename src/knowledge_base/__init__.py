"""
Knowledge Base module.

Handles administration, ingestion, and management of global PDF documents
used for Retrieval-Augmented Generation (RAG).
"""
from .dependency import get_pdf_service
from .model import PDFDocument
from .repository import PDFRepository
from .router import router as kb_router
from .service import KnowledgeBase

__all__ = [
    "kb_router",
    "get_pdf_service",
    "KnowledgeBase",
    "PDFRepository",
    "PDFDocument",
]