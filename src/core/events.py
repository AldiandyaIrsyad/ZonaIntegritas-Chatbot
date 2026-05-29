"""
Event enumerations for logging and tracing.
"""
from enum import Enum


class LogEvent(str, Enum):
    """Structured event names for consistent logging across the application."""
    ADMIN_UPLOAD_PDF = "admin_upload_pdf"
    USER_UPLOAD_PDF = "user_upload_pdf"
    RAG_INGESTION = "rag_ingestion"
    LLM_GENERATION = "llm_generation"
    VECTOR_UPSERT = "vector_upsert"
    VECTOR_SEARCH = "vector_search"
    RAG_RETRIEVAL = "rag_retrieval"
