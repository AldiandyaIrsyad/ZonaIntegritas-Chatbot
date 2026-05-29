"""
RAG (Retrieval-Augmented Generation) Module.

Exposes core components for document ingestion and context retrieval.
"""
from .chunking import create_parent_chunks, split_into_children
from .dependency import (
    get_document_parser,
    get_embedding_provider,
    get_ingestion_service,
    get_reranker,
    get_retrieval_service,
    get_vector_store,
)
from .ingestion import IngestionService
from .model import IngestionTask, ParentChunk
from .repository import RAGRepository
from .retrieval import RetrievalService, RetrievedContext

__all__ = [
    "get_retrieval_service",
    "get_ingestion_service",
    "get_document_parser",
    "get_embedding_provider",
    "get_vector_store",
    "get_reranker",
    "IngestionService",
    "RetrievalService",
    "RetrievedContext",
    "RAGRepository",
    "IngestionTask",
    "ParentChunk",
    "create_parent_chunks",
    "split_into_children",
]
