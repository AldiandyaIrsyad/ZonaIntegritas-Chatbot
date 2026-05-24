"""
Dependency injection for the RAG pipeline.

Provides singleton factory functions for infrastructure adapters
and request-scoped factories for services that need a DB session.
"""
from fastapi import Depends
from functools import lru_cache
from sqlalchemy.ext.asyncio import AsyncSession

from src.core.config import (
    get_infinity_settings,
    get_qdrant_settings,
    get_unstructured_settings,
)
from src.core.database import get_db
from src.infra.document_parser import DocumentParser
from src.infra.embedding_provider import EmbeddingProvider
from src.infra.reranker import Reranker
from src.infra.vector_store import QdrantStore
from src.rag.ingestion import IngestionService
from src.rag.retrieval import RetrievalService


@lru_cache
def get_vector_store() -> QdrantStore:
    """Singleton QdrantStore instance."""
    settings = get_qdrant_settings()
    return QdrantStore(
        host=settings.host,
        port=settings.port,
        collection_name=settings.collection_name,
    )


@lru_cache
def get_embedding_provider() -> EmbeddingProvider:
    """Singleton EmbeddingProvider instance."""
    settings = get_infinity_settings()
    return EmbeddingProvider(
        base_url=settings.base_url,
        model=settings.embedding_model,
    )


@lru_cache
def get_reranker() -> Reranker:
    """Singleton Reranker instance."""
    settings = get_infinity_settings()
    return Reranker(
        base_url=settings.base_url,
        model=settings.reranker_model,
    )


@lru_cache
def get_document_parser() -> DocumentParser:
    """Singleton DocumentParser instance."""
    settings = get_unstructured_settings()
    return DocumentParser(base_url=settings.base_url)


def get_retrieval_service(
    db: AsyncSession = Depends(get_db),
    embedding_provider: EmbeddingProvider = Depends(get_embedding_provider),
    vector_store: QdrantStore = Depends(get_vector_store),
    reranker: Reranker = Depends(get_reranker),
) -> RetrievalService:
    """Request-scoped RetrievalService with DB session."""
    return RetrievalService(
        db=db,
        embedding_provider=embedding_provider,
        vector_store=vector_store,
        reranker=reranker,
    )


def get_ingestion_service(
    db: AsyncSession = Depends(get_db),
    document_parser: DocumentParser = Depends(get_document_parser),
    embedding_provider: EmbeddingProvider = Depends(get_embedding_provider),
    vector_store: QdrantStore = Depends(get_vector_store),
) -> IngestionService:
    """Request-scoped IngestionService with DB session."""
    return IngestionService(
        db=db,
        document_parser=document_parser,
        embedding_provider=embedding_provider,
        vector_store=vector_store,
    )
