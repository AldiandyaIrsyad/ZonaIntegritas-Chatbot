"""
Dependency injection for the RAG pipeline.

Provides singleton factory functions for infrastructure adapters
and request-scoped factories for services that need a DB session.
"""
from functools import lru_cache

from fastapi import Depends
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import (
    get_infinity_settings,
    get_qdrant_settings,
    get_unstructured_settings,
)
from app.infra.db import get_db_session
from app.infra.document_parser import DocumentParser
from app.infra.embedding_provider import EmbeddingProvider
from app.infra.vector_store import QdrantStore
from app.infra.reranker import Reranker

from app.core.interfaces.ai import IEmbeddingProvider, IReranker
from app.core.interfaces.infra import IDocumentParser, IVectorStore

from .ingestion import IngestionService
from .retrieval import RetrievalService


@lru_cache
def get_vector_store() -> IVectorStore:
    """Singleton QdrantStore instance.

    Returns:
        QdrantStore: The singleton vector store instance.
    """
    settings = get_qdrant_settings()
    return QdrantStore(
        host=settings.host,
        port=settings.port,
        collection_name=settings.collection_name,
    )


@lru_cache
def get_embedding_provider() -> IEmbeddingProvider:
    """Singleton EmbeddingProvider instance.

    Returns:
        EmbeddingProvider: The singleton embedding provider instance.
    """
    settings = get_infinity_settings()
    return EmbeddingProvider(
        base_url=settings.base_url,
        model=settings.embedding_model,
    )


@lru_cache
def get_reranker() -> IReranker:
    """Singleton Reranker instance.

    Returns:
        Reranker: The singleton reranker instance.
    """
    settings = get_infinity_settings()
    return Reranker(
        base_url=settings.base_url,
        model=settings.reranker_model,
    )


@lru_cache
def get_document_parser() -> IDocumentParser:
    """Singleton DocumentParser instance.

    Returns:
        DocumentParser: The singleton document parser instance.
    """
    settings = get_unstructured_settings()
    return DocumentParser(base_url=settings.base_url)


def get_retrieval_service(
    db: AsyncSession = Depends(get_db_session),
    embedding_provider: IEmbeddingProvider = Depends(get_embedding_provider),
    vector_store: IVectorStore = Depends(get_vector_store),
    reranker: IReranker = Depends(get_reranker),
) -> RetrievalService:
    """Request-scoped RetrievalService with DB session.

    Args:
        db (AsyncSession): The database session.
        embedding_provider (IEmbeddingProvider): The embedding provider instance.
        vector_store (IVectorStore): The vector store instance.
        reranker (IReranker): The reranker instance.

    Returns:
        RetrievalService: The request-scoped retrieval service.
    """
    return RetrievalService(
        db=db,
        embedding_provider=embedding_provider,
        vector_store=vector_store,
        reranker=reranker,
    )


def get_ingestion_service(
    db: AsyncSession = Depends(get_db_session),
    document_parser: IDocumentParser = Depends(get_document_parser),
    embedding_provider: IEmbeddingProvider = Depends(get_embedding_provider),
    vector_store: IVectorStore = Depends(get_vector_store),
) -> IngestionService:
    """Request-scoped IngestionService with DB session.

    Args:
        db (AsyncSession): The database session.
        document_parser (IDocumentParser): The document parser instance.
        embedding_provider (IEmbeddingProvider): The embedding provider instance.
        vector_store (IVectorStore): The vector store instance.

    Returns:
        IngestionService: The request-scoped ingestion service.
    """
    return IngestionService(
        db=db,
        document_parser=document_parser,
        embedding_provider=embedding_provider,
        vector_store=vector_store,
    )
