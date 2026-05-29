"""
Dependency injection for the knowledge base module.

Provides request-scoped and cached singletons for KB document operations.
"""
from functools import lru_cache

from fastapi import Depends
from sqlalchemy.ext.asyncio import AsyncSession

from src.core import get_db, get_storage_settings
from src.infra import LocalStorageProvider, QdrantStore, StorageProvider
from src.rag import IngestionService, get_ingestion_service, get_vector_store

from .repository import PDFRepository
from .service import KnowledgeBase


@lru_cache
def get_storage_provider() -> StorageProvider:
    """Singleton StorageProvider for admin-uploaded global documents.

    Returns:
        StorageProvider: The storage provider instance.
    """
    settings = get_storage_settings()
    return LocalStorageProvider(settings.admin_upload_dir)

def get_pdf_repository(db: AsyncSession = Depends(get_db)) -> PDFRepository:
    """Request-scoped PDFRepository.

    Args:
        db (AsyncSession): The database session.

    Returns:
        PDFRepository: The configured PDF repository.
    """
    return PDFRepository(db)

def get_pdf_service(
    repository: PDFRepository = Depends(get_pdf_repository),
    storage: StorageProvider = Depends(get_storage_provider),
    vector_store: QdrantStore = Depends(get_vector_store),
    ingestion_service: IngestionService = Depends(get_ingestion_service),
) -> KnowledgeBase:
    """Request-scoped KnowledgeBase service.

    Args:
        repository (PDFRepository): The PDF repository.
        storage (StorageProvider): The storage provider.
        vector_store (QdrantStore): The global RAG vector store.
        ingestion_service (IngestionService): The ingestion service.

    Returns:
        KnowledgeBase: The configured KB service.
    """
    return KnowledgeBase(
        repository=repository,
        storage=storage,
        vector_store=vector_store,
        ingestion_service=ingestion_service,
    )