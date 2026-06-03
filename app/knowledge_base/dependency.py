"""
Dependency injection for the knowledge base module.

Provides request-scoped and cached singletons for KB document operations.
"""
from functools import lru_cache

from fastapi import Depends
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import get_storage_settings
from app.infra.db import get_db_session
from app.infra.storage import LocalStorageProvider
from app.core.interfaces.infra import IStorageProvider, IVectorStore
from app.core.interfaces.rag import IIngestionService
from app.core.interfaces.knowledge_base import IKnowledgeBaseRepository, IKnowledgeBaseService
from app.rag.dependency import get_ingestion_service, get_vector_store

from .repository import PDFRepository
from .service import KnowledgeBase


@lru_cache
def get_storage_provider() -> IStorageProvider:
    """Singleton StorageProvider for admin-uploaded global documents.

    Returns:
        IStorageProvider: The storage provider instance.
    """
    settings = get_storage_settings()
    return LocalStorageProvider(settings.kb_upload_dir)

def get_pdf_repository(db: AsyncSession = Depends(get_db_session)) -> IKnowledgeBaseRepository:
    """Request-scoped PDFRepository.

    Args:
        db (AsyncSession): The database session.

    Returns:
        IKnowledgeBaseRepository: The configured PDF repository.
    """
    return PDFRepository(db)

def get_pdf_service(
    repository: IKnowledgeBaseRepository = Depends(get_pdf_repository),
    storage: IStorageProvider = Depends(get_storage_provider),
    vector_store: IVectorStore = Depends(get_vector_store),
    ingestion_service: IIngestionService = Depends(get_ingestion_service),
) -> IKnowledgeBaseService:
    """Request-scoped KnowledgeBase service.

    Args:
        repository (PDFRepository): The PDF repository.
        storage (IStorageProvider): The storage provider.
        vector_store (IVectorStore): The global RAG vector store.
        ingestion_service (IIngestionService): The ingestion service.

    Returns:
        IKnowledgeBaseService: The configured KB service.
    """
    return KnowledgeBase(
        repository=repository,
        storage=storage,
        vector_store=vector_store,
        ingestion_service=ingestion_service,
    )