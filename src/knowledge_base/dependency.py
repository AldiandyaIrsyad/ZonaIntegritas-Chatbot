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
    settings = get_storage_settings()
    return LocalStorageProvider(settings.admin_upload_dir)

def get_pdf_repository(db: AsyncSession = Depends(get_db)) -> PDFRepository:
    return PDFRepository(db)

def get_pdf_service(
    repository: PDFRepository = Depends(get_pdf_repository),
    storage: StorageProvider = Depends(get_storage_provider),
    vector_store: QdrantStore = Depends(get_vector_store),
    ingestion_service: IngestionService = Depends(get_ingestion_service),
) -> KnowledgeBase:
    return KnowledgeBase(
        repository=repository,
        storage=storage,
        vector_store=vector_store,
        ingestion_service=ingestion_service,
    )