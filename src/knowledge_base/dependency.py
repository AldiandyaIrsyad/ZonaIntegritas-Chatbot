from fastapi import Depends
from sqlalchemy.ext.asyncio import AsyncSession
from src.core.database import get_db
from src.knowledge_base.repository import PDFRepository
from src.knowledge_base.service import KnowledgeBase
from src.infra.storage import StorageProvider, LocalStorageProvider
from src.rag.dependency import get_vector_store, get_ingestion_service
from src.infra.vector_store import QdrantStore
from src.rag.ingestion import IngestionService
from src.core.config import get_storage_settings
from functools import lru_cache

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