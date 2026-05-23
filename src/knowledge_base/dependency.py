from fastapi import Depends
from sqlalchemy.ext.asyncio import AsyncSession
from src.core.database import get_db
from src.knowledge_base.repository import PDFRepository
from src.knowledge_base.service import KnowledgeBase
from src.infra.storage import StorageProvider, LocalStorageProvider
from functools import lru_cache

@lru_cache
def get_storage_provider() -> StorageProvider:
    return LocalStorageProvider()

def get_pdf_repository(db: AsyncSession = Depends(get_db)) -> PDFRepository:
    return PDFRepository(db)

def get_pdf_service(
    repository: PDFRepository = Depends(get_pdf_repository),
    storage: StorageProvider = Depends(get_storage_provider)
) -> KnowledgeBase:
    return KnowledgeBase(repository=repository, storage=storage)