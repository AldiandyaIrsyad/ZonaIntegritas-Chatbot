from functools import lru_cache

from fastapi import Depends
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import get_qdrant_settings, get_storage_settings
from app.infra.db import get_db_session as get_db
from app.infra import (
    LocalStorageProvider,
    QdrantStore,
)
from app.rag import (
    get_document_parser,
    get_embedding_provider,
)
from app.ivm.dependency import get_ivm_service

from app.core.interfaces.ai import IEmbeddingProvider
from app.core.interfaces.infra import IDocumentParser, IStorageProvider, IVectorStore
from app.core.interfaces.ivm import IIVMService

from app.chat.repository import ChatRepository
from app.chat.service import ChatService


@lru_cache
def get_session_vector_store() -> IVectorStore:
    """Singleton QdrantStore instance for session-specific documents."""
    settings = get_qdrant_settings()
    return QdrantStore(
        host=settings.host,
        port=settings.port,
        collection_name=settings.collection_name,
    )

@lru_cache
def get_user_storage_provider() -> IStorageProvider:
    """Singleton StorageProvider for user-uploaded documents."""
    settings = get_storage_settings()
    return LocalStorageProvider(settings.user_upload_dir)

def get_chat_repository(db: AsyncSession = Depends(get_db)) -> ChatRepository:
    """Request-scoped ChatRepository."""
    return ChatRepository(db)

def get_chat_service(
    repository: ChatRepository = Depends(get_chat_repository),
    storage: IStorageProvider = Depends(get_user_storage_provider),
) -> ChatService:
    """Request-scoped ChatService."""
    return ChatService(
        repository=repository,
        storage=storage,
    )