from functools import lru_cache

from fastapi import Depends
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import get_qdrant_settings, get_storage_settings
from app.infra.db import get_db_session as get_db
from app.infra import (
    LocalStorageProvider,
    QdrantStore,
)
from app.llm import get_llm_service
from app.rag import (
    get_document_parser,
    get_embedding_provider,
    get_reranker,
    get_retrieval_service,
)
from app.ivm.dependency import get_ivm_service
from app.ram.dependency import get_ram_service

from app.core.interfaces.ai import IEmbeddingProvider, IReranker
from app.core.interfaces.infra import IDocumentParser, IStorageProvider, IVectorStore
from app.core.interfaces.ivm import IIVMService
from app.core.interfaces.llm import ILLMService
from app.core.interfaces.rag import IRetrievalService
from app.core.interfaces.ram import IRAMService

from app.chat.repository import ChatRepository
from app.chat.service import ChatService
from app.chat.pipeline import ChatPipeline


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
    document_parser: IDocumentParser = Depends(get_document_parser),
    vector_store: IVectorStore = Depends(get_session_vector_store),
    embedding_provider: IEmbeddingProvider = Depends(get_embedding_provider),
    ivm_service: IIVMService = Depends(get_ivm_service),
) -> ChatService:
    """Request-scoped ChatService."""
    return ChatService(
        repository=repository,
        storage=storage,
        document_parser=document_parser,
        vector_store=vector_store,
        embedding_provider=embedding_provider,
        ivm_service=ivm_service,
    )

def get_chat_pipeline(
    repository: ChatRepository = Depends(get_chat_repository),
    llm_service: ILLMService = Depends(get_llm_service),
    retrieval_service: IRetrievalService = Depends(get_retrieval_service),
    reranker: IReranker = Depends(get_reranker),
    vector_store: IVectorStore = Depends(get_session_vector_store),
    embedding_provider: IEmbeddingProvider = Depends(get_embedding_provider),
    ivm_service: IIVMService = Depends(get_ivm_service),
    ram_service: IRAMService = Depends(get_ram_service),
) -> ChatPipeline:
    """Request-scoped ChatPipeline."""
    return ChatPipeline(
        repository=repository,
        llm_service=llm_service,
        retrieval_service=retrieval_service,
        reranker=reranker,
        vector_store=vector_store,
        embedding_provider=embedding_provider,
        ivm_service=ivm_service,
        ram_service=ram_service,
    )