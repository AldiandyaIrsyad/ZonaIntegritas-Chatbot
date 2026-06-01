 
from functools import lru_cache

from fastapi import Depends
from sqlalchemy.ext.asyncio import AsyncSession

from src.core import get_db, get_qdrant_settings, get_storage_settings
from src.infra import (
    DocumentParser,
    EmbeddingProvider,
    LocalStorageProvider,
    QdrantStore,
    Reranker,
    StorageProvider,
)
from src.llm import get_llm_service
from src.rag import (
    RetrievalService,
    get_document_parser,
    get_embedding_provider,
    get_reranker,
    get_retrieval_service,
)
from src.ivm.service import IVMService
from src.ivm.dependency import get_ivm_service
from src.ram.service import RAMService
from src.ram.dependency import get_ram_service

from .repository import ChatRepository
from .service import ChatService



@lru_cache
def get_session_vector_store() -> QdrantStore:
    """Singleton QdrantStore instance for session-specific documents.

    Returns:
        QdrantStore: The vector store instance.
    """
    settings = get_qdrant_settings()
    return QdrantStore(
        host=settings.host,
        port=settings.port,
        collection_name=settings.session_collection_name,
    )

@lru_cache
def get_user_storage_provider() -> StorageProvider:
    """Singleton StorageProvider for user-uploaded documents.

    Returns:
        StorageProvider: The storage provider instance.
    """
    settings = get_storage_settings()
    return LocalStorageProvider(settings.user_upload_dir)

def get_chat_repository(db: AsyncSession = Depends(get_db)) -> ChatRepository:
    """Request-scoped ChatRepository.

    Args:
        db (AsyncSession): The database session.

    Returns:
        ChatRepository: The configured chat repository.
    """
    return ChatRepository(db)

def get_chat_service(
    repository: ChatRepository = Depends(get_chat_repository),
    llm_service = Depends(get_llm_service),
    retrieval_service: RetrievalService = Depends(get_retrieval_service),
    storage: StorageProvider = Depends(get_user_storage_provider),
    document_parser: DocumentParser = Depends(get_document_parser),
    reranker: Reranker = Depends(get_reranker),
    vector_store: QdrantStore = Depends(get_session_vector_store),
    embedding_provider: EmbeddingProvider = Depends(get_embedding_provider),
    ivm_service: IVMService = Depends(get_ivm_service),
    ram_service: RAMService = Depends(get_ram_service),
) -> ChatService:
    """Request-scoped ChatService.

    Args:
        repository (ChatRepository): The chat repository.
        llm_service (LLMService): The LLM connection service.
        retrieval_service (RetrievalService): The global RAG retrieval service.
        storage (StorageProvider): The storage provider.
        document_parser (DocumentParser): The document parser.
        reranker (Reranker): The reranking service.
        vector_store (QdrantStore): The session vector store.
        embedding_provider (EmbeddingProvider): The embedding service.
        ivm_service (IVMService): The Input Validation Module service.
        ram_service (RAMService): The Response Assessment Module service.

    Returns:
        ChatService: The configured chat service.
    """
    return ChatService(
        repository=repository,
        llm_service=llm_service,
        retrieval_service=retrieval_service,
        storage=storage,
        document_parser=document_parser,
        reranker=reranker,
        vector_store=vector_store,
        embedding_provider=embedding_provider,
        ivm_service=ivm_service,
        ram_service=ram_service,
    )