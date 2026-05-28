from fastapi import Depends
from sqlalchemy.ext.asyncio import AsyncSession
from src.core.database import get_db
from src.llm.dependency import get_llm_service
from src.rag.dependency import get_retrieval_service, get_document_parser, get_reranker
from src.chat.repository import ChatRepository
from src.chat.service import ChatService
from src.rag.retrieval import RetrievalService
from src.infra.storage import StorageProvider, LocalStorageProvider
from src.core.config import get_storage_settings
from src.infra.document_parser import DocumentParser
from src.infra.reranker import Reranker
from functools import lru_cache
from src.infra.vector_store import QdrantStore
from src.core.config import get_qdrant_settings
from src.infra.embedding_provider import EmbeddingProvider
from src.rag.dependency import get_embedding_provider

@lru_cache
def get_session_vector_store() -> QdrantStore:
    settings = get_qdrant_settings()
    return QdrantStore(
        host=settings.host,
        port=settings.port,
        collection_name=settings.session_collection_name,
    )

@lru_cache
def get_user_storage_provider() -> StorageProvider:
    settings = get_storage_settings()
    return LocalStorageProvider(settings.user_upload_dir)

def get_chat_repository(db: AsyncSession = Depends(get_db)) -> ChatRepository:
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
) -> ChatService:
    return ChatService(
        repository=repository,
        llm_service=llm_service,
        retrieval_service=retrieval_service,
        storage=storage,
        document_parser=document_parser,
        reranker=reranker,
        vector_store=vector_store,
        embedding_provider=embedding_provider,
    )