# src/chat/dependencies.py
from fastapi import Depends
from sqlalchemy.ext.asyncio import AsyncSession
from src.core.database import get_db
from src.llm.dependency import get_llm_service
from src.rag.dependency import get_retrieval_service
from src.chat.repository import ChatRepository
from src.chat.service import ChatService
from src.rag.retrieval import RetrievalService

def get_chat_repository(db: AsyncSession = Depends(get_db)) -> ChatRepository:
    return ChatRepository(db)

def get_chat_service(
    repository: ChatRepository = Depends(get_chat_repository),
    llm_service = Depends(get_llm_service),
    retrieval_service: RetrievalService = Depends(get_retrieval_service),
) -> ChatService:
    return ChatService(
        repository=repository,
        llm_service=llm_service,
        retrieval_service=retrieval_service,
    )