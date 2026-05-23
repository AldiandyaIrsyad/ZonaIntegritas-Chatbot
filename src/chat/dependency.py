# src/chat/dependencies.py
from fastapi import Depends
from sqlalchemy.ext.asyncio import AsyncSession
from src.core.database import get_db
from src.core.config import get_settings, LLMSettings
from src.infra.llm_provider import get_llm_client, LLM
from src.chat.repository import ChatRepository
from src.chat.service import ChatService

def get_chat_repository(db: AsyncSession = Depends(get_db)) -> ChatRepository:
    return ChatRepository(db)

def get_chat_service(
    repository: ChatRepository = Depends(get_chat_repository), 
    settings: LLMSettings = Depends(get_settings)
) -> ChatService:
    llm_client = get_llm_client(settings)
    return ChatService(repository=repository, llm=llm_client)