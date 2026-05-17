from fastapi import Depends
from sqlalchemy.ext.asyncio import AsyncSession
from backend.config import get_db
from backend.repository import ChatRepository, PDFRepository
from backend.service import ChatService, PDFService
from backend.controller import ChatController, PDFController
from settings import get_settings, LLMSettings
from services.LLM import LLM

def get_llm_client(settings: LLMSettings) -> LLM:
    """Centralized logic for evaluating model configuration."""
    return LLM(
        model=settings.model,
        base_url=settings.base_url,
        api_key=settings.api_key
    )


def get_chat_repository(db: AsyncSession = Depends(get_db)) -> ChatRepository:
    return ChatRepository(db)

def get_chat_service(repository: ChatRepository = Depends(get_chat_repository), settings: LLMSettings = Depends(get_settings)) -> ChatService:
    return ChatService(repository=repository, llm=get_llm_client(settings))

def get_chat_controller(service: ChatService = Depends(get_chat_service)) -> ChatController:
    return ChatController(service)

def get_pdf_repository(db: AsyncSession = Depends(get_db)) -> PDFRepository:
    return PDFRepository(db)

def get_pdf_service(repository: PDFRepository = Depends(get_pdf_repository)) -> PDFService:
    return PDFService(repository=repository)

def get_pdf_controller(service: PDFService = Depends(get_pdf_service)) -> PDFController:
    return PDFController(service)