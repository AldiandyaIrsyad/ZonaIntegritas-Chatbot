from fastapi import Depends
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker

from backend.repository import ChatRepository, PDFRepository
from backend.service import ChatService, KnowledgeBase
from backend.controller import ChatController, PDFController
from settings import get_settings, LLMSettings, get_db_settings
from services.LLM import LLM

db_settings = get_db_settings()
engine = create_async_engine(db_settings.database_url, echo=False)
async_session = async_sessionmaker(engine, expire_on_commit=False, class_=AsyncSession)

async def get_db():
    async with async_session() as session:
        yield session

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

def get_pdf_service(repository: PDFRepository = Depends(get_pdf_repository)) -> KnowledgeBase:
    return KnowledgeBase(repository=repository)

def get_pdf_controller(service: KnowledgeBase = Depends(get_pdf_service)) -> PDFController:
    return PDFController(service)

