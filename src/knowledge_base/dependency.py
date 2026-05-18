from fastapi import Depends
from sqlalchemy.ext.asyncio import AsyncSession
from src.core.database import get_db
from src.knowledge_base.repository import PDFRepository
from src.knowledge_base.service import KnowledgeBase

def get_pdf_repository(db: AsyncSession = Depends(get_db)) -> PDFRepository:
    return PDFRepository(db)

def get_pdf_service(repository: PDFRepository = Depends(get_pdf_repository)) -> KnowledgeBase:
    return KnowledgeBase(repository=repository)