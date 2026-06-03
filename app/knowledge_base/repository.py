"""
Database repository for the knowledge base module.

Handles CRUD operations for global PDF documents.
"""
import structlog
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from sqlalchemy.orm import selectinload

from app.core.interfaces.knowledge_base import IKnowledgeBaseRepository
from .model import PDFDocument

logger = structlog.get_logger(__name__)


class PDFRepository(IKnowledgeBaseRepository):
    """Database operations for the knowledge base domain."""
    
    def __init__(self, db: AsyncSession):
        self.db = db
        
    async def get_all_pdfs(self) -> list[PDFDocument]:
        """Retrieve all knowledge base PDF documents.

        Returns:
            list[PDFDocument]: List of all PDF documents ordered by creation date descending.
        """
        result = await self.db.execute(select(PDFDocument).order_by(PDFDocument.created_at.desc()))
        return list(result.scalars().all())
        
    async def get_pdf_by_id(self, pdf_id: str) -> PDFDocument | None:
        """Fetch a PDF document by its ID.

        Args:
            pdf_id (str): UUID of the document.

        Returns:
            PDFDocument | None: The requested document, or None if not found.
        """
        query = select(PDFDocument).where(PDFDocument.id == pdf_id)
        result = await self.db.execute(query)
        return result.scalars().first()
        
    async def create_pdf(self, title: str, description: str, pdf_path: str) -> PDFDocument:
        """Create a new PDF document record.

        Args:
            title (str): Title of the document.
            description (str): Description of the document.
            pdf_path (str): File path on local storage.

        Returns:
            PDFDocument: The created document object.
        """
        new_pdf = PDFDocument(title=title, description=description, pdf_path=pdf_path)
        self.db.add(new_pdf)
        await self.db.commit()
        await self.db.refresh(new_pdf)
        logger.info("PDF document created in database", pdf_id=new_pdf.id, title=title)
        return new_pdf
        
    async def update_pdf_active_status(self, pdf_id: str, active: bool) -> PDFDocument | None:
        """Update the active status of a PDF document.

        Args:
            pdf_id (str): UUID of the document.
            active (bool): Whether the document should be actively used in RAG.

        Returns:
            PDFDocument | None: The updated document, or None if not found.
        """
        pdf = await self.get_pdf_by_id(pdf_id)
        if pdf:
            pdf.active = active
            await self.db.commit()
            await self.db.refresh(pdf)
            logger.info("PDF active status updated", pdf_id=pdf_id, active=active)
            return pdf
        logger.warning("Attempted to update active status for non-existent PDF", pdf_id=pdf_id)
        return None
        
    async def delete_pdf(self, pdf_id: str) -> bool:
        """Delete a PDF document from the database.

        Args:
            pdf_id (str): UUID of the document.

        Returns:
            bool: True if deleted successfully, False if not found.
        """
        pdf = await self.get_pdf_by_id(pdf_id)
        if pdf:
            await self.db.delete(pdf)
            await self.db.commit()
            logger.info("PDF document deleted from database", pdf_id=pdf_id)
            return True
        logger.warning("Attempted to delete non-existent PDF document", pdf_id=pdf_id)
        return False
