"""
Knowledge Base domain interfaces.
"""
from typing import TYPE_CHECKING, List, Optional, Protocol, Dict, Any

from fastapi import BackgroundTasks, UploadFile

if TYPE_CHECKING:
    from app.knowledge_base.model import PDFDocument


class IKnowledgeBaseRepository(Protocol):
    """Database operations for the knowledge base domain."""

    async def get_all_pdfs(self) -> List["PDFDocument"]:
        """Retrieve all knowledge base PDF documents."""
        ...

    async def get_pdf_by_id(self, pdf_id: str) -> Optional["PDFDocument"]:
        """Fetch a PDF document by its ID."""
        ...

    async def create_pdf(self, title: str, description: str, pdf_path: str) -> "PDFDocument":
        """Create a new PDF document record."""
        ...

    async def update_pdf_active_status(self, pdf_id: str, active: bool) -> Optional["PDFDocument"]:
        """Update the active status of a PDF document."""
        ...

    async def delete_pdf(self, pdf_id: str) -> bool:
        """Delete a PDF document from the database."""
        ...


class IKnowledgeBaseService(Protocol):
    """Core business logic for global knowledge base documents."""

    async def list_pdfs(self) -> List[Dict[str, Any]]:
        """List all available PDF documents."""
        ...

    async def upload_pdf(
        self, 
        title: str, 
        description: str, 
        file: UploadFile, 
        background_tasks: BackgroundTasks
    ) -> "PDFDocument":
        """Upload a PDF document and queue it for ingestion."""
        ...

    async def update_pdf_status(self, pdf_id: str, active: bool) -> Optional["PDFDocument"]:
        """Toggle a document's active state."""
        ...

    async def delete_pdf(self, pdf_id: str) -> bool:
        """Delete a PDF and its associated vectors from both Postgres and Qdrant."""
        ...

    async def get_ingestion_status(self, pdf_id: str) -> Optional[Dict[str, Any]]:
        """Get the current ingestion status of a document."""
        ...
