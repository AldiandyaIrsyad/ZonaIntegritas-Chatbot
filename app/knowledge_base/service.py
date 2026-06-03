"""
Service layer for the knowledge base module.

Orchestrates PDF document uploads, ingestion background tasks, and active status.
"""
import os
from typing import Optional

import structlog
from fastapi import BackgroundTasks, UploadFile

from app.infra.db import async_session_maker
from app.core.interfaces.infra import IStorageProvider, IVectorStore
from app.core.interfaces.rag import IIngestionService
from app.rag.dependency import (
    get_document_parser,
    get_embedding_provider,
    get_vector_store,
)
from app.rag.ingestion import IngestionService

from app.core.interfaces.knowledge_base import IKnowledgeBaseRepository, IKnowledgeBaseService
from .repository import PDFRepository

logger = structlog.get_logger(__name__)

async def run_ingestion_background(pdf_id: str) -> None:
    """Run ingestion in the background with an independent DB session.

    Args:
        pdf_id (str): UUID of the document to ingest.
    """
    async with async_session_maker() as db:
        parser = get_document_parser()
        embedder = get_embedding_provider()
        vector_store = get_vector_store()
        
        service = IngestionService(
            db=db,
            document_parser=parser,
            embedding_provider=embedder,
            vector_store=vector_store
        )
        await service.ingest_document(pdf_id)


class KnowledgeBase(IKnowledgeBaseService):
    """Core business logic for global knowledge base documents."""
    
    def __init__(
        self,
        repository: IKnowledgeBaseRepository,
        storage: IStorageProvider,
        vector_store: IVectorStore,
        ingestion_service: IIngestionService,
    ):
        self.repository = repository
        self.storage = storage
        self.vector_store = vector_store
        self.ingestion_service = ingestion_service
        
    async def list_pdfs(self) -> list[dict]:
        """List all available PDF documents.

        Returns:
            list[dict]: List of document summaries.
        """
        pdfs = await self.repository.get_all_pdfs()
        return [
            {
                "id": p.id,
                "title": p.title,
                "description": p.description,
                "pdf_path": p.pdf_path,
                "active": p.active,
                "ingestion_status": p.ingestion_status,
            }
            for p in pdfs
        ]
        
    async def upload_pdf(self, title: str, description: str, file: UploadFile, background_tasks: BackgroundTasks):
        """Upload a PDF document and queue it for ingestion.

        Args:
            title (str): Document title.
            description (str): Document description.
            file (UploadFile): The uploaded file.
            background_tasks (BackgroundTasks): FastAPI background task manager.

        Returns:
            PDFDocument: The created document record.

        Raises:
            ValueError: If the file is not a PDF.
        """
        file_extension = os.path.splitext(file.filename or "")[1].lower()
        if file.content_type != "application/pdf" or file_extension != ".pdf":
            logger.error(
                "Admin PDF upload failed: Invalid file type",
                action="ADMIN_UPLOAD_PDF",
                title=title,
                file_extension=file_extension,
                status="failed",
                reason="invalid_type",
                exc_info=True
            )
            raise ValueError("Only PDF files are allowed")
            
        try:
            file_path = await self.storage.save_file(file, file_extension)
            pdf = await self.repository.create_pdf(title, description, file_path)

            # Enqueue async ingestion in the background
            background_tasks.add_task(run_ingestion_background, pdf.id)

            logger.info(
                "Admin PDF upload successful",
                action="ADMIN_UPLOAD_PDF",
                pdf_id=pdf.id,
                title=pdf.title,
                description_length=len(description) if description else 0,
                file_extension=file_extension,
                status="success"
            )
            return pdf
        except Exception as e:
            logger.error(
                "Admin PDF upload failed: Exception occurred",
                action="ADMIN_UPLOAD_PDF",
                title=title,
                file_extension=file_extension,
                status="failed",
                reason=type(e).__name__,
                exc_info=True
            )
            raise

    async def update_pdf_status(self, pdf_id: str, active: bool):
        """Toggle a document's active state in both Postgres and Qdrant.

        Implements the state synchronization guardrail:
        1. Update PostgreSQL first (within transaction)
        2. Update Qdrant payload (is_active)
        3. On Qdrant failure, revert PostgreSQL change

        Args:
            pdf_id (str): UUID of the document.
            active (bool): True to activate, False to deactivate.

        Returns:
            PDFDocument | None: The updated document, or None if not found.
        """
        pdf = await self.repository.update_pdf_active_status(pdf_id, active)
        if not pdf:
            return None

        try:
            await self.vector_store.update_payload(
                doc_id=pdf_id,
                payload={"is_active": active},
            )
        except Exception as e:
            logger.error(
                "Qdrant payload update failed for doc_id='%s': %s. "
                "Reverting PostgreSQL state.",
                pdf_id,
                type(e).__name__,
                exc_info=True,
            )
            # Revert Postgres to prevent state drift
            await self.repository.update_pdf_active_status(pdf_id, not active)
            raise

        return pdf
        
    async def delete_pdf(self, pdf_id: str) -> bool:
        """Delete a PDF and its associated vectors from both Postgres and Qdrant.

        Args:
            pdf_id (str): UUID of the document to delete.

        Returns:
            bool: True if deleted successfully, False if not found.
        """
        pdf = await self.repository.get_pdf_by_id(pdf_id)
        if not pdf:
            return False

        # Delete vectors from Qdrant first (non-critical if it fails)
        try:
            await self.vector_store.delete_by_doc_id(pdf_id)
        except Exception as e:
            logger.error(
                "Failed to delete vectors from Qdrant for doc_id='%s': %s",
                pdf_id,
                type(e).__name__,
                exc_info=True,
            )
            raise

        # Delete file from storage
        if pdf.pdf_path:
            await self.storage.delete_file(pdf.pdf_path)

        # Delete from Postgres (cascades to parent_chunks and ingestion_tasks)
        return await self.repository.delete_pdf(pdf_id)

    async def get_ingestion_status(self, pdf_id: str) -> Optional[dict]:
        """Get the current ingestion status of a document.

        Args:
            pdf_id (str): UUID of the document.

        Returns:
            Optional[dict]: The document's ID, title, and ingestion status.
        """
        pdf = await self.repository.get_pdf_by_id(pdf_id)
        if not pdf:
            return None
        return {
            "id": pdf.id,
            "title": pdf.title,
            "ingestion_status": pdf.ingestion_status,
        }