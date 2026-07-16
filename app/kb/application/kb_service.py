"""
Application service for KB administration workflows.
"""

import os
import shutil
import uuid
import structlog
from typing import List, Optional, Any
from fastapi import BackgroundTasks, UploadFile

from typing import Optional

from app.kb.domain.interfaces import IKBRepository, IVectorStore
from app.kb.domain.models import PDFDocument
from app.kb.application.ingest_worker import IngestWorker

logger = structlog.get_logger(__name__)

class KBApplicationService:
    """Service handling administration of PDF documents and triggering ingestion."""

    def __init__(
        self,
        kb_repo: IKBRepository,
        vector_store: IVectorStore,
        ingest_worker: IngestWorker,
        upload_dir: str = "./uploads/knowledge_base",
    ):
        self.kb_repo = kb_repo
        self.vector_store = vector_store
        self.ingest_worker = ingest_worker
        self.upload_dir = upload_dir

        if not os.path.exists(self.upload_dir):
            os.makedirs(self.upload_dir, exist_ok=True)

    async def list_pdfs(self) -> List[PDFDocument]:
        return await self.kb_repo.get_all_pdfs()

    async def upload_pdf(self, title: str, description: str, file: UploadFile, bg_tasks: BackgroundTasks) -> PDFDocument:
        safe_filename = file.filename or "upload.pdf"
        file_path = os.path.join(self.upload_dir, f"{title}_{safe_filename}")
        
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
            
        pdf_doc = await self.kb_repo.create_pdf(title=title, description=description, pdf_path=file_path)
        
        # Trigger ingestion in the background
        bg_tasks.add_task(self.ingest_worker.ingest_document, doc_id=str(pdf_doc.id))
        
        return pdf_doc

    async def upload_pdfs_batch(
        self,
        files: List[UploadFile],
        titles: List[str],
        descriptions: List[str],
        bg_tasks: BackgroundTasks,
    ) -> List[PDFDocument]:
        """Upload multiple PDFs and trigger background ingestion for each.

        Uses a UUID prefix on the saved filename to avoid collisions when
        multiple files share the same original filename or title.

        Args:
            files: List of uploaded PDF files.
            titles: List of titles, one per file (matched by index).
            descriptions: List of descriptions, one per file (matched by index).
            bg_tasks: FastAPI BackgroundTasks for async ingestion.

        Returns:
            List of created PDFDocument objects (one per file).
        """
        results: List[PDFDocument] = []
        for file, title, description in zip(files, titles, descriptions):
            safe_filename = file.filename or "upload.pdf"
            # UUID prefix prevents filename collisions across batch uploads
            unique_prefix = str(uuid.uuid4())[:8]
            file_path = os.path.join(self.upload_dir, f"{unique_prefix}_{title}_{safe_filename}")

            with open(file_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)

            pdf_doc = await self.kb_repo.create_pdf(
                title=title,
                description=description,
                pdf_path=file_path,
            )
            bg_tasks.add_task(self.ingest_worker.ingest_document, doc_id=str(pdf_doc.id))
            results.append(pdf_doc)

        logger.info("kb.upload.batch_complete", count=len(results))
        return results

    async def update_pdf_status(self, pdf_id: str, active: bool) -> Optional[PDFDocument]:
        doc = await self.kb_repo.update_pdf_active_status(pdf_id, active)
        if doc:
            await self.vector_store.update_payload(pdf_id, {"is_active": active})
        return doc

    async def delete_pdf(self, pdf_id: str) -> bool:
        doc = await self.kb_repo.get_pdf_by_id(pdf_id)
        if not doc:
            return False
            
        # Delete from postgres
        await self.kb_repo.delete_pdf(pdf_id)

        # Delete from qdrant
        await self.vector_store.delete_by_doc_id(pdf_id)

        # Remove file
        if os.path.exists(str(doc.pdf_path)):
            os.remove(str(doc.pdf_path))
            
        return True

    async def get_ingestion_status(self, pdf_id: str) -> Optional[dict[str, Any]]:
        task = await self.kb_repo.get_ingestion_task_by_doc_id(pdf_id)
        if not task:
            return None
        return {
            "status": task.status,
            "error_message": task.error_message,
            "created_at": task.created_at,
            "completed_at": task.completed_at,
        }
