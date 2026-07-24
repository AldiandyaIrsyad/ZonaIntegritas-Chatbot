"""
Application service for KB administration workflows.
"""

import os
import re
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

# Cap on the on-disk filename's stem (before the UUID prefix/extension are
# added). Two independent things break on long filenames, both discovered
# empirically: (1) the local filesystem raises ENAMETOOLONG around 255 bytes
# for the full path component, and (2) Unstructured Cloud's job API — which
# receives this same filename, since IngestWorker/UnstructuredClient read it
# back off disk — silently completes with output_node_files: None (no error
# at all) once the filename gets long (reproduced at 240 chars; titles in
# this app's real corpus routinely exceed that). The original filename and
# title are preserved in PDFDocument.title/description, not lost — only the
# on-disk/upstream-API-visible name is shortened.
_MAX_FILENAME_STEM_LEN = 60


def _safe_upload_filename(original_filename: str) -> str:
    """Build a short, unique, filesystem- and Unstructured-Cloud-safe
    filename for on-disk storage. Always UUID-prefixed for uniqueness
    (previously only the batch path had a prefix, and even that was too
    weak against long titles)."""
    stem, ext = os.path.splitext(original_filename or "upload.pdf")
    ext = ext or ".pdf"
    safe_stem = re.sub(r"[^A-Za-z0-9._-]", "_", stem)[:_MAX_FILENAME_STEM_LEN]
    return f"{uuid.uuid4().hex[:12]}_{safe_stem}{ext}"


class KBApplicationService:
    """Service handling administration of PDF documents and triggering ingestion."""

    def __init__(
        self,
        kb_repo: IKBRepository,
        vector_store: IVectorStore,
        ingest_worker: IngestWorker,
        upload_dir: str = "./uploads/knowledge_base",
    ):
        """Wire the service's collaborators and ensure the upload directory exists.

        Args:
            kb_repo: Postgres-backed :class:`IKBRepository` for document/chunk
                persistence.
            vector_store: :class:`IVectorStore` (Qdrant) for chunk vectors.
            ingest_worker: Orchestrates the parse→embed→upsert pipeline;
                triggered as a FastAPI background task, not awaited here.
            upload_dir: Directory where uploaded PDFs are saved on disk.
        """
        self.kb_repo = kb_repo
        self.vector_store = vector_store
        self.ingest_worker = ingest_worker
        self.upload_dir = upload_dir

        if not os.path.exists(self.upload_dir):
            os.makedirs(self.upload_dir, exist_ok=True)

    async def list_pdfs(self) -> List[PDFDocument]:
        """Return all KB documents (passthrough to ``IKBRepository``)."""
        return await self.kb_repo.get_all_pdfs()

    async def naive_title_search(self, query: str) -> List[PDFDocument]:
        """Passthrough to ``IKBRepository.search_titles_naive`` — see there
        for why this literal ILIKE search exists alongside the real
        hybrid-search pipeline."""
        return await self.kb_repo.search_titles_naive(query)

    async def upload_pdf(self, title: str, description: str, file: UploadFile, bg_tasks: BackgroundTasks) -> PDFDocument:
        """Save one uploaded PDF to disk, record it, and schedule ingestion.

        Coordinates two systems: the filesystem (saves the file under
        ``upload_dir`` with a safe name) and ``IKBRepository`` (creates the
        ``PDFDocument`` row). Vector storage isn't touched here — embedding
        and upserting into Qdrant happen later, asynchronously, inside
        ``ingest_worker.ingest_document`` via ``bg_tasks``.
        """
        file_path = os.path.join(self.upload_dir, _safe_upload_filename(file.filename))

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
    ) -> tuple[List[PDFDocument], List[dict[str, str]]]:
        """Upload multiple PDFs and trigger background ingestion for each.

        Saved filenames are UUID-prefixed and length-capped (see
        _safe_upload_filename) to avoid both filesystem ENAMETOOLONG errors
        and a silent Unstructured Cloud failure mode on long filenames.

        Each file is isolated: on success its PDFDocument row is committed
        immediately, so a later file's failure (disk full, bad filename,
        DB error) can't roll back files that already succeeded — previously
        the whole batch shared one uncommitted transaction, so any failure
        anywhere in the loop discarded every already-processed file. On
        failure, the session is rolled back (to clear its errored state) and
        the loop continues with the next file instead of aborting.

        Args:
            files: List of uploaded PDF files.
            titles: List of titles, one per file (matched by index).
            descriptions: List of descriptions, one per file (matched by index).
            bg_tasks: FastAPI BackgroundTasks for async ingestion.

        Returns:
            Tuple of (successfully created PDFDocuments, [{"filename", "error"}, ...]).
        """
        results: List[PDFDocument] = []
        failures: List[dict[str, str]] = []

        for file, title, description in zip(files, titles, descriptions):
            safe_filename = file.filename or "upload.pdf"
            try:
                file_path = os.path.join(self.upload_dir, _safe_upload_filename(file.filename))

                with open(file_path, "wb") as buffer:
                    shutil.copyfileobj(file.file, buffer)

                pdf_doc = await self.kb_repo.create_pdf(
                    title=title,
                    description=description,
                    pdf_path=file_path,
                )
                bg_tasks.add_task(self.ingest_worker.ingest_document, doc_id=str(pdf_doc.id))
                await self.kb_repo.commit()
                results.append(pdf_doc)
            except Exception as exc:
                logger.error("kb.upload.file_failed", filename=safe_filename, error=str(exc))
                await self.kb_repo.rollback()
                failures.append({"filename": safe_filename, "error": str(exc)})

        logger.info("kb.upload.batch_complete", succeeded=len(results), failed=len(failures))
        return results, failures

    async def update_pdf_status(self, pdf_id: str, active: bool) -> Optional[PDFDocument]:
        """Toggle a document's active flag in both Postgres and Qdrant.

        Inactive documents are excluded from retrieval (``hybrid_search``
        filters on the ``is_active`` payload field), so the flag must be
        mirrored into the vector store's payload, not just the Postgres row.
        """
        doc = await self.kb_repo.update_pdf_active_status(pdf_id, active)
        if doc:
            await self.vector_store.update_payload(pdf_id, {"is_active": active})
        return doc

    async def delete_pdf(self, pdf_id: str) -> bool:
        """Delete a document across all three systems it lives in: the
        Postgres row (and cascaded chunks), its vectors in Qdrant, and the
        PDF file on disk. Returns False without side effects if the document
        doesn't exist."""
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
        """Return the latest ingestion task's status for a document, or
        None if no ingestion task has been recorded for it."""
        task = await self.kb_repo.get_ingestion_task_by_doc_id(pdf_id)
        if not task:
            return None
        return {
            "status": task.status,
            "error_message": task.error_message,
            "created_at": task.created_at,
            "completed_at": task.completed_at,
        }
