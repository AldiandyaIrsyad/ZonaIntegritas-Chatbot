"""PostgreSQL repository for the Knowledge Base."""

import structlog
from datetime import datetime, timezone
from typing import List, Optional

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from sqlalchemy.orm import joinedload

from app.kb.domain.interfaces import IKBRepository
from app.kb.domain.models import PDFDocument, ParentChunk, IngestionTask, ChildChunk

logger = structlog.get_logger(__name__)


class PostgresKBRepository(IKBRepository):
    """Database operations for the Knowledge Base domain."""

    def __init__(self, db: AsyncSession):
        self.db = db

    async def get_all_pdfs(self) -> List[PDFDocument]:
        result = await self.db.execute(select(PDFDocument).order_by(PDFDocument.created_at.desc()))
        return list(result.scalars().all())

    async def get_pdf_by_id(self, pdf_id: str) -> Optional[PDFDocument]:
        query = select(PDFDocument).where(PDFDocument.id == pdf_id)
        result = await self.db.execute(query)
        return result.scalars().first()

    async def create_pdf(self, title: str, description: str, pdf_path: str) -> PDFDocument:
        new_pdf = PDFDocument(title=title, description=description, pdf_path=pdf_path)
        self.db.add(new_pdf)
        await self.db.flush()
        await self.db.refresh(new_pdf)
        logger.info("kb.repository.pdf_created", pdf_id=new_pdf.id, title=title)
        return new_pdf

    async def update_pdf_active_status(self, pdf_id: str, active: bool) -> Optional[PDFDocument]:
        pdf = await self.get_pdf_by_id(pdf_id)
        if pdf:
            pdf.active = active  # type: ignore
            await self.db.flush()
            await self.db.refresh(pdf)
            logger.info("kb.repository.pdf_status_updated", pdf_id=pdf_id, active=active)
            return pdf
        return None

    async def delete_pdf(self, pdf_id: str) -> bool:
        pdf = await self.get_pdf_by_id(pdf_id)
        if pdf:
            await self.db.delete(pdf)
            await self.db.flush()
            logger.info("kb.repository.pdf_deleted", pdf_id=pdf_id)
            return True
        return False

    async def get_pdfs_by_ids(self, pdf_ids: List[str]) -> List[PDFDocument]:
        """Fetch multiple PDF documents by their IDs in a single query.

        Args:
            pdf_ids: List of PDFDocument primary key strings.

        Returns:
            List[PDFDocument]: All found documents (order not guaranteed).
        """
        if not pdf_ids:
            return []
        result = await self.db.execute(
            select(PDFDocument).where(PDFDocument.id.in_(pdf_ids))
        )
        return list(result.scalars().all())

    async def save_parent_chunks(self, chunks: List[ParentChunk]) -> List[ParentChunk]:
        if not chunks:
            return []
        self.db.add_all(chunks)
        await self.db.flush()
        logger.info("kb.repository.chunks_saved", count=len(chunks))
        return chunks

    async def get_parent_chunks_by_ids(self, chunk_ids: List[str]) -> List[ParentChunk]:
        if not chunk_ids:
            return []
        result = await self.db.execute(
            select(ParentChunk)
            .options(joinedload(ParentChunk.document))
            .where(ParentChunk.id.in_(chunk_ids))
            .order_by(ParentChunk.chunk_index)
        )
        return list(result.scalars().all())

    async def delete_parent_chunks_by_doc_id(self, doc_id: str) -> int:
        result = await self.db.execute(select(ParentChunk).where(ParentChunk.doc_id == doc_id))
        chunks = result.scalars().all()
        count = len(chunks)
        for chunk in chunks:
            await self.db.delete(chunk)
        await self.db.flush()
        logger.info("kb.repository.chunks_deleted", count=count, doc_id=doc_id)
        return count

    async def save_child_chunks(self, chunks: List[ChildChunk]) -> List[ChildChunk]:
        """Persist child chunk records to the database."""
        if not chunks:
            return []
        self.db.add_all(chunks)
        await self.db.flush()
        logger.info("kb.repository.child_chunks_saved", count=len(chunks))
        return chunks

    async def get_child_chunks_by_ids(self, chunk_ids: List[str]) -> List[ChildChunk]:
        """Fetch child chunks by their primary key IDs."""
        if not chunk_ids:
            return []
        result = await self.db.execute(
            select(ChildChunk).where(ChildChunk.id.in_(chunk_ids))
        )
        return list(result.scalars().all())

    async def get_child_chunks_by_parent_ids(self, parent_ids: List[str]) -> List[ChildChunk]:
        """Fetch child chunks by their parent chunk IDs, ordered by ordinal."""
        if not parent_ids:
            return []
        result = await self.db.execute(
            select(ChildChunk)
            .where(ChildChunk.parent_chunk_id.in_(parent_ids))
            .order_by(ChildChunk.parent_chunk_id, ChildChunk.ordinal)
        )
        return list(result.scalars().all())

    async def get_sibling_chunks(self, parent_id: str) -> List[ParentChunk]:
        """Fetch sibling parent chunks sharing the same parent_id."""
        result = await self.db.execute(
            select(ParentChunk)
            .where(ParentChunk.parent_id == parent_id)
            .order_by(ParentChunk.ordinal)
        )
        return list(result.scalars().all())

    async def get_chunks_by_path_prefix(self, path_prefix: str) -> List[ParentChunk]:
        """Fetch parent chunks whose path starts with the given prefix.

        Uses LIKE 'prefix%' which works on both Postgres and SQLite.
        """
        if not path_prefix:
            return []
        result = await self.db.execute(
            select(ParentChunk)
            .where(ParentChunk.path.like(f"{path_prefix}%"))
            .order_by(ParentChunk.ordinal)
        )
        return list(result.scalars().all())

    async def create_ingestion_task(self, doc_id: str) -> IngestionTask:
        task = IngestionTask(doc_id=doc_id, status="pending")
        self.db.add(task)
        await self.db.flush()
        await self.db.refresh(task)
        logger.info("kb.repository.task_created", task_id=task.id, doc_id=doc_id)
        return task

    async def update_ingestion_task(
        self, task_id: str, status: str, error_message: Optional[str] = None
    ) -> Optional[IngestionTask]:
        result = await self.db.execute(select(IngestionTask).where(IngestionTask.id == task_id))
        task = result.scalars().first()
        if not task:
            return None

        task.status = status
        if error_message is not None:
            task.error_message = error_message
        if status in ("completed", "failed"):
            task.completed_at = datetime.now(timezone.utc)
        else:
            task.completed_at = None

        await self.db.flush()
        logger.info("kb.repository.task_updated", task_id=task_id, status=status)
        return task

    async def get_ingestion_task_by_doc_id(self, doc_id: str) -> Optional[IngestionTask]:
        result = await self.db.execute(
            select(IngestionTask)
            .where(IngestionTask.doc_id == doc_id)
            .order_by(IngestionTask.created_at.desc())
            .limit(1)
        )
        return result.scalars().first()
