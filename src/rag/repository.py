"""
RAG-specific database operations.

Handles CRUD for ParentChunk and IngestionTask models that support
the async ingestion pipeline and Small-to-Big retrieval strategy.
"""
import logging
from datetime import datetime, timezone
from typing import List, Optional

from sqlalchemy import update
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select

from src.rag.model import IngestionTask, ParentChunk

logger = logging.getLogger(__name__)


class RAGRepository:
    """Database operations for the RAG pipeline."""

    def __init__(self, db: AsyncSession):
        self.db = db

    # --- Parent Chunk Operations ---

    async def save_parent_chunks(
        self, chunks: List[ParentChunk]
    ) -> List[ParentChunk]:
        """Batch insert parent chunks into PostgreSQL.

        Args:
            chunks: List of ParentChunk ORM instances to persist.

        Returns:
            The persisted chunks (same objects, now attached to session).
        """
        if not chunks:
            return []

        self.db.add_all(chunks)
        await self.db.flush()
        logger.info("Saved %d parent chunks to PostgreSQL", len(chunks))
        return chunks

    async def get_parent_chunks_by_ids(
        self, chunk_ids: List[str]
    ) -> List[ParentChunk]:
        """Fetch parent chunks by their IDs.

        Used during retrieval: after Qdrant returns child chunk matches,
        we extract unique parent_chunk_ids and fetch the full parent texts.

        Args:
            chunk_ids: List of parent chunk UUIDs.

        Returns:
            List of ParentChunk objects with full text.
        """
        if not chunk_ids:
            return []

        result = await self.db.execute(
            select(ParentChunk)
            .where(ParentChunk.id.in_(chunk_ids))
            .order_by(ParentChunk.chunk_index)
        )
        return list(result.scalars().all())

    async def delete_parent_chunks_by_doc_id(self, doc_id: str) -> int:
        """Delete all parent chunks for a document.

        Called when re-ingesting or deleting a document.

        Returns:
            Number of deleted chunks.
        """
        result = await self.db.execute(
            select(ParentChunk).where(ParentChunk.doc_id == doc_id)
        )
        chunks = result.scalars().all()
        count = len(chunks)
        for chunk in chunks:
            await self.db.delete(chunk)
        await self.db.flush()
        logger.info(
            "Deleted %d parent chunks for doc_id='%s'", count, doc_id
        )
        return count

    # --- Ingestion Task Operations ---

    async def create_ingestion_task(self, doc_id: str) -> IngestionTask:
        """Create a new ingestion task for tracking PDF processing.

        Args:
            doc_id: UUID of the PDFDocument being ingested.

        Returns:
            The newly created IngestionTask.
        """
        task = IngestionTask(doc_id=doc_id, status="pending")
        self.db.add(task)
        await self.db.flush()
        await self.db.refresh(task)
        logger.info(
            "Created ingestion task '%s' for doc_id='%s'", task.id, doc_id
        )
        return task

    async def update_ingestion_task(
        self,
        task_id: str,
        status: str,
        error_message: Optional[str] = None,
    ) -> Optional[IngestionTask]:
        """Update the status of an ingestion task.

        Args:
            task_id: UUID of the ingestion task.
            status: New status (processing, completed, failed).
            error_message: Error details if status is 'failed'.

        Returns:
            The updated IngestionTask, or None if not found.
        """
        result = await self.db.execute(
            select(IngestionTask).where(IngestionTask.id == task_id)
        )
        task = result.scalars().first()
        if not task:
            return None

        task.status = status
        if error_message is not None:
            task.error_message = error_message
        if status in ("completed", "failed"):
            task.completed_at = datetime.now(timezone.utc)

        await self.db.flush()
        logger.info(
            "Updated ingestion task '%s' to status='%s'", task_id, status
        )
        return task

    async def get_ingestion_task_by_doc_id(
        self, doc_id: str
    ) -> Optional[IngestionTask]:
        """Get the most recent ingestion task for a document.

        Args:
            doc_id: UUID of the PDFDocument.

        Returns:
            The most recent IngestionTask, or None.
        """
        result = await self.db.execute(
            select(IngestionTask)
            .where(IngestionTask.doc_id == doc_id)
            .order_by(IngestionTask.created_at.desc())
            .limit(1)
        )
        return result.scalars().first()
