"""
Asynchronous document ingestion pipeline.

Orchestrates the full write path: PDF parsing → hierarchical chunking →
embedding → vector storage. Designed to run as an async background task
via Procrastinate or direct invocation.
"""
import logging
from typing import Optional

from sqlalchemy.ext.asyncio import AsyncSession

from src.infra.document_parser import DocumentParser
from src.infra.embedding_provider import EmbeddingProvider
from src.infra.vector_store import ChunkVector, QdrantStore
from src.rag.chunking import (
    ChildChunkData,
    ParentChunkData,
    create_parent_chunks,
    split_into_children,
)
from src.rag.model import ParentChunk
from src.rag.repository import RAGRepository

logger = logging.getLogger(__name__)


class IngestionService:
    """
    Orchestrates the async document ingestion pipeline.

    Pipeline:
    1. Fetch PDFDocument from Postgres (get file path)
    2. Send PDF to unstructured-api → structured elements
    3. Create parent chunks → save to PostgreSQL
    4. Split into child chunks
    5. Embed child chunks via Infinity → dense + sparse vectors
    6. Upsert vectors to Qdrant with parent/doc metadata
    7. Update document ingestion status

    Transaction strategy:
    - PostgreSQL writes (parent chunks + status) happen within a single
      DB session transaction.
    - Qdrant upserts happen after Postgres commit.
    - If Qdrant fails after Postgres commit, the document status is set
      to 'failed' and parent chunks remain (can be retried).
    """

    def __init__(
        self,
        db: AsyncSession,
        document_parser: DocumentParser,
        embedding_provider: EmbeddingProvider,
        vector_store: QdrantStore,
    ):
        self.db = db
        self.document_parser = document_parser
        self.embedding_provider = embedding_provider
        self.vector_store = vector_store
        self.rag_repo = RAGRepository(db)

    async def ingest_document(self, doc_id: str) -> None:
        """Run the full ingestion pipeline for a document.

        This is the main entry point, designed to be called from a
        background task (Procrastinate) or directly for synchronous testing.

        Args:
            doc_id: UUID of the PDFDocument to ingest.

        Raises:
            FileNotFoundError: If the PDF file doesn't exist on disk.
            Exception: Any error during parsing, embedding, or storage.
        """
        task = await self.rag_repo.create_ingestion_task(doc_id)
        await self.db.commit()

        try:
            # Mark as processing
            await self.rag_repo.update_ingestion_task(task.id, "processing")
            await self._update_document_status(doc_id, "processing")
            await self.db.commit()

            # Step 1: Fetch document record
            pdf_doc = await self._get_document(doc_id)
            if not pdf_doc:
                raise ValueError(f"PDFDocument not found: {doc_id}")

            logger.info(
                "Starting ingestion for '%s' (doc_id=%s)",
                pdf_doc.title,
                doc_id,
            )

            # Step 2: Parse PDF via unstructured-api
            elements = await self.document_parser.parse_pdf(pdf_doc.pdf_path)
            if not elements:
                logger.warning(
                    "No elements extracted from PDF '%s'", pdf_doc.title
                )
                await self.rag_repo.update_ingestion_task(
                    task.id, "completed"
                )
                await self._update_document_status(doc_id, "completed")
                await self.db.commit()
                return

            # Step 3: Create parent chunks and save to PostgreSQL
            parent_chunk_data = create_parent_chunks(elements, doc_id)
            parent_chunk_models = [
                ParentChunk(
                    id=pc.id,
                    doc_id=pc.doc_id,
                    text=pc.text,
                    chunk_index=pc.chunk_index,
                )
                for pc in parent_chunk_data
            ]
            await self.rag_repo.save_parent_chunks(parent_chunk_models)
            await self.db.commit()

            # Step 4: Split into child chunks
            all_children: list[ChildChunkData] = []
            for parent in parent_chunk_data:
                children = split_into_children(parent)
                all_children.extend(children)

            if not all_children:
                logger.warning(
                    "No child chunks generated for doc_id='%s'", doc_id
                )
                await self.rag_repo.update_ingestion_task(
                    task.id, "completed"
                )
                await self._update_document_status(doc_id, "completed")
                await self.db.commit()
                return

            # Step 5: Embed child chunks via Infinity
            child_texts = [c.text for c in all_children]
            embeddings = await self.embedding_provider.embed_texts(child_texts)

            if len(embeddings) != len(all_children):
                raise RuntimeError(
                    f"Embedding count mismatch for doc_id='{doc_id}': "
                    f"expected {len(all_children)}, got {len(embeddings)}"
                )

            # Step 6: Upsert to Qdrant
            chunk_vectors = []
            for child, embedding in zip(all_children, embeddings):
                chunk_vectors.append(
                    ChunkVector(
                        chunk_id=child.id,
                        parent_chunk_id=child.parent_chunk_id,
                        doc_id=child.doc_id,
                        dense_vector=embedding.dense,
                        sparse_indices=embedding.sparse_indices,
                        sparse_values=embedding.sparse_values,
                    )
                )

            await self.vector_store.upsert_chunks(chunk_vectors)

            # Step 7: Mark as completed
            await self.rag_repo.update_ingestion_task(task.id, "completed")
            await self._update_document_status(doc_id, "completed")
            await self.db.commit()

            logger.info(
                "Ingestion completed for doc_id='%s': "
                "%d parent chunks, %d child chunks embedded",
                doc_id,
                len(parent_chunk_data),
                len(all_children),
            )

        except Exception as e:
            await self.db.rollback()
            logger.error(
                "Ingestion failed for doc_id='%s': %s",
                doc_id,
                str(e),
                exc_info=True,
            )
            # Attempt to mark as failed
            try:
                await self.rag_repo.update_ingestion_task(
                    task.id, "failed", error_message=str(e)
                )
                await self._update_document_status(doc_id, "failed")
                await self.db.commit()
            except Exception as rollback_err:
                logger.error(
                    "Failed to update task status after error: %s",
                    str(rollback_err),
                )
            raise

    async def _get_document(self, doc_id: str):
        """Fetch a PDFDocument by ID."""
        from sqlalchemy.future import select
        from src.knowledge_base.model import PDFDocument

        result = await self.db.execute(
            select(PDFDocument).where(PDFDocument.id == doc_id)
        )
        return result.scalars().first()

    async def _update_document_status(
        self, doc_id: str, status: str
    ) -> None:
        """Update the ingestion_status field on PDFDocument."""
        from sqlalchemy.future import select
        from src.knowledge_base.model import PDFDocument

        result = await self.db.execute(
            select(PDFDocument).where(PDFDocument.id == doc_id)
        )
        doc = result.scalars().first()
        if doc:
            doc.ingestion_status = status
