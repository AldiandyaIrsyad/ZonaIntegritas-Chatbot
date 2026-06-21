"""
Asynchronous document ingestion workflow for the KB domain.
"""

import structlog
from sqlalchemy.ext.asyncio import AsyncSession

from app.kb.domain.interfaces import (
    IDocumentParser,
    ITextEmbedder,
    IVectorStore,
    IKBRepository,
    ChunkVector,
)
from app.kb.domain.models import ParentChunk
from app.thesis.chunking.logic import create_parent_chunks, split_into_children
from app.thesis.chunking.models import ChildChunkData, ParentChunkData

logger = structlog.get_logger(__name__)


class IngestWorker:
    """Orchestrates the ingestion pipeline for a document into the Knowledge Base."""

    def __init__(
        self,
        db: AsyncSession,
        document_parser: IDocumentParser,
        text_embedder: ITextEmbedder,
        vector_store: IVectorStore,
        kb_repo: IKBRepository,
    ):
        self.db = db
        self.document_parser = document_parser
        self.text_embedder = text_embedder
        self.vector_store = vector_store
        self.kb_repo = kb_repo

    async def ingest_document(self, doc_id: str) -> None:
        """Run the full ingestion pipeline for a document."""
        task = await self.kb_repo.create_ingestion_task(doc_id)
        await self.db.commit()

        try:
            await self.kb_repo.update_ingestion_task(task.id, "processing")
            pdf_doc = await self.kb_repo.get_pdf_by_id(doc_id)
            if not pdf_doc:
                raise ValueError(f"PDFDocument not found: {doc_id}")
            pdf_doc.ingestion_status = "processing"  # type: ignore
            await self.db.commit()

            logger.info("kb.ingest.started", doc_id=doc_id, stage="parsing")

            # 1. Parse PDF
            elements = await self.document_parser.parse_pdf(str(pdf_doc.pdf_path))
            if not elements:
                await self.kb_repo.update_ingestion_task(task.id, "completed")
                pdf_doc.ingestion_status = "completed"  # type: ignore
                await self.db.commit()
                return

            # 2. Pure Chunking Algorithm (Thesis)
            parent_chunk_data = create_parent_chunks(elements, doc_id)

            # 3. Save Parent Chunks to Postgres
            parent_chunk_models = [
                ParentChunk(
                    id=pc.id,
                    doc_id=pc.doc_id,
                    text=pc.text,
                    chunk_index=pc.chunk_index,
                    page=pc.page,
                )
                for pc in parent_chunk_data
            ]
            await self.kb_repo.save_parent_chunks(parent_chunk_models)

            # 4. Split into children
            all_children: list[ChildChunkData] = []
            for parent in parent_chunk_data:
                all_children.extend(split_into_children(parent))

            if not all_children:
                await self.kb_repo.update_ingestion_task(task.id, "completed")
                pdf_doc.ingestion_status = "completed"  # type: ignore
                await self.db.commit()
                return

            logger.info("kb.ingest.chunking_complete", doc_id=doc_id, parent_chunks=len(parent_chunk_data), child_chunks=len(all_children))

            # 5. Embed children
            child_texts = [c.text for c in all_children]
            embeddings = await self.text_embedder.embed_texts(child_texts)

            # 6. Upsert to Qdrant
            chunk_vectors = [
                ChunkVector(
                    chunk_id=child.id,
                    parent_chunk_id=child.parent_chunk_id,
                    doc_id=child.doc_id,
                    dense_vector=emb.dense,
                    sparse_indices=emb.sparse_indices,
                    sparse_values=emb.sparse_values,
                )
                for child, emb in zip(all_children, embeddings)
            ]
            await self.vector_store.upsert_chunks(chunk_vectors)

            # 7. Complete
            await self.kb_repo.update_ingestion_task(task.id, "completed")
            pdf_doc.ingestion_status = "completed"  # type: ignore
            await self.db.commit()

            logger.info("kb.ingest.completed", doc_id=doc_id)

        except Exception as e:
            await self.db.rollback()
            logger.error("kb.ingest.failed", doc_id=doc_id, error=str(e))
            try:
                await self.kb_repo.update_ingestion_task(task.id, "failed", error_message=str(e))
                pdf_doc = await self.kb_repo.get_pdf_by_id(doc_id)
                if pdf_doc:
                    pdf_doc.ingestion_status = "failed"  # type: ignore
                await self.db.commit()
            except Exception as rollback_err:
                logger.error("kb.ingest.status_update_failed", error=str(rollback_err))
            raise
