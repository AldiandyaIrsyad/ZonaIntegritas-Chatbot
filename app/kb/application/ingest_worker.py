"""
Asynchronous document ingestion workflow for the KB domain.
"""

import os
import structlog
from sqlalchemy.ext.asyncio import AsyncSession
from typing import Optional

from app.kb.domain.interfaces import (
    IDocumentParser,
    ITextEmbedder,
    IVectorStore,
    IKBRepository,
    ChunkVector,
)
from app.kb.domain.models import ParentChunk
from app.thesis.chunking.logic import create_parent_chunks, split_into_children
from app.thesis.chunking.models import ChildChunkData, ContentType, ParentChunkData, ParsedElement
from app.thesis.chunking.router import classify_element
from app.thesis.chunking.interfaces import IVLMEnricher
from app.thesis.ivm.service import IVMService, IrrelevantDocumentException

logger = structlog.get_logger(__name__)


class IngestWorker:
    """Orchestrates the ingestion pipeline for a document into the Knowledge Base.

    Pipeline stages:
    1. Parse PDF → ParsedElement[]
    1b. IVM document relevance validation (optional)
    2. VLM enrichment — for FIGURE elements, generate text descriptions
    3. Create parent chunks (table-aware, figure-aware)
    4. Save parent chunks to Postgres
    5. Split into children (content-type-aware dispatcher)
    6. Embed children
    7. Upsert to Qdrant
    8. Complete
    """

    def __init__(
        self,
        db: AsyncSession,
        document_parser: IDocumentParser,
        text_embedder: ITextEmbedder,
        vector_store: IVectorStore,
        kb_repo: IKBRepository,
        ivm_service: Optional[IVMService] = None,
        vlm_enricher: Optional[IVLMEnricher] = None,
        image_dir: str = "./uploads/knowledge_base/images",
    ):
        self.db = db
        self.document_parser = document_parser
        self.text_embedder = text_embedder
        self.vector_store = vector_store
        self.kb_repo = kb_repo
        self.ivm_service = ivm_service
        self.vlm_enricher = vlm_enricher
        self.image_dir = image_dir

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

            # 1b. Document relevance validation (IVM) — sample up to 3 chunks
            if self.ivm_service is not None:
                sample_texts = [
                    el.text for el in elements if el.text and el.text.strip()
                ][:3]
                if sample_texts:
                    try:
                        await self.ivm_service.validate_document_relevance(sample_texts)
                    except IrrelevantDocumentException:
                        logger.warning("kb.ingest.document_irrelevant", doc_id=doc_id)
                        await self.kb_repo.update_ingestion_task(
                            task.id, "failed",
                            error_message="Document is not relevant to the knowledge base domain.",
                        )
                        pdf_doc.ingestion_status = "failed"  # type: ignore
                        await self.db.commit()
                        return

            # 1c. VLM enrichment — generate text descriptions for FIGURE elements
            elements = await self._enrich_figures(elements, str(pdf_doc.pdf_path), doc_id)

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
                    breadcrumbs=pc.breadcrumbs,
                    content_type=pc.content_type.value,
                    element_metadata=pc.element_metadata,
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
                    breadcrumbs=child.breadcrumbs,
                    content_type=child.content_type.value,
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

    async def _enrich_figures(
        self,
        elements: list[ParsedElement],
        pdf_path: str,
        doc_id: str,
    ) -> list[ParsedElement]:
        """Enrich FIGURE elements with VLM-generated text descriptions.

        For each element classified as FIGURE (Image/Figure type from the
        parser), this method:

        1. Extracts the page image using PyMuPDF (if the parser didn't
           already provide an image path).
        2. Calls the VLM enricher to generate a structured text description.
        3. Replaces the element's (likely empty) text with the description.

        If no VLM enricher is configured, figure elements with empty text
        are filtered out (they can't be embedded or searched without text).

        **Fail-closed**: If VLM enrichment fails for a specific figure,
        the error is logged and the element is kept with whatever text it
        has (possibly empty). The chunker will skip empty-text elements.

        Args:
            elements: Parsed elements from the document parser.
            pdf_path: Path to the source PDF (for fallback image extraction).
            doc_id: Document UUID for logging.

        Returns:
            The elements list with FIGURE elements enriched (text filled in).
        """
        if not elements:
            return elements

        # Classify all elements first
        for el in elements:
            el.content_type = classify_element(el)

        figure_elements = [el for el in elements if el.content_type == ContentType.FIGURE]
        if not figure_elements:
            return elements

        logger.info(
            "kb.ingest.vlm_enrichment_started",
            doc_id=doc_id,
            figure_count=len(figure_elements),
        )

        if self.vlm_enricher is None:
            # No VLM configured — filter out empty-text figures (can't embed them)
            kept = [el for el in elements if not (el.content_type == ContentType.FIGURE and not el.text.strip())]
            skipped = len(elements) - len(kept)
            if skipped > 0:
                logger.warning(
                    "kb.ingest.figures_skipped",
                    doc_id=doc_id,
                    skipped=skipped,
                    reason="no_vlm_enricher",
                )
            return kept

        # Set PDF path for fallback mode
        if hasattr(self.vlm_enricher, "set_pdf_path"):
            self.vlm_enricher.set_pdf_path(pdf_path)  # type: ignore[attr-defined]

        os.makedirs(self.image_dir, exist_ok=True)
        enriched_count = 0

        for el in figure_elements:
            # If the element already has text (e.g. from a previous run), skip
            if el.text.strip():
                enriched_count += 1
                continue

            # Get the image path from metadata, or extract the page
            image_path = el.metadata.get("image_path")
            page_number = el.metadata.get("page_number", 1)

            if not image_path:
                # Extract the page as an image using PyMuPDF
                image_path = self._extract_page_image(pdf_path, page_number, doc_id)
                if image_path:
                    el.metadata["image_path"] = image_path

            if not image_path:
                logger.warning(
                    "kb.ingest.figure_no_image",
                    doc_id=doc_id,
                    page=page_number,
                    reason="image_extraction_failed",
                )
                continue

            try:
                description = await self.vlm_enricher.describe_image(image_path)
                if description and description.strip():
                    el.text = description.strip()
                    enriched_count += 1
                    logger.info(
                        "kb.ingest.figure_enriched",
                        doc_id=doc_id,
                        page=page_number,
                        desc_len=len(description),
                    )
                else:
                    logger.warning(
                        "kb.ingest.vlm_empty_description",
                        doc_id=doc_id,
                        page=page_number,
                    )
            except Exception as exc:
                logger.error(
                    "kb.ingest.vlm_enrichment_failed",
                    doc_id=doc_id,
                    page=page_number,
                    error=str(exc),
                )

        logger.info(
            "kb.ingest.vlm_enrichment_completed",
            doc_id=doc_id,
            enriched=enriched_count,
            total_figures=len(figure_elements),
        )

        # Filter out figures that still have no text (can't embed them)
        return [el for el in elements if el.text.strip()]

    def _extract_page_image(
        self,
        pdf_path: str,
        page_number: int,
        doc_id: str,
    ) -> Optional[str]:
        """Extract a PDF page as a PNG image using PyMuPDF.

        Args:
            pdf_path: Path to the source PDF.
            page_number: 1-indexed page number.
            doc_id: Document UUID (for naming the output file).

        Returns:
            Path to the extracted PNG, or None if extraction failed.
        """
        try:
            import fitz  # PyMuPDF

            doc = fitz.open(pdf_path)
            page_idx = max(0, page_number - 1)
            if page_idx >= len(doc):
                doc.close()
                return None

            page = doc[page_idx]
            mat = fitz.Matrix(150 / 72, 150 / 72)  # 150 DPI
            pix = page.get_pixmap(matrix=mat)

            output_path = os.path.join(
                self.image_dir,
                f"{doc_id}_page_{page_number}.png",
            )
            pix.save(output_path)
            doc.close()
            return output_path
        except Exception as exc:
            logger.error(
                "kb.ingest.image_extraction_failed",
                page=page_number,
                error=str(exc),
            )
            return None
