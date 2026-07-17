"""Ingestion pipeline visualization runner.

Runs the REAL ingestion pipeline (parse → route → chunk → embed → upsert)
against a temporary SQLite database and an ephemeral Qdrant collection,
capturing a snapshot of every intermediate stage for HTML report generation.

This module intentionally imports from :mod:`app.kb.infra` (real HTTP
adapters) to produce authentic output. It is research tooling, not
production thesis code.
"""

from __future__ import annotations

import os
import structlog
from typing import List, Optional

from sqlalchemy.ext.asyncio import AsyncSession

from app.kb.domain.interfaces import ChunkVector
from app.kb.domain.models import PDFDocument, ParentChunk, ChildChunk, IngestionTask
from app.kb.infra.infinity_embeddings import InfinityEmbeddings
from app.kb.infra.postgres_repo import PostgresKBRepository
from app.kb.infra.qdrant_store import QdrantStore
from app.kb.infra.unstructured_client import UnstructuredClient
from app.thesis.chunking import create_parent_chunks, split_into_children
from app.thesis.vlm.interfaces import IVLMEnricher
from app.thesis.chunking.models import ParsedElement
from app.thesis.chunking.router import classify_element
from app.thesis.chunking.table_converter import html_table_to_markdown

from .capture import (
    ChildChunkSnapshot,
    EmbeddingSnapshot,
    IngestionSnapshot,
    ParentChunkSnapshot,
    ParsedElementSnapshot,
    QdrantPointSnapshot,
)

logger = structlog.get_logger(__name__)

# How many dense vector floats to show in the preview
_DENSE_PREVIEW_LEN = 8

# How many sparse tokens to show (sorted by weight descending)
_SPARSE_TOP_N = 5

# Maximum text preview length for parsed elements
_ELEMENT_TEXT_PREVIEW = 200


async def run_ingestion(
    *,
    pdf_path: str,
    pdf_title: str,
    session: AsyncSession,
    sqlite_path: str,
    qdrant_collection: str,
    unstructured_url: str,
    infinity_url: str,
    qdrant_host: str,
    qdrant_port: int,
    embedding_model: str = "BAAI/bge-m3",
    unstructured_api_key: str = "",
    vlm_enricher: Optional[IVLMEnricher] = None,
    image_dir: str = "./uploads/knowledge_base/images",
) -> IngestionSnapshot:
    """Run the full ingestion pipeline and capture every stage.

    Args:
        pdf_path: Absolute path to the source PDF.
        pdf_title: Human-readable title for the PDFDocument row.
        session: Async SQLAlchemy session bound to the temp SQLite DB.
        sqlite_path: Path to the SQLite file (for reporting).
        qdrant_collection: Ephemeral Qdrant collection name.
        unstructured_url: Base URL of the Unstructured API.
        infinity_url: Base URL of the Infinity embedding server.
        qdrant_host: Qdrant host.
        qdrant_port: Qdrant HTTP port.
        embedding_model: BGE-M3 model identifier for Infinity.
        vlm_enricher: Optional VLM enricher for figure descriptions.
            If None, figures get a placeholder text.
        image_dir: Directory for extracted page images (VLM enrichment).

    Returns:
        An :class:`IngestionSnapshot` with all captured stage data.
    """
    document_parser = UnstructuredClient(
        base_url=unstructured_url, extract_images=True, api_key=unstructured_api_key
    )
    text_embedder = InfinityEmbeddings(
        base_url=infinity_url, model=embedding_model, batch_size=8
    )
    vector_store = QdrantStore(
        host=qdrant_host, port=qdrant_port, collection_name=qdrant_collection
    )
    kb_repo = PostgresKBRepository(session)

    try:
        return await _run_pipeline(
            pdf_path=pdf_path,
            pdf_title=pdf_title,
            session=session,
            kb_repo=kb_repo,
            document_parser=document_parser,
            text_embedder=text_embedder,
            vector_store=vector_store,
            sqlite_path=sqlite_path,
            qdrant_collection=qdrant_collection,
            unstructured_api_key=unstructured_api_key,
            vlm_enricher=vlm_enricher,
            image_dir=image_dir,
        )
    finally:
        await document_parser.close()
        await text_embedder.close()
        await vector_store.close()
        if vlm_enricher is not None:
            await vlm_enricher.close()


async def _run_pipeline(
    *,
    pdf_path: str,
    pdf_title: str,
    session: AsyncSession,
    kb_repo: PostgresKBRepository,
    document_parser: UnstructuredClient,
    text_embedder: InfinityEmbeddings,
    vector_store: QdrantStore,
    sqlite_path: str,
    qdrant_collection: str,
    unstructured_api_key: str = "",
    vlm_enricher: Optional[IVLMEnricher] = None,
    image_dir: str = "./uploads/knowledge_base/images",
) -> IngestionSnapshot:
    """Execute the ingestion pipeline stages and capture snapshots."""

    # ── Stage 0: Create PDFDocument + IngestionTask ──────────────────
    pdf_doc = await kb_repo.create_pdf(
        title=pdf_title, description="Visualization run", pdf_path=pdf_path
    )
    doc_id: str = str(pdf_doc.id)
    task = await kb_repo.create_ingestion_task(doc_id=doc_id)
    updated_task = await kb_repo.update_ingestion_task(task.id, status="processing")
    if updated_task is not None:
        task = updated_task
    await session.commit()
    logger.info("viz.ingest.pdf_created", doc_id=doc_id, title=pdf_title)

    # ── Stage 1: Parse PDF ──────────────────────────────────────────
    raw_elements = await document_parser.parse_pdf(pdf_path)
    logger.info("viz.ingest.parsed", element_count=len(raw_elements))

    element_snapshots: List[ParsedElementSnapshot] = []
    element_type_counts: dict[str, int] = {}
    total_element_chars = 0

    for i, el in enumerate(raw_elements):
        text_preview = el.text[:_ELEMENT_TEXT_PREVIEW]
        if len(el.text) > _ELEMENT_TEXT_PREVIEW:
            text_preview += "…"
        element_snapshots.append(
            ParsedElementSnapshot(
                index=i,
                element_type=el.element_type,
                text_preview=text_preview,
                text_length=len(el.text),
                page=el.metadata.get("page_number"),
                metadata_keys=sorted(el.metadata.keys()),
            )
        )
        element_type_counts[el.element_type] = (
            element_type_counts.get(el.element_type, 0) + 1
        )
        total_element_chars += len(el.text)

    # ── Stage 2: Route & enrich elements ─────────────────────────────
    # Classify each element; convert HTML tables to Markdown;
    # enrich figures via VLM (if configured) or use placeholder text.
    if vlm_enricher is not None and hasattr(vlm_enricher, "set_pdf_path"):
        vlm_enricher.set_pdf_path(pdf_path)  # type: ignore[attr-defined]

    os.makedirs(image_dir, exist_ok=True)
    enriched_elements: List[ParsedElement] = []
    vlm_enriched_count = 0
    for el in raw_elements:
        el.content_type = classify_element(el)

        if el.content_type.value == "table" and el.text:
            # Convert HTML table → Markdown for better embeddings
            markdown = html_table_to_markdown(el.text)
            if markdown and markdown.strip():
                el.text = markdown

        if el.content_type.value == "figure" and not el.text.strip():
            if vlm_enricher is not None:
                # Extract page image and call VLM for description
                page_number = el.metadata.get("page_number", 1)
                image_path = el.metadata.get("image_path")
                if not image_path:
                    image_path = _extract_page_image(
                        pdf_path, page_number, doc_id, image_dir
                    )
                    if image_path:
                        el.metadata["image_path"] = image_path

                if image_path:
                    try:
                        description = await vlm_enricher.describe_image(image_path)
                        if description and description.strip():
                            el.text = description.strip()
                            vlm_enriched_count += 1
                            logger.info(
                                "viz.ingest.figure_enriched",
                                page=page_number,
                                desc_len=len(description),
                            )
                        else:
                            el.text = "[Figure — VLM returned empty description]"
                    except Exception as exc:
                        logger.error(
                            "viz.ingest.vlm_failed",
                            page=page_number,
                            error=str(exc),
                        )
                        el.text = f"[Figure — VLM enrichment failed: {exc}]"
                else:
                    el.text = "[Figure — image extraction failed]"
            else:
                el.text = "[Figure — VLM enrichment skipped (no VLM configured)]"

        if el.text.strip():
            enriched_elements.append(el)

    logger.info(
        "viz.ingest.routed",
        original=len(raw_elements),
        enriched=len(enriched_elements),
        vlm_enriched=vlm_enriched_count,
    )

    # ── Stage 3: Create parent chunks ────────────────────────────────
    parent_chunk_data = create_parent_chunks(enriched_elements, doc_id=doc_id)
    logger.info("viz.ingest.parents_created", count=len(parent_chunk_data))

    parent_snapshots: List[ParentChunkSnapshot] = []
    content_type_counts: dict[str, int] = {}
    total_parent_chars = 0
    for pc in parent_chunk_data:
        parent_snapshots.append(
            ParentChunkSnapshot(
                id=pc.id,
                chunk_index=pc.chunk_index,
                page=pc.page,
                breadcrumbs=list(pc.breadcrumbs),
                content_type=pc.content_type.value,
                text=pc.text,
                text_length=len(pc.text),
                parent_id=pc.parent_id,
                ordinal=pc.ordinal,
                path=pc.path,
                depth=pc.depth,
            )
        )
        content_type_counts[pc.content_type.value] = (
            content_type_counts.get(pc.content_type.value, 0) + 1
        )
        total_parent_chars += len(pc.text)

    # ── Stage 4: Persist parent chunks to SQLite ────────────────────
    parent_models: List[ParentChunk] = [
        ParentChunk(
            id=pc.id,
            doc_id=pc.doc_id,
            text=pc.text,
            chunk_index=pc.chunk_index,
            page=pc.page,
            breadcrumbs=list(pc.breadcrumbs),
            content_type=pc.content_type.value,
            element_metadata=dict(pc.element_metadata),
            parent_id=pc.parent_id,
            ordinal=pc.ordinal,
            path=pc.path,
            depth=pc.depth,
        )
        for pc in parent_chunk_data
    ]
    await kb_repo.save_parent_chunks(parent_models)
    await session.commit()
    logger.info("viz.ingest.parents_saved", count=len(parent_models))

    # ── Stage 5: Split into children ─────────────────────────────────
    all_children = []
    child_snapshots: List[ChildChunkSnapshot] = []
    total_child_chars = 0
    for pc in parent_chunk_data:
        children = split_into_children(pc)
        for child in children:
            all_children.append(child)
            child_snapshots.append(
                ChildChunkSnapshot(
                    id=child.id,
                    parent_chunk_id=child.parent_chunk_id,
                    parent_index=pc.chunk_index,
                    text=child.text,
                    text_length=len(child.text),
                    content_type=child.content_type.value,
                    ordinal=child.ordinal,
                    path=child.path,
                )
            )
            total_child_chars += len(child.text)

    logger.info("viz.ingest.children_created", count=len(all_children))

    # ── Stage 5b: Save child chunks to SQLite ────────────────────────
    child_models: List[ChildChunk] = [
        ChildChunk(
            id=child.id,
            parent_chunk_id=child.parent_chunk_id,
            doc_id=child.doc_id,
            text=child.text,
            ordinal=child.ordinal,
            path=child.path,
            page=child.page,
            content_type=child.content_type.value,
        )
        for child in all_children
    ]
    await kb_repo.save_child_chunks(child_models)
    await session.commit()
    logger.info("viz.ingest.children_saved", count=len(child_models))

    # ── Stage 6: Embed children ─────────────────────────────────────
    child_texts = [c.text for c in all_children]
    embeddings = await text_embedder.embed_texts(child_texts)

    embedding_snapshots: List[EmbeddingSnapshot] = []
    for child, emb in zip(all_children, embeddings):
        # Sort sparse tokens by weight descending, take top N
        sparse_pairs = sorted(
            zip(emb.sparse_indices, emb.sparse_values),
            key=lambda x: x[1],
            reverse=True,
        )
        embedding_snapshots.append(
            EmbeddingSnapshot(
                child_id=child.id,
                dense_dim=len(emb.dense),
                dense_preview=list(emb.dense[:_DENSE_PREVIEW_LEN]),
                sparse_nnz=len(emb.sparse_indices),
                sparse_top5=sparse_pairs[:_SPARSE_TOP_N],
            )
        )

    logger.info(
        "viz.ingest.embedded",
        count=len(embeddings),
        dense_dim=len(embeddings[0].dense) if embeddings else 0,
    )

    # ── Stage 7: Upsert to Qdrant ───────────────────────────────────
    await vector_store.ensure_collection()

    chunk_vectors: List[ChunkVector] = [
        ChunkVector(
            chunk_id=child.id,
            parent_chunk_id=child.parent_chunk_id,
            doc_id=child.doc_id,
            dense_vector=emb.dense,
            sparse_indices=emb.sparse_indices,
            sparse_values=emb.sparse_values,
            breadcrumbs=list(child.breadcrumbs),
            content_type=child.content_type.value,
            text=child.text,
        )
        for child, emb in zip(all_children, embeddings)
    ]
    await vector_store.upsert_chunks(chunk_vectors)

    # Verify point count
    count_info = await vector_store._client.count(
        collection_name=qdrant_collection
    )
    qdrant_point_count = count_info.count

    qdrant_snapshots: List[QdrantPointSnapshot] = [
        QdrantPointSnapshot(
            point_id=cv.chunk_id,
            parent_chunk_id=cv.parent_chunk_id,
            doc_id=cv.doc_id,
            content_type=cv.content_type,
            dense_dim=len(cv.dense_vector),
            sparse_nnz=len(cv.sparse_indices),
        )
        for cv in chunk_vectors
    ]

    logger.info("viz.ingest.upserted", points=qdrant_point_count)

    # ── Stage 8: Complete ───────────────────────────────────────────
    await kb_repo.update_ingestion_task(task.id, status="completed")
    pdf_doc.ingestion_status = "completed"  # type: ignore[assignment]
    await session.commit()

    return IngestionSnapshot(
        doc_id=doc_id,
        doc_title=pdf_title,
        pdf_path=pdf_path,
        sqlite_path=sqlite_path,
        qdrant_collection=qdrant_collection,
        elements=element_snapshots,
        element_type_counts=dict(element_type_counts),
        parents=parent_snapshots,
        children=child_snapshots,
        embeddings=embedding_snapshots,
        qdrant_points=qdrant_snapshots,
        qdrant_point_count=qdrant_point_count,
        content_type_counts=dict(content_type_counts),
        total_element_chars=total_element_chars,
        total_parent_chars=total_parent_chars,
        total_child_chars=total_child_chars,
    )


def _extract_page_image(
    pdf_path: str,
    page_number: int,
    doc_id: str,
    image_dir: str,
) -> Optional[str]:
    """Extract a PDF page as a PNG image using PyMuPDF.

    Args:
        pdf_path: Path to the source PDF.
        page_number: 1-indexed page number.
        doc_id: Document UUID (for naming the output file).
        image_dir: Directory to save the extracted image.

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
            image_dir,
            f"{doc_id}_page_{page_number}.png",
        )
        pix.save(output_path)
        doc.close()
        return output_path
    except Exception as exc:
        logger.error("viz.ingest.image_extract_failed", page=page_number, error=str(exc))
        return None
