"""Production ingestion visualization runner.

Unlike :mod:`ingestion_viz` (which reimplements the pipeline stages inline
against an ephemeral SQLite DB, and therefore never exercises the real
per-page ``classify_page`` → VLM routing decision), this module drives the
real, unmodified :class:`app.kb.application.ingest_worker.IngestWorker`
against a real Postgres session and a real Qdrant collection, using its
optional ``on_stage`` hook to capture a snapshot of every stage — including
the real page-classifier verdicts and VLM output for scanned/visual pages.

Intended for documentation/research use against an isolated database and
Qdrant collection (see ``writing/visualization/ukt/_common.py``), not the
production knowledge base.
"""

from __future__ import annotations

from typing import List, Optional

import structlog
from sqlalchemy.ext.asyncio import AsyncSession

from app.kb.application.ingest_worker import IngestWorker
from app.kb.config import get_bge_m3_settings
from app.kb.domain.interfaces import ChunkVector
from app.kb.infra.bge_m3_embeddings import BGEM3Embeddings
from app.kb.infra.postgres_repo import PostgresKBRepository
from app.kb.infra.qdrant_store import QdrantStore
from app.kb.infra.unstructured_client import UnstructuredClient
from app.thesis.chunking.page_classifier import PageClassification
from app.thesis.vlm.interfaces import IVLMEnricher

from .capture import (
    ChildChunkSnapshot,
    EmbeddingSnapshot,
    IngestionSnapshot,
    PageClassificationSnapshot,
    ParentChunkSnapshot,
    ParsedElementSnapshot,
    QdrantPointSnapshot,
)

logger = structlog.get_logger(__name__)

_DENSE_PREVIEW_LEN = 8
_SPARSE_TOP_N = 5
_ELEMENT_TEXT_PREVIEW = 200


async def run_production_ingestion(
    *,
    pdf_path: str,
    pdf_title: str,
    session: AsyncSession,
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
    """Ingest one PDF through the real ``IngestWorker`` and capture every stage.

    Args:
        pdf_path: Absolute path to the source PDF.
        pdf_title: Human-readable title for the PDFDocument row.
        session: Async SQLAlchemy session bound to the isolated Postgres DB.
        qdrant_collection: The isolated Qdrant collection name.
        unstructured_url: Base URL of the (self-hosted) Unstructured API.
        infinity_url: Base URL of the Infinity server — no longer used for
            embedding (see ``text_embedder`` below), kept for signature
            compatibility with callers; still relevant if a reranker is
            added to this capture path in the future.
        qdrant_host: Qdrant host.
        qdrant_port: Qdrant HTTP port.
        embedding_model: Unused now that embedding runs in-process via
            ``BGEM3Embeddings``/``get_bge_m3_settings()``; kept for
            signature compatibility.
        unstructured_api_key: Bearer token, empty for local self-hosted.
        vlm_enricher: Real VLM enricher (e.g. ``OpenRouterVLMClient``) — pass
            the same instance ``app.kb.dependency.get_vlm_enricher()``
            would build, so scanned/VISUAL pages get real VLM extraction.
        image_dir: Directory for extracted page images (VLM enrichment).

    Returns:
        An :class:`IngestionSnapshot` with all captured stage data,
        including real ``page_classifications``.
    """
    document_parser = UnstructuredClient(
        base_url=unstructured_url, extract_images=True, api_key=unstructured_api_key
    )
    bge_m3_cfg = get_bge_m3_settings()
    text_embedder = BGEM3Embeddings(
        model_name=bge_m3_cfg.model,
        device=bge_m3_cfg.device,
        use_fp16=bge_m3_cfg.use_fp16,
        batch_size=bge_m3_cfg.batch_size,
    )
    vector_store = QdrantStore(host=qdrant_host, port=qdrant_port, collection_name=qdrant_collection)
    kb_repo = PostgresKBRepository(session)

    # ── Capture state, populated by the on_stage hook as IngestWorker runs ──
    captured: dict = {
        "elements": None,
        "routed_elements": None,
        "page_classifications": [],
        "parent_chunk_data": None,
        "all_children": None,
        "embeddings": None,
        "chunk_vectors": None,
    }

    def on_stage(stage: str, data: dict) -> None:
        if stage == "parsed":
            captured["elements"] = data["elements"]
        elif stage == "page_classified":
            captured["page_classifications"].append(data["classification"])
        elif stage == "routed":
            captured["routed_elements"] = data["elements"]
        elif stage == "parent_chunks":
            captured["parent_chunk_data"] = data["parent_chunk_data"]
        elif stage == "children":
            captured["all_children"] = data["all_children"]
        elif stage == "embeddings":
            captured["embeddings"] = data["embeddings"]
        elif stage == "upserted":
            captured["chunk_vectors"] = data["chunk_vectors"]

    try:
        pdf_doc = await kb_repo.create_pdf(
            title=pdf_title, description="UKT documentation run", pdf_path=pdf_path
        )
        doc_id: str = str(pdf_doc.id)
        await session.commit()
        logger.info("prod_viz.ingest.pdf_created", doc_id=doc_id, title=pdf_title)

        worker = IngestWorker(
            db=session,
            document_parser=document_parser,
            text_embedder=text_embedder,
            vector_store=vector_store,
            kb_repo=kb_repo,
            vlm_enricher=vlm_enricher,
            image_dir=image_dir,
            on_stage=on_stage,
        )
        await worker.ingest_document(doc_id)

        return _assemble_snapshot(
            doc_id=doc_id,
            pdf_title=pdf_title,
            pdf_path=pdf_path,
            qdrant_collection=qdrant_collection,
            captured=captured,
        )
    finally:
        await document_parser.close()
        await text_embedder.close()
        await vector_store.close()
        if vlm_enricher is not None:
            await vlm_enricher.close()


def _assemble_snapshot(
    *, doc_id: str, pdf_title: str, pdf_path: str, qdrant_collection: str, captured: dict
) -> IngestionSnapshot:
    elements = captured["elements"] or []
    routed_elements = captured["routed_elements"] or []
    parent_chunk_data = captured["parent_chunk_data"] or []
    all_children = captured["all_children"] or []
    embeddings = captured["embeddings"] or []
    chunk_vectors: List[ChunkVector] = captured["chunk_vectors"] or []
    page_classifications: List[PageClassification] = captured["page_classifications"]

    element_snapshots: List[ParsedElementSnapshot] = []
    element_type_counts: dict[str, int] = {}
    total_element_chars = 0
    for i, el in enumerate(elements):
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
        element_type_counts[el.element_type] = element_type_counts.get(el.element_type, 0) + 1
        total_element_chars += len(el.text)

    page_classification_snapshots = [
        PageClassificationSnapshot(
            page_number=pc.page_number,
            page_type=pc.page_type.value,
            element_count=pc.element_count,
            image_count=pc.image_count,
            table_count=pc.table_count,
            garbage_image_count=pc.garbage_image_count,
            image_ratio=pc.image_ratio,
            garbage_ratio=pc.garbage_ratio,
        )
        for pc in page_classifications
    ]

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
        content_type_counts[pc.content_type.value] = content_type_counts.get(pc.content_type.value, 0) + 1
        total_parent_chars += len(pc.text)

    child_snapshots: List[ChildChunkSnapshot] = []
    total_child_chars = 0
    parent_index_by_id = {pc.id: pc.chunk_index for pc in parent_chunk_data}
    for child in all_children:
        child_snapshots.append(
            ChildChunkSnapshot(
                id=child.id,
                parent_chunk_id=child.parent_chunk_id,
                parent_index=parent_index_by_id.get(child.parent_chunk_id, -1),
                text=child.text,
                text_length=len(child.text),
                content_type=child.content_type.value,
                ordinal=child.ordinal,
                path=child.path,
            )
        )
        total_child_chars += len(child.text)

    embedding_snapshots: List[EmbeddingSnapshot] = []
    for child, emb in zip(all_children, embeddings):
        sparse_pairs = sorted(zip(emb.sparse_indices, emb.sparse_values), key=lambda x: x[1], reverse=True)
        embedding_snapshots.append(
            EmbeddingSnapshot(
                child_id=child.id,
                dense_dim=len(emb.dense),
                dense_preview=list(emb.dense[:_DENSE_PREVIEW_LEN]),
                sparse_nnz=len(emb.sparse_indices),
                sparse_top5=sparse_pairs[:_SPARSE_TOP_N],
            )
        )

    qdrant_snapshots = [
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

    return IngestionSnapshot(
        doc_id=doc_id,
        doc_title=pdf_title,
        pdf_path=pdf_path,
        sqlite_path="",  # not applicable — this run uses the isolated Postgres DB
        qdrant_collection=qdrant_collection,
        elements=element_snapshots,
        element_type_counts=dict(element_type_counts),
        parents=parent_snapshots,
        children=child_snapshots,
        embeddings=embedding_snapshots,
        qdrant_points=qdrant_snapshots,
        qdrant_point_count=len(qdrant_snapshots),
        content_type_counts=dict(content_type_counts),
        total_element_chars=total_element_chars,
        total_parent_chars=total_parent_chars,
        total_child_chars=total_child_chars,
        page_classifications=page_classification_snapshots,
    )
