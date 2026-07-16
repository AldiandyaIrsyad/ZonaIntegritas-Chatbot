"""Combined multi-PDF retrieval visualization orchestrator.

Ingests multiple PDFs into a single shared Qdrant collection, then runs
multiple queries against that collection to show how documents interact
during hybrid retrieval (dense + sparse + RRF fusion).

This module reuses the existing :func:`run_ingestion` and
:func:`run_retrieval` functions as-is — it only orchestrates them in a
loop and aggregates the results into a :class:`CombinedSnapshot`.

Key design:
    - All PDFs are ingested into the SAME Qdrant collection
      (``ensure_collection()`` is idempotent — only the first call creates
      it; subsequent calls just upsert new points).
    - Each point's payload carries ``doc_id``, so search results naturally
      identify which document each chunk came from.
    - Retrieval searches across ALL documents' points (no ``doc_id`` filter).
"""

from __future__ import annotations

import structlog
from typing import Dict, List, Optional

from sqlalchemy.ext.asyncio import AsyncSession

from app.thesis.vlm.interfaces import IVLMEnricher

from .capture import (
    CombinedSnapshot,
    DocIngestionSummary,
    IngestionSnapshot,
    QueryRetrievalSummary,
    RetrievalSnapshot,
)
from .ingestion_viz import run_ingestion
from .retrieval_viz import run_retrieval

logger = structlog.get_logger(__name__)


async def run_combined(
    *,
    pdf_paths: List[str],
    pdf_titles: List[str],
    queries: List[str],
    session: AsyncSession,
    sqlite_path: str,
    qdrant_collection: str,
    unstructured_url: str,
    unstructured_api_key: str,
    infinity_url: str,
    qdrant_host: str,
    qdrant_port: int,
    embedding_model: str = "BAAI/bge-m3",
    top_k: int = 10,
    vlm_enricher: Optional[IVLMEnricher] = None,
    image_dir: str = "./uploads/knowledge_base/images",
) -> CombinedSnapshot:
    """Run multi-PDF ingestion + multi-query retrieval and capture snapshots.

    Args:
        pdf_paths: List of PDF file paths to ingest.
        pdf_titles: Human-readable titles (same length as ``pdf_paths``).
        queries: List of search queries to run.
        session: Async SQLAlchemy session bound to the shared SQLite DB.
        sqlite_path: Path to the SQLite file (for reporting).
        qdrant_collection: Shared Qdrant collection name.
        unstructured_url: Base URL of the Unstructured API.
        unstructured_api_key: API key for Unstructured Cloud (empty for local).
        infinity_url: Base URL of the Infinity embedding server.
        qdrant_host: Qdrant host.
        qdrant_port: Qdrant HTTP port.
        embedding_model: BGE-M3 model identifier for Infinity.
        top_k: Number of search results per mode per query.
        vlm_enricher: Optional VLM enricher for figure descriptions.
        image_dir: Directory for extracted page images (VLM enrichment).

    Returns:
        A :class:`CombinedSnapshot` with all per-doc and per-query data.
    """
    if len(pdf_paths) != len(pdf_titles):
        raise ValueError(
            f"pdf_paths ({len(pdf_paths)}) and pdf_titles ({len(pdf_titles)}) "
            "must have the same length."
        )

    total_docs = len(pdf_paths)
    total_queries = len(queries)

    logger.info(
        "viz.combined.start",
        total_docs=total_docs,
        total_queries=total_queries,
        qdrant_collection=qdrant_collection,
    )

    # ── Phase 1: Ingest all PDFs into the shared collection ──────────
    ingestions: List[IngestionSnapshot] = []
    for i, (pdf_path, pdf_title) in enumerate(zip(pdf_paths, pdf_titles), start=1):
        logger.info(
            "viz.combined.ingest.progress",
            doc_index=i,
            total_docs=total_docs,
            doc_title=pdf_title,
        )
        print(f"   [{i}/{total_docs}] Ingesting: {pdf_title}")

        ingestion = await run_ingestion(
            pdf_path=pdf_path,
            pdf_title=pdf_title,
            session=session,
            sqlite_path=sqlite_path,
            qdrant_collection=qdrant_collection,
            unstructured_url=unstructured_url,
            infinity_url=infinity_url,
            qdrant_host=qdrant_host,
            qdrant_port=qdrant_port,
            embedding_model=embedding_model,
            unstructured_api_key=unstructured_api_key,
            vlm_enricher=vlm_enricher,
            image_dir=image_dir,
        )
        ingestions.append(ingestion)

        logger.info(
            "viz.combined.ingest.done",
            doc_index=i,
            doc_title=pdf_title,
            elements=len(ingestion.elements),
            parents=len(ingestion.parents),
            children=len(ingestion.children),
            points=ingestion.qdrant_point_count,
        )
        print(
            f"      ✓ {len(ingestion.elements)} elements → "
            f"{len(ingestion.parents)} parents → "
            f"{len(ingestion.children)} children → "
            f"{ingestion.qdrant_point_count} points"
        )

    # ── Phase 2: Run all queries against the shared collection ───────
    # run_retrieval() only uses ingestion.qdrant_collection, so we can pass
    # any ingestion snapshot — the collection is shared.
    reference_ingestion = ingestions[0]

    retrievals: List[RetrievalSnapshot] = []
    for i, query in enumerate(queries, start=1):
        logger.info(
            "viz.combined.retrieval.progress",
            query_index=i,
            total_queries=total_queries,
            query=query,
        )
        print(f"   [{i}/{total_queries}] Querying: \"{query}\"")

        retrieval = await run_retrieval(
            query=query,
            ingestion=reference_ingestion,
            session=session,
            infinity_url=infinity_url,
            qdrant_host=qdrant_host,
            qdrant_port=qdrant_port,
            embedding_model=embedding_model,
            top_k=top_k,
        )
        retrievals.append(retrieval)

        logger.info(
            "viz.combined.retrieval.done",
            query_index=i,
            query=query,
            hybrid_results=len(retrieval.hybrid_results),
            retrieved_parents=len(retrieval.retrieved_parents),
        )
        print(
            f"      ✓ Dense: {len(retrieval.dense_results)} | "
            f"Sparse: {len(retrieval.sparse_results)} | "
            f"Hybrid: {len(retrieval.hybrid_results)} results"
        )

    # ── Phase 3: Build aggregated summaries ──────────────────────────
    doc_summaries = _build_doc_summaries(ingestions)
    query_summaries = _build_query_summaries(retrievals)

    logger.info(
        "viz.combined.complete",
        total_docs=len(ingestions),
        total_queries=len(retrievals),
        total_points=sum(s.qdrant_point_count for s in doc_summaries),
    )

    return CombinedSnapshot(
        timestamp=ingestions[0].doc_id,  # placeholder; set by caller
        qdrant_collection=qdrant_collection,
        sqlite_path=sqlite_path,
        ingestions=ingestions,
        retrievals=retrievals,
        doc_summaries=doc_summaries,
        query_summaries=query_summaries,
        queries=list(queries),
    )


def _build_doc_summaries(
    ingestions: List[IngestionSnapshot],
) -> List[DocIngestionSummary]:
    """Extract per-document ingestion summaries.

    Args:
        ingestions: List of full ingestion snapshots.

    Returns:
        List of :class:`DocIngestionSummary` with aggregated counts.
    """
    summaries: List[DocIngestionSummary] = []
    for ing in ingestions:
        summaries.append(
            DocIngestionSummary(
                doc_id=ing.doc_id,
                doc_title=ing.doc_title,
                pdf_path=ing.pdf_path,
                element_count=len(ing.elements),
                parent_count=len(ing.parents),
                child_count=len(ing.children),
                qdrant_point_count=ing.qdrant_point_count,
                content_type_counts=dict(ing.content_type_counts),
                total_chars=ing.total_parent_chars,
            )
        )
    return summaries


def _build_query_summaries(
    retrievals: List[RetrievalSnapshot],
) -> List[QueryRetrievalSummary]:
    """Aggregate per-query retrieval stats by document.

    For each query, counts how many results came from each document and
    tracks the best (highest) RRF score per document.

    Args:
        retrievals: List of full retrieval snapshots.

    Returns:
        List of :class:`QueryRetrievalSummary` with per-doc breakdowns.
    """
    summaries: List[QueryRetrievalSummary] = []
    for ret in retrievals:
        per_doc_counts: Dict[str, int] = {}
        per_doc_best_score: Dict[str, float] = {}

        for rp in ret.retrieved_parents:
            title = rp.source_title
            per_doc_counts[title] = per_doc_counts.get(title, 0) + 1
            current_best = per_doc_best_score.get(title, 0.0)
            if rp.rrf_score > current_best:
                per_doc_best_score[title] = rp.rrf_score

        # Top doc = first retrieved parent's source
        top_doc_title = (
            ret.retrieved_parents[0].source_title
            if ret.retrieved_parents
            else "—"
        )
        top_doc_score = (
            ret.retrieved_parents[0].rrf_score
            if ret.retrieved_parents
            else 0.0
        )

        summaries.append(
            QueryRetrievalSummary(
                query=ret.query,
                total_results=len(ret.retrieved_parents),
                per_doc_counts=dict(per_doc_counts),
                per_doc_best_score=dict(per_doc_best_score),
                top_doc_title=top_doc_title,
                top_doc_score=top_doc_score,
            )
        )
    return summaries
