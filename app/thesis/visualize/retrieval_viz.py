"""Retrieval pipeline visualization runner.

Runs the REAL 6-step retrieval pipeline against the Qdrant collection
and SQLite DB populated during ingestion, capturing scores for dense,
sparse, and hybrid (RRF) modes, plus reranking, sibling hydration, and
cross-reference detection.

Pipeline stages captured:
1. (Optional) HyDE query expansion
2. Embed query → dense + sparse vectors
3. Hybrid search (top_k=50) → SearchResult[]
4. Fetch child chunks → get child text + breadcrumbs
5. Cross-encoder rerank chunks → top-8
6. Hydrate parents + siblings + cross-refs → merge + dedupe
"""

from __future__ import annotations

import structlog
from typing import Dict, List, Optional

from sqlalchemy.ext.asyncio import AsyncSession

from app.kb.domain.interfaces import SearchResult
from app.kb.domain.models import ParentChunk, RetrievedContext
from app.kb.infra.infinity_embeddings import InfinityEmbeddings
from app.kb.infra.infinity_reranker import InfinityReranker
from app.kb.infra.postgres_repo import PostgresKBRepository
from app.kb.infra.qdrant_store import QdrantStore
from app.kb.application.search_service import SearchService

from .capture import (
    IngestionSnapshot,
    RetrievedParentSnapshot,
    RetrievalSnapshot,
    SearchResultSnapshot,
)

logger = structlog.get_logger(__name__)

# How many dense vector floats to show in the query embedding preview
_DENSE_PREVIEW_LEN = 8

# Default top_k for search
_DEFAULT_TOP_K = 10

# Reranker model name
_RERANKER_MODEL = "BAAI/bge-reranker-v2-m3"


async def run_retrieval(
    *,
    query: str,
    ingestion: IngestionSnapshot,
    session: AsyncSession,
    infinity_url: str,
    qdrant_host: str,
    qdrant_port: int,
    embedding_model: str = "BAAI/bge-m3",
    top_k: int = _DEFAULT_TOP_K,
) -> RetrievalSnapshot:
    """Run the 6-step retrieval pipeline and capture all stages.

    Args:
        query: The search query text.
        ingestion: The ingestion snapshot (provides collection name, doc_id).
        session: Async SQLAlchemy session bound to the temp SQLite DB.
        infinity_url: Base URL of the Infinity embedding server.
        qdrant_host: Qdrant host.
        qdrant_port: Qdrant HTTP port.
        embedding_model: BGE-M3 model identifier.
        top_k: Number of results per search mode.

    Returns:
        A :class:`RetrievalSnapshot` with all pipeline stage data.
    """
    text_embedder = InfinityEmbeddings(
        base_url=infinity_url, model=embedding_model, batch_size=8
    )
    vector_store = QdrantStore(
        host=qdrant_host, port=qdrant_port, collection_name=ingestion.qdrant_collection
    )
    kb_repo = PostgresKBRepository(session)
    reranker = InfinityReranker(base_url=infinity_url, model=_RERANKER_MODEL)

    try:
        return await _run_search(
            query=query,
            ingestion=ingestion,
            kb_repo=kb_repo,
            text_embedder=text_embedder,
            vector_store=vector_store,
            reranker=reranker,
            top_k=top_k,
        )
    finally:
        await text_embedder.close()
        await vector_store.close()
        await reranker.close()


async def _run_search(
    *,
    query: str,
    ingestion: IngestionSnapshot,
    kb_repo: PostgresKBRepository,
    text_embedder: InfinityEmbeddings,
    vector_store: QdrantStore,
    reranker: InfinityReranker,
    top_k: int,
) -> RetrievalSnapshot:
    """Execute the 6-step retrieval pipeline and capture snapshots."""

    # ── Stage 1: Embed the query ────────────────────────────────────
    embeddings = await text_embedder.embed_texts([query])
    query_emb = embeddings[0]

    logger.info(
        "viz.retrieval.query_embedded",
        dense_dim=len(query_emb.dense),
        sparse_nnz=len(query_emb.sparse_indices),
    )

    # ── Stage 2: Search in 3 modes (for comparison) ─────────────────
    dense_results = await vector_store.hybrid_search(
        dense_vector=query_emb.dense,
        sparse_indices=query_emb.sparse_indices,
        sparse_values=query_emb.sparse_values,
        top_k=top_k,
        mode="dense",
    )
    sparse_results = await vector_store.hybrid_search(
        dense_vector=query_emb.dense,
        sparse_indices=query_emb.sparse_indices,
        sparse_values=query_emb.sparse_values,
        top_k=top_k,
        mode="sparse",
    )
    hybrid_results = await vector_store.hybrid_search(
        dense_vector=query_emb.dense,
        sparse_indices=query_emb.sparse_indices,
        sparse_values=query_emb.sparse_values,
        top_k=top_k,
        mode="hybrid",
    )

    logger.info(
        "viz.retrieval.searched",
        dense=len(dense_results),
        sparse=len(sparse_results),
        hybrid=len(hybrid_results),
    )

    # ── Stage 3: Capture search result snapshots ────────────────────
    dense_snaps = _to_snapshots(dense_results)
    sparse_snaps = _to_snapshots(sparse_results)
    hybrid_snaps = _to_snapshots(hybrid_results)

    # ── Stage 4-6: Run the full SearchService pipeline ───────────────
    # Use the real SearchService to get reranking, sibling hydration,
    # and cross-reference detection.
    search_service = SearchService(
        text_embedder=text_embedder,
        vector_store=vector_store,
        kb_repo=kb_repo,
        reranker=reranker,
    )
    contexts: List[RetrievedContext] = await search_service.search(
        query=query, top_k=top_k, mode="hybrid"
    )

    # Build score lookup maps: chunk_id → score, for each mode
    dense_score_map: Dict[str, float] = {s.chunk_id: s.score for s in dense_snaps}
    sparse_score_map: Dict[str, float] = {s.chunk_id: s.score for s in sparse_snaps}

    # Convert RetrievedContext → RetrievedParentSnapshot
    retrieved_parents: List[RetrievedParentSnapshot] = []
    for rank, ctx in enumerate(contexts):
        retrieved_parents.append(
            RetrievedParentSnapshot(
                rank=rank,
                parent_chunk_id=ctx.parent_chunk_id,
                rrf_score=ctx.score,
                dense_score=dense_score_map.get(ctx.chunk_id),
                sparse_score=sparse_score_map.get(ctx.chunk_id),
                source_title=ctx.source_title,
                page=ctx.page,
                breadcrumbs=list(ctx.breadcrumbs),
                content_type=ctx.content_type,
                text=ctx.text,
                child_text=ctx.child_text or "",
                path=ctx.path,
            )
        )

    return RetrievalSnapshot(
        query=query,
        query_dense_preview=list(query_emb.dense[:_DENSE_PREVIEW_LEN]),
        query_sparse_nnz=len(query_emb.sparse_indices),
        dense_results=dense_snaps,
        sparse_results=sparse_snaps,
        hybrid_results=hybrid_snaps,
        retrieved_parents=retrieved_parents,
        rerank_input_count=len(hybrid_results),
        rerank_output_count=len(contexts),
        final_context_count=len(contexts),
    )


def _to_snapshots(
    results: List[SearchResult],
) -> List[SearchResultSnapshot]:
    """Convert SearchResult objects to frozen snapshots.

    Args:
        results: List of SearchResult dataclass instances.

    Returns:
        List of SearchResultSnapshot with rank assigned.
    """
    return [
        SearchResultSnapshot(
            rank=i,
            chunk_id=r.chunk_id,
            parent_chunk_id=r.parent_chunk_id,
            doc_id=r.doc_id,
            score=r.score,
        )
        for i, r in enumerate(results)
    ]
