"""Production retrieval visualization runner.

Runs the REAL 6-step ``SearchService`` pipeline (optionally with HyDE query
expansion, matching production ``ChatService`` behavior) against a real
Postgres session and a real Qdrant collection — as opposed to
:mod:`retrieval_viz`, which is built for the ephemeral single-doc demo and
never wires HyDE.

Provenance tagging (which of the final results are "primary" search matches
vs. "sibling" same-parent hydration vs. "cross_ref" Pasal/BAB/Ayat lookups)
is captured by wrapping the real ``PostgresKBRepository`` with a thin
recording proxy around exactly the two read methods
(``get_sibling_chunks``, ``get_chunks_by_path_prefix``) that
``SearchService._hydrate_siblings`` / ``_detect_and_fetch_cross_refs`` call
internally — the real, unmodified ``SearchService.search()`` still runs
exactly once, so there is no risk of a second (possibly different) HyDE
generation or a duplicated LLM call skewing the "official" result set.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import structlog
from sqlalchemy.ext.asyncio import AsyncSession

from app.kb.config import get_bge_m3_settings
from app.kb.domain.interfaces import IQueryExpander, SearchResult
from app.kb.domain.models import ParentChunk, RetrievedContext
from app.kb.infra.bge_m3_embeddings import BGEM3Embeddings
from app.kb.infra.infinity_reranker import InfinityReranker
from app.kb.infra.postgres_repo import PostgresKBRepository
from app.kb.infra.qdrant_store import QdrantStore
from app.kb.application.search_service import SearchService

from .capture import RetrievedParentSnapshot, RetrievalSnapshot, SearchResultSnapshot

logger = structlog.get_logger(__name__)

_DENSE_PREVIEW_LEN = 8


class _ProvenanceCapturingKBRepo:
    """Wraps a real ``PostgresKBRepository`` to record, for provenance
    tagging only, which parent-chunk IDs were fetched via sibling hydration
    vs. cross-reference lookup. Delegates every other method unchanged."""

    def __init__(self, inner: PostgresKBRepository):
        self._inner = inner
        self.sibling_ids: set[str] = set()
        self.cross_ref_ids: set[str] = set()
        self.detected_prefixes: List[str] = []

    async def get_sibling_chunks(self, parent_id: str) -> List[ParentChunk]:
        result = await self._inner.get_sibling_chunks(parent_id)
        self.sibling_ids.update(c.id for c in result)
        return result

    async def get_chunks_by_path_prefix(self, path_prefix: str) -> List[ParentChunk]:
        result = await self._inner.get_chunks_by_path_prefix(path_prefix)
        self.detected_prefixes.append(path_prefix)
        self.cross_ref_ids.update(c.id for c in result)
        return result

    def __getattr__(self, name):
        return getattr(self._inner, name)


async def run_production_retrieval(
    *,
    query: str,
    session: AsyncSession,
    qdrant_collection: str,
    infinity_url: str,
    qdrant_host: str,
    qdrant_port: int,
    embedding_model: str = "BAAI/bge-m3",
    reranker_model: str = "BAAI/bge-reranker-v2-m3",
    top_k: int = 15,
    query_expander: Optional[IQueryExpander] = None,
) -> RetrievalSnapshot:
    """Run the real retrieval pipeline for one query and capture all stages.

    Args:
        query: The search query text.
        session: Async SQLAlchemy session bound to the isolated Postgres DB.
        qdrant_collection: The isolated Qdrant collection name.
        infinity_url: Base URL of the Infinity reranker server (embedding no
            longer runs through Infinity — see ``text_embedder`` below).
        qdrant_host: Qdrant host.
        qdrant_port: Qdrant HTTP port.
        embedding_model: Unused now that embedding runs in-process via
            ``BGEM3Embeddings``/``get_bge_m3_settings()``; kept for
            signature compatibility.
        reranker_model: Cross-encoder reranker model identifier.
        top_k: Final result count (production chat uses 15).
        query_expander: Real ``HyDEExpander`` to match production behavior,
            or None to search on the raw query only.

    Returns:
        A :class:`RetrievalSnapshot` with dense/sparse/hybrid comparison,
        the final reranked+hydrated results, and real provenance tags.
    """
    bge_m3_cfg = get_bge_m3_settings()
    text_embedder = BGEM3Embeddings(
        model_name=bge_m3_cfg.model,
        device=bge_m3_cfg.device,
        use_fp16=bge_m3_cfg.use_fp16,
        batch_size=bge_m3_cfg.batch_size,
    )
    vector_store = QdrantStore(host=qdrant_host, port=qdrant_port, collection_name=qdrant_collection)
    real_kb_repo = PostgresKBRepository(session)
    capturing_repo = _ProvenanceCapturingKBRepo(real_kb_repo)
    reranker = InfinityReranker(base_url=infinity_url, model=reranker_model)

    try:
        # ── Stage 1-2: embed the raw query (for the 3-mode comparison —
        # deliberately NOT HyDE-expanded here, so this comparison reflects
        # the literal query the user typed, independent of query expansion).
        embeddings = await text_embedder.embed_texts([query])
        query_emb = embeddings[0]

        dense_results = await vector_store.hybrid_search(
            dense_vector=query_emb.dense, sparse_indices=query_emb.sparse_indices,
            sparse_values=query_emb.sparse_values, top_k=top_k, mode="dense",
        )
        sparse_results = await vector_store.hybrid_search(
            dense_vector=query_emb.dense, sparse_indices=query_emb.sparse_indices,
            sparse_values=query_emb.sparse_values, top_k=top_k, mode="sparse",
        )
        hybrid_results = await vector_store.hybrid_search(
            dense_vector=query_emb.dense, sparse_indices=query_emb.sparse_indices,
            sparse_values=query_emb.sparse_values, top_k=top_k, mode="hybrid",
        )

        dense_snaps = _to_snapshots(dense_results)
        sparse_snaps = _to_snapshots(sparse_results)
        hybrid_snaps = _to_snapshots(hybrid_results)

        # ── Stage 3-6: the real, unmodified 6-step SearchService pipeline,
        # run exactly once (so HyDE, if enabled, is generated only once and
        # the "primary" set below is byte-identical to what production
        # ChatService would retrieve for this same query).
        search_service = SearchService(
            text_embedder=text_embedder,
            vector_store=vector_store,
            kb_repo=capturing_repo,  # type: ignore[arg-type]
            reranker=reranker,
            query_expander=query_expander,
        )
        contexts: List[RetrievedContext] = await search_service.search(query=query, top_k=top_k, mode="hybrid")

        dense_score_map: Dict[str, float] = {s.chunk_id: s.score for s in dense_snaps}
        sparse_score_map: Dict[str, float] = {s.chunk_id: s.score for s in sparse_snaps}

        retrieved_parents: List[RetrievedParentSnapshot] = []
        for rank, ctx in enumerate(contexts):
            if ctx.parent_chunk_id in capturing_repo.cross_ref_ids:
                provenance = "cross_ref"
            elif ctx.parent_chunk_id in capturing_repo.sibling_ids:
                provenance = "sibling"
            else:
                provenance = "primary"
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
                    provenance=provenance,
                )
            )

        sibling_count = sum(1 for p in retrieved_parents if p.provenance == "sibling")
        cross_ref_count = sum(1 for p in retrieved_parents if p.provenance == "cross_ref")

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
            sibling_count=sibling_count,
            cross_ref_count=cross_ref_count,
            final_context_count=len(contexts),
            detected_cross_refs=list(dict.fromkeys(capturing_repo.detected_prefixes)),
        )
    finally:
        await text_embedder.close()
        await vector_store.close()
        await reranker.close()


def _to_snapshots(results: List[SearchResult]) -> List[SearchResultSnapshot]:
    return [
        SearchResultSnapshot(rank=i, chunk_id=r.chunk_id, parent_chunk_id=r.parent_chunk_id, doc_id=r.doc_id, score=r.score)
        for i, r in enumerate(results)
    ]
