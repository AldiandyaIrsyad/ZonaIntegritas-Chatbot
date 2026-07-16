"""Frozen dataclasses capturing snapshots of each pipeline stage.

These are plain data containers (no logic) used to pass captured data
from the ingestion/retrieval runners to the HTML report builder.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass(frozen=True)
class ParsedElementSnapshot:
    """Snapshot of a single parsed element from the Unstructured API."""

    index: int
    element_type: str
    text_preview: str
    text_length: int
    page: Optional[int]
    metadata_keys: List[str]


@dataclass(frozen=True)
class ParentChunkSnapshot:
    """Snapshot of a parent chunk after ``create_parent_chunks``.

    Attributes:
        id: UUID of the parent chunk.
        chunk_index: Sequential index within the document.
        page: Page number of the first element.
        breadcrumbs: Hierarchical section path.
        content_type: Structural type (text/table/figure/hybrid).
        text: Full text content.
        text_length: Character count of text.
        parent_id: UUID of the parent section (None for root).
        ordinal: Position within the parent section.
        path: ltree-style dot path.
        depth: Heading depth (0 = root).
    """

    id: str
    chunk_index: int
    page: Optional[int]
    breadcrumbs: List[str]
    content_type: str
    text: str
    text_length: int
    parent_id: Optional[str] = None
    ordinal: int = 0
    path: str = ""
    depth: int = 0


@dataclass(frozen=True)
class ChildChunkSnapshot:
    """Snapshot of a child chunk after ``split_into_children``.

    Attributes:
        id: UUID of the child chunk.
        parent_chunk_id: UUID of the parent chunk.
        parent_index: Chunk index of the parent.
        text: Child text (may include breadcrumb prefix).
        text_length: Character count of text.
        content_type: Structural type inherited from parent.
        ordinal: Position within the parent chunk.
        path: ltree-style path (parent path + ".c" + ordinal).
    """

    id: str
    parent_chunk_id: str
    parent_index: int
    text: str
    text_length: int
    content_type: str
    ordinal: int = 0
    path: str = ""


@dataclass(frozen=True)
class EmbeddingSnapshot:
    """Snapshot of an embedding result for a child chunk."""

    child_id: str
    dense_dim: int
    dense_preview: List[float]
    sparse_nnz: int
    sparse_top5: List[tuple[int, float]]


@dataclass(frozen=True)
class QdrantPointSnapshot:
    """Snapshot of a Qdrant point after upsert."""

    point_id: str
    parent_chunk_id: str
    doc_id: str
    content_type: str
    dense_dim: int
    sparse_nnz: int


@dataclass(frozen=True)
class IngestionSnapshot:
    """Complete snapshot of the ingestion pipeline run."""

    doc_id: str
    doc_title: str
    pdf_path: str
    sqlite_path: str
    qdrant_collection: str
    elements: List[ParsedElementSnapshot]
    element_type_counts: Dict[str, int]
    parents: List[ParentChunkSnapshot]
    children: List[ChildChunkSnapshot]
    embeddings: List[EmbeddingSnapshot]
    qdrant_points: List[QdrantPointSnapshot]
    qdrant_point_count: int
    content_type_counts: Dict[str, int]
    total_element_chars: int
    total_parent_chars: int
    total_child_chars: int


@dataclass(frozen=True)
class SearchResultSnapshot:
    """Snapshot of a single search result in one retrieval mode."""

    rank: int
    chunk_id: str
    parent_chunk_id: str
    doc_id: str
    score: float


@dataclass(frozen=True)
class RetrievedParentSnapshot:
    """Snapshot of a retrieved parent chunk with its score.

    Attributes:
        rank: Final rank after reranking and hydration.
        parent_chunk_id: UUID of the parent chunk.
        rrf_score: RRF fusion score from hybrid search.
        dense_score: Dense cosine similarity score, if available.
        sparse_score: Sparse BM25 score, if available.
        rerank_score: Cross-encoder reranker score, if available.
        source_title: Display name of the source document.
        page: Page number of the chunk.
        breadcrumbs: Hierarchical section path.
        content_type: Structural type.
        text: Full parent chunk text.
        child_text: The matching child chunk text.
        path: ltree-style path of the parent.
        sibling_ids: List of sibling chunk IDs (same parent).
        cross_refs: List of cross-reference strings detected in the text.
    """

    rank: int
    parent_chunk_id: str
    rrf_score: float
    dense_score: Optional[float]
    sparse_score: Optional[float]
    source_title: str
    page: Optional[int]
    breadcrumbs: List[str]
    content_type: str
    text: str
    rerank_score: Optional[float] = None
    child_text: str = ""
    path: str = ""
    sibling_ids: List[str] = field(default_factory=list)
    cross_refs: List[str] = field(default_factory=list)


@dataclass(frozen=True)
class RetrievalSnapshot:
    """Complete snapshot of the retrieval pipeline run.

    Captures all 6 stages of the retrieval pipeline:
    1. Query embedding
    2. Hybrid search (dense + sparse + RRF)
    3. Child chunk text fetch
    4. Cross-encoder reranking
    5. Parent/sibling/cross-ref hydration
    6. Merge + dedupe

    Attributes:
        query: The search query text.
        query_dense_preview: First 8 floats of the query dense embedding.
        query_sparse_nnz: Number of non-zero sparse tokens in the query.
        dense_results: Dense search results.
        sparse_results: Sparse search results.
        hybrid_results: Hybrid (RRF) search results.
        retrieved_parents: Final hydrated parent chunks after reranking.
        rerank_input_count: Number of candidates before reranking.
        rerank_output_count: Number of candidates after reranking.
        sibling_count: Total sibling chunks hydrated.
        cross_ref_count: Total cross-reference chunks fetched.
        final_context_count: Final deduplicated context count.
    """

    query: str
    query_dense_preview: List[float]
    query_sparse_nnz: int
    dense_results: List[SearchResultSnapshot]
    sparse_results: List[SearchResultSnapshot]
    hybrid_results: List[SearchResultSnapshot]
    retrieved_parents: List[RetrievedParentSnapshot]
    rerank_input_count: int = 0
    rerank_output_count: int = 0
    sibling_count: int = 0
    cross_ref_count: int = 0
    final_context_count: int = 0


@dataclass(frozen=True)
class PipelineSnapshot:
    """Top-level snapshot combining ingestion + retrieval."""

    timestamp: str
    ingestion: IngestionSnapshot
    retrieval: RetrievalSnapshot


# ── Combined multi-PDF visualization snapshots ───────────────────────


@dataclass(frozen=True)
class DocIngestionSummary:
    """Per-document ingestion summary for the combined view.

    Attributes:
        doc_id: UUID of the document.
        doc_title: Human-readable title.
        pdf_path: Path to the source PDF.
        element_count: Number of parsed elements.
        parent_count: Number of parent chunks.
        child_count: Number of child chunks.
        qdrant_point_count: Number of Qdrant points contributed.
        content_type_counts: Breakdown by content type (text/table/figure).
        total_chars: Total characters across all parent chunks.
    """

    doc_id: str
    doc_title: str
    pdf_path: str
    element_count: int
    parent_count: int
    child_count: int
    qdrant_point_count: int
    content_type_counts: Dict[str, int]
    total_chars: int


@dataclass(frozen=True)
class QueryRetrievalSummary:
    """Per-query retrieval summary for the combined view.

    Attributes:
        query: The search query text.
        total_results: Total number of retrieved parent chunks.
        per_doc_counts: Mapping of doc_title → result count.
        per_doc_best_score: Mapping of doc_title → best RRF score.
        top_doc_title: Title of the top-ranked document.
        top_doc_score: RRF score of the top-ranked result.
    """

    query: str
    total_results: int
    per_doc_counts: Dict[str, int]
    per_doc_best_score: Dict[str, float]
    top_doc_title: str
    top_doc_score: float


@dataclass(frozen=True)
class CombinedSnapshot:
    """Top-level snapshot for the combined multi-PDF visualization.

    Holds the full per-document ingestion snapshots and per-query
    retrieval snapshots, plus aggregated summaries for quick rendering.

    Attributes:
        timestamp: UTC timestamp string.
        qdrant_collection: Shared Qdrant collection name.
        sqlite_path: Path to the shared SQLite DB.
        ingestions: Full per-doc ingestion snapshots.
        retrievals: Full per-query retrieval snapshots.
        doc_summaries: Aggregated per-doc summaries.
        query_summaries: Aggregated per-query summaries.
        queries: List of query strings (mirrors retrievals order).
    """

    timestamp: str
    qdrant_collection: str
    sqlite_path: str
    ingestions: List[IngestionSnapshot]
    retrievals: List[RetrievalSnapshot]
    doc_summaries: List[DocIngestionSummary]
    query_summaries: List[QueryRetrievalSummary]
    queries: List[str]
