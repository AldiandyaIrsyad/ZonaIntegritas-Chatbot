"""Domain ports (Protocol interfaces) for the Knowledge Base bounded context.

This module defines the abstract contracts the KB application layer depends
on. Per the Dependency Inversion Principle, the application and domain
layers import only these Protocols; the concrete adapters live in
``app/kb/infra/`` and are injected at the composition root
(``app/kb/dependency.py``).

Ports → adapters map:
    - :class:`IDocumentParser` → ``app/kb/infra/unstructured_client.py::UnstructuredClient``
    - :class:`ITextEmbedder`   → ``app/kb/infra/bge_m3_embeddings.py::BGEM3Embeddings``
                                (or ``infinity_embeddings.py::InfinityEmbeddings``)
    - :class:`IQueryExpander`  → ``app/chat/infra/hyde_expander.py::HyDEExpander``
                                (note: implemented in the *chat* context and
                                injected across the boundary by
                                ``app/chat/dependency.py``)
    - :class:`IReranker`       → ``app/kb/infra/infinity_reranker.py::InfinityReranker``
    - :class:`IVectorStore`    → ``app/kb/infra/qdrant_store.py::QdrantStore``
    - :class:`IKBRepository`   → ``app/kb/infra/postgres_repo.py::PostgresKBRepository``
"""

from typing import Protocol, List, Optional, Any, Dict
from dataclasses import dataclass
from app.kb.domain.models import PDFDocument, ParentChunk, IngestionTask, ChildChunk
from app.thesis.chunking.models import ParsedElement


class IDocumentParser(Protocol):
    """Port for parsing unstructured PDFs into semantic elements.

    Implemented by: ``app/kb/infra/unstructured_client.py::UnstructuredClient``
    (wired in ``app/kb/dependency.py::get_document_parser``).
    """

    async def parse_pdf(self, file_path: str) -> List[ParsedElement]:
        """Parse a PDF into a list of typed semantic elements.

        Args:
            file_path: Absolute path to the PDF on disk.

        Returns:
            Ordered list of :class:`ParsedElement` (text, tables, figures).
        """
        ...

    async def close(self) -> None:
        """Release the underlying HTTP client / connection pool."""
        ...


@dataclass
class EmbeddingResult:
    """Result of an embedding operation.

    Attributes:
        dense: Dense float vector (BGE-M3, 1024-dim).
        sparse_indices: BM25-style sparse term indices.
        sparse_values: Sparse term weights aligned with ``sparse_indices``.
    """

    dense: List[float]
    sparse_indices: List[int]
    sparse_values: List[float]


class ITextEmbedder(Protocol):
    """Port for generating text embeddings (dense + sparse).

    Implemented by: ``app/kb/infra/bge_m3_embeddings.py::BGEM3Embeddings``
    (in-process model; wired in ``app/kb/dependency.py::get_text_embedder``).
    An alternative HTTP-backed implementation lives at
    ``app/kb/infra/infinity_embeddings.py::InfinityEmbeddings``.
    """

    async def embed_texts(self, texts: List[str]) -> List[EmbeddingResult]:
        """Embed a batch of texts into dense + sparse vectors.

        Args:
            texts: Texts to embed.

        Returns:
            One :class:`EmbeddingResult` per input text, in order.
        """
        ...

    async def close(self) -> None:
        """Release the model / HTTP client."""
        ...


class IQueryExpander(Protocol):
    """Port for query expansion (e.g. HyDE).

    Implementations generate a hypothetical document from the user's raw
    query. The expanded text is then embedded instead of the raw query to
    improve retrieval recall (HyDE — Hypothetical Document Embeddings).

    Implemented by: ``app/chat/infra/hyde_expander.py::HyDEExpander``
    (wired in ``app/chat/dependency.py::get_query_expander`` and injected
    across the chat→kb boundary into ``SearchService``).
    """

    async def expand(self, query: str) -> str:
        """Generate an expanded/hypothetical document for the query.

        Args:
            query: The user's raw search query.

        Returns:
            A hypothetical answer document text to be embedded for retrieval.
        """
        ...

    async def close(self) -> None:
        """Release any underlying LLM connection."""
        ...


@dataclass
class RerankResult:
    """Result of a reranking operation for a single document.

    Attributes:
        index: Original position of the document in the input list.
        score: Relevance score assigned by the reranker (higher = more relevant).
    """

    index: int
    score: float


class IReranker(Protocol):
    """Port for reranking retrieved documents by relevance to a query.

    Implemented by: ``app/kb/infra/infinity_reranker.py::InfinityReranker``
    (wired in ``app/kb/dependency.py::get_reranker``).
    """

    async def rerank(self, query: str, documents: List[str], top_k: Optional[int] = None) -> List[RerankResult]:
        """Rerank documents by relevance to the query.

        Args:
            query: The search query.
            documents: List of document texts in their original retrieval order.
            top_k: If provided, return only the top-k most relevant documents.

        Returns:
            List of RerankResult ordered by descending relevance score.
        """
        ...

    async def close(self) -> None:
        """Release the underlying HTTP client."""
        ...


@dataclass
class ChunkVector:
    """A vectorized child chunk to be stored in the vector database.

    Attributes:
        chunk_id: UUID of the child chunk.
        parent_chunk_id: UUID of the parent chunk this child belongs to.
        doc_id: UUID of the source PDF document.
        dense_vector: Dense embedding (BGE-M3, 1024-dim).
        sparse_indices: BM25 sparse term indices.
        sparse_values: BM25 sparse term weights.
        breadcrumbs: Hierarchical section path (e.g. ["BAB I", "Pasal 5"]).
        content_type: Structural type ("text", "table", "figure").
        session_id: Optional chat session scope (for per-session collections).
        text: The child chunk text (stored as payload for retrieval display).
    """

    chunk_id: str
    parent_chunk_id: str
    doc_id: str
    dense_vector: List[float]
    sparse_indices: List[int]
    sparse_values: List[float]
    breadcrumbs: List[str]
    content_type: str = "text"
    session_id: Optional[str] = None
    text: str = ""


@dataclass
class SearchResult:
    """A raw match returned from the vector store.

    Attributes:
        chunk_id: UUID of the matched child chunk.
        parent_chunk_id: UUID of the parent chunk to hydrate (Small-to-Big).
        doc_id: UUID of the source PDF document.
        score: Hybrid (dense + sparse) similarity score.
    """

    chunk_id: str
    parent_chunk_id: str
    doc_id: str
    score: float


class IVectorStore(Protocol):
    """Port for vector database operations (hybrid dense + sparse search).

    Implemented by: ``app/kb/infra/qdrant_store.py::QdrantStore``
    (wired in ``app/kb/dependency.py::get_vector_store``).
    """

    async def ensure_collection(self) -> None:
        """Create the Qdrant collection if it doesn't exist (idempotent)."""
        ...

    async def upsert_chunks(self, chunks: List[ChunkVector]) -> None:
        """Upsert a batch of vectorized child chunks into the collection."""
        ...

    async def hybrid_search(
        self,
        dense_vector: List[float],
        sparse_indices: List[int],
        sparse_values: List[float],
        top_k: int = 15,
        session_id: Optional[str] = None,
        mode: str = "hybrid",
    ) -> List[SearchResult]:
        """Run a hybrid (dense + sparse) search against the collection.

        Args:
            dense_vector: Query dense embedding.
            sparse_indices: Query sparse term indices.
            sparse_values: Query sparse term weights.
            top_k: Maximum number of results to return.
            session_id: If set, restrict to a per-session collection.
            mode: Fusion mode — "hybrid" (RRF), "dense", or "sparse".

        Returns:
            List of :class:`SearchResult` ranked by hybrid score.
        """
        ...

    async def update_payload(self, doc_id: str, payload: Dict[str, Any]) -> None:
        """Update payload fields for all points of a given doc_id."""
        ...

    async def delete_by_doc_id(self, doc_id: str) -> None:
        """Delete all points belonging to a document (cascade on doc removal)."""
        ...

    async def close(self) -> None:
        """Release the underlying Qdrant client."""
        ...


class IKBRepository(Protocol):
    """Port for KB relational database operations (Postgres).

    Implemented by: ``app/kb/infra/postgres_repo.py::PostgresKBRepository``
    (wired in ``app/kb/dependency.py::get_kb_repo``).
    """

    async def get_all_pdfs(self) -> List[PDFDocument]:
        """Return all KB documents."""
        ...

    async def get_pdf_by_id(self, pdf_id: str) -> Optional[PDFDocument]:
        """Fetch a single document by ID."""
        ...

    async def get_pdfs_by_ids(self, pdf_ids: List[str]) -> List[PDFDocument]:
        """Fetch multiple documents by ID (preserving order is not guaranteed)."""
        ...

    async def search_titles_naive(self, query: str) -> List[PDFDocument]:
        """Naive ILIKE title search (used to ground HyDE with active doc titles)."""
        ...

    async def create_pdf(self, title: str, description: str, pdf_path: str) -> PDFDocument:
        """Create a new KB document record."""
        ...

    async def update_pdf_active_status(self, pdf_id: str, active: bool) -> Optional[PDFDocument]:
        """Toggle a document's active flag (inactive docs are excluded from retrieval)."""
        ...

    async def delete_pdf(self, pdf_id: str) -> bool:
        """Delete a document and its chunks. Returns True if deleted."""
        ...

    async def save_parent_chunks(self, chunks: List[ParentChunk]) -> List[ParentChunk]:
        """Persist parent chunks (full-text sections for LLM context)."""
        ...

    async def get_parent_chunks_by_ids(self, chunk_ids: List[str]) -> List[ParentChunk]:
        """Fetch parent chunks by ID (Small-to-Big hydration step)."""
        ...

    async def save_child_chunks(self, chunks: List[ChildChunk]) -> List[ChildChunk]:
        """Persist child chunks (sentence-level, indexed in Qdrant)."""
        ...

    async def get_child_chunks_by_ids(self, chunk_ids: List[str]) -> List[ChildChunk]:
        """Fetch child chunks by ID."""
        ...

    async def get_sibling_chunks(self, parent_id: str) -> List[ParentChunk]:
        """Fetch sibling parent chunks sharing the same parent (cross-ref hydration)."""
        ...

    async def get_chunks_by_path_prefix(self, path_prefix: str) -> List[ParentChunk]:
        """Fetch chunks whose ltree path starts with the given prefix."""
        ...

    async def create_ingestion_task(self, doc_id: str) -> IngestionTask:
        """Create an ingestion task record for tracking pipeline progress."""
        ...

    async def update_ingestion_task(self, task_id: str, status: str, error_message: Optional[str] = None) -> Optional[IngestionTask]:
        """Update an ingestion task's status (and optional error message)."""
        ...

    async def get_ingestion_task_by_doc_id(self, doc_id: str) -> Optional[IngestionTask]:
        """Fetch the ingestion task for a document, if any."""
        ...

    async def commit(self) -> None:
        """Commit the current transaction."""
        ...

    async def rollback(self) -> None:
        """Roll back the current transaction."""
        ...
