from typing import Protocol, List, Optional, Any, Dict
from dataclasses import dataclass
from app.kb.domain.models import PDFDocument, ParentChunk, IngestionTask, ChildChunk
from app.thesis.chunking.models import ParsedElement

class IDocumentParser(Protocol):
    """Protocol for parsing unstructured documents into semantic elements."""
    async def parse_pdf(self, file_path: str) -> List[ParsedElement]:
        ...
    async def close(self) -> None:
        ...

@dataclass
class EmbeddingResult:
    """Result of an embedding operation."""
    dense: List[float]
    sparse_indices: List[int]
    sparse_values: List[float]

class ITextEmbedder(Protocol):
    """Protocol for generating text embeddings (dense and sparse)."""
    async def embed_texts(self, texts: List[str]) -> List[EmbeddingResult]:
        ...
    async def close(self) -> None:
        ...


class IQueryExpander(Protocol):
    """Protocol for query expansion (e.g. HyDE).

    Implementations generate a hypothetical document from the user's raw
    query. The expanded text is then embedded instead of the raw query to
    improve retrieval recall (HyDE — Hypothetical Document Embeddings).
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
    """Protocol for reranking retrieved documents by relevance to a query."""
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
        ...

@dataclass
class ChunkVector:
    """A vectorized child chunk to be stored in the vector database."""
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
    """A raw match returned from the vector store."""
    chunk_id: str
    parent_chunk_id: str
    doc_id: str
    score: float

class IVectorStore(Protocol):
    """Protocol for vector database operations."""
    async def ensure_collection(self) -> None:
        ...
    async def upsert_chunks(self, chunks: List[ChunkVector]) -> None:
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
        ...
    async def update_payload(self, doc_id: str, payload: Dict[str, Any]) -> None:
        ...
    async def delete_by_doc_id(self, doc_id: str) -> None:
        ...
    async def close(self) -> None:
        ...

class IKBRepository(Protocol):
    """Protocol for KB relational database operations."""
    async def get_all_pdfs(self) -> List[PDFDocument]: ...
    async def get_pdf_by_id(self, pdf_id: str) -> Optional[PDFDocument]: ...
    async def get_pdfs_by_ids(self, pdf_ids: List[str]) -> List[PDFDocument]: ...
    async def create_pdf(self, title: str, description: str, pdf_path: str) -> PDFDocument: ...
    async def update_pdf_active_status(self, pdf_id: str, active: bool) -> Optional[PDFDocument]: ...
    async def delete_pdf(self, pdf_id: str) -> bool: ...

    async def save_parent_chunks(self, chunks: List[ParentChunk]) -> List[ParentChunk]: ...
    async def get_parent_chunks_by_ids(self, chunk_ids: List[str]) -> List[ParentChunk]: ...
    async def delete_parent_chunks_by_doc_id(self, doc_id: str) -> int: ...

    async def save_child_chunks(self, chunks: List[ChildChunk]) -> List[ChildChunk]: ...
    async def get_child_chunks_by_ids(self, chunk_ids: List[str]) -> List[ChildChunk]: ...
    async def get_child_chunks_by_parent_ids(self, parent_ids: List[str]) -> List[ChildChunk]: ...
    async def get_sibling_chunks(self, parent_id: str) -> List[ParentChunk]: ...
    async def get_chunks_by_path_prefix(self, path_prefix: str) -> List[ParentChunk]: ...

    async def create_ingestion_task(self, doc_id: str) -> IngestionTask: ...
    async def update_ingestion_task(self, task_id: str, status: str, error_message: Optional[str] = None) -> Optional[IngestionTask]: ...
    async def get_ingestion_task_by_doc_id(self, doc_id: str) -> Optional[IngestionTask]: ...
