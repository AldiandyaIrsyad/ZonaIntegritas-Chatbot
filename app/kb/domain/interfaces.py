from typing import Protocol, List, Optional, Any, Dict
from dataclasses import dataclass
from app.kb.domain.models import PDFDocument, ParentChunk, IngestionTask
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

@dataclass
class ChunkVector:
    """A vectorized child chunk to be stored in the vector database."""
    chunk_id: str
    parent_chunk_id: str
    doc_id: str
    dense_vector: List[float]
    sparse_indices: List[int]
    sparse_values: List[float]
    session_id: Optional[str] = None

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
    
    async def create_ingestion_task(self, doc_id: str) -> IngestionTask: ...
    async def update_ingestion_task(self, task_id: str, status: str, error_message: Optional[str] = None) -> Optional[IngestionTask]: ...
    async def get_ingestion_task_by_doc_id(self, doc_id: str) -> Optional[IngestionTask]: ...
