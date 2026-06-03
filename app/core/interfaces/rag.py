"""
RAG-specific domain interfaces.
"""
from dataclasses import dataclass
from typing import TYPE_CHECKING, List, Optional, Protocol, Any

if TYPE_CHECKING:
    from app.rag.model import IngestionTask, ParentChunk


@dataclass
class RetrievedContext:
    """A single context block ready to be injected into the LLM prompt."""
    text: str
    doc_id: str
    score: float
    source_title: str
    parent_chunk_id: str  # Used by RAMService to identify NLI evidence source
    page: Optional[int] = None


class IRAGRepository(Protocol):
    """Database operations for the RAG pipeline."""

    async def save_parent_chunks(
        self, chunks: List["ParentChunk"]
    ) -> List["ParentChunk"]:
        """Batch insert parent chunks into PostgreSQL."""
        ...

    async def get_parent_chunks_by_ids(
        self, chunk_ids: List[str]
    ) -> List["ParentChunk"]:
        """Fetch parent chunks by their IDs."""
        ...

    async def delete_parent_chunks_by_doc_id(self, doc_id: str) -> int:
        """Delete all parent chunks for a document."""
        ...

    async def create_ingestion_task(self, doc_id: str) -> "IngestionTask":
        """Create a new ingestion task for tracking PDF processing."""
        ...

    async def update_ingestion_task(
        self,
        task_id: str,
        status: str,
        error_message: Optional[str] = None,
    ) -> Optional["IngestionTask"]:
        """Update the status of an ingestion task."""
        ...

    async def get_ingestion_task_by_doc_id(
        self, doc_id: str
    ) -> Optional["IngestionTask"]:
        """Get the most recent ingestion task for a document."""
        ...


class IIngestionService(Protocol):
    """Orchestrates the async document ingestion pipeline."""

    async def ingest_document(self, doc_id: str) -> None:
        """Run the full ingestion pipeline for a document."""
        ...


class IRetrievalService(Protocol):
    """Orchestrates the hybrid retrieval and reranking pipeline."""

    async def retrieve_context(
        self,
        query: str,
        top_k_search: int = 15,
        top_k_rerank: int = 3,
    ) -> List[RetrievedContext]:
        """Execute the full retrieval pipeline for a user query."""
        ...
