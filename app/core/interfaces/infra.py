"""Infrastructure layer interface contracts.

Defines result dataclasses and structural :class:`typing.Protocol` types for
non-AI infrastructure adapters: document parsing, file storage, vector search,
LLM streaming, and thumbnail generation.

Consumers (services, tests, analysis) depend on these abstractions — never on
concrete infra implementations directly.

Example::

    from app.core.interfaces.infra import IVectorStore, ChunkVector, SearchResult
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, AsyncIterator, Optional, Protocol, runtime_checkable

from fastapi import UploadFile


# ---------------------------------------------------------------------------
# Result / value types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ParsedElement:
    """A single semantic element extracted from a parsed document.

    Attributes:
        element_type: Semantic category assigned by the parser (e.g.
                      ``"Title"``, ``"NarrativeText"``, ``"Table"``,
                      ``"ListItem"``).
        text: Extracted text content, already stripped of leading/trailing
              whitespace.
        metadata: Raw metadata dict from the parser (page number, coordinates,
                  etc.).  Structure varies by parser implementation.
    """

    element_type: str
    text: str
    metadata: dict[str, Any]


@dataclass
class ChunkVector:
    """A child document chunk paired with its dense and sparse embeddings.

    This is a mutable dataclass — ``session_id`` may be ``None`` for
    knowledge-base documents and set for ephemeral session uploads.

    Attributes:
        chunk_id: Unique identifier for this child chunk (UUID string).
        parent_chunk_id: Reference to the parent chunk stored in PostgreSQL.
        doc_id: Reference to the source document (UUID string).
        dense_vector: Dense float vector (must match the configured embedding
                      model dimension, e.g. 1024 for BAAI/bge-m3).
        sparse_indices: Token indices for the BM25 sparse representation.
        sparse_values: Corresponding weights for each sparse index.
        session_id: Optional session scope for ephemeral uploads.  ``None``
                    for permanent knowledge-base documents.
    """

    chunk_id: str
    parent_chunk_id: str
    doc_id: str
    dense_vector: list[float]
    sparse_indices: list[int]
    sparse_values: list[float]
    session_id: Optional[str] = None


@dataclass(frozen=True)
class SearchResult:
    """A single hit returned from a vector similarity or hybrid search.

    Attributes:
        chunk_id: Identifier of the matched child chunk.
        parent_chunk_id: Reference to the parent chunk in PostgreSQL (carries
                         the full text for display/context).
        doc_id: Reference to the source document.
        score: Relevance score from the vector index (higher is more relevant).
    """

    chunk_id: str
    parent_chunk_id: str
    doc_id: str
    score: float


# ---------------------------------------------------------------------------
# Protocols
# ---------------------------------------------------------------------------


@runtime_checkable
class IDocumentParser(Protocol):
    """Structural contract for document parsing adapters.

    Any object that provides :meth:`parse_pdf` and :meth:`close` with the
    correct signatures satisfies this Protocol — no subclassing required.
    """

    async def parse_pdf(self, file_path: str) -> list[ParsedElement]:
        """Parse a PDF file into ordered semantic text elements.

        Args:
            file_path: Absolute path to the PDF file on disk.

        Returns:
            Ordered list of :class:`ParsedElement` objects.  Empty elements
            (no text) are excluded from the result.

        Raises:
            FileNotFoundError: If no file exists at ``file_path``.
        """
        ...

    async def close(self) -> None:
        """Release any underlying connections or resources."""
        ...


@runtime_checkable
class IStorageProvider(Protocol):
    """Structural contract for file storage adapters.

    Any object that provides :meth:`save_file` and :meth:`delete_file` with
    the correct signatures satisfies this Protocol — no subclassing required.
    """

    async def save_file(self, file: UploadFile, file_extension: str) -> str:
        """Persist an uploaded file and return its storage path or URI.

        Args:
            file: The FastAPI :class:`~fastapi.UploadFile` object to persist.
            file_extension: File extension including the leading dot
                            (e.g. ``".pdf"``).

        Returns:
            The absolute path or URI where the file was stored.
        """
        ...

    async def delete_file(self, file_path: str) -> bool:
        """Remove a previously stored file.

        Args:
            file_path: Path or URI of the file to remove.

        Returns:
            ``True`` if the file was deleted; ``False`` if it was not found or
            the path was empty.
        """
        ...


@runtime_checkable
class IVectorStore(Protocol):
    """Structural contract for vector database adapters.

    Any object that provides the full set of methods with the correct
    signatures satisfies this Protocol — no subclassing required.
    """

    async def ensure_collection(self) -> None:
        """Create the vector collection if it does not already exist."""
        ...

    async def upsert_chunks(self, chunks: list[ChunkVector]) -> None:
        """Batch-insert or update chunk vectors in the store.

        Args:
            chunks: Child chunks to persist, including their embeddings and
                    metadata payloads.
        """
        ...

    async def hybrid_search(
        self,
        dense_vector: list[float],
        sparse_indices: list[int],
        sparse_values: list[float],
        top_k: int = 15,
        session_id: Optional[str] = None,
    ) -> list[SearchResult]:
        """Execute a hybrid dense + sparse search.

        Args:
            dense_vector: Dense embedding of the query.
            sparse_indices: BM25 sparse token indices for the query.
            sparse_values: Corresponding BM25 weights.
            top_k: Maximum number of results to return.
            session_id: If provided, restricts the search to this session
                        scope.

        Returns:
            Up to ``top_k`` :class:`SearchResult` items by relevance.
        """
        ...

    async def update_payload(
        self, doc_id: str, payload: dict[str, Any]
    ) -> None:
        """Update metadata payload fields for all vectors of a document.

        Args:
            doc_id: UUID of the target document.
            payload: Key-value pairs to set on all matching vector points.
        """
        ...

    async def delete_by_doc_id(self, doc_id: str) -> None:
        """Delete all vectors associated with a document.

        Args:
            doc_id: UUID of the document whose vectors should be removed.
        """
        ...

    async def close(self) -> None:
        """Release any underlying connections or resources."""
        ...


@runtime_checkable
class ILLMConnection(Protocol):
    """Structural contract for LLM streaming connection adapters.

    Any object that provides :meth:`stream_chat` and :meth:`close` with the
    correct signatures satisfies this Protocol — no subclassing required.
    """

    def stream_chat(
        self,
        model: str,
        messages: list[dict[str, str]],
        max_tokens: int,
    ) -> AsyncIterator[str]:
        """Stream a chat completion from the LLM backend.

        Args:
            model: Model identifier (e.g. ``"openai/gpt-4o"`` or
                   ``"llama3.1:8b"``).
            messages: Conversation history in OpenAI message format.
            max_tokens: Maximum number of tokens to generate.

        Yields:
            Non-empty text delta chunks as produced by the model.
        """
        ...

    async def close(self) -> None:
        """Release any underlying connections or resources."""
        ...


@runtime_checkable
class IThumbnailStrategy(Protocol):
    """Structural contract for per-format thumbnail generation strategies.

    Any object that provides :meth:`generate` with the correct signature
    satisfies this Protocol — no subclassing required.
    """

    def generate(self, file_path: str) -> str | None:
        """Generate a base64 PNG data-URI thumbnail for the given file.

        Args:
            file_path: Absolute path to the source file.

        Returns:
            A ``data:image/png;base64,...`` string on success, or ``None``
            if the file type is unsupported or generation fails.
        """
        ...
