"""Domain ports (Protocol interfaces) for the Chat bounded context.

This module defines the abstract contracts the chat application layer depends
on. Per the Dependency Inversion Principle, the application and domain layers
import only these Protocols; the concrete adapters live in ``app/chat/infra/``
and are injected at the composition root (``app/chat/dependency.py``).

Ports → adapters map:
    - :class:`ILLMConnection`        → ``app/chat/infra/llm_connection.py::LLMConnection``
    - :class:`IChatRepository`       → ``app/chat/infra/postgres_chat_repo.py::PostgresChatRepository``
    - :class:`IAttachmentExtractor` → ``app/chat/infra/pdf_text_extractor.py::PdfTextExtractor``
"""

from dataclasses import dataclass
from typing import Protocol, AsyncIterator, List, Optional, Dict, Any
from app.chat.domain.models import Session, Message


class ILLMConnection(Protocol):
    """Port for interacting with a Language Model backend.

    Implemented by: ``app/chat/infra/llm_connection.py::LLMConnection``
    (an async OpenAI-compatible client; wired in
    ``app/chat/dependency.py::get_llm_connection``).

    Also re-implemented for the thesis eval harness by
    ``app/thesis/_eval/_shared/clients.py::EvalLLMClient`` so experiments
    can hit a separate model endpoint without touching production wiring.
    """

    def stream_chat(
        self,
        model: str,
        messages: List[Dict[str, str]],
        max_tokens: int,
        temperature: float = 0.0,
    ) -> AsyncIterator[str]:
        """Stream a chat completion token-by-token.

        Args:
            model: Model identifier.
            messages: Chat messages.
            max_tokens: Maximum tokens for the response.
            temperature: Sampling temperature.

        Yields:
            Incremental response text fragments.
        """
        ...

    async def generate(
        self,
        model: str,
        messages: List[Dict[str, str]],
        max_tokens: int,
        temperature: float = 0.0,
    ) -> str:
        """Generate a complete (non-streaming) chat completion.

        Used for tasks that need the full response as a string before
        proceeding (e.g. HyDE hypothetical document generation).

        Args:
            model: Model identifier.
            messages: Chat messages.
            max_tokens: Maximum tokens for the response.
            temperature: Sampling temperature.

        Returns:
            The full response text.
        """
        ...

    async def close(self) -> None:
        """Release the underlying HTTP client / connection pool."""
        ...


class IChatRepository(Protocol):
    """Port for chat session/message persistence.

    Implemented by: ``app/chat/infra/postgres_chat_repo.py::PostgresChatRepository``
    (wired in ``app/chat/dependency.py::get_chat_repo``).
    """

    async def get_all_sessions(self) -> List[Session]:
        """Return all chat sessions, newest first."""
        ...

    async def create_session(self, session_id: str, title: str) -> Session:
        """Create and persist a new chat session."""
        ...

    async def get_session_by_id(self, session_id: str, load_messages: bool = False) -> Optional[Session]:
        """Fetch a session by ID, optionally eager-loading its messages."""
        ...

    async def update_session_title(self, session: Session, new_title: str) -> Session:
        """Rename an existing session."""
        ...

    async def create_message(
        self,
        session_id: str,
        role: str,
        content: str,
        raw_content: Optional[str] = None,
        context: Optional[str] = None,
        sources: Optional[List[Dict[str, Any]]] = None,
        attachment_filename: Optional[str] = None,
    ) -> Message:
        """Persist a single message within a session.

        Args:
            session_id: Parent session ID.
            role: ``"user"``, ``"assistant"``, or ``"system"``.
            content: Final rendered message text (with citations).
            raw_content: Original LLM output before citation formatting.
            context: Concatenated RAG context text used to ground the answer.
            sources: JSON-serialisable list of source citations.
            attachment_filename: Filename of the uploaded PDF, if any.
        """
        ...

    async def delete_session(self, session_id: str) -> bool:
        """Delete a session and all its messages. Returns True if deleted."""
        ...


@dataclass(frozen=True)
class ExtractedPdfText:
    """Result of extracting text from an uploaded PDF.

    Attributes:
        text: The extracted plain text.
        page_count: Number of pages in the source PDF.
        char_count: Length of ``text`` in characters.
        truncated: True if ``text`` was cut short to respect a char cap.
    """

    text: str
    page_count: int
    char_count: int
    truncated: bool


class IAttachmentExtractor(Protocol):
    """Port for extracting text from a chat-uploaded document.

    Unlike ``app.kb.domain.interfaces.IDocumentParser`` (used for permanent
    KB ingestion via Unstructured + VLM, which can take minutes), this is
    expected to complete synchronously within a single chat request — so
    implementations should favor fast native text extraction over OCR/VLM
    enrichment.

    Implemented by: ``app/chat/infra/pdf_text_extractor.py::PdfTextExtractor``
    (wired in ``app/chat/dependency.py::get_pdf_text_extractor``).
    """

    def extract(self, file_bytes: bytes) -> ExtractedPdfText:
        """Extract text from the given PDF bytes.

        Args:
            file_bytes: Raw PDF file content.

        Returns:
            The extracted text and metadata.

        Raises:
            PdfCorruptError: The PDF is unreadable.
            PdfNoTextError: The PDF has no extractable text (e.g. scanned).
            PdfTooManyPagesError: The PDF exceeds the page limit.
        """
        ...
