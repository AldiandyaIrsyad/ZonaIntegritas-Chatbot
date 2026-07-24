"""
Application service for chat PDF attachments.

Extracts text from an uploaded PDF and runs it through the same IVM safety
check (``IVMService.check_malicious``) that the typed chat message goes
through, since attachment text becomes part of the LLM prompt just like the
message does. This is a fail-fast UX check — the authoritative check happens
again inside ``ChatService.process_chat_message`` against the combined
message+attachment text before generation, so a client cannot bypass safety
scanning by tampering with the extracted text between the two requests.
"""
from dataclasses import dataclass

import structlog

from app.chat.domain.interfaces import IAttachmentExtractor
from app.thesis.ivm.service import IVMService

logger = structlog.get_logger(__name__)


@dataclass(frozen=True)
class AttachmentResult:
    """Outcome of processing one chat attachment upload.

    Attributes:
        filename: Original uploaded filename (display only).
        text: Extracted (possibly truncated) PDF text, already safety-checked.
        page_count: Number of pages in the source PDF.
        char_count: Length of ``text`` in characters.
        truncated: True if ``text`` was cut short to respect a char cap.
    """

    filename: str
    text: str
    page_count: int
    char_count: int
    truncated: bool


class AttachmentService:
    """Application service: validates, extracts, and safety-checks a
    chat-uploaded PDF before it's handed back to the client for inclusion
    in a subsequent ``/stream`` request.
    """

    def __init__(self, extractor: IAttachmentExtractor, ivm_service: IVMService):
        """Wire the extraction and safety-check collaborators.

        Args:
            extractor: Port for extracting text from the uploaded PDF
                (``IAttachmentExtractor``, e.g. ``PdfTextExtractor``).
            ivm_service: Safety checker reused from the main chat pipeline
                so the attachment text goes through the same
                prompt-injection scan as typed messages.
        """
        self.extractor = extractor
        self.ivm_service = ivm_service

    async def process_upload(self, filename: str, file_bytes: bytes) -> AttachmentResult:
        """Extract text from the PDF and scan it for malicious content.

        Args:
            filename: The original uploaded filename (for display only).
            file_bytes: The raw PDF file content.

        Returns:
            An AttachmentResult with the extracted (possibly truncated) text.

        Raises:
            PdfExtractionError: If the file can't be parsed, has too many
                pages, or has no extractable text (see pdf_text_extractor.py).
            MaliciousPromptException: If the extracted text fails the IVM
                safety check.
        """
        extracted = self.extractor.extract(file_bytes)

        await self.ivm_service.check_malicious(extracted.text)

        return AttachmentResult(
            filename=filename,
            text=extracted.text,
            page_count=extracted.page_count,
            char_count=extracted.char_count,
            truncated=extracted.truncated,
        )
