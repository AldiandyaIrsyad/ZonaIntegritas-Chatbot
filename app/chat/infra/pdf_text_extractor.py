"""
Fast, synchronous PDF text extraction for chat attachments.

Uses PyMuPDF (``fitz``) to pull native embedded text per page. Deliberately
does not reuse ``app.kb.infra.unstructured_client.UnstructuredClient`` (the
KB ingestion parser) — that pipeline runs OCR/VLM enrichment and can take
up to ~15 minutes, which is incompatible with completing inside a single
chat request. This trades off scanned/image-only PDF support for speed,
which is an acceptable fit for this app's domain (born-digital legal/
regulatory documents).

Fulfills: ``app/chat/domain/interfaces.py::IAttachmentExtractor``.
Wired in: ``app/chat/dependency.py::get_pdf_text_extractor``.
"""
import fitz  # PyMuPDF
import structlog

from app.chat.domain.interfaces import ExtractedPdfText

logger = structlog.get_logger(__name__)


class PdfExtractionError(Exception):
    """Base exception for PDF attachment extraction failures.

    These are ``PdfTextExtractor``-specific and are not declared on
    ``IAttachmentExtractor.extract`` directly, but the three concrete
    subclasses below correspond exactly to the ``Raises:`` documented on
    that port — see ``app/chat/domain/interfaces.py::IAttachmentExtractor.extract``.
    """


class PdfCorruptError(PdfExtractionError):
    """The file could not be opened/parsed as a PDF.

    Corresponds to ``IAttachmentExtractor.extract``'s documented
    ``PdfCorruptError`` case.
    """


class PdfTooManyPagesError(PdfExtractionError):
    """The PDF has more pages than the configured cap.

    Corresponds to ``IAttachmentExtractor.extract``'s documented
    ``PdfTooManyPagesError`` case.
    """

    def __init__(self, page_count: int, max_pages: int):
        self.page_count = page_count
        self.max_pages = max_pages
        super().__init__(f"PDF has {page_count} pages, exceeding the cap of {max_pages}.")


class PdfNoTextError(PdfExtractionError):
    """No extractable text was found (e.g. a scanned/image-only PDF).

    Corresponds to ``IAttachmentExtractor.extract``'s documented
    ``PdfNoTextError`` case.
    """


class PdfTextExtractor:
    """Extracts native embedded text from a PDF using PyMuPDF.

    Fulfills: ``app/chat/domain/interfaces.py::IAttachmentExtractor``.
    """

    def __init__(self, max_pages: int, max_chars: int):
        """Set the extraction limits enforced by ``extract``.

        Args:
            max_pages: Page count above which ``extract`` raises
                ``PdfTooManyPagesError`` instead of parsing the document.
            max_chars: Character cap applied to the extracted text; text
                beyond this is truncated (see ``ExtractedPdfText.truncated``).
        """
        self.max_pages = max_pages
        self.max_chars = max_chars

    def extract(self, file_bytes: bytes) -> ExtractedPdfText:
        """Extract native text from a PDF's bytes.

        Args:
            file_bytes: The raw PDF file content.

        Returns:
            ExtractedPdfText with the (possibly truncated) text, page count,
            character count, and a truncated flag.

        Raises:
            PdfCorruptError: If the file can't be opened as a PDF.
            PdfTooManyPagesError: If the page count exceeds ``max_pages``.
            PdfNoTextError: If no extractable text is found on any page.
        """
        try:
            doc = fitz.open(stream=file_bytes, filetype="pdf")
        except Exception as e:
            logger.warning("pdf_extractor.open_failed", error=str(e))
            raise PdfCorruptError("Could not open file as a PDF.") from e

        try:
            page_count = doc.page_count
            if page_count > self.max_pages:
                raise PdfTooManyPagesError(page_count, self.max_pages)

            page_texts = [page.get_text() for page in doc]
        finally:
            doc.close()

        full_text = "\n\n".join(t.strip() for t in page_texts if t.strip())

        if not full_text.strip():
            raise PdfNoTextError(
                "No extractable text found — the PDF may be scanned/image-only."
            )

        truncated = len(full_text) > self.max_chars
        text = full_text[: self.max_chars]

        return ExtractedPdfText(
            text=text,
            page_count=page_count,
            char_count=len(text),
            truncated=truncated,
        )
