"""Thumbnail generation strategies.

Implements the Strategy pattern for producing base64-encoded PNG thumbnail
data URIs from various file types.  The public entry point is
:class:`ThumbnailContext`, which selects the correct strategy from the file
extension and delegates generation.

Supported formats:
- ``".pdf"`` → :class:`PDFThumbnailStrategy` (PyMuPDF first-page render)
- ``".jpg"`` / ``".jpeg"`` / ``".png"`` → :class:`ImageThumbnailStrategy` (Pillow)
- Everything else → :class:`DefaultThumbnailStrategy` (returns ``None``)

All strategies treat generation as best-effort: failures are logged as warnings
and ``None`` is returned rather than raising, so the caller can proceed without
a thumbnail.
"""

from __future__ import annotations

import abc
import base64
import os
from io import BytesIO

# PIL is imported at module level — lightweight and always available.
# fitz (PyMuPDF) is imported lazily inside PDFThumbnailStrategy.generate()
# so the module remains importable even when PyMuPDF is not installed.
import structlog
from PIL import Image

from app.core.interfaces.infra import IThumbnailStrategy

logger = structlog.get_logger(__name__)


class ThumbnailStrategy(abc.ABC):
    """Abstract base for file-type-specific thumbnail generators.

    Subclasses implement :meth:`generate` for a specific file format.
    All implementations satisfy the
    :class:`~app.core.interfaces.infra.IThumbnailStrategy` Protocol
    structurally.
    """

    @abc.abstractmethod
    def generate(self, file_path: str) -> str | None:
        """Generate a base64 PNG data-URI thumbnail for the given file.

        Args:
            file_path: Absolute path to the source file.

        Returns:
            A ``data:image/png;base64,...`` string on success, or ``None``
            if generation fails or is unsupported.
        """


class PDFThumbnailStrategy(ThumbnailStrategy):
    """Renders the first page of a PDF as a PNG thumbnail.

    Uses PyMuPDF (``fitz``) at 1.5× scale (~108 DPI) — compact yet readable.
    Returns ``None`` (with a warning log) if the PDF is empty or rendering
    fails for any reason.
    """

    def generate(self, file_path: str) -> str | None:
        """Render the first PDF page as a base64 PNG data URI.

        Args:
            file_path: Absolute path to the PDF file.

        Returns:
            A ``data:image/png;base64,...`` string, or ``None`` on failure.
        """
        try:
            import fitz as _fitz  # lazy: PyMuPDF may not be installed in all envs
            doc = _fitz.open(file_path)
            if not doc.page_count:
                logger.warning(
                    "thumbnail.pdf.empty_document",
                    file_path=file_path,
                )
                return None
            page = doc.load_page(0)
            pix = page.get_pixmap(matrix=_fitz.Matrix(1.5, 1.5))
            b64 = base64.b64encode(pix.tobytes("png")).decode("utf-8")
            return f"data:image/png;base64,{b64}"
        except Exception as exc:
            logger.warning(
                "thumbnail.pdf.failed",
                file_path=file_path,
                error=str(exc),
            )
            return None


class ImageThumbnailStrategy(ThumbnailStrategy):
    """Scales a raster image to fit within 200×200 and returns it as a PNG thumbnail.

    Uses Pillow's :meth:`~PIL.Image.Image.thumbnail` which preserves the
    aspect ratio.  Returns ``None`` (with a warning log) on any Pillow error.
    """

    def generate(self, file_path: str) -> str | None:
        """Scale an image to 200×200 and return it as a base64 PNG data URI.

        Args:
            file_path: Absolute path to the image file.

        Returns:
            A ``data:image/png;base64,...`` string, or ``None`` on failure.
        """
        try:
            with Image.open(file_path) as img:
                img.thumbnail((200, 200))
                buf = BytesIO()
                img.save(buf, format="PNG")
                b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
                return f"data:image/png;base64,{b64}"
        except Exception as exc:
            logger.warning(
                "thumbnail.image.failed",
                file_path=file_path,
                error=str(exc),
            )
            return None


class DefaultThumbnailStrategy(ThumbnailStrategy):
    """No-op strategy for unsupported file types.

    Always returns ``None`` without raising or logging — the absence of a
    thumbnail is expected and not an error for unknown formats.
    """

    def generate(self, file_path: str) -> str | None:
        """Return ``None`` — no thumbnail available for this file type.

        Args:
            file_path: Unused; accepted for interface compatibility.

        Returns:
            Always ``None``.
        """
        return None


class ThumbnailContext:
    """Selects and delegates to the appropriate thumbnail strategy by file extension.

    Supports ``.pdf``, ``.jpg``, ``.jpeg``, and ``.png``.  All other
    extensions delegate to :class:`DefaultThumbnailStrategy`, which returns
    ``None``.

    Example::

        ctx = ThumbnailContext()
        data_uri = ctx.generate("/uploads/report.pdf")
        # → "data:image/png;base64,..." or None
    """

    def __init__(self) -> None:
        self._strategies: dict[str, ThumbnailStrategy] = {
            ".pdf": PDFThumbnailStrategy(),
            ".jpg": ImageThumbnailStrategy(),
            ".jpeg": ImageThumbnailStrategy(),
            ".png": ImageThumbnailStrategy(),
        }
        self._default = DefaultThumbnailStrategy()
        logger.info(
            "ThumbnailContext initialised",
            supported_extensions=list(self._strategies.keys()),
        )

    def generate(self, file_path: str) -> str | None:
        """Generate a thumbnail by selecting the strategy for the file's extension.

        The extension lookup is case-insensitive (``".PDF"`` is treated the
        same as ``".pdf"``).

        Args:
            file_path: Absolute path to the source file.

        Returns:
            A ``data:image/png;base64,...`` data URI, or ``None`` if the file
            type is unsupported or thumbnail generation fails.
        """
        _, ext = os.path.splitext(file_path)
        strategy = self._strategies.get(ext.lower(), self._default)
        return strategy.generate(file_path)
