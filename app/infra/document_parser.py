"""Unstructured API document parser adapter.

Wraps HTTP calls to the self-hosted ``unstructured-api`` container for
layout-aware PDF parsing.  The container uses YOLOX for layout detection and
Tesseract for OCR, returning typed semantic elements that preserve document
structure.

The adapter is intentionally thin — all model lifecycle, batching, and GPU
scheduling are handled server-side.
"""

from __future__ import annotations

import os

import httpx
import structlog

from app.core.interfaces.infra import IDocumentParser, ParsedElement

logger = structlog.get_logger(__name__)


class DocumentParser:
    """HTTP adapter for the ``unstructured-api`` container.

    Sends PDF files to ``/general/v0/general`` using the ``"hi_res"``
    strategy, which enables YOLOX layout detection and Tesseract OCR.
    Returns typed :class:`~app.core.interfaces.infra.ParsedElement` objects
    with empty elements filtered out.  Satisfies the
    :class:`~app.core.interfaces.infra.IDocumentParser` Protocol structurally.

    Args:
        base_url: Base URL of the unstructured-api container
                  (e.g. ``"http://unstructured:8000"``).
    """

    def __init__(self, base_url: str) -> None:
        self._client = httpx.AsyncClient(
            base_url=base_url.rstrip("/"),
            # PDF parsing with layout detection + OCR can take several minutes
            timeout=httpx.Timeout(900.0, connect=30.0),
        )
        logger.info("DocumentParser initialised", base_url=base_url)

    async def parse_pdf(self, file_path: str) -> list[ParsedElement]:
        """Parse a PDF file into ordered semantic text elements.

        Resolves symlinks before opening to prevent path traversal.  Empty
        elements (no text after stripping) are excluded from the result.

        Args:
            file_path: Absolute path to the PDF file on disk.

        Returns:
            Ordered list of :class:`~app.core.interfaces.infra.ParsedElement`
            objects with non-empty text content.

        Raises:
            FileNotFoundError: If no regular file exists at ``file_path``
                after symlink resolution.
            httpx.HTTPStatusError: If the unstructured-api returns a non-2xx
                response.
            Exception: Re-raised for any other network or I/O error.
        """
        resolved = os.path.realpath(file_path)
        if not os.path.isfile(resolved):
            raise FileNotFoundError(f"PDF not found: {resolved!r}")

        filename = os.path.basename(resolved)
        log = logger.bind(filename=filename)

        try:
            with open(resolved, "rb") as fh:
                response = await self._client.post(
                    "/general/v0/general",
                    files={"files": (filename, fh, "application/pdf")},
                    data={"strategy": "auto"}, # hi_res, auto, fast
                )
            response.raise_for_status()
        except FileNotFoundError:
            raise
        except httpx.HTTPStatusError as exc:
            log.error(
                "parse.http_error",
                status_code=exc.response.status_code,
                error=str(exc),
            )
            raise
        except Exception as exc:
            log.error("parse.request_failed", error=str(exc))
            raise

        elements: list[ParsedElement] = []
        for elem in response.json():
            text = elem.get("text", "").strip()
            if not text:
                continue
            elements.append(
                ParsedElement(
                    element_type=elem.get("type", "UncategorizedText"),
                    text=text,
                    metadata=elem.get("metadata") or {},
                )
            )

        log.debug("parse.complete", element_count=len(elements))
        return elements

    async def close(self) -> None:
        """Close the underlying HTTP client."""
        await self._client.aclose()
