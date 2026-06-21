"""Unstructured API document parser adapter.

Wraps HTTP calls to the self-hosted ``unstructured-api`` container for
layout-aware PDF parsing.  The container uses YOLOX for layout detection and
Tesseract for OCR, returning typed semantic elements that preserve document
structure.

The adapter is intentionally thin — all model lifecycle, batching, and GPU
scheduling are handled server-side.
"""

from __future__ import annotations

import hashlib
import os
import time
from collections import Counter

import httpx
import structlog

from app.core.interfaces.infra import IDocumentParser, ParsedElement
from app.observability import ResearchLogger

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
        self._research_logger = ResearchLogger()
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

        # Compute SHA256 hash for provenance
        sha256_hash = hashlib.sha256()
        with open(resolved, "rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        doc_hash = sha256_hash.hexdigest()

        # Track execution time
        start_time = time.perf_counter()

        # Define strategy and routing flags
        strategy = "hi_res"
        routing_flags: dict[str, str] = {}  # Empty for now, but logged as per requirement

        try:
            with open(resolved, "rb") as fh:
                response = await self._client.post(
                    "/general/v0/general",
                    files={"files": (filename, fh, "application/pdf")},
                    data={"strategy": strategy, **routing_flags},
                )
            response.raise_for_status()
            raw_output = response.json()
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

        execution_time = time.perf_counter() - start_time

        # Try to extract API version from headers, defaulting to "unknown"
        api_version = response.headers.get("server", response.headers.get("x-unstructured-version", "unknown"))

        # Save raw output
        raw_output_path = self._research_logger.save_raw_output(
            component="document_parser",
            document_name=filename,
            doc_hash=doc_hash,
            payload=raw_output,
        )

        elements: list[ParsedElement] = []
        element_types: Counter[str] = Counter()
        for elem in raw_output:
            text = elem.get("text", "").strip()
            if not text:
                continue

            elem_type = elem.get("type", "UncategorizedText")
            element_types[elem_type] += 1

            elements.append(
                ParsedElement(
                    element_type=elem_type,
                    text=text,
                    metadata=elem.get("metadata") or {},
                )
            )

        log.info(
            "parse.success",
            provenance={
                "document_path": file_path,
                "document_name": filename,
                "sha256": doc_hash,
                "execution_time_seconds": round(execution_time, 2),
            },
            unstructured_api_version=api_version,
            strategy=strategy,
            routing_flags=routing_flags,
            metrics={
                "total_elements_extracted": len(elements),
                "element_type_distribution": dict(element_types),
            },
            raw_output_path=raw_output_path,
        )

        return elements

    async def close(self) -> None:
        """Close the underlying HTTP client."""
        await self._client.aclose()
