"""
Unstructured API document parser client.

Wraps HTTP calls to the self-hosted unstructured-api container for
layout-aware PDF parsing using YOLOX and Tesseract.
"""
import logging
import os
from dataclasses import dataclass
from typing import List

import httpx

logger = logging.getLogger(__name__)


@dataclass
class ParsedElement:
    """A single element extracted from a PDF by the unstructured parser.

    Attributes:
        element_type: The type of element (e.g., "Title", "NarrativeText",
                      "Table", "ListItem", "Header", "Footer").
        text: The extracted text content.
        metadata: Additional metadata from unstructured (page number, etc).
    """
    element_type: str
    text: str
    metadata: dict


class DocumentParser:
    """
    HTTP client for the unstructured-api container.

    Sends PDFs to the `/general/v0/general` endpoint for layout-aware
    parsing that preserves document structure (titles, sections, tables).
    """

    def __init__(self, base_url: str):
        self.base_url = base_url.rstrip("/")
        self._client = httpx.AsyncClient(
            base_url=self.base_url,
            # PDF parsing can be slow, especially with layout detection
            timeout=httpx.Timeout(300.0, connect=30.0),
        )

    async def parse_pdf(self, file_path: str) -> List[ParsedElement]:
        """Parse a PDF file into structured elements via unstructured-api.

        Sends the file to the unstructured container which uses YOLOX for
        layout detection and Tesseract for OCR when needed. Returns typed
        elements preserving the document's semantic structure.

        Args:
            file_path: Absolute path to the PDF file on disk.

        Returns:
            List of ParsedElement with types and text content.

        Raises:
            FileNotFoundError: If the PDF file doesn't exist.
            httpx.HTTPStatusError: If the unstructured-api returns an error.
        """
        # Validate file exists before sending
        resolved_path = os.path.realpath(file_path)
        if not os.path.isfile(resolved_path):
            raise FileNotFoundError(f"PDF file not found: {resolved_path}")

        filename = os.path.basename(resolved_path)

        with open(resolved_path, "rb") as f:
            response = await self._client.post(
                "/general/v0/general",
                files={"files": (filename, f, "application/pdf")},
                data={
                    "strategy": "hi_res",
                    "hi_res_model_name": "yolox",
                    "pdf_infer_table_structure": "true",
                    "ocr_languages": "indonesian",
                },
            )


        response.raise_for_status()
        elements_data = response.json()

        elements = []
        for elem in elements_data:
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

        logger.info(
            "Parsed PDF '%s': extracted %d elements", filename, len(elements)
        )
        return elements

    async def close(self) -> None:
        """Close the HTTP client."""
        await self._client.aclose()
