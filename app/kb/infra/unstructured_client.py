"""Unstructured API document parser adapter."""

import os
import time
import httpx
import structlog
from typing import List

from app.kb.domain.interfaces import IDocumentParser
from app.thesis.chunking.models import ParsedElement

logger = structlog.get_logger(__name__)

class UnstructuredClient(IDocumentParser):
    """HTTP adapter for the unstructured-api container.

    When ``extract_images=True``, the parser sends
    ``extract_image_block_types=["Image", "Table"]`` to the unstructured
    API, which causes it to return ``Image`` elements with image paths
    in their metadata. These are later enriched by a VLM during ingestion.
    """

    def __init__(self, base_url: str, extract_images: bool = True) -> None:
        self._client = httpx.AsyncClient(
            base_url=base_url.rstrip("/"),
            timeout=httpx.Timeout(900.0, connect=30.0),
        )
        self._extract_images = extract_images
        logger.info(
            "UnstructuredClient initialized",
            base_url=base_url,
            extract_images=extract_images,
        )

    async def parse_pdf(self, file_path: str) -> List[ParsedElement]:
        resolved = os.path.realpath(file_path)
        if not os.path.isfile(resolved):
            raise FileNotFoundError(f"PDF not found: {resolved!r}")

        filename = os.path.basename(resolved)
        log = logger.bind(filename=filename)
        start_time = time.perf_counter()

        strategy = "hi_res"

        # Build form data — add image extraction when enabled
        form_data: dict[str, str] = {"strategy": strategy}
        if self._extract_images:
            form_data["extract_image_block_types"] = '["Image", "Table"]'
            form_data["extract_image_block_to_payload"] = "false"

        try:
            with open(resolved, "rb") as fh:
                response = await self._client.post(
                    "/general/v0/general",
                    files={"files": (filename, fh, "application/pdf")},
                    data=form_data,
                )
            response.raise_for_status()
            raw_output = response.json()
        except Exception as exc:
            log.error("parse.failed", error=str(exc))
            raise

        elements: List[ParsedElement] = []
        image_count = 0
        table_count = 0

        for elem in raw_output:
            text = elem.get("text", "").strip()

            elem_type = elem.get("type", "UncategorizedText")
            metadata = elem.get("metadata") or {}

            # For tables, if text_as_html is available, prefer it over plain text
            if elem_type == "Table" and metadata.get("text_as_html"):
                text = metadata["text_as_html"]
                table_count += 1

            # For Image elements, the text may be empty but we still want to
            # keep the element so the VLM enricher can process it. The image
            # path is in metadata["image_path"].
            if elem_type == "Image":
                image_count += 1
                # Keep the element even if text is empty — the VLM will fill it
                elements.append(
                    ParsedElement(
                        element_type=elem_type,
                        text=text,  # May be empty — VLM will enrich
                        metadata=metadata,
                    )
                )
                continue

            if not text:
                continue

            elements.append(
                ParsedElement(
                    element_type=elem_type,
                    text=text,
                    metadata=metadata,
                )
            )

        log.info(
            "parse.success",
            elements_count=len(elements),
            image_count=image_count,
            table_count=table_count,
            execution_time_sec=round(time.perf_counter() - start_time, 2)
        )
        return elements

    async def close(self) -> None:
        await self._client.aclose()
