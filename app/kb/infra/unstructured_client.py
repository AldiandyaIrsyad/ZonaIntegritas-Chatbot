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
    """HTTP adapter for the unstructured-api container."""

    def __init__(self, base_url: str) -> None:
        self._client = httpx.AsyncClient(
            base_url=base_url.rstrip("/"),
            timeout=httpx.Timeout(900.0, connect=30.0),
        )
        logger.info("UnstructuredClient initialized", base_url=base_url)

    async def parse_pdf(self, file_path: str) -> List[ParsedElement]:
        resolved = os.path.realpath(file_path)
        if not os.path.isfile(resolved):
            raise FileNotFoundError(f"PDF not found: {resolved!r}")

        filename = os.path.basename(resolved)
        log = logger.bind(filename=filename)
        start_time = time.perf_counter()

        strategy = "hi_res"
        
        try:
            with open(resolved, "rb") as fh:
                response = await self._client.post(
                    "/general/v0/general",
                    files={"files": (filename, fh, "application/pdf")},
                    data={"strategy": strategy},
                )
            response.raise_for_status()
            raw_output = response.json()
        except Exception as exc:
            log.error("parse.failed", error=str(exc))
            raise

        elements: List[ParsedElement] = []
        for elem in raw_output:
            text = elem.get("text", "").strip()
            if not text:
                continue

            elem_type = elem.get("type", "UncategorizedText")
            elements.append(
                ParsedElement(
                    element_type=elem_type,
                    text=text,
                    metadata=elem.get("metadata") or {},
                )
            )

        log.info(
            "parse.success",
            elements_count=len(elements),
            execution_time_sec=round(time.perf_counter() - start_time, 2)
        )
        return elements

    async def close(self) -> None:
        await self._client.aclose()
