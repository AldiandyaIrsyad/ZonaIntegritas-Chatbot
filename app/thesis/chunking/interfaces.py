"""Protocol interfaces for the chunking pipeline.

This module defines Protocol interfaces for capabilities that the pure
``thesis/chunking`` module needs but cannot import (Dependency Inversion).
Infrastructure adapters in ``kb/infra`` implement these protocols.

The VLM enricher converts visual elements (flowcharts, annotated
screenshots) into structured text descriptions during ingestion, *before*
the chunker processes them. This keeps the chunker pure — it only sees
text, never images.
"""

from __future__ import annotations

from typing import Protocol


class IVLMEnricher(Protocol):
    """Protocol for Vision-Language Model enrichment of visual elements.

    Implementations wrap a VLM service (cloud API like Gemini/GPT-4o,
    or local model like LLaVA via Ollama, or a text-only fallback using
    PyMuPDF drawing analysis).

    The enricher is called during ingestion for each ``ParsedElement``
    whose ``content_type`` is :attr:`ContentType.FIGURE`. The returned
    text description replaces the element's (likely empty) text field,
    allowing the standard chunker to process it as narrative text.
    """

    async def describe_image(
        self,
        image_path: str,
        prompt: str = (
            "Describe this image in detail. If it is a flowchart or process "
            "diagram, describe each step, the actors involved, decision "
            "points, and the sequence of events. If it is an annotated "
            "screenshot, describe what each annotation points to."
        ),
    ) -> str:
        """Generate a structured text description of a visual element.

        Args:
            image_path: Absolute path to the extracted page/region image.
            prompt: Instruction for the VLM. The default prompt is
                optimised for flowcharts and annotated screenshots.

        Returns:
            Structured text description preserving the visual logic.

        Raises:
            Exception: If the VLM call fails. Callers should handle this
                gracefully (fail-closed: skip enrichment, keep original text).
        """
        ...

    async def close(self) -> None:
        """Release resources (HTTP clients, etc.)."""
        ...


class ITableSummarizer(Protocol):
    """Protocol for generating natural-language summaries of tables.

    Table summaries are used as child chunks for embedding (Small-to-Big
    retrieval). The full table HTML/Markdown is stored as the parent
    chunk; the summary is what gets vector-searched.

    Implementations typically wrap an LLM API (the same LLM connection
    used for generation, or a separate smaller model).
    """

    async def summarize_table(
        self,
        table_html: str,
        context: str = "",
    ) -> str:
        """Generate a natural-language summary of a table.

        Args:
            table_html: The table content in HTML or Markdown format.
            context: Optional breadcrumb/section context to ground the summary.

        Returns:
            A concise summary describing what the table contains, its
            columns, and key data points. E.g.:
            "This table details the work programs for Zone Integrity
            Bidang Penataan Tatalaksana, covering September to October
            2023, with columns for Indikator, Program Kerja, and Waktu
            Pelaksanaan."

        Raises:
            Exception: If the summarisation call fails. Callers should
                fall back to using the table's first row + headers as
                the child text.
        """
        ...

    async def close(self) -> None:
        """Release resources."""
        ...
