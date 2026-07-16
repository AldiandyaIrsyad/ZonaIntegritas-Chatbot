"""Protocol interfaces for the chunking pipeline.

This module defines Protocol interfaces for capabilities that the pure
``thesis/chunking`` module needs but cannot import (Dependency Inversion).
Infrastructure adapters in ``kb/infra`` implement these protocols.
"""

from __future__ import annotations

from typing import Protocol


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
