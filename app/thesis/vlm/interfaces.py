"""Protocol interfaces for Vision-Language Model enrichment.

This module defines the Protocol interface that the pure ``thesis``
research core needs but cannot import (Dependency Inversion). The
concrete adapters implementing this protocol live in
``thesis/vlm/client.py``.

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

    Implemented by (all in ``app/thesis/vlm/client.py``):
        - ``OpenRouterVLMClient``  — cloud VLM via OpenRouter (default).
        - ``OllamaVLMClient``      — local VLM via Ollama.
        - ``FallbackVLMClient``   — text-only fallback using PyMuPDF drawing
          analysis (no external model; used when no VLM endpoint is configured).

    Wired in ``app/kb/dependency.py::get_vlm_enricher``.

    The enricher is called during ingestion for each ``ParsedElement``
    whose ``content_type`` is :attr:`ContentType.FIGURE`. The returned
    text description replaces the element's (likely empty) text field,
    allowing the standard chunker to process it as narrative text.
    """

    async def describe_image(
        self,
        image_path: str,
        prompt: str = (
            "Deskripsikan gambar ini secara detail. Jika berupa bagan alir "
            "(flowchart) atau diagram proses, jelaskan setiap langkah, pihak/aktor "
            "yang terlibat, titik keputusan, dan urutan kejadiannya. Jika berupa "
            "tangkapan layar (screenshot) yang diberi anotasi, jelaskan apa yang "
            "ditunjuk oleh setiap anotasi."
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
