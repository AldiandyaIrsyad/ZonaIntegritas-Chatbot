"""Tests for the citation format helper in ChatService.

Verifies that ``_format_citation()`` produces the canonical citation format
``*(STATUS: SCORE; SOURCE; Page N)*`` for all three NLI labels, with and
without source/page information.

Canonical format:
    *(Supported: 0.92; Pedoman ZI UPI; Page 12)*
    *(Contradiction: 0.88; Pedoman ZI UPI; Page 12)*
    *(Neutral: 0.75)*
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional
from unittest.mock import MagicMock

import pytest

from app.chat.application.chat_service import ChatService
from app.thesis.ram.interfaces import NLIResult


def _make_nli_result(
    label: str = "entailment",
    entailment_score: float = 0.92,
    contradiction_score: float = 0.05,
    neutral_score: float = 0.03,
    source_title: str = "Pedoman ZI UPI",
    page: Optional[int] = 12,
) -> NLIResult:
    """Create an NLIResult for testing."""
    return NLIResult(
        label=label,
        entailment_score=entailment_score,
        contradiction_score=contradiction_score,
        neutral_score=neutral_score,
        source_title=source_title,
        page=page,
    )


class TestFormatCitation:
    """Tests for ChatService._format_citation()."""

    def test_entailment_with_source_and_page(self) -> None:
        """Supported label with source and page."""
        result = _make_nli_result(
            label="entailment",
            entailment_score=0.92,
            source_title="Pedoman ZI UPI",
            page=12,
        )
        citation = ChatService._format_citation(result)
        assert "Supported" in citation
        assert "0.92" in citation
        assert "Pedoman ZI UPI" in citation
        assert "Page 12" in citation

    def test_contradiction_with_source_and_page(self) -> None:
        """Contradiction label with source and page."""
        result = _make_nli_result(
            label="contradiction",
            entailment_score=0.05,
            contradiction_score=0.88,
            source_title="Pedoman ZI UPI",
            page=5,
        )
        citation = ChatService._format_citation(result)
        assert "Contradiction" in citation
        assert "0.88" in citation
        assert "Pedoman ZI UPI" in citation
        assert "Page 5" in citation

    def test_neutral_with_source_and_page(self) -> None:
        """Neutral label with source and page."""
        result = _make_nli_result(
            label="neutral",
            entailment_score=0.10,
            contradiction_score=0.15,
            neutral_score=0.75,
            source_title="Pedoman ZI UPI",
            page=3,
        )
        citation = ChatService._format_citation(result)
        assert "Neutral" in citation
        assert "0.75" in citation
        assert "Pedoman ZI UPI" in citation
        assert "Page 3" in citation

    def test_without_source_title(self) -> None:
        """Citation without source title should omit it."""
        result = _make_nli_result(
            label="entailment",
            entailment_score=0.90,
            source_title="",
            page=7,
        )
        citation = ChatService._format_citation(result)
        assert "Supported" in citation
        assert "0.90" in citation
        assert "Page 7" in citation
        # Should not have empty source segment
        assert ";;" not in citation

    def test_without_page(self) -> None:
        """Citation without page number should omit it."""
        result = _make_nli_result(
            label="entailment",
            entailment_score=0.85,
            source_title="Pedoman ZI",
            page=None,
        )
        citation = ChatService._format_citation(result)
        assert "Supported" in citation
        assert "0.85" in citation
        assert "Pedoman ZI" in citation
        assert "Page" not in citation

    def test_without_source_and_page(self) -> None:
        """Citation with neither source nor page."""
        result = _make_nli_result(
            label="neutral",
            neutral_score=0.60,
            source_title="",
            page=None,
        )
        citation = ChatService._format_citation(result)
        assert "Neutral" in citation
        assert "0.60" in citation
        assert "Page" not in citation

    def test_none_result_returns_empty(self) -> None:
        """None result should return empty string."""
        citation = ChatService._format_citation(None)
        assert citation == ""

    def test_score_two_decimals(self) -> None:
        """Score should always be formatted to 2 decimal places."""
        result = _make_nli_result(
            label="entailment",
            entailment_score=0.123456,
            source_title="Doc",
            page=1,
        )
        citation = ChatService._format_citation(result)
        assert "0.12" in citation

    def test_citation_wrapped_in_asterisk_parens(self) -> None:
        """Citation should be wrapped in *(...)* format."""
        result = _make_nli_result(
            label="entailment",
            entailment_score=0.90,
            source_title="Doc",
            page=1,
        )
        citation = ChatService._format_citation(result)
        assert citation.startswith(" *(")
        assert citation.endswith(")*")

    def test_unknown_label_defaults_to_neutral(self) -> None:
        """Unknown label should default to 'Neutral'."""
        result = _make_nli_result(
            label="unknown_label",
            neutral_score=0.50,
            source_title="Doc",
            page=1,
        )
        citation = ChatService._format_citation(result)
        assert "Neutral" in citation
