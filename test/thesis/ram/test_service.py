"""Tests for RAMService's evidence-snippet sanitization helper."""
from __future__ import annotations

from app.thesis.ram.service import _sanitize_snippet, EVIDENCE_SNIPPET_MAX_CHARS


class TestSanitizeSnippet:
    def test_collapses_whitespace(self) -> None:
        assert _sanitize_snippet("Ada   banyak\nspasi.") == "Ada banyak spasi."

    def test_strips_characters_that_break_the_citation_marker_grammar(self) -> None:
        result = _sanitize_snippet('Isi (a); (b) * penting "kutipan"')
        assert ";" not in result
        assert ")" not in result
        assert "*" not in result
        assert '"' not in result

    def test_truncates_long_text_with_ellipsis(self) -> None:
        text = "kata " * 100
        result = _sanitize_snippet(text)
        assert len(result) <= EVIDENCE_SNIPPET_MAX_CHARS
        assert result.endswith("…")

    def test_short_text_is_unchanged_besides_whitespace(self) -> None:
        assert _sanitize_snippet("Teks singkat.") == "Teks singkat."
