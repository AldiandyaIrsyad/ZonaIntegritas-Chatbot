"""Tests for the shared sentence-splitting helper used by RAM."""
from __future__ import annotations

from app.thesis.ram.text_utils import split_sentences


class TestSplitSentences:
    def test_splits_on_sentence_boundaries(self) -> None:
        result = split_sentences("Ini kalimat pertama. Ini kalimat kedua!")
        assert result == ["Ini kalimat pertama.", "Ini kalimat kedua!"]

    def test_numbered_list_markers_stay_attached_to_their_item(self) -> None:
        text = "1. Langkah pertama. 2. Langkah kedua."
        result = split_sentences(text)
        assert result == ["1. Langkah pertama.", "2. Langkah kedua."]

    def test_numbered_list_without_trailing_period_not_split_at_marker(self) -> None:
        # No real sentence boundary here besides the list markers' own
        # periods — the whole thing should stay as one unit rather than
        # being torn apart at "1." / "2.".
        text = "1. Ajukan surat ke bagian TU 2. Tunggu verifikasi"
        result = split_sentences(text)
        assert result == [text]

    def test_newline_separated_list_items_split_cleanly(self) -> None:
        text = "1. Langkah pertama\n2. Langkah kedua\n3. Langkah ketiga"
        result = split_sentences(text)
        assert result == ["1. Langkah pertama", "2. Langkah kedua", "3. Langkah ketiga"]

    def test_multiple_newlines_collapse_to_single_boundary(self) -> None:
        result = split_sentences("Paragraf satu.\n\n\nParagraf dua.")
        assert result == ["Paragraf satu.", "Paragraf dua."]

    def test_empty_string_returns_empty_list(self) -> None:
        assert split_sentences("") == []

    def test_whitespace_only_returns_empty_list(self) -> None:
        assert split_sentences("   \n\n  ") == []
