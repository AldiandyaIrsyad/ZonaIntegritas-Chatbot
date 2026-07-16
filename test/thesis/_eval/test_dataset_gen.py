"""Tests for the dataset generation pipeline.

Verifies:
    - EvaluatorPanel majority voting (≥4/5 accept, <4/5 reject)
    - EvaluatorPanel fail-closed on errors (defaults to NO)
    - _parse_yes_no() parsing logic
    - DatasetGenerator JSONL parsing
    - DatasetGenerator handles markdown fences
"""

from __future__ import annotations

import json
from typing import Any, Dict, List
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from app.thesis._eval._dataset_gen.config import DatasetGenSettings
from app.thesis._eval._dataset_gen.generator import DatasetGenerator, GeneratedItem
from app.thesis._eval._dataset_gen.panel import EvaluatorPanel, PanelVerdict


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_settings(
    api_key: str = "test-key",
    panel_models: str = "model-a,model-b,model-c,model-d,model-e",
    threshold: int = 4,
) -> DatasetGenSettings:
    """Create DatasetGenSettings for testing."""
    return DatasetGenSettings(
        openrouter_api_key=api_key,
        openrouter_base_url="https://openrouter.ai/api/v1",
        generator_model="deepseek/deepseek-chat",
        generator_temperature=0.0,
        panel_models=panel_models,
        panel_temperature=0.0,
        acceptance_threshold=threshold,
    )


def _mock_chat_response(content: str) -> MagicMock:
    """Create a mock OpenRouter chat completion response."""
    resp = MagicMock()
    resp.raise_for_status = MagicMock()
    resp.json.return_value = {
        "choices": [{"message": {"content": content}}]
    }
    return resp


# ---------------------------------------------------------------------------
# EvaluatorPanel tests
# ---------------------------------------------------------------------------


class TestEvaluatorPanel:
    """Tests for the EvaluatorPanel majority voting."""

    @pytest.mark.asyncio
    async def test_accept_when_four_yes(self) -> None:
        """4/5 YES → accepted=True."""
        settings = _make_settings()
        panel = EvaluatorPanel(settings)
        panel._client = MagicMock(spec=httpx.AsyncClient)

        # 4 YES, 1 NO
        responses = [
            _mock_chat_response("YES"),
            _mock_chat_response("YES"),
            _mock_chat_response("YES"),
            _mock_chat_response("YES"),
            _mock_chat_response("NO"),
        ]
        panel._client.post = AsyncMock(side_effect=responses)

        verdict = await panel.evaluate("Is this valid?", "context here")

        assert verdict.accepted is True
        assert verdict.yes_count == 4
        assert verdict.no_count == 1
        assert verdict.acceptance_threshold == 4

    @pytest.mark.asyncio
    async def test_reject_when_three_yes(self) -> None:
        """3/5 YES → accepted=False (below threshold)."""
        settings = _make_settings()
        panel = EvaluatorPanel(settings)
        panel._client = MagicMock(spec=httpx.AsyncClient)

        responses = [
            _mock_chat_response("YES"),
            _mock_chat_response("YES"),
            _mock_chat_response("YES"),
            _mock_chat_response("NO"),
            _mock_chat_response("NO"),
        ]
        panel._client.post = AsyncMock(side_effect=responses)

        verdict = await panel.evaluate("Is this valid?", "context here")

        assert verdict.accepted is False
        assert verdict.yes_count == 3
        assert verdict.no_count == 2

    @pytest.mark.asyncio
    async def test_unanimous_yes(self) -> None:
        """5/5 YES → accepted=True."""
        settings = _make_settings()
        panel = EvaluatorPanel(settings)
        panel._client = MagicMock(spec=httpx.AsyncClient)

        responses = [_mock_chat_response("YES") for _ in range(5)]
        panel._client.post = AsyncMock(side_effect=responses)

        verdict = await panel.evaluate("Is this valid?", "context")
        assert verdict.accepted is True
        assert verdict.yes_count == 5

    @pytest.mark.asyncio
    async def test_unanimous_no(self) -> None:
        """0/5 YES → accepted=False."""
        settings = _make_settings()
        panel = EvaluatorPanel(settings)
        panel._client = MagicMock(spec=httpx.AsyncClient)

        responses = [_mock_chat_response("NO") for _ in range(5)]
        panel._client.post = AsyncMock(side_effect=responses)

        verdict = await panel.evaluate("Is this valid?", "context")
        assert verdict.accepted is False
        assert verdict.yes_count == 0

    @pytest.mark.asyncio
    async def test_fail_closed_on_error(self) -> None:
        """On API error, the vote should default to NO (fail-closed)."""
        settings = _make_settings()
        panel = EvaluatorPanel(settings)
        panel._client = MagicMock(spec=httpx.AsyncClient)

        # 3 succeed with YES, 2 fail with exception
        responses = [
            _mock_chat_response("YES"),
            _mock_chat_response("YES"),
            _mock_chat_response("YES"),
            httpx.ConnectError("Connection refused"),
            httpx.ConnectError("Timeout"),
        ]
        panel._client.post = AsyncMock(side_effect=responses)

        verdict = await panel.evaluate("Is this valid?", "context")

        # 3 YES + 2 error→NO = 3/5 → rejected
        assert verdict.accepted is False
        assert verdict.yes_count == 3
        assert verdict.no_count == 2

    @pytest.mark.asyncio
    async def test_votes_recorded(self) -> None:
        """Individual votes should be recorded in the verdict."""
        settings = _make_settings()
        panel = EvaluatorPanel(settings)
        panel._client = MagicMock(spec=httpx.AsyncClient)

        responses = [_mock_chat_response("YES") for _ in range(5)]
        panel._client.post = AsyncMock(side_effect=responses)

        verdict = await panel.evaluate("prompt", "context")

        assert len(verdict.votes) == 5
        assert all(v.parsed for v in verdict.votes)
        assert all(v.model in ["model-a", "model-b", "model-c", "model-d", "model-e"]
                    for v in verdict.votes)


class TestParseYesNo:
    """Tests for EvaluatorPanel._parse_yes_no()."""

    def test_explicit_yes(self) -> None:
        """Explicit YES → True."""
        assert EvaluatorPanel._parse_yes_no("YES") is True
        assert EvaluatorPanel._parse_yes_no("yes") is True
        assert EvaluatorPanel._parse_yes_no("Yes") is True

    def test_explicit_no(self) -> None:
        """Explicit NO → False."""
        assert EvaluatorPanel._parse_yes_no("NO") is False
        assert EvaluatorPanel._parse_yes_no("no") is False
        assert EvaluatorPanel._parse_yes_no("No") is False

    def test_yes_in_sentence(self) -> None:
        """YES embedded in text → True."""
        assert EvaluatorPanel._parse_yes_no("Yes, this is correct.") is True

    def test_no_in_sentence(self) -> None:
        """NO embedded in text → False."""
        assert EvaluatorPanel._parse_yes_no("No, this is wrong.") is False

    def test_ambiguous(self) -> None:
        """Ambiguous text → False (fail-closed)."""
        assert EvaluatorPanel._parse_yes_no("Maybe") is False
        assert EvaluatorPanel._parse_yes_no("") is False
        assert EvaluatorPanel._parse_yes_no("I'm not sure") is False


# ---------------------------------------------------------------------------
# DatasetGenerator tests
# ---------------------------------------------------------------------------


class TestDatasetGenerator:
    """Tests for the DatasetGenerator JSONL parsing."""

    def test_parse_jsonl_simple(self) -> None:
        """Parse simple JSONL output."""
        text = '{"question": "Apa itu ZI?", "answer": "Zona Integritas"}\n{"question": "Apa WBK?", "answer": "Wilayah Birokrasi Bersih"}'
        items = DatasetGenerator._parse_jsonl(text)

        assert len(items) == 2
        assert isinstance(items[0].parsed, dict)
        assert items[0].parsed["question"] == "Apa itu ZI?"
        assert items[1].parsed["answer"] == "Wilayah Birokrasi Bersih"

    def test_parse_jsonl_with_markdown_fences(self) -> None:
        """Parse JSONL wrapped in markdown code fences."""
        text = '```json\n{"question": "Apa itu ZI?"}\n{"question": "Apa WBK?"}\n```'
        items = DatasetGenerator._parse_jsonl(text)

        assert len(items) == 2
        assert items[0].parsed["question"] == "Apa itu ZI?"

    def test_parse_jsonl_empty_lines(self) -> None:
        """Empty lines should be skipped."""
        text = '{"q": "1"}\n\n{"q": "2"}\n'
        items = DatasetGenerator._parse_jsonl(text)

        assert len(items) == 2

    def test_parse_jsonl_invalid_json_fallback(self) -> None:
        """Invalid JSON lines should be kept as raw strings."""
        text = '{"q": "1"}\nnot json at all\n{"q": "3"}'
        items = DatasetGenerator._parse_jsonl(text)

        assert len(items) == 3
        assert items[0].parsed == {"q": "1"}
        assert items[1].parsed == "not json at all"
        assert items[2].parsed == {"q": "3"}

    def test_parse_jsonl_empty(self) -> None:
        """Empty text → empty list."""
        items = DatasetGenerator._parse_jsonl("")
        assert items == []

    def test_parse_jsonl_extract_json_from_line(self) -> None:
        """Extract JSON object from a line with surrounding text."""
        text = 'The answer is: {"q": "test"} done'
        items = DatasetGenerator._parse_jsonl(text)

        assert len(items) == 1
        assert items[0].parsed == {"q": "test"}

    @pytest.mark.asyncio
    async def test_generate_calls_api(self) -> None:
        """generate() should call the OpenRouter API and parse results."""
        settings = _make_settings()
        gen = DatasetGenerator(settings)
        gen._client = MagicMock(spec=httpx.AsyncClient)

        jsonl_output = '{"question": "Q1"}\n{"question": "Q2"}'
        gen._client.post = AsyncMock(return_value=_mock_chat_response(jsonl_output))

        items = await gen.generate("Generate questions", count=2)

        assert len(items) == 2
        assert items[0].parsed["question"] == "Q1"
        gen._client.post.assert_called_once()

    @pytest.mark.asyncio
    async def test_generate_single(self) -> None:
        """generate_single() should return raw text."""
        settings = _make_settings()
        gen = DatasetGenerator(settings)
        gen._client = MagicMock(spec=httpx.AsyncClient)
        gen._client.post = AsyncMock(return_value=_mock_chat_response("raw response"))

        result = await gen.generate_single("Tell me about ZI")
        assert result == "raw response"
