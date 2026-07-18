"""Tests for LLMJudge (app/thesis/ivm/judge.py)."""
from unittest.mock import AsyncMock, MagicMock

import pytest

from app.thesis.ivm.judge import DEFAULT_RELEVANCE_JUDGE_USER_TEMPLATE, LLMJudge


async def _fake_stream(chunks: list[str]):
    for chunk in chunks:
        yield chunk


def _make_llm_connection(response_chunks: list[str]) -> MagicMock:
    conn = MagicMock()
    conn.stream_chat = MagicMock(return_value=_fake_stream(response_chunks))
    return conn


class TestLLMJudgeUserTemplate:
    @pytest.mark.asyncio
    async def test_default_user_template_used_when_not_provided(self) -> None:
        conn = _make_llm_connection(["YES"])
        judge = LLMJudge(llm_connection=conn)

        await judge.evaluate_relevance(query="apa itu cuti?", context="SOP cuti pegawai")

        messages = conn.stream_chat.call_args.kwargs["messages"]
        assert messages[1]["content"] == DEFAULT_RELEVANCE_JUDGE_USER_TEMPLATE.replace(
            "{context}", "SOP cuti pegawai"
        ).replace("{query}", "apa itu cuti?")

    @pytest.mark.asyncio
    async def test_custom_user_template_is_used(self) -> None:
        conn = _make_llm_connection(["NO"])
        judge = LLMJudge(
            llm_connection=conn,
            user_template="CTX={context} | Q={query}",
        )

        result = await judge.evaluate_relevance(query="sepeda listrik", context="SOP cuti")

        messages = conn.stream_chat.call_args.kwargs["messages"]
        assert messages[1]["content"] == "CTX=SOP cuti | Q=sepeda listrik"
        assert result is False

    @pytest.mark.asyncio
    async def test_template_substitution_survives_braces_in_context(self) -> None:
        conn = _make_llm_connection(["YES"])
        judge = LLMJudge(llm_connection=conn)

        await judge.evaluate_relevance(
            query="q", context="contains {literal braces} in text"
        )

        messages = conn.stream_chat.call_args.kwargs["messages"]
        assert "contains {literal braces} in text" in messages[1]["content"]
