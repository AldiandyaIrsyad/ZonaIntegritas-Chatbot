"""Chat / IVM / RAM visualization runner.

Drives the real, unmodified ``ChatService.process_chat_message()`` — the
literal production chat pipeline (IVM safety check -> IVM relevance
precheck -> full retrieval -> prompt build -> streamed generation ->
per-sentence RAM assessment -> citation formatting) — against an isolated
Postgres session and isolated Qdrant collection, and captures the
intermediate verdicts that pipeline normally keeps internal:

- The real ``PromptGuardClient.check_prompt()`` result for every
  sliding-window chunk the IVM safety check examines (label/score/message),
  not just the final raise/no-raise outcome.
- The real LLM-judge relevance transcript (the literal system prompt, user
  message, and raw YES/NO response text) — ``LLMJudge.evaluate_relevance``
  normally discards the text after computing the boolean.
- Every per-sentence ``NLIResult`` the RAM module produces, not just the
  citation-marker string spliced into the streamed answer.

This is done by composing the real service classes (``IVMService``,
``RelevanceService``, ``RAMService``, ``ChatService`` are used entirely
unmodified) with thin recording proxies around the *adapters* they're
constructed with (``ISafetyModel``, ``ILLMJudgeConnection``, and
``RAMService`` itself) — production code is not touched.
"""

from __future__ import annotations

import dataclasses
import json
import uuid
from typing import Any, AsyncIterator, Dict, List, Optional

import structlog
from sqlalchemy.ext.asyncio import AsyncSession

from app.chat.application.chat_service import ChatService
from app.chat.infra.llm_connection import LLMConnection
from app.chat.infra.nli_client import NLIClient
from app.chat.infra.postgres_chat_repo import PostgresChatRepository
from app.chat.infra.prompt_guard_client import PromptGuardClient
from app.chat.infra.hyde_expander import HyDEExpander
from app.kb.application.search_service import SearchService
from app.kb.config import get_bge_m3_settings
from app.kb.domain.interfaces import IQueryExpander
from app.kb.infra.bge_m3_embeddings import BGEM3Embeddings
from app.kb.infra.infinity_reranker import InfinityReranker
from app.kb.infra.postgres_repo import PostgresKBRepository
from app.kb.infra.qdrant_store import QdrantStore
from app.thesis.ivm.checkers import LLMJudgeRelevanceChecker
from app.thesis.ivm.interfaces import SafetyResult
from app.thesis.ivm.judge import LLMJudge
from app.thesis.ivm.relevance_service import RelevanceService
from app.thesis.ivm.service import IVMService
from app.thesis.ram.interfaces import NLIResult
from app.thesis.ram.service import RAMService

logger = structlog.get_logger(__name__)


# ── Recording proxies (wrap real adapters, delegate everything) ──────────


class _CapturingSafetyModel:
    """Wraps a real ``PromptGuardClient``, recording every sliding-window
    ``check_prompt`` call's input chunk + result."""

    def __init__(self, inner: PromptGuardClient):
        self._inner = inner
        self.calls: List[Dict[str, Any]] = []

    async def check_prompt(self, text: str) -> SafetyResult:
        result = await self._inner.check_prompt(text)
        self.calls.append({"chunk_preview": text[:80], "is_safe": result.is_safe, "message": result.message})
        return result


class _CapturingJudgeConnection:
    """Wraps the judge's dedicated ``LLMConnection`` (a separate instance
    from the main chat-generation connection — see
    ``app/chat/dependency.py:get_relevance_checker``), recording the exact
    system/user messages and raw response text of every judge call."""

    def __init__(self, inner: LLMConnection):
        self._inner = inner
        self.calls: List[Dict[str, Any]] = []

    def stream_chat(self, model: str, messages: List[Dict[str, Any]], max_tokens: int = 100) -> AsyncIterator[str]:
        return self._record_and_stream(model, messages, max_tokens)

    async def _record_and_stream(self, model: str, messages: List[Dict[str, Any]], max_tokens: int):
        chunks: List[str] = []
        async for piece in self._inner.stream_chat(model=model, messages=messages, max_tokens=max_tokens):
            chunks.append(piece)
            yield piece
        self.calls.append({
            "system_prompt": next((m["content"] for m in messages if m["role"] == "system"), ""),
            "user_message": next((m["content"] for m in messages if m["role"] == "user"), ""),
            "raw_response": "".join(chunks),
        })


class _CapturingRAMService:
    """Wraps a real ``RAMService``, recording every per-sentence
    ``assess_sentence`` call's sentence + full ``NLIResult``."""

    def __init__(self, inner: RAMService):
        self._inner = inner
        self.assessments: List[Dict[str, Any]] = []

    def build_premise(self, contexts) -> str:
        return self._inner.build_premise(contexts)

    async def assess_sentence(self, sentence: str, premise: str, contexts) -> NLIResult:
        result = await self._inner.assess_sentence(sentence, premise, contexts)
        self.assessments.append({"sentence": sentence, "result": dataclasses.asdict(result)})
        return result


# ── Snapshot dataclass ────────────────────────────────────────────────────


@dataclasses.dataclass(frozen=True)
class ChatCaptureSnapshot:
    query: str
    outcome: str  # "answered" | "blocked_unsafe" | "blocked_irrelevant" | "error"
    safety_calls: List[Dict[str, Any]]
    judge_calls: List[Dict[str, Any]]
    context_payload: Optional[Dict[str, Any]]
    answer_text: str
    ram_assessments: List[Dict[str, Any]]
    ndjson_events: List[Dict[str, Any]]


# ── Real adapter bundle, built once and reused across queries ────────────


@dataclasses.dataclass
class ChatRealAdapters:
    """Real, unmodified infra adapters shared across queries — expensive to
    construct (httpx clients) so built once by the caller and passed in."""

    session: AsyncSession
    qdrant_collection: str
    infinity_url: str
    qdrant_host: str
    qdrant_port: int
    embedding_model: str
    reranker_model: str
    prompt_guard_model: str
    nli_model: str
    security_threshold: float
    llm_base_url: str
    llm_api_key: Any
    llm_model: str
    system_prompt: str
    llm_temperature: float
    hyde_enabled: bool
    hyde_max_tokens: int
    hyde_temperature: float
    hyde_prompt_template: str
    hyde_system_prompt: str


async def run_chat_capture(*, query: str, adapters: ChatRealAdapters) -> ChatCaptureSnapshot:
    """Drive the real chat pipeline for one query and capture IVM/RAM detail.

    Builds a fresh ``ChatService`` (and fresh IVM/relevance/RAM service
    instances) per call, wired with recording proxies, but reuses the
    caller's real HTTP-client-backed adapters — cheap to construct, safe to
    call once per query without cross-query bleed in the capture logs.
    """
    bge_m3_cfg = get_bge_m3_settings()
    text_embedder = BGEM3Embeddings(
        model_name=bge_m3_cfg.model,
        device=bge_m3_cfg.device,
        use_fp16=bge_m3_cfg.use_fp16,
        batch_size=bge_m3_cfg.batch_size,
    )
    vector_store = QdrantStore(host=adapters.qdrant_host, port=adapters.qdrant_port, collection_name=adapters.qdrant_collection)
    reranker = InfinityReranker(base_url=adapters.infinity_url, model=adapters.reranker_model)
    kb_repo = PostgresKBRepository(adapters.session)

    query_expander: Optional[IQueryExpander] = None
    main_llm_conn = LLMConnection(base_url=adapters.llm_base_url, api_key=adapters.llm_api_key)
    if adapters.hyde_enabled:
        query_expander = HyDEExpander(
            llm=main_llm_conn, model=adapters.llm_model,
            prompt_template=adapters.hyde_prompt_template,
            system_prompt=adapters.hyde_system_prompt,
            max_tokens=adapters.hyde_max_tokens, temperature=adapters.hyde_temperature,
        )

    search_service = SearchService(
        text_embedder=text_embedder, vector_store=vector_store, kb_repo=kb_repo,
        reranker=reranker, query_expander=query_expander,
    )

    real_safety = PromptGuardClient(base_url=adapters.infinity_url, model=adapters.prompt_guard_model, security_threshold=adapters.security_threshold)
    capturing_safety = _CapturingSafetyModel(real_safety)
    ivm_service = IVMService(safety_model=capturing_safety)

    real_judge_llm = LLMConnection(base_url=adapters.llm_base_url, api_key=adapters.llm_api_key)
    capturing_judge_conn = _CapturingJudgeConnection(real_judge_llm)
    judge = LLMJudge(llm_connection=capturing_judge_conn, model=adapters.llm_model)
    relevance_service = RelevanceService(relevance_checker=LLMJudgeRelevanceChecker(judge=judge))

    real_nli = NLIClient(base_url=adapters.infinity_url, model=adapters.nli_model)
    real_ram = RAMService(nli_model=real_nli, reranker_model=reranker, enabled=True)
    capturing_ram = _CapturingRAMService(real_ram)

    chat_repo = PostgresChatRepository(adapters.session)
    chat_service = ChatService(
        chat_repo=chat_repo,
        llm_conn=main_llm_conn,
        search_service=search_service,
        ivm_service=ivm_service,  # type: ignore[arg-type]
        relevance_service=relevance_service,
        ram_service=capturing_ram,  # type: ignore[arg-type]
        model_name=adapters.llm_model,
        system_prompt=adapters.system_prompt,
        temperature=adapters.llm_temperature,
    )

    session_id = str(uuid.uuid4())
    events: List[Dict[str, Any]] = []
    context_payload: Optional[Dict[str, Any]] = None
    answer_chunks: List[str] = []
    outcome = "answered"

    try:
        async for line in chat_service.process_chat_message(session_id, query, skip_guardrails=False):
            evt = json.loads(line)
            events.append(evt)
            if evt["type"] == "context":
                context_payload = {"content": evt.get("content", ""), "chunks": evt.get("chunks", [])}
            elif evt["type"] == "chunk":
                answer_chunks.append(evt["content"])
            elif evt["type"] == "error":
                msg = evt.get("content", "")
                if "safety filters" in msg:
                    outcome = "blocked_unsafe"
                elif "only answer questions" in msg:
                    outcome = "blocked_irrelevant"
                else:
                    outcome = "error"
        await adapters.session.commit()
    finally:
        await text_embedder.close()
        await vector_store.close()
        await reranker.close()
        await main_llm_conn.close()
        await real_judge_llm.close()
        await real_safety.close()
        await real_nli.close()

    return ChatCaptureSnapshot(
        query=query,
        outcome=outcome,
        safety_calls=capturing_safety.calls,
        judge_calls=capturing_judge_conn.calls,
        context_payload=context_payload,
        answer_text="".join(answer_chunks),
        ram_assessments=capturing_ram.assessments,
        ndjson_events=events,
    )
