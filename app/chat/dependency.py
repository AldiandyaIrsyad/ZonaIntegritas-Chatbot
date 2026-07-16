"""
Dependency injection for the Chat module.

This is the composition root where ``chat/infra`` adapters are injected into
KB-domain services (e.g. HyDEExpander → SearchService). This file may import
from both ``chat/`` and ``kb/`` modules — it is the boundary where the
dependency inversion is resolved.
"""

from typing import Optional

from fastapi import Depends
from sqlalchemy.ext.asyncio import AsyncSession

from app.shared.db import get_db_session
from app.chat.config import get_chat_config
from app.chat.infra import LLMConnection, PromptGuardClient, NLIClient, PostgresChatRepository, EmbeddingClient
from app.chat.infra.hyde_expander import HyDEExpander
from app.chat.application.chat_service import ChatService
from app.kb.dependency import (
    get_kb_repo,
    get_vector_store,
    get_text_embedder,
    get_reranker,
)
from app.kb.infra import PostgresKBRepository, QdrantStore, InfinityEmbeddings, InfinityReranker
from app.kb.application.search_service import SearchService
from app.kb.domain.interfaces import IQueryExpander

from app.thesis.ivm.checkers import LLMJudgeRelevanceChecker
from app.thesis.ivm.interfaces import IRelevanceChecker
from app.thesis.ivm.judge import LLMJudge
from app.thesis.ivm.relevance_service import RelevanceService
from app.thesis.ivm.service import IVMService
from app.thesis.ram.service import RAMService

async def get_chat_repo(db: AsyncSession = Depends(get_db_session)) -> PostgresChatRepository:
    return PostgresChatRepository(db)

def get_llm_connection() -> LLMConnection:
    config = get_chat_config()
    return LLMConnection(base_url=config.llm_base_url, api_key=config.llm_api_key)

def get_prompt_guard_client() -> PromptGuardClient:
    config = get_chat_config()
    return PromptGuardClient(base_url=config.infinity_url, model=config.prompt_guard_model, security_threshold=config.security_threshold)

def get_nli_client() -> NLIClient:
    config = get_chat_config()
    return NLIClient(base_url=config.infinity_url, model=config.nli_model)

def get_embedding_client() -> EmbeddingClient:
    config = get_chat_config()
    return EmbeddingClient(base_url=config.infinity_url, model="BAAI/bge-m3")

def get_ivm_service(
    safety_client: PromptGuardClient = Depends(get_prompt_guard_client)
) -> IVMService:
    return IVMService(safety_model=safety_client)

def get_relevance_checker() -> IRelevanceChecker:
    """Overrides ``app.kb.dependency.get_relevance_checker`` (registered via
    ``app.main``'s ``dependency_overrides``) since the LLM-as-judge relevance
    check needs a cloud LLM (``chat/infra``) that ``kb/`` is not allowed to
    import.
    """
    config = get_chat_config()
    judge_llm = LLMConnection(base_url=config.llm_base_url, api_key=config.llm_api_key)
    judge = LLMJudge(llm_connection=judge_llm, model=config.llm_model)
    return LLMJudgeRelevanceChecker(judge=judge)

def get_relevance_service(
    checker: IRelevanceChecker = Depends(get_relevance_checker),
) -> RelevanceService:
    return RelevanceService(relevance_checker=checker)

def get_query_expander(
    llm_conn: LLMConnection = Depends(get_llm_connection),
) -> Optional[IQueryExpander]:
    """Build a HyDEExpander if HyDE is enabled in ChatConfig.

    Overrides the default ``None`` provider in ``kb/dependency.py``. This is
    the composition layer where ``chat/infra`` adapters are legally injected
    into KB-domain services.
    """
    config = get_chat_config()
    if not config.hyde_enabled:
        return None
    return HyDEExpander(
        llm=llm_conn,
        model=config.llm_model,
        prompt_template=config.hyde_prompt_template,
        max_tokens=config.hyde_max_tokens,
        temperature=config.hyde_temperature,
    )

def get_ram_service(
    nli_client: NLIClient = Depends(get_nli_client),
    reranker: Optional[InfinityReranker] = Depends(get_reranker)
) -> RAMService:
    if reranker is None:
        raise RuntimeError("Reranker is required for RAMService.")
    return RAMService(
        nli_model=nli_client,
        reranker_model=reranker,
        enabled=True
    )

async def get_search_service(
    repo: PostgresKBRepository = Depends(get_kb_repo),
    vstore: QdrantStore = Depends(get_vector_store),
    embedder: InfinityEmbeddings = Depends(get_text_embedder),
    reranker: Optional[InfinityReranker] = Depends(get_reranker),
    query_expander: Optional[IQueryExpander] = Depends(get_query_expander),
) -> SearchService:
    """Override of kb.dependency.get_search_service injecting HyDE expander."""
    return SearchService(
        text_embedder=embedder,
        vector_store=vstore,
        kb_repo=repo,
        reranker=reranker,
        query_expander=query_expander,
    )

async def get_chat_service(
    chat_repo: PostgresChatRepository = Depends(get_chat_repo),
    llm_conn: LLMConnection = Depends(get_llm_connection),
    search_service: SearchService = Depends(get_search_service),
    ivm_service: IVMService = Depends(get_ivm_service),
    relevance_service: RelevanceService = Depends(get_relevance_service),
    ram_service: RAMService = Depends(get_ram_service)
) -> ChatService:
    config = get_chat_config()
    return ChatService(
        chat_repo=chat_repo,
        llm_conn=llm_conn,
        search_service=search_service,
        ivm_service=ivm_service,
        relevance_service=relevance_service,
        ram_service=ram_service,
        model_name=config.llm_model,
        system_prompt=config.system_prompt,
        temperature=config.llm_temperature,
    )
