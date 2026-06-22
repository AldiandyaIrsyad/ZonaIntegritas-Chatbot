"""
Dependency injection for the Chat module.
"""

from fastapi import Depends
from sqlalchemy.ext.asyncio import AsyncSession

from app.shared.db import get_db_session
from app.chat.config import get_chat_config
from app.chat.infra import LLMConnection, PromptGuardClient, NLIClient, PostgresChatRepository, EmbeddingClient
from app.chat.application.chat_service import ChatService
from app.kb.dependency import get_search_service
from app.kb.application.search_service import SearchService

from app.thesis.ivm.service import IVMService
from app.thesis.ivm.judge import LLMJudge
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
    return EmbeddingClient(base_url=config.infinity_url, model="BBAAI/bge-m3")

def get_ivm_service(
    safety_client: PromptGuardClient = Depends(get_prompt_guard_client)
) -> IVMService:
    config = get_chat_config()
    # Create a separate instance of LLM connection for the Judge
    judge_llm = LLMConnection(base_url=config.llm_base_url, api_key=config.llm_api_key)
    judge = LLMJudge(llm_connection=judge_llm, model=config.llm_model)
    
    return IVMService(
        safety_model=safety_client,
        judge=judge
    )

def get_ram_service(
    nli_client: NLIClient = Depends(get_nli_client),
    embedding_client: EmbeddingClient = Depends(get_embedding_client)
) -> RAMService:
    return RAMService(
        nli_model=nli_client,
        embedding_model=embedding_client,
        enabled=True
    )

async def get_chat_service(
    chat_repo: PostgresChatRepository = Depends(get_chat_repo),
    llm_conn: LLMConnection = Depends(get_llm_connection),
    search_service: SearchService = Depends(get_search_service),
    ivm_service: IVMService = Depends(get_ivm_service),
    ram_service: RAMService = Depends(get_ram_service)
) -> ChatService:
    config = get_chat_config()
    return ChatService(
        chat_repo=chat_repo,
        llm_conn=llm_conn,
        search_service=search_service,
        ivm_service=ivm_service,
        ram_service=ram_service,
        model_name=config.llm_model,
        system_prompt=config.system_prompt
    )
