"""Chat pipeline orchestrator."""

import uuid
from typing import AsyncGenerator, List, Optional

import structlog
from fastapi import HTTPException
from fastapi.responses import StreamingResponse

from app.core.interfaces.ai import IEmbeddingProvider, IReranker
from app.core.interfaces.infra import IVectorStore
from app.core.interfaces.ivm import IIVMService
from app.core.interfaces.llm import ILLMService
from app.core.interfaces.rag import IRetrievalService, RetrievedContext
from app.core.interfaces.ram import IRAMService

from app.chat.repository import ChatRepository
from app.chat.prompt_builder import build_secure_system_prompt
from app.chat.yield_handler import nli_streaming_generate
from app.chat.output_checker import check_and_persist

logger = structlog.get_logger(__name__)


class ChatPipeline:
    """Orchestrates the chat process from input validation to output persistence."""

    def __init__(
        self,
        repository: ChatRepository,
        llm_service: ILLMService,
        retrieval_service: IRetrievalService,
        reranker: IReranker,
        vector_store: IVectorStore,
        embedding_provider: IEmbeddingProvider,
        ivm_service: IIVMService,
        ram_service: IRAMService,
    ) -> None:
        self.repository = repository
        self.llm_service = llm_service
        self.retrieval_service = retrieval_service
        self.reranker = reranker
        self.vector_store = vector_store
        self.embedding_provider = embedding_provider
        self.ivm_service = ivm_service
        self.ram_service = ram_service

    async def process(self, session_id: str, message_text: str) -> StreamingResponse:
        """Process an incoming user message and yield a streaming LLM response."""
        # 1. Input validation
        await self.ivm_service.validate_prompt(message_text)

        # 2. Session housekeeping
        session = await self.repository.get_session_by_id(session_id, load_messages=True)
        if not session:
            await self.repository.create_session(session_id, message_text[:20] + "...")
            session = await self.repository.get_session_by_id(session_id, load_messages=True)
            if not session:
                raise HTTPException(status_code=500, detail="Failed to initialize session")

        if session.title == "New Chat":
            new_title = message_text[:30] + ("..." if len(message_text) > 30 else "")
            await self.repository.update_session_title(session, new_title)

        await self.repository.create_message(session_id, "user", message_text, raw_content=message_text)
        raw_history = [{"role": m.role, "content": m.raw_content if m.raw_content is not None else m.content} for m in session.messages]

        # 3. Session-document context (user-uploaded PDF)
        session_texts: List[str] = []
        try:
            if session.documents:
                session_texts = await self._retrieve_session_context(session_id, message_text)
        except Exception:
            logger.warning("Failed to retrieve session context in process_chat_message", exc_info=True)

        # 4. RAG retrieval (augmented with session-doc context)
        rag_query = message_text
        try:
            if session_texts:
                valid_texts = [str(t) for t in session_texts if t][:3]
                if valid_texts:
                    rag_context_str = "\n".join(valid_texts)
                    rag_query = f"{message_text}\n\nRelated Document Context:\n{rag_context_str}"
        except Exception:
            logger.warning("Failed to construct augmented RAG query", exc_info=True)
            rag_query = message_text

        contexts = await self._retrieve_rag_context(rag_query)

        # 5. System-prompt construction
        system_content = build_secure_system_prompt(contexts, session_texts)
        raw_history.insert(0, {"role": "system", "content": system_content})
        raw_history.append({"role": "user", "content": message_text})

        # 6 & 7. NLI streaming → output check & persist
        async def generate() -> AsyncGenerator[str, None]:
            response_content = ""

            async for chunk in nli_streaming_generate(
                raw_history=raw_history,
                contexts=contexts,
                ram_service=self.ram_service,
                llm_service=self.llm_service,
            ):
                response_content += chunk
                yield chunk

            if response_content.strip():
                # Attempt to get pruned prompt if the method exists, else use raw
                pruned_prompt = raw_history
                if hasattr(self.llm_service, "_prune_context"):
                    pruned_prompt = self.llm_service._prune_context(raw_history)
                    
                logger.info("LLM generation completed", 
                    action="LLM_GENERATION",
                    session_id=session_id,
                    model=getattr(self.llm_service, "model", "unknown"),
                    rag_context_included=len(contexts) > 0,
                    session_context_included=bool(session.documents),
                    raw_prompt=pruned_prompt,
                    generated_output=response_content,
                )

                await check_and_persist(
                    session_id=session_id,
                    initial_prompt=message_text,
                    final_output=response_content,
                    repository=self.repository,
                )

        return StreamingResponse(generate(), media_type="text/plain")

    async def _retrieve_session_context(self, session_id: str, query: str) -> List[str]:
        try:
            query_embeddings = await self.embedding_provider.embed_texts([query])
            if not query_embeddings:
                return []

            query_emb = query_embeddings[0]
            search_results = await self.vector_store.hybrid_search(
                dense_vector=query_emb.dense,
                sparse_indices=query_emb.sparse_indices,
                sparse_values=query_emb.sparse_values,
                top_k=15,
                session_id=session_id,
            )
        except Exception:
            logger.warning(f"Failed to retrieve session context for session {session_id}", exc_info=True)
            return []

        if not search_results:
            return []

        chunk_ids = [res.chunk_id for res in search_results]
        try:
            session_chunks = await self.repository.get_session_chunks_by_ids(chunk_ids)
            retrieved_texts = [str(c.text) for c in session_chunks if c.text]
        except Exception:
            logger.warning(f"Failed to fetch session chunks for session {session_id}", exc_info=True)
            return []

        if not retrieved_texts:
            return []

        try:
            ranked_results = await self.reranker.rerank(query, retrieved_texts, top_k=5)
            return [r.text for r in ranked_results] if ranked_results else []
        except Exception:
            logger.warning("Reranking session chunks failed", exc_info=True)
            return []

    async def _retrieve_rag_context(self, query: str) -> List[RetrievedContext]:
        try:
            return await self.retrieval_service.retrieve_context(query)
        except Exception as e:
            logger.warning(
                "RAG retrieval failed, proceeding without context",
                error_type=type(e).__name__,
                exc_info=True,
            )
            return []
