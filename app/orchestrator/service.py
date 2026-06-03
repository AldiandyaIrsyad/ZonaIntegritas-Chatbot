"""Chat orchestrator."""

import asyncio
import uuid
from typing import AsyncGenerator, List, Callable, Awaitable, Optional, Any
import structlog

from app.core.interfaces.ai import IEmbeddingProvider, IReranker
from app.core.interfaces.infra import IVectorStore, IDocumentParser, ChunkVector
from app.core.interfaces.ivm import IIVMService
from app.core.interfaces.llm import ILLMService
from app.core.interfaces.rag import IRetrievalService, RetrievedContext
from app.core.interfaces.ram import IRAMService
from app.core.interfaces.chat import ISessionChunkProvider, ChunkData

from app.rag import create_parent_chunks, split_into_children

from app.orchestrator.prompt_builder import build_secure_system_prompt
from app.orchestrator.yield_handler import nli_streaming_generate

logger = structlog.get_logger(__name__)

class ChatOrchestrator:
    """Orchestrates IVM, RAG, System Prompt, and LLM streaming."""

    def __init__(
        self,
        llm_service: ILLMService,
        retrieval_service: IRetrievalService,
        reranker: IReranker,
        vector_store: IVectorStore,
        embedding_provider: IEmbeddingProvider,
        ivm_service: IIVMService,
        ram_service: IRAMService,
        document_parser: IDocumentParser,
    ) -> None:
        self.llm_service = llm_service
        self.retrieval_service = retrieval_service
        self.reranker = reranker
        self.vector_store = vector_store
        self.embedding_provider = embedding_provider
        self.ivm_service = ivm_service
        self.ram_service = ram_service
        self.document_parser = document_parser

    async def process_session_document(self, session_id: str, doc_id: str, file_path: str, chunk_provider: ISessionChunkProvider) -> None:
        elements = await self.document_parser.parse_pdf(file_path)
        parent_chunks = create_parent_chunks(elements, doc_id)
        
        chunk_models = []
        chunk_idx = 0
        for parent in parent_chunks:
            children = split_into_children(parent)
            for child in children:
                chunk_models.append(
                    ChunkData(
                        id=str(uuid.uuid4()),
                        text=child.text,
                        chunk_index=chunk_idx,
                        page=child.page,
                    )
                )
                chunk_idx += 1

        if chunk_models:
            await chunk_provider.save_document_chunks(doc_id, chunk_models)
            
            chunk_texts = [c.text for c in chunk_models]
            embeddings = await self.embedding_provider.embed_texts(chunk_texts)
            
            if not embeddings:
                raise ValueError("Failed to generate embeddings for all document chunks")

            await self.ivm_service.validate_document_relevance(embeddings)
            
            chunk_vectors = []
            for model, emb in zip(chunk_models, embeddings):
                chunk_vectors.append(
                    ChunkVector(
                        chunk_id=model.id,
                        parent_chunk_id=doc_id,
                        doc_id=doc_id,
                        dense_vector=emb.dense,
                        sparse_indices=emb.sparse_indices,
                        sparse_values=emb.sparse_values,
                        session_id=session_id,
                    )
                )
            await self.vector_store.upsert_chunks(chunk_vectors)

    async def delete_document_vectors(self, doc_ids: List[str]) -> None:

        async def _delete(doc_id: str) -> None:
            try:
                await self.vector_store.delete_by_doc_id(doc_id)
            except Exception:
                logger.error(f"Failed to delete Qdrant vectors for doc {doc_id}", exc_info=True)
        await asyncio.gather(*[_delete(doc_id) for doc_id in doc_ids])

    async def process(
        self, 
        session_id: str, 
        message_text: str, 
        raw_history: List[dict[str, str]],
        has_session_documents: bool,
        chunk_provider: ISessionChunkProvider,
        on_finish: Callable[[str], Awaitable[None]]
    ) -> AsyncGenerator[str, None]:
        """Process an incoming user message and yield a streaming LLM response."""
        # 1. Input validation
        await self.ivm_service.validate_prompt(message_text)

        # 2. Session-document context (user-uploaded PDF)
        session_texts: List[str] = []
        try:
            if has_session_documents:
                session_texts = await self._retrieve_session_context(session_id, message_text, chunk_provider)
        except Exception:
            logger.warning("Failed to retrieve session context in orchestrator", exc_info=True)

        # 3. RAG retrieval (augmented with session-doc context)
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

        # 4. System-prompt construction
        system_content = build_secure_system_prompt(contexts, session_texts)
        raw_history.insert(0, {"role": "system", "content": system_content})
        raw_history.append({"role": "user", "content": message_text})

        # 5. NLI streaming & Custom Yield
        async for chunk in self._generate_stream(session_id, raw_history, contexts, has_session_documents, on_finish):
            yield chunk

    async def _generate_stream(
        self, 
        session_id: str, 
        raw_history: List[dict[str, str]], 
        contexts: List[RetrievedContext],
        has_session_documents: bool,
        on_finish: Callable[[str], Awaitable[None]]
    ) -> AsyncGenerator[str, None]:
        """Custom yield that API will consume."""
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
            pruned_prompt = raw_history
            if hasattr(self.llm_service, "_prune_context"):
                pruned_prompt = self.llm_service._prune_context(raw_history)
                
            logger.info("LLM generation completed", 
                action="LLM_GENERATION",
                session_id=session_id,
                model=getattr(self.llm_service, "model", "unknown"),
                rag_context_included=len(contexts) > 0,
                session_context_included=has_session_documents,
                raw_prompt=pruned_prompt,
                generated_output=response_content,
            )

            await on_finish(response_content)

    async def _retrieve_session_context(self, session_id: str, query: str, chunk_provider: ISessionChunkProvider) -> List[str]:
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
            session_chunks = await chunk_provider.get_session_chunks_by_ids(chunk_ids)
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
