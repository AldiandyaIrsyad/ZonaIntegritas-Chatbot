"""
Service layer for the chat module.

Orchestrates LLM communication, RAG retrieval, NLI validation, and session state.
The three LLM-pipeline stages are delegated to focused sub-modules:

- :mod:`src.chat.prompt_builder`  — salted system-prompt construction
- :mod:`src.chat.yield_handler`   — NLI sentence-buffered streaming
- :mod:`src.chat.output_checker`  — output validation and DB persistence
"""
import os
import uuid
from typing import AsyncGenerator, List, Optional

import anyio
from fastapi import HTTPException, UploadFile
from fastapi.responses import StreamingResponse

from src.core import LogEvent, get_logger
from src.infra import (
    ChunkVector,
    DocumentParser,
    EmbeddingProvider,
    QdrantStore,
    Reranker,
    StorageProvider,
)
from src.infra.thumbnail import ThumbnailContext
from src.llm import LLMService
from src.rag import (
    RetrievalService,
    RetrievedContext,
    create_parent_chunks,
    split_into_children,
)
from src.ivm.service import IVMService
from src.ram.service import RAMService

from .model import SessionDocumentChunk
from .repository import ChatRepository
from .prompt_builder import build_secure_system_prompt
from .yield_handler import nli_streaming_generate
from .output_checker import check_and_persist

logger = get_logger(__name__)


class ChatService:
    """Core business logic for chat interactions.

    Acts as the orchestrator for the full LLM pipeline:
    input validation → RAG retrieval → prompt construction →
    NLI-annotated streaming → output checking & persistence.
    """

    def __init__(
        self,
        repository: ChatRepository,
        llm_service: LLMService,
        retrieval_service: RetrievalService,
        storage: StorageProvider,
        document_parser: DocumentParser,
        reranker: Reranker,
        vector_store: QdrantStore,
        embedding_provider: EmbeddingProvider,
        ivm_service: IVMService,
        ram_service: RAMService,
    ) -> None:
        self.repository = repository
        self.llm_service = llm_service
        self.retrieval_service = retrieval_service
        self.storage = storage
        self.document_parser = document_parser
        self.reranker = reranker
        self.vector_store = vector_store
        self.embedding_provider = embedding_provider
        self.ivm_service = ivm_service
        self.ram_service = ram_service
        self.thumbnail_context = ThumbnailContext()

    async def list_sessions(self) -> List[dict]:
        """List all available chat sessions.

        Returns:
            List[dict]: A list of session summaries.
        """
        sessions = await self.repository.get_all_sessions()
        return [{"id": s.id, "title": s.title} for s in sessions]

    async def create_new_session(self) -> dict:
        """Create a new, empty chat session.

        Returns:
            dict: The newly created session details (id, title).
        """
        session_id = str(uuid.uuid4())
        new_session = await self.repository.create_session(session_id, "New Chat")
        return {"id": new_session.id, "title": new_session.title}

    async def get_session_details(self, session_id: str) -> Optional[dict]:
        """Retrieve full details of a specific chat session.

        Args:
            session_id (str): UUID of the session.

        Returns:
            Optional[dict]: Dictionary with session messages and documents, or None if not found.
        """
        session = await self.repository.get_session_by_id(session_id, load_messages=True)
        if not session:
            return None
        return {
            "title": session.title,
            "messages": [{"role": m.role, "content": m.content} for m in session.messages],
            "documents": [{"id": d.id, "filename": d.filename, "thumbnail": d.thumbnail} for d in session.documents],
        }

    async def upload_pdf(self, session_id: str, file: UploadFile) -> dict:
        """Upload, parse, and chunk a PDF file for a specific chat session.

        Requires the session to already exist. Call POST /api/sessions first
        if you need to create one.

        Args:
            session_id (str): UUID of the session.
            file (UploadFile): The file to upload.

        Returns:
            dict: Metadata about the uploaded document (id, filename, thumbnail).

        Raises:
            HTTPException: If the session doesn't exist, file upload fails, or if max docs reached.
        """
        session = await self.repository.get_session_by_id(session_id, load_messages=True)
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")

        if len(session.documents) >= 1:
            raise HTTPException(status_code=400, detail="Only one PDF is allowed per session for now.")

        # Determine extension from the original filename robustly.
        _, file_extension = os.path.splitext(file.filename or "")
        file_extension = file_extension.lower()

        file_path = await self.storage.save_file(file, file_extension)

        # Generate thumbnail
        try:
            thumbnail = await anyio.to_thread.run_sync(
                self.thumbnail_context.generate_thumbnail, file_path
            )
        except Exception:
            logger.error(f"Failed to generate thumbnail for {file.filename}", exc_info=True)
            await self.storage.delete_file(file_path)
            raise

        # Create session document
        doc = await self.repository.create_session_document(
            session_id=session_id,
            filename=file.filename,
            file_path=file_path,
            thumbnail=thumbnail,
        )

        # Parse and chunk
        if file_extension == ".pdf":
            try:
                elements = await self.document_parser.parse_pdf(file_path)
                parent_chunks = create_parent_chunks(elements, doc.id)
                chunk_models = []
                chunk_idx = 0
                for parent in parent_chunks:
                    children = split_into_children(parent)
                    for child in children:
                        chunk_models.append(
                            SessionDocumentChunk(
                                id=str(uuid.uuid4()),
                                session_document_id=doc.id,
                                text=child.text,
                                chunk_index=chunk_idx,
                            )
                        )
                        chunk_idx += 1

                if chunk_models:
                    await self.repository.save_session_document_chunks(chunk_models)

                    # Embed and store in Qdrant for Session RAG
                    chunk_texts = [c.text for c in chunk_models]
                    embeddings = await self.embedding_provider.embed_texts(chunk_texts)

                    if embeddings:
                        # Validate the document's relevance before saving its vectors
                        await self.ivm_service.validate_document_relevance(embeddings)

                        chunk_vectors = []
                        for model, emb in zip(chunk_models, embeddings):
                            chunk_vectors.append(
                                ChunkVector(
                                    chunk_id=model.id,
                                    parent_chunk_id=doc.id,  # flat: use doc_id as parent
                                    doc_id=doc.id,
                                    dense_vector=emb.dense,
                                    sparse_indices=emb.sparse_indices,
                                    sparse_values=emb.sparse_values,
                                    session_id=session_id,
                                )
                            )
                        await self.vector_store.upsert_chunks(chunk_vectors)
            except HTTPException:
                # Cleanup on relevance validation failure before re-raising
                try:
                    await self.storage.delete_file(file_path)
                except Exception:
                    pass
                try:
                    await self.repository.delete_session_document(doc.id)
                except Exception:
                    pass
                raise
            except Exception as e:
                logger.error(f"Failed to parse PDF {file.filename} for session {session_id}", exc_info=True)

                try:
                    await self.storage.delete_file(file_path)
                except Exception:
                    logger.error(f"Failed to delete failed upload {file.filename} for session {session_id}", exc_info=True)

                try:
                    await self.repository.delete_session_document(doc.id)
                except Exception:
                    logger.error(f"Failed to delete session document DB record {doc.id} for session {session_id}", exc_info=True)

                logger.error("User PDF upload failed: Exception occurred", exc_info=True, extra={
                    "event": LogEvent.USER_UPLOAD_PDF.value,
                    "session_id": session_id,
                    "upload_filename": file.filename,
                    "file_extension": file_extension,
                    "status": "failed",
                    "reason": type(e).__name__,
                })
                raise HTTPException(status_code=500, detail="Failed to process PDF")

        logger.info("User PDF upload successful", extra={
            "event": LogEvent.USER_UPLOAD_PDF.value,
            "session_id": session_id,
            "document_id": doc.id,
            "upload_filename": doc.filename,
            "file_extension": file_extension,
            "status": "success",
        })

        return {
            "id": doc.id,
            "filename": doc.filename,
            "thumbnail": doc.thumbnail,
        }

    async def process_chat_message(self, session_id: str, message_text: str) -> StreamingResponse:
        """Process an incoming user message and yield a streaming LLM response.

        Pipeline:
            1. IVM prompt validation (malicious-prompt / relevance check).
            2. Session lookup / creation and title update.
            3. Session-document context retrieval (if a PDF was uploaded).
            4. RAG knowledge-base retrieval using the augmented query.
            5. Salted system-prompt construction via :func:`.prompt_builder.build_secure_system_prompt`.
            6. NLI-annotated streaming via :func:`.yield_handler.nli_streaming_generate`.
            7. Output checking and DB persistence via :func:`.output_checker.check_and_persist`.

        Args:
            session_id (str): UUID of the chat session.
            message_text (str): The raw text of the user's message.

        Returns:
            StreamingResponse: Text stream of the NLI-annotated LLM response.
        """
        # ── 1. Input validation ──────────────────────────────────────
        await self.ivm_service.validate_prompt(message_text)

        # ── 2. Session housekeeping ──────────────────────────────────
        session = await self.repository.get_session_by_id(session_id, load_messages=True)
        if not session:
            session = await self.repository.create_session(session_id, message_text[:20] + "...")

        if session.title == "New Chat":
            new_title = message_text[:30] + ("..." if len(message_text) > 30 else "")
            await self.repository.update_session_title(session, new_title)

        await self.repository.create_message(session_id, "user", message_text)
        raw_history = [{"role": m.role, "content": m.content} for m in session.messages]

        # ── 3. Session-document context (user-uploaded PDF) ──────────
        session_texts: List[str] = []
        try:
            if session.documents:
                session_texts = await self._retrieve_session_context(session_id, message_text)
        except Exception:
            logger.warning("Failed to retrieve session context in process_chat_message", exc_info=True)

        # ── 4. RAG retrieval (augmented with session-doc context) ────
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

        # ── 5. System-prompt construction ────────────────────────────
        system_content = build_secure_system_prompt(contexts, session_texts)
        raw_history.insert(0, {"role": "system", "content": system_content})
        raw_history.append({"role": "user", "content": message_text})

        # ── 6 & 7. NLI streaming → output check & persist ────────────
        async def generate() -> AsyncGenerator[str, None]:
            """Drive NLI streaming and persist the completed response."""
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
                pruned_prompt = self.llm_service._prune_context(raw_history)
                logger.info("LLM generation completed", extra={
                    "event": LogEvent.LLM_GENERATION.value,
                    "session_id": session_id,
                    "model": getattr(self.llm_service, "model", "unknown"),
                    "rag_context_included": len(contexts) > 0,
                    "session_context_included": bool(session.documents),
                    "raw_prompt": pruned_prompt,
                    "generated_output": response_content,
                })

                await check_and_persist(
                    session_id=session_id,
                    initial_prompt=message_text,
                    final_output=response_content,
                    repository=self.repository,
                )

        return StreamingResponse(generate(), media_type="text/plain")

    async def _retrieve_session_context(self, session_id: str, query: str) -> List[str]:
        """Retrieve and rerank session-document chunks, returning raw texts.

        Returns an empty list on any failure so that the caller (and the
        prompt builder) never has to worry about None handling.

        Args:
            session_id (str): UUID of the chat session.
            query (str): The user's prompt query.

        Returns:
            List[str]: A list of retrieved session context strings.
        """
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
            retrieved_texts = [c.text for c in session_chunks if c.text]
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

    async def delete_session(self, session_id: str) -> bool:
        """Delete a chat session and all its associated data.

        Args:
            session_id (str): UUID of the session to delete.

        Returns:
            bool: True if the session was successfully deleted, False otherwise.
        """
        session = await self.repository.get_session_by_id(session_id)
        if session:
            # Clean up associated session documents (files and vector stores)
            for doc in session.documents:
                # 1. Delete vectors from Qdrant
                try:
                    await self.vector_store.delete_by_doc_id(doc.id)
                except Exception:
                    logger.error(f"Failed to delete Qdrant vectors for doc {doc.id}", exc_info=True)

                # 2. Delete file from storage
                try:
                    await self.storage.delete_file(doc.file_path)
                except Exception:
                    logger.error(f"Failed to delete file {doc.file_path}", exc_info=True)

            # 3. Delete session from PostgreSQL (will cascade delete documents and messages)
            return await self.repository.delete_session(session_id)
        return False

    async def _retrieve_rag_context(self, query: str) -> List[RetrievedContext]:
        """Attempt to retrieve RAG context; return empty list on failure.

        RAG failures should not break the chat — the LLM can still
        respond without knowledge base context.

        Args:
            query (str): The user's query string for retrieval.

        Returns:
            List[RetrievedContext]: The successfully retrieved chunks, or empty list.
        """
        try:
            return await self.retrieval_service.retrieve_context(query)
        except Exception as e:
            logger.warning(
                "RAG retrieval failed, proceeding without context: %s",
                type(e).__name__,
                exc_info=True,
            )
            return []
