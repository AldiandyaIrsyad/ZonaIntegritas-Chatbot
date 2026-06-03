"""
Service layer for the chat module.

Handles session state and user-uploaded PDF processing.
"""
import os
import uuid
from typing import List, Optional, Any

import anyio
import structlog
from fastapi import HTTPException, UploadFile

from app.core.interfaces.ai import IEmbeddingProvider
from app.core.interfaces.infra import IDocumentParser, IStorageProvider, IVectorStore, ChunkVector
from app.core.interfaces.ivm import IIVMService
from app.infra.thumbnail import ThumbnailContext
from app.rag import create_parent_chunks, split_into_children

from app.chat.model import SessionDocumentChunk
from app.chat.repository import ChatRepository

logger = structlog.get_logger(__name__)


class ChatService:
    """Core business logic for chat interactions and file uploads."""

    def __init__(
        self,
        repository: ChatRepository,
        storage: IStorageProvider,
        document_parser: IDocumentParser,
        vector_store: IVectorStore,
        embedding_provider: IEmbeddingProvider,
        ivm_service: IIVMService,
    ) -> None:
        self.repository = repository
        self.storage = storage
        self.document_parser = document_parser
        self.vector_store = vector_store
        self.embedding_provider = embedding_provider
        self.ivm_service = ivm_service
        self.thumbnail_context = ThumbnailContext()

    async def list_sessions(self) -> List[dict[str, Any]]:
        """List all available chat sessions."""
        sessions = await self.repository.get_all_sessions()
        return [{"id": s.id, "title": s.title} for s in sessions]

    async def create_new_session(self) -> dict[str, Any]:
        """Create a new, empty chat session."""
        session_id = str(uuid.uuid4())
        new_session = await self.repository.create_session(session_id, "New Chat")
        return {"id": new_session.id, "title": new_session.title}

    async def get_session_details(self, session_id: str) -> Optional[dict[str, Any]]:
        """Retrieve full details of a specific chat session."""
        session = await self.repository.get_session_by_id(session_id, load_messages=True)
        if not session:
            return None
        return {
            "title": session.title,
            "messages": [{"role": m.role, "content": m.content} for m in session.messages],
            "documents": [{"id": d.id, "filename": d.filename, "thumbnail": d.thumbnail} for d in session.documents],
        }

    async def upload_pdf(self, session_id: str, file: UploadFile) -> dict[str, Any]:
        """Upload, parse, and chunk a PDF file for a specific chat session."""
        session = await self.repository.get_session_by_id(session_id, load_messages=True)
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")

        if len(session.documents) >= 1:
            raise HTTPException(status_code=400, detail="Only one PDF is allowed per session for now.")

        _, file_extension = os.path.splitext(file.filename or "")
        file_extension = file_extension.lower()

        file_path = await self.storage.save_file(file, file_extension)

        try:
            thumbnail = await anyio.to_thread.run_sync(
                self.thumbnail_context.generate, file_path
            )
        except Exception:
            logger.error(f"Failed to generate thumbnail for {file.filename}", exc_info=True)
            await self.storage.delete_file(file_path)
            raise

        doc = await self.repository.create_session_document(
            session_id=session_id,
            filename=file.filename or "unknown",
            file_path=file_path,
            thumbnail=thumbnail,
        )

        if file_extension == ".pdf":
            try:
                elements = await self.document_parser.parse_pdf(file_path)
                parent_chunks = create_parent_chunks(elements, str(doc.id))
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
                                page=child.page,
                            )
                        )
                        chunk_idx += 1

                if chunk_models:
                    await self.repository.save_session_document_chunks(chunk_models)

                    chunk_texts = [str(c.text) for c in chunk_models]
                    embeddings = await self.embedding_provider.embed_texts(chunk_texts)

                    if embeddings:
                        await self.ivm_service.validate_document_relevance(embeddings)

                        chunk_vectors = []
                        for model, emb in zip(chunk_models, embeddings):
                            chunk_vectors.append(
                                ChunkVector(
                                    chunk_id=str(model.id),
                                    parent_chunk_id=str(doc.id),
                                    doc_id=str(doc.id),
                                    dense_vector=emb.dense,
                                    sparse_indices=emb.sparse_indices,
                                    sparse_values=emb.sparse_values,
                                    session_id=session_id,
                                )
                            )
                        await self.vector_store.upsert_chunks(chunk_vectors)
            except HTTPException:
                try:
                    await self.storage.delete_file(file_path)
                except Exception:
                    pass
                try:
                    await self.repository.delete_session_document(str(doc.id))
                except Exception:
                    pass
                raise
            except Exception as e:
                logger.error(f"Failed to parse PDF {file.filename} for session {session_id}", exc_info=True)
                try:
                    await self.storage.delete_file(file_path)
                except Exception:
                    pass
                try:
                    await self.repository.delete_session_document(str(doc.id))
                except Exception:
                    pass
                logger.error("User PDF upload failed: Exception occurred",
                    action="USER_UPLOAD_PDF",
                    session_id=session_id,
                    upload_filename=file.filename,
                    file_extension=file_extension,
                    status="failed",
                    reason=type(e).__name__,
                )
                raise HTTPException(status_code=500, detail="Failed to process PDF")

        logger.info("User PDF upload successful",
            action="USER_UPLOAD_PDF",
            session_id=session_id,
            document_id=doc.id,
            upload_filename=doc.filename,
            file_extension=file_extension,
            status="success",
        )

        return {
            "id": doc.id,
            "filename": doc.filename,
            "thumbnail": doc.thumbnail,
        }

    async def delete_session(self, session_id: str) -> bool:
        """Delete a chat session and all its associated data."""
        session = await self.repository.get_session_by_id(session_id)
        if session:
            for doc in session.documents:
                try:
                    await self.vector_store.delete_by_doc_id(doc.id)
                except Exception:
                    logger.error(f"Failed to delete Qdrant vectors for doc {doc.id}", exc_info=True)
                try:
                    await self.storage.delete_file(doc.file_path)
                except Exception:
                    logger.error(f"Failed to delete file {doc.file_path}", exc_info=True)
            return await self.repository.delete_session(session_id)
        return False
