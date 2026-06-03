"""
Service layer for the chat module.

Handles session state and user-uploaded PDF processing.
"""
import os
import uuid
from typing import List, Optional, Any, AsyncGenerator

import anyio
import structlog
from fastapi import HTTPException, UploadFile

from app.core.interfaces.infra import IStorageProvider
from app.infra.thumbnail import ThumbnailContext

from app.chat.model import SessionDocumentChunk
from app.chat.repository import ChatRepository
from app.chat.output_checker import check_and_persist

from app.orchestrator.service import ChatOrchestrator

logger = structlog.get_logger(__name__)


class ChatService:
    """Core business logic for chat interactions and file uploads."""

    def __init__(
        self,
        repository: ChatRepository,
        storage: IStorageProvider,
    ) -> None:
        self.repository = repository
        self.storage = storage
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

    async def upload_pdf(self, session_id: str, file: UploadFile, orchestrator: "ChatOrchestrator") -> dict[str, Any]:
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
                await orchestrator.process_session_document(session_id, str(doc.id), file_path, self.repository)
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
                logger.error(f"Failed to process PDF {file.filename} for session {session_id}", exc_info=True)
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

    async def delete_session(self, session_id: str, orchestrator: "ChatOrchestrator") -> bool:
        """Delete a chat session and all its associated data."""
        session = await self.repository.get_session_by_id(session_id)
        if session:
            if session.documents:
                await orchestrator.delete_document_vectors([str(doc.id) for doc in session.documents])
            for doc in session.documents:
                try:
                    await self.storage.delete_file(doc.file_path)
                except Exception:
                    logger.error(f"Failed to delete file {doc.file_path}", exc_info=True)
            return await self.repository.delete_session(session_id)
        return False

    async def process_chat_message(self, session_id: str, message_text: str, orchestrator: "ChatOrchestrator") -> AsyncGenerator[str, None]:
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
        
        has_session_documents = bool(session.documents)

        async def on_finish(final_output: str) -> None:
            await check_and_persist(
                session_id=session_id,
                initial_prompt=message_text,
                final_output=final_output,
                repository=self.repository,
            )

        stream = orchestrator.process(
            session_id=session_id,
            message_text=message_text,
            raw_history=raw_history,
            has_session_documents=has_session_documents,
            chunk_provider=self.repository,
            on_finish=on_finish
        )
        async for chunk in stream:
            yield chunk
