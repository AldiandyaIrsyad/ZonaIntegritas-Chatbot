"""
Database repository for the chat module.

Handles CRUD operations for sessions, messages, and session documents.
"""
from typing import List

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from sqlalchemy.orm import selectinload

from .model import Message as DBMessage
from .model import Session as DBSession
from .model import SessionDocument, SessionDocumentChunk


class ChatRepository:
    """Database operations for the chat domain."""
    
    def __init__(self, db: AsyncSession):
        self.db = db

    async def get_all_sessions(self) -> List[DBSession]:
        """Retrieve all chat sessions, ordered by creation date.

        Returns:
            List[DBSession]: List of all chat sessions.
        """
        result = await self.db.execute(select(DBSession).order_by(DBSession.created_at.asc()))
        return list(result.scalars().all())

    async def create_session(self, session_id: str, title: str) -> DBSession:
        """Create a new chat session.

        Args:
            session_id (str): UUID for the new session.
            title (str): Title for the session.

        Returns:
            DBSession: The created session object.
        """
        new_session = DBSession(id=session_id, title=title)
        self.db.add(new_session)
        await self.db.commit()
        await self.db.refresh(new_session)
        return new_session

    async def get_session_by_id(self, session_id: str, load_messages: bool = False) -> DBSession | None:
        """Fetch a chat session by its ID.

        Args:
            session_id (str): UUID of the session.
            load_messages (bool, optional): Whether to eagerly load the messages. Defaults to False.

        Returns:
            DBSession | None: The requested session, or None if not found.
        """
        query = select(DBSession).where(DBSession.id == session_id)
        if load_messages:
            query = query.options(selectinload(DBSession.messages), selectinload(DBSession.documents))
        else:
            # Always load documents for the document-guard check in service.py
            query = query.options(selectinload(DBSession.documents))
        result = await self.db.execute(query)
        return result.scalars().first()

    async def update_session_title(self, session: DBSession, new_title: str) -> DBSession:
        """Update the title of an existing session.

        Args:
            session (DBSession): The session object.
            new_title (str): The new title.

        Returns:
            DBSession: The updated session object.
        """
        session.title = new_title # type: ignore
        await self.db.commit()
        return session

    async def create_message(self, session_id: str, role: str, content: str, raw_content: str | None = None) -> DBMessage:
        """Create a new message in a session.

        Args:
            session_id (str): UUID of the session.
            role (str): Role of the message sender (user, assistant, system).
            content (str): Text content of the message.
            raw_content (str | None): Original text content without citations.

        Returns:
            DBMessage: The created message object.
        """
        new_msg = DBMessage(session_id=session_id, role=role, content=content, raw_content=raw_content)
        self.db.add(new_msg)
        await self.db.commit()
        return new_msg

    async def delete_session(self, session_id: str) -> bool:
        """Delete a chat session and its cascade contents.

        Args:
            session_id (str): UUID of the session.

        Returns:
            bool: True if the session was deleted, False if not found.
        """
        session = await self.get_session_by_id(session_id)
        if session:
            await self.db.delete(session)
            await self.db.commit()
            return True
        return False

    async def create_session_document(self, session_id: str, filename: str, file_path: str, thumbnail: str | None = None) -> SessionDocument:
        """Record an uploaded document for a session.

        Args:
            session_id (str): UUID of the session.
            filename (str): Original filename.
            file_path (str): Saved path on storage.
            thumbnail (str | None, optional): Base64 string for the thumbnail. Defaults to None.

        Returns:
            SessionDocument: The created session document object.
        """
        doc = SessionDocument(session_id=session_id, filename=filename, file_path=file_path, thumbnail=thumbnail)
        self.db.add(doc)
        await self.db.commit()
        await self.db.refresh(doc)
        return doc

    async def save_session_document_chunks(self, chunks: List[SessionDocumentChunk]) -> None:
        """Save text chunks of a session document.

        Args:
            chunks (List[SessionDocumentChunk]): List of chunk ORM instances to save.
        """
        self.db.add_all(chunks)
        await self.db.commit()

    async def get_session_document_chunks(self, session_id: str) -> List[SessionDocumentChunk]:
        """Fetch all document chunks associated with a session.

        Args:
            session_id (str): UUID of the session.

        Returns:
            List[SessionDocumentChunk]: The ordered list of chunks.
        """
        query = (
            select(SessionDocumentChunk)
            .join(SessionDocument)
            .where(SessionDocument.session_id == session_id)
            .order_by(SessionDocument.created_at, SessionDocumentChunk.chunk_index)
        )
        result = await self.db.execute(query)
        return list(result.scalars().all())

    async def get_session_chunks_by_ids(self, chunk_ids: List[str]) -> List[SessionDocumentChunk]:
        """Fetch specific session document chunks by their IDs.
        
        More efficient than loading all chunks when we already know which IDs
        we want from Qdrant search results.

        Args:
            chunk_ids (List[str]): List of chunk UUIDs to load.

        Returns:
            List[SessionDocumentChunk]: The chunks preserving requested order.
        """
        if not chunk_ids:
            return []
        query = select(SessionDocumentChunk).where(
            SessionDocumentChunk.id.in_(chunk_ids)
        )
        result = await self.db.execute(query)
        # Preserve the order from the Qdrant search results
        rows = {str(c.id): c for c in result.scalars().all()}
        return [rows[cid] for cid in chunk_ids if cid in rows]

    async def delete_session_document(self, document_id: str) -> bool:
        """Delete a single session document from the database.

        Args:
            document_id (str): UUID of the document.

        Returns:
            bool: True if deleted, False if not found.
        """
        query = select(SessionDocument).where(SessionDocument.id == document_id)
        result = await self.db.execute(query)
        doc = result.scalars().first()
        if doc:
            await self.db.delete(doc)
            await self.db.commit()
            return True
        return False