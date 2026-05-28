from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from sqlalchemy.orm import selectinload
from src.chat.model import Session as DBSession, Message as DBMessage, SessionDocument, SessionDocumentChunk
from typing import List

class ChatRepository:
    def __init__(self, db: AsyncSession):
        self.db = db

    async def get_all_sessions(self):
        result = await self.db.execute(select(DBSession).order_by(DBSession.created_at.asc()))
        return result.scalars().all()

    async def create_session(self, session_id: str, title: str):
        new_session = DBSession(id=session_id, title=title)
        self.db.add(new_session)
        await self.db.commit()
        await self.db.refresh(new_session)
        return new_session

    async def get_session_by_id(self, session_id: str, load_messages: bool = False):
        query = select(DBSession).where(DBSession.id == session_id)
        if load_messages:
            query = query.options(selectinload(DBSession.messages), selectinload(DBSession.documents))
        else:
            # Always load documents for the document-guard check in service.py
            query = query.options(selectinload(DBSession.documents))
        result = await self.db.execute(query)
        return result.scalars().first()

    async def update_session_title(self, session: DBSession, new_title: str):
        session.title = new_title
        await self.db.commit()
        return session

    async def create_message(self, session_id: str, role: str, content: str):
        new_msg = DBMessage(session_id=session_id, role=role, content=content)
        self.db.add(new_msg)
        await self.db.commit()
        return new_msg

    async def delete_session(self, session_id: str):
        session = await self.get_session_by_id(session_id)
        if session:
            await self.db.delete(session)
            await self.db.commit()
            return True
        return False

    async def create_session_document(self, session_id: str, filename: str, file_path: str, thumbnail: str | None = None) -> SessionDocument:
        doc = SessionDocument(session_id=session_id, filename=filename, file_path=file_path, thumbnail=thumbnail)
        self.db.add(doc)
        await self.db.commit()
        await self.db.refresh(doc)
        return doc

    async def save_session_document_chunks(self, chunks: List[SessionDocumentChunk]):
        self.db.add_all(chunks)
        await self.db.commit()

    async def get_session_document_chunks(self, session_id: str) -> List[SessionDocumentChunk]:
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
        """
        if not chunk_ids:
            return []
        query = select(SessionDocumentChunk).where(
            SessionDocumentChunk.id.in_(chunk_ids)
        )
        result = await self.db.execute(query)
        # Preserve the order from the Qdrant search results
        rows = {c.id: c for c in result.scalars().all()}
        return [rows[cid] for cid in chunk_ids if cid in rows]

    async def delete_session_document(self, document_id: str):
        query = select(SessionDocument).where(SessionDocument.id == document_id)
        result = await self.db.execute(query)
        doc = result.scalars().first()
        if doc:
            await self.db.delete(doc)
            await self.db.commit()
            return True
        return False