"""Database repository for the chat module."""

from typing import Any, Dict, List, Optional

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from sqlalchemy.orm import selectinload

from app.chat.domain.models import Message, Session
from app.chat.domain.interfaces import IChatRepository

class PostgresChatRepository(IChatRepository):
    """Database operations for the chat domain."""
    
    def __init__(self, db: AsyncSession):
        self.db = db

    async def get_all_sessions(self) -> List[Session]:
        result = await self.db.execute(select(Session).order_by(Session.created_at.asc()))
        return list(result.scalars().all())

    async def create_session(self, session_id: str, title: str) -> Session:
        new_session = Session(id=session_id, title=title)
        self.db.add(new_session)
        await self.db.flush()
        await self.db.refresh(new_session)
        return new_session

    async def get_session_by_id(self, session_id: str, load_messages: bool = False) -> Optional[Session]:
        query = select(Session).where(Session.id == session_id)
        if load_messages:
            query = query.options(selectinload(Session.messages))
        result = await self.db.execute(query)
        return result.scalars().first()

    async def update_session_title(self, session: Session, new_title: str) -> Session:
        session.title = new_title
        await self.db.flush()
        return session

    async def create_message(
        self,
        session_id: str,
        role: str,
        content: str,
        raw_content: Optional[str] = None,
        context: Optional[str] = None,
        sources: Optional[List[Dict[str, Any]]] = None,
    ) -> Message:
        new_msg = Message(
            session_id=session_id,
            role=role,
            content=content,
            raw_content=raw_content,
            context=context,
            sources=sources,
        )
        self.db.add(new_msg)
        await self.db.flush()
        return new_msg

    async def delete_session(self, session_id: str) -> bool:
        session = await self.get_session_by_id(session_id)
        if session:
            await self.db.delete(session)
            await self.db.flush()
            return True
        return False
