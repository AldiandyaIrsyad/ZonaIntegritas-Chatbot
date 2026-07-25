"""Postgres repository adapter for the chat bounded context.

Persists :class:`Session` and :class:`Message` ORM entities via the shared
async SQLAlchemy session from ``app/shared/db.py``. Fulfills
``app/chat/domain/interfaces.py::IChatRepository``; wired in
``app/chat/dependency.py::get_chat_repo``.
"""

from typing import Any, Dict, List, Optional

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from sqlalchemy.orm import selectinload

from app.chat.domain.models import Message, Session
from app.chat.domain.interfaces import IChatRepository


class PostgresChatRepository(IChatRepository):
    """Database operations for the chat domain."""

    def __init__(self, db: AsyncSession):
        """Wrap a request-scoped async session. Commit/rollback is managed by
        the caller's request scope, not here.
        """
        self.db = db

    async def get_all_sessions(self) -> List[Session]:
        """Return all sessions ordered oldest-first; does not eager-load
        messages (each ``Session.messages`` stays a lazy relationship)."""
        result = await self.db.execute(select(Session).order_by(Session.created_at.asc()))
        return list(result.scalars().all())

    async def create_session(self, session_id: str, title: str) -> Session:
        """Insert a new session row and flush + refresh it so server-side
        defaults (e.g. ``created_at``) are populated on the returned
        instance. Does not commit — the caller's request-scoped session
        handles the transaction boundary."""
        new_session = Session(id=session_id, title=title)
        self.db.add(new_session)
        await self.db.flush()
        await self.db.refresh(new_session)
        return new_session

    async def get_session_by_id(self, session_id: str, load_messages: bool = False) -> Optional[Session]:
        """Fetch a session by ID. When ``load_messages`` is True, eager-loads
        ``Session.messages`` via ``selectinload`` (a separate SELECT) to
        avoid a lazy-load in async context, which would otherwise raise a
        ``greenlet_spawn`` error once the session detaches."""
        query = select(Session).where(Session.id == session_id)
        if load_messages:
            query = query.options(selectinload(Session.messages))
        result = await self.db.execute(query)
        return result.scalars().first()

    async def update_session_title(self, session: Session, new_title: str) -> Session:
        """Rename an already-loaded session in place and flush the change."""
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
        attachment_filename: Optional[str] = None,
    ) -> Message:
        """Insert a new message row and flush (see ``IChatRepository`` for
        the meaning of each field)."""
        new_msg = Message(
            session_id=session_id,
            role=role,
            content=content,
            raw_content=raw_content,
            context=context,
            sources=sources,
            attachment_filename=attachment_filename,
        )
        self.db.add(new_msg)
        await self.db.flush()
        return new_msg

    async def delete_session(self, session_id: str) -> bool:
        """Delete a session (and, via the ORM's ``cascade="all,
        delete-orphan"`` on ``Session.messages``, all of its messages).
        Returns False without raising if no such session exists."""
        session = await self.get_session_by_id(session_id)
        if session:
            await self.db.delete(session)
            await self.db.flush()
            return True
        return False
