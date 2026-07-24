"""
Domain models for the Chat module.
Contains SQLAlchemy entities for database storage.

These entities are the concrete row shape persisted by
``app/chat/infra/postgres_chat_repo.py::PostgresChatRepository`` (the
``IChatRepository`` implementation) and are also referenced directly by
:mod:`app.chat.domain.interfaces` type hints — so despite the ORM coupling,
domain/application code is allowed to depend on ``Session``/``Message`` as
plain data shapes.
"""

import uuid
from datetime import datetime, timezone
from typing import List, Optional

from sqlalchemy import JSON, Column, DateTime, ForeignKey, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.shared.db import Base


class Session(Base):
    """Stores information about a chat session.

    A session is the container for one conversation thread shown in the
    sidebar; its ``messages`` are loaded on demand (see
    ``IChatRepository.get_session_by_id``'s ``load_messages`` flag) rather
    than always eager-loaded, since most session-list views only need
    ``id``/``title``.
    """
    __tablename__ = "sessions"

    # Client-generated UUID string (not a DB-generated serial) so the
    # frontend can create a session id before the first message round-trip.
    id: Mapped[str] = mapped_column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    # Sidebar display title; starts as "New Chat" and is rewritten from the
    # first user message once the session has one (see ChatService).
    title: Mapped[str] = mapped_column(String, default="New Chat")
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc)
    )

    # One-to-many, oldest-first. cascade="all, delete-orphan" means deleting
    # a Session also deletes all of its Messages (see delete_session()).
    messages = relationship(
        "Message",
        back_populates="session",
        cascade="all, delete-orphan",
        order_by="Message.created_at",
    )


class Message(Base):
    """Stores a single message within a chat session."""
    __tablename__ = "messages"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    session_id: Mapped[str] = mapped_column(String, ForeignKey("sessions.id"))
    # "user", "assistant", or "system" — see IChatRepository.create_message.
    role: Mapped[str] = mapped_column(String)
    # Original text before any post-processing. For an assistant message
    # this is the LLM's raw streamed output before citation markers were
    # appended; for a user message it's currently identical to `content`.
    # Kept distinct so future formatting changes to `content` don't lose the
    # unmodified source text.
    raw_content: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    # Final rendered text as shown in the UI — for assistant messages this
    # includes the inline "*(STATUS: SCORE; ...)*" citation markers that
    # ChatService._format_citation appends per assessed proposition.
    content: Mapped[str] = mapped_column(Text)
    # Concatenated RAG context text (all retrieved chunks joined) that was
    # used to ground this assistant message, so the "view RAG context" UI
    # panel survives a page refresh. Null for user messages.
    context: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    # JSON-serialized list of per-chunk source dicts (title/page/breadcrumbs/
    # text) paired with `context` — the richer, per-source structure the
    # chat UI uses to render individual citations rather than one text blob.
    sources: Mapped[Optional[list]] = mapped_column(JSON, nullable=True)
    # Original filename of a chat-uploaded PDF attached to this (user)
    # message, if any. Only the filename is persisted — the extracted
    # attachment text itself is single-turn and never stored (see
    # app/chat/application/attachment_service.py).
    attachment_filename: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc)
    )

    session = relationship("Session", back_populates="messages")
