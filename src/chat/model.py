import uuid
from datetime import datetime, timezone
from sqlalchemy import Column, String, DateTime, ForeignKey, Text, Integer
from sqlalchemy.orm import relationship
from src.core.database import Base

class Session(Base):
    __tablename__ = "sessions"
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    title = Column(String, default="New Chat")
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    
    messages = relationship(
        "Message", 
        back_populates="session", 
        cascade="all, delete-orphan", 
        order_by="Message.created_at"
    )

    documents = relationship(
        "SessionDocument",
        back_populates="session",
        cascade="all, delete-orphan",
        order_by="SessionDocument.created_at"
    )

class Message(Base):
    __tablename__ = "messages"
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    session_id = Column(String, ForeignKey("sessions.id"))
    role = Column(String) 
    content = Column(Text)
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    
    session = relationship("Session", back_populates="messages")

class SessionDocument(Base):
    __tablename__ = "session_documents"
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    session_id = Column(String, ForeignKey("sessions.id"))
    filename = Column(String)
    file_path = Column(String)
    thumbnail = Column(Text, nullable=True)
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))

    session = relationship("Session", back_populates="documents")
    chunks = relationship(
        "SessionDocumentChunk",
        back_populates="document",
        cascade="all, delete-orphan",
        order_by="SessionDocumentChunk.chunk_index"
    )

class SessionDocumentChunk(Base):
    __tablename__ = "session_document_chunks"
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    session_document_id = Column(String, ForeignKey("session_documents.id"))
    text = Column(Text)
    chunk_index = Column(Integer)

    document = relationship("SessionDocument", back_populates="chunks")