"""
Domain models for the Knowledge Base.
Contains SQLAlchemy entities for database storage and Pydantic models for data transfer.
"""

import uuid
from datetime import datetime, timezone
from typing import List, Optional

from pydantic import BaseModel
from sqlalchemy import Boolean, Column, DateTime, ForeignKey, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.shared.db import Base

class PDFDocument(Base):
    """Stores metadata and ingestion status for a global knowledge base document."""
    __tablename__ = "pdf_documents"
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    title = Column(String)
    description = Column(Text)
    pdf_path = Column(String)
    active = Column(Boolean, default=True)
    ingestion_status = Column(String, default="pending")
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))

    parent_chunks = relationship(
        "ParentChunk",
        back_populates="document",
        cascade="all, delete-orphan",
    )


class ParentChunk(Base):
    """Stores full-text parent chunks for the Small-to-Big retrieval strategy.

    Child chunks (sentence-level) are indexed in Qdrant for retrieval precision.
    When a child chunk matches a query, the corresponding parent chunk's full
    text is fetched from this table to provide broader context to the LLM.
    """
    __tablename__ = "parent_chunks"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    doc_id: Mapped[str] = mapped_column(
        String,
        ForeignKey("pdf_documents.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    text: Mapped[str] = mapped_column(Text, nullable=False)
    chunk_index: Mapped[int] = mapped_column(Integer, nullable=False)
    page: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc)
    )

    document = relationship("PDFDocument", back_populates="parent_chunks")


class IngestionTask(Base):
    """Tracks the async processing status of a PDF document's ingestion.

    States: pending → processing → completed | failed
    """
    __tablename__ = "ingestion_tasks"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    doc_id: Mapped[str] = mapped_column(
        String,
        ForeignKey("pdf_documents.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    status: Mapped[str] = mapped_column(String, default="pending", nullable=False)
    error_message: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc)
    )
    completed_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)


class RetrievedContext(BaseModel):
    """A clean domain object representing a context retrieved from the Knowledge Base."""
    chunk_id: str
    parent_chunk_id: str
    doc_id: str
    text: str
    score: float
    source_title: str = ""
    page: Optional[int] = None
    dense_vector: Optional[List[float]] = None
    
