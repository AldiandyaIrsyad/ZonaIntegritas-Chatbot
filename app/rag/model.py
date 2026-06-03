"""
SQLAlchemy models for the RAG pipeline.

ParentChunk stores the full-text parent chunks in PostgreSQL for LLM context.
IngestionTask tracks the async processing status of uploaded documents.
"""
import uuid
from datetime import datetime, timezone

from sqlalchemy import DateTime, ForeignKey, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.infra.db import Base


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
    page: Mapped[int | None] = mapped_column(Integer, nullable=True)
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
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc)
    )
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
