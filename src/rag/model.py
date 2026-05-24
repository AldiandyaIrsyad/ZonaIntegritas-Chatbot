"""
SQLAlchemy models for the RAG pipeline.

ParentChunk stores the full-text parent chunks in PostgreSQL for LLM context.
IngestionTask tracks the async processing status of uploaded documents.
"""
import uuid
from datetime import datetime, timezone

from sqlalchemy import Column, DateTime, ForeignKey, Integer, String, Text
from sqlalchemy.orm import relationship

from src.core.database import Base


class ParentChunk(Base):
    """Stores full-text parent chunks for the Small-to-Big retrieval strategy.

    Child chunks (sentence-level) are indexed in Qdrant for retrieval precision.
    When a child chunk matches a query, the corresponding parent chunk's full
    text is fetched from this table to provide broader context to the LLM.
    """
    __tablename__ = "parent_chunks"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    doc_id = Column(
        String,
        ForeignKey("pdf_documents.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    text = Column(Text, nullable=False)
    chunk_index = Column(Integer, nullable=False)
    created_at = Column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc)
    )

    document = relationship("PDFDocument", back_populates="parent_chunks")


class IngestionTask(Base):
    """Tracks the async processing status of a PDF document's ingestion.

    States: pending → processing → completed | failed
    """
    __tablename__ = "ingestion_tasks"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    doc_id = Column(
        String,
        ForeignKey("pdf_documents.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    status = Column(String, default="pending", nullable=False)
    error_message = Column(Text, nullable=True)
    created_at = Column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc)
    )
    completed_at = Column(DateTime(timezone=True), nullable=True)
