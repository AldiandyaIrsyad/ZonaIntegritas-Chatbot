"""
SQLAlchemy models for the knowledge base module.

Defines schemas for globally available PDF documents used in RAG.
"""
import uuid
from datetime import datetime, timezone

from sqlalchemy import Boolean, Column, DateTime, String, Text
from sqlalchemy.orm import relationship

from app.infra.db import Base


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