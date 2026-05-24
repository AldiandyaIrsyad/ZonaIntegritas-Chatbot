import uuid
from datetime import datetime, timezone
from sqlalchemy import Column, String, DateTime, Text, Boolean
from sqlalchemy.orm import relationship
from src.core.database import Base

class PDFDocument(Base):
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