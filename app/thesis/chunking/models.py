from typing import Any, Dict, Optional

from pydantic import BaseModel


class ParsedElement(BaseModel):
    """An element parsed from an unstructured document."""
    element_type: str
    text: str
    metadata: Dict[str, Any] = {}


class ParentChunkData(BaseModel):
    """A parent chunk representing a logical section of the document."""
    id: str
    doc_id: str
    text: str
    chunk_index: int
    page: Optional[int] = None


class ChildChunkData(BaseModel):
    """A child chunk representing a sentence-level split of a parent chunk."""
    id: str
    parent_chunk_id: str
    doc_id: str
    text: str
    page: Optional[int] = None
