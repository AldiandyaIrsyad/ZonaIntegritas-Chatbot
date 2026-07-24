"""Data models for the hierarchical chunking pipeline.

Defines the data structures that flow through the chunking pipeline:
- ParsedElement: output of the document parser (unstructured.io)
- ParentChunkData: logical sections of the document (for LLM context)
- ChildChunkData: sentence-level splits of each parent (for retrieval precision)

Each model carries a ``content_type`` field so downstream components
(embedding, vector store, retrieval, citation) can distinguish narrative
text from tables and VLM-enriched figure descriptions.

Pure ``pydantic`` data models, no infra imports — part of the ``thesis/``
research core purity rule (see ``docs/02-arsitektur.md`` §2.2).
"""

from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel


class ContentType(str, Enum):
    """The structural type of a chunk's content.

    Attributes:
        TEXT: Narrative prose — standard paragraph text.
        TABLE: HTML/Markdown table — must not be character-split.
        FIGURE: VLM-generated description of a visual element (flowchart,
            annotated screenshot, diagram).
        HYBRID: Mixed content — a parent chunk containing both text and
            table/figure elements.
    """

    TEXT = "text"
    TABLE = "table"
    FIGURE = "figure"
    HYBRID = "hybrid"


class ParsedElement(BaseModel):
    """An element parsed from an unstructured document.

    Attributes:
        element_type: The type assigned by the parser (e.g. "Title",
            "NarrativeText", "Table", "Image").
        text: The text content. For tables, this may be HTML. For images,
            this may be empty until VLM enrichment.
        metadata: Parser metadata (page_number, category_depth,
            text_as_html, image_path, etc.).
        content_type: The classified content type for routing. Defaults
            to TEXT; set by the content-type router before chunking.
    """

    element_type: str
    text: str
    metadata: Dict[str, Any] = {}
    content_type: ContentType = ContentType.TEXT


class ParentChunkData(BaseModel):
    """A parent chunk representing a logical section of the document.

    Parent chunks are the unit of context returned to the LLM at query
    time (Small-to-Big retrieval). They preserve the full text of a
    section, including tables and figure descriptions.

    Attributes:
        id: Unique UUID for this chunk.
        doc_id: UUID of the source document.
        text: The full text content (may include HTML tables, VLM
            descriptions). Pure body text — no inline breadcrumb tag;
            see ``breadcrumbs`` for the structured hierarchical path.
        chunk_index: Sequential index within the document.
        page: Page number of the first element in this chunk.
        breadcrumbs: Hierarchical section path (e.g. ["Chapter 1", "Section 1.1"]).
        content_type: The structural type of this chunk's content.
        element_metadata: Preserved metadata from the source element
            (e.g. raw HTML for tables, image path for figures).
        parent_id: UUID of the parent section's representative chunk
            (None for root sections).
        ordinal: Position within the parent section (uses chunk_index).
        path: ltree-style dot path (e.g., "doc_id.bab_i.pasal_5").
        depth: Heading depth (0 = root, higher = deeper).
    """

    id: str
    doc_id: str
    text: str
    chunk_index: int
    page: Optional[int] = None
    breadcrumbs: List[str] = []
    content_type: ContentType = ContentType.TEXT
    element_metadata: Dict[str, Any] = {}

    # ltree hierarchy fields
    parent_id: Optional[str] = None
    ordinal: int = 0
    path: str = ""
    depth: int = 0


class ChildChunkData(BaseModel):
    """A child chunk representing a sentence-level split of a parent chunk.

    Child chunks are the unit of vector search. They are embedded and
    stored in Qdrant. When a child matches a query, its parent's full
    text is retrieved for LLM context.

    Attributes:
        id: Unique UUID for this chunk.
        parent_chunk_id: UUID of the parent chunk this child belongs to.
        doc_id: UUID of the source document.
        text: The child text, prefixed with a breadcrumb tag (e.g.
            "BAB II > Pasal 5\n\n") when the parent has breadcrumbs —
            embedding-only, never displayed to the user or LLM.
        page: Page number inherited from the parent.
        breadcrumbs: Hierarchical section path inherited from the parent.
        content_type: The structural type inherited from the parent.
        ordinal: Position within the parent chunk.
        path: ltree-style path (parent path + ".c" + ordinal).
    """

    id: str
    parent_chunk_id: str
    doc_id: str
    text: str
    page: Optional[int] = None
    breadcrumbs: List[str] = []
    content_type: ContentType = ContentType.TEXT

    # ltree hierarchy fields
    ordinal: int = 0
    path: str = ""
