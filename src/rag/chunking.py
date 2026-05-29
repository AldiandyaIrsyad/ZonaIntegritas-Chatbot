"""
Parent-child chunking strategy for Small-to-Big retrieval.

Takes structured elements from the unstructured parser and organizes them
into a two-level hierarchy:
- Parent chunks: logical sections of the document (for LLM context)
- Child chunks: sentence-level splits of each parent (for retrieval precision)
"""
import logging
import uuid
from dataclasses import dataclass, field
from typing import List

from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.infra import ParsedElement

logger = logging.getLogger(__name__)

# Element types that indicate section boundaries
SECTION_BOUNDARY_TYPES = {"Title", "Header"}

# Default chunking parameters
DEFAULT_PARENT_MAX_CHARS = 2000
DEFAULT_CHILD_MAX_CHARS = 512
DEFAULT_CHILD_OVERLAP_CHARS = 50


@dataclass
class ParentChunkData:
    """A parent chunk ready for PostgreSQL storage."""
    id: str
    doc_id: str
    text: str
    chunk_index: int


@dataclass
class ChildChunkData:
    """A child chunk ready for embedding and Qdrant storage."""
    id: str
    parent_chunk_id: str
    doc_id: str
    text: str


def create_parent_chunks(
    elements: List[ParsedElement],
    doc_id: str,
    max_chars: int = DEFAULT_PARENT_MAX_CHARS,
) -> List[ParentChunkData]:
    """Group parsed elements into logical parent chunks.

    Uses section boundary elements (Title, Header) from the unstructured
    parser to create semantically meaningful parent chunks. Each parent
    chunk aggregates content under a section heading until the next heading
    or until the max character limit is reached.

    Args:
        elements (List[ParsedElement]): Structured elements from DocumentParser.parse_pdf().
        doc_id (str): UUID of the source PDFDocument.
        max_chars (int): Maximum character length per parent chunk.

    Returns:
        List[ParentChunkData]: Ordered list of ParentChunkData for storage in PostgreSQL.
    """
    if not elements:
        return []

    parent_chunks: List[ParentChunkData] = []
    current_texts: List[str] = []
    current_length = 0
    chunk_index = 0

    def _flush_current():
        nonlocal current_texts, current_length, chunk_index
        if not current_texts:
            return
        combined_text = "\n\n".join(current_texts).strip()
        if combined_text:
            parent_chunks.append(
                ParentChunkData(
                    id=str(uuid.uuid4()),
                    doc_id=doc_id,
                    text=combined_text,
                    chunk_index=chunk_index,
                )
            )
            chunk_index += 1
        current_texts = []
        current_length = 0

    for element in elements:
        text = element.text.strip()
        if not text:
            continue

        # Start a new parent chunk at section boundaries
        if element.element_type in SECTION_BOUNDARY_TYPES and current_texts:
            _flush_current()

        # If adding this element would exceed the limit, flush first
        if current_length + len(text) > max_chars and current_texts:
            _flush_current()

        current_texts.append(text)
        current_length += len(text)

    # Don't forget the last accumulated chunk
    _flush_current()

    logger.info(
        "Created %d parent chunks from %d elements for doc_id='%s'",
        len(parent_chunks),
        len(elements),
        doc_id,
    )
    return parent_chunks


def split_into_children(
    parent: ParentChunkData,
    max_chars: int = DEFAULT_CHILD_MAX_CHARS,
    overlap_chars: int = DEFAULT_CHILD_OVERLAP_CHARS,
) -> List[ChildChunkData]:
    """Split a parent chunk into sentence-level child chunks.

    Uses RecursiveCharacterTextSplitter from langchain for intelligent
    splitting that respects sentence and word boundaries.

    Args:
        parent (ParentChunkData): The parent chunk to split.
        max_chars (int): Maximum characters per child chunk.
        overlap_chars (int): Overlap between consecutive child chunks.

    Returns:
        List[ChildChunkData]: List of ChildChunkData, each referencing its parent.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=max_chars,
        chunk_overlap=overlap_chars,
        separators=["\n\n", "\n", ". ", "? ", "! ", "; ", ", ", " ", ""],
        length_function=len,
    )

    child_texts = splitter.split_text(parent.text)

    children = []
    for text in child_texts:
        text = text.strip()
        if not text:
            continue
        children.append(
            ChildChunkData(
                id=str(uuid.uuid4()),
                parent_chunk_id=parent.id,
                doc_id=parent.doc_id,
                text=text,
            )
        )

    logger.debug(
        "Split parent chunk '%s' into %d children", parent.id, len(children)
    )
    return children
