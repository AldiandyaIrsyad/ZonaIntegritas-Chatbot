"""
Parent-child chunking strategy for Small-to-Big retrieval.

Takes structured elements from the unstructured parser and organizes them
into a two-level hierarchy:
- Parent chunks: logical sections of the document (for LLM context)
- Child chunks: sentence-level splits of each parent (for retrieval precision)
"""
import uuid
import structlog
from typing import List, Optional

from langchain_text_splitters import RecursiveCharacterTextSplitter

from .models import ChildChunkData, ParentChunkData, ParsedElement

logger = structlog.get_logger(__name__)

# Element types that indicate section boundaries
SECTION_BOUNDARY_TYPES = {"Title"}

# Element types to ignore (noise, page numbers, repeating headers/footers)
IGNORE_ELEMENT_TYPES = {"Header", "Footer", "PageNumber"}

# Default chunking parameters
DEFAULT_PARENT_MAX_CHARS = 2000
DEFAULT_CHILD_MAX_CHARS = 512
DEFAULT_CHILD_OVERLAP_CHARS = 50


def create_parent_chunks(
    elements: List[ParsedElement],
    doc_id: str,
    max_chars: int = DEFAULT_PARENT_MAX_CHARS,
) -> List[ParentChunkData]:
    """Group parsed elements into logical parent chunks.

    Uses section boundary elements (Title) from the unstructured
    parser to create semantically meaningful parent chunks. Each parent
    chunk aggregates content under a section heading until the next heading
    or until the max character limit is reached.

    Consecutive headers are grouped together, and page artifacts are ignored.

    Args:
        elements (List[ParsedElement]): Structured elements.
        doc_id (str): UUID of the source document.
        max_chars (int): Maximum character length per parent chunk.

    Returns:
        List[ParentChunkData]: Ordered list of ParentChunkData.
    """
    if not elements:
        return []

    parent_chunks: List[ParentChunkData] = []
    current_texts: List[str] = []
    current_length = 0
    chunk_index = 0
    current_page: Optional[int] = None
    has_body_text = False

    def _flush_current() -> None:
        nonlocal current_texts, current_length, chunk_index, current_page, has_body_text
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
                    page=current_page,
                )
            )
            chunk_index += 1
        current_texts = []
        current_length = 0
        current_page = None
        has_body_text = False

    for element in elements:
        text = element.text.strip()
        if not text:
            continue

        # Ignore noisy elements like page headers/footers
        if element.element_type in IGNORE_ELEMENT_TYPES:
            continue

        # Treat as boundary only if it's a Title and has substantial text
        # (filters out 1-letter artifacts like bullets misclassified as Titles)
        is_boundary = element.element_type in SECTION_BOUNDARY_TYPES and len(text) > 3

        # Start a new parent chunk at section boundaries, but only if we already
        # have body text in the current chunk. This prevents consecutive headers
        # from being split into separate tiny chunks.
        if is_boundary and has_body_text:
            _flush_current()

        # If adding this element would exceed the limit, flush first
        if current_length + len(text) > max_chars and current_texts:
            _flush_current()

        current_texts.append(text)
        current_length += len(text)
        
        if current_page is None:
            current_page = element.metadata.get("page_number")
            
        if not is_boundary:
            has_body_text = True

    # Don't forget the last accumulated chunk
    _flush_current()

    logger.info(
        "thesis.chunking.parents_created",
        parent_count=len(parent_chunks),
        element_count=len(elements),
        doc_id=doc_id,
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
                page=parent.page,
            )
        )

    logger.debug(
        "thesis.chunking.children_created", parent_id=parent.id, child_count=len(children)
    )
    return children
