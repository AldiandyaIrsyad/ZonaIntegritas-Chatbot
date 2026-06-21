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
    Maintains a heading stack to track the hierarchical path (breadcrumbs) of the document.

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
    
    # Track the hierarchical path: list of (depth, title_text)
    heading_stack: List[tuple[int, str]] = []
    current_breadcrumbs: List[str] = []

    def _flush_current() -> None:
        nonlocal current_texts, current_length, chunk_index, current_page, has_body_text
        if not current_texts:
            return
            
        combined_text = "\n\n".join(current_texts).strip()
        
        # Prepend breadcrumbs to the parent text so the LLM gets the context
        if current_breadcrumbs and combined_text:
            context_header = f"[Context: {' > '.join(current_breadcrumbs)}]\n\n"
            combined_text = context_header + combined_text
            
        if combined_text:
            parent_chunks.append(
                ParentChunkData(
                    id=str(uuid.uuid4()),
                    doc_id=doc_id,
                    text=combined_text,
                    chunk_index=chunk_index,
                    page=current_page,
                    breadcrumbs=list(current_breadcrumbs),
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

        if is_boundary:
            # Start a new parent chunk at section boundaries, but only if we already
            # have body text in the current chunk. This prevents consecutive headers
            # from being split into separate tiny chunks.
            if has_body_text:
                _flush_current()

            # Update heading stack AFTER flushing the previous section
            depth = element.metadata.get("category_depth") or 0
            
            # Pop elements from stack that are at the same or deeper level
            while heading_stack and heading_stack[-1][0] >= depth:
                heading_stack.pop()
                
            heading_stack.append((depth, text))
            current_breadcrumbs = [h[1] for h in heading_stack]

        # If adding this element would exceed the limit, flush first
        if current_length + len(text) > max_chars and current_texts:
            _flush_current()

        # Don't add boundary titles to the body text if we just used them for breadcrumbs,
        # unless it's the only way to keep them. Actually, keeping them in text is fine 
        # and helps with readability.
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
    
    Injects the parent's breadcrumbs into every child chunk so that vector
    search always has the full hierarchical context.

    Args:
        parent (ParentChunkData): The parent chunk to split.
        max_chars (int): Maximum characters per child chunk.
        overlap_chars (int): Overlap between consecutive child chunks.

    Returns:
        List[ChildChunkData]: List of ChildChunkData, each referencing its parent.
    """
    # If the parent text starts with the context header, we should ideally split the 
    # text WITHOUT the context header, and then re-prepend it to each child.
    # This prevents the context header from being cut off in the middle of a child.
    
    text_to_split = parent.text
    context_prefix = ""
    
    if parent.breadcrumbs:
        context_str = f"[Context: {' > '.join(parent.breadcrumbs)}]\n\n"
        if text_to_split.startswith(context_str):
            text_to_split = text_to_split[len(context_str):]
            context_prefix = context_str
            
    # Adjust max_chars to account for the context prefix that we will add to every child
    # Use at least max_chars // 4 for body text to avoid thousands of micro-children
    min_body_chars = max(10, max_chars // 4)
    effective_max_chars = max(min_body_chars, max_chars - len(context_prefix))

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=effective_max_chars,
        chunk_overlap=overlap_chars,
        separators=["\n\n", "\n", ". ", "? ", "! ", "; ", ", ", " ", ""],
        length_function=len,
    )

    child_texts = splitter.split_text(text_to_split)

    children = []
    for child_text in child_texts:
        child_text = child_text.strip()
        if not child_text:
            continue
            
        # Re-inject the context prefix into every child chunk
        final_child_text = context_prefix + child_text
            
        children.append(
            ChildChunkData(
                id=str(uuid.uuid4()),
                parent_chunk_id=parent.id,
                doc_id=parent.doc_id,
                text=final_child_text,
                page=parent.page,
                breadcrumbs=parent.breadcrumbs,
            )
        )

    logger.debug(
        "thesis.chunking.children_created", parent_id=parent.id, child_count=len(children)
    )
    return children
