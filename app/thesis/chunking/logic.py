"""
Parent-child chunking strategy for Small-to-Big retrieval.

Takes structured elements from the unstructured parser and organizes them
into a two-level hierarchy:
- Parent chunks: logical sections of the document (for LLM context)
- Child chunks: sentence-level splits of each parent (for retrieval precision)

Content-type aware: tables are protected from character splitting (the
full table becomes its own parent chunk), and figure descriptions are
split at sentence boundaries. Narrative text uses the standard
RecursiveCharacterTextSplitter.
"""
import uuid
import structlog
from typing import List, Optional

from langchain_text_splitters import RecursiveCharacterTextSplitter

from .models import ChildChunkData, ContentType, ParentChunkData, ParsedElement
from .router import (
    IGNORE_ELEMENT_TYPES,
    SECTION_BOUNDARY_TYPES,
    TABLE_ELEMENT_TYPES,
    FIGURE_ELEMENT_TYPES,
    classify_element,
)

logger = structlog.get_logger(__name__)

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

    **Table-aware**: When a ``Table`` element is encountered, the current
    text buffer is flushed first (so the table is not mixed with prose),
    and the table becomes its own parent chunk with
    ``content_type=ContentType.TABLE``. This prevents the character
    splitter from fragmenting table HTML.

    Consecutive headers are grouped together, and page artifacts are ignored.
    Maintains a heading stack to track the hierarchical path (breadcrumbs).

    Args:
        elements (List[ParsedElement]): Structured elements.
        doc_id (str): UUID of the source document.
        max_chars (int): Maximum character length per parent chunk.

    Returns:
        List[ParentChunkData]: Ordered list of ParentChunkData.
    """
    if not elements:
        return []

    # Classify all elements by content type
    for el in elements:
        el.content_type = classify_element(el)

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
                    content_type=ContentType.TEXT,
                )
            )
            chunk_index += 1
            
        current_texts = []
        current_length = 0
        current_page = None
        has_body_text = False

    def _flush_table(element: ParsedElement) -> None:
        """Create a standalone parent chunk for a table element."""
        nonlocal chunk_index
        text = element.text.strip()
        if not text:
            return

        # Prepend breadcrumbs for context
        if current_breadcrumbs:
            context_header = f"[Context: {' > '.join(current_breadcrumbs)}]\n\n"
            text = context_header + text

        parent_chunks.append(
            ParentChunkData(
                id=str(uuid.uuid4()),
                doc_id=doc_id,
                text=text,
                chunk_index=chunk_index,
                page=element.metadata.get("page_number"),
                breadcrumbs=list(current_breadcrumbs),
                content_type=ContentType.TABLE,
                element_metadata=dict(element.metadata),
            )
        )
        chunk_index += 1

    def _flush_figure(element: ParsedElement) -> None:
        """Create a standalone parent chunk for a figure/VLM description."""
        nonlocal chunk_index
        text = element.text.strip()
        if not text:
            return

        if current_breadcrumbs:
            context_header = f"[Context: {' > '.join(current_breadcrumbs)}]\n\n"
            text = context_header + text

        parent_chunks.append(
            ParentChunkData(
                id=str(uuid.uuid4()),
                doc_id=doc_id,
                text=text,
                chunk_index=chunk_index,
                page=element.metadata.get("page_number"),
                breadcrumbs=list(current_breadcrumbs),
                content_type=ContentType.FIGURE,
                element_metadata=dict(element.metadata),
            )
        )
        chunk_index += 1

    for element in elements:
        text = element.text.strip()
        if not text:
            continue

        # Ignore noisy elements like page headers/footers
        if element.element_type in IGNORE_ELEMENT_TYPES:
            continue

        # --- Table routing: flush current prose, emit table as own parent ---
        if element.content_type == ContentType.TABLE:
            _flush_current()
            _flush_table(element)
            continue

        # --- Figure routing: flush current prose, emit figure as own parent ---
        if element.content_type == ContentType.FIGURE:
            _flush_current()
            _flush_figure(element)
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

        # Titles ARE kept in body text (helps readability)
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

    **Content-type aware dispatcher**. Routes to the appropriate splitting
    strategy based on ``parent.content_type``:

    - :attr:`ContentType.TEXT` → :func:`_split_text_children`
      (RecursiveCharacterTextSplitter, respects sentence boundaries)
    - :attr:`ContentType.TABLE` → :func:`_split_table_children`
      (no splitting — full table as single child; preserves HTML structure)
    - :attr:`ContentType.FIGURE` → :func:`_split_figure_children`
      (sentence-level split on VLM description)
    - :attr:`ContentType.HYBRID` → :func:`_split_text_children`
      (treat as text — the hybrid content is already linearised)

    Injects the parent's breadcrumbs into every child chunk so that vector
    search always has the full hierarchical context.

    Args:
        parent (ParentChunkData): The parent chunk to split.
        max_chars (int): Maximum characters per child chunk.
        overlap_chars (int): Overlap between consecutive child chunks.

    Returns:
        List[ChildChunkData]: List of ChildChunkData, each referencing its parent.
    """
    if parent.content_type == ContentType.TABLE:
        children = _split_table_children(parent)
    elif parent.content_type == ContentType.FIGURE:
        children = _split_figure_children(parent, max_chars, overlap_chars)
    else:
        # TEXT and HYBRID both use the standard text splitter
        children = _split_text_children(parent, max_chars, overlap_chars)

    logger.debug(
        "thesis.chunking.children_created",
        parent_id=parent.id,
        child_count=len(children),
        content_type=parent.content_type.value,
    )
    return children


def _build_context_prefix(breadcrumbs: List[str]) -> str:
    """Build the context header string from breadcrumbs.

    Args:
        breadcrumbs: Hierarchical section path.

    Returns:
        Context header string (empty if no breadcrumbs).
    """
    if not breadcrumbs:
        return ""
    return f"[Context: {' > '.join(breadcrumbs)}]\n\n"


def _strip_context_prefix(text: str, breadcrumbs: List[str]) -> tuple[str, str]:
    """Separate the context prefix from the body text.

    Args:
        text: The parent text (may start with a context header).
        breadcrumbs: Breadcrumbs used to build the context header.

    Returns:
        Tuple of (body_text, context_prefix).
    """
    context_prefix = _build_context_prefix(breadcrumbs)
    if context_prefix and text.startswith(context_prefix):
        return text[len(context_prefix):], context_prefix
    return text, context_prefix


def _split_text_children(
    parent: ParentChunkData,
    max_chars: int = DEFAULT_CHILD_MAX_CHARS,
    overlap_chars: int = DEFAULT_CHILD_OVERLAP_CHARS,
) -> List[ChildChunkData]:
    """Split narrative text into child chunks using RecursiveCharacterTextSplitter.

    Respects sentence and word boundaries. Re-injects the breadcrumb context
    prefix into every child so vector search always has hierarchical context.

    Args:
        parent: The parent chunk to split.
        max_chars: Maximum characters per child chunk.
        overlap_chars: Overlap between consecutive child chunks.

    Returns:
        List of child chunks.
    """
    text_to_split, context_prefix = _strip_context_prefix(parent.text, parent.breadcrumbs)

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

    children: List[ChildChunkData] = []
    for child_text in child_texts:
        child_text = child_text.strip()
        if not child_text:
            continue

        final_child_text = context_prefix + child_text

        children.append(
            ChildChunkData(
                id=str(uuid.uuid4()),
                parent_chunk_id=parent.id,
                doc_id=parent.doc_id,
                text=final_child_text,
                page=parent.page,
                breadcrumbs=parent.breadcrumbs,
                content_type=parent.content_type,
            )
        )
    return children


def _split_table_children(
    parent: ParentChunkData,
) -> List[ChildChunkData]:
    """Create child chunk(s) for a table parent.

    **Tables are NOT character-split** — the full table HTML/Markdown is
    preserved as a single child chunk. This prevents the splitter from
    slicing through ``<tr>``/``<td>`` tags and destroying the table structure.

    If a table summary is available in ``element_metadata`` (generated by
    an :class:`ITableSummarizer`), it is added as an additional child
    chunk — the summary is what gets embedded for vector search, while
    the full table is retrieved as the parent context.

    Args:
        parent: The table parent chunk.

    Returns:
        List of child chunks: always at least one (the full table).
        May include a second child with the table summary if available.
    """
    context_prefix = _build_context_prefix(parent.breadcrumbs)
    children: List[ChildChunkData] = []

    # Child 1: Full table (preserves structure for parent retrieval)
    table_text = parent.text
    if not table_text.strip():
        return children

    children.append(
        ChildChunkData(
            id=str(uuid.uuid4()),
            parent_chunk_id=parent.id,
            doc_id=parent.doc_id,
            text=table_text,
            page=parent.page,
            breadcrumbs=parent.breadcrumbs,
            content_type=ContentType.TABLE,
        )
    )

    # Child 2 (optional): Table summary for better embedding match
    # The summary is generated during ingestion by an ITableSummarizer
    # and stored in element_metadata["table_summary"]
    table_summary = parent.element_metadata.get("table_summary")
    if table_summary and isinstance(table_summary, str) and table_summary.strip():
        summary_text = context_prefix + table_summary.strip()
        children.append(
            ChildChunkData(
                id=str(uuid.uuid4()),
                parent_chunk_id=parent.id,
                doc_id=parent.doc_id,
                text=summary_text,
                page=parent.page,
                breadcrumbs=parent.breadcrumbs,
                content_type=ContentType.TABLE,
            )
        )

    return children


def _split_figure_children(
    parent: ParentChunkData,
    max_chars: int = DEFAULT_CHILD_MAX_CHARS,
    overlap_chars: int = DEFAULT_CHILD_OVERLAP_CHARS,
) -> List[ChildChunkData]:
    """Split a VLM figure description into child chunks.

    Figure descriptions are natural-language text (generated by a VLM),
    so they can be split at sentence boundaries like narrative text.
    However, if the description is short enough to fit in a single
    child chunk (common case), no splitting is applied — this preserves
    the full description as one retrievable unit.

    Args:
        parent: The figure parent chunk (text = VLM description).
        max_chars: Maximum characters per child chunk.
        overlap_chars: Overlap between consecutive child chunks.

    Returns:
        List of child chunks.
    """
    text_to_split, context_prefix = _strip_context_prefix(parent.text, parent.breadcrumbs)

    # If the description fits in a single child, don't split — preserve
    # the full VLM description as one retrievable unit
    if len(text_to_split) <= max_chars:
        return [
            ChildChunkData(
                id=str(uuid.uuid4()),
                parent_chunk_id=parent.id,
                doc_id=parent.doc_id,
                text=parent.text,
                page=parent.page,
                breadcrumbs=parent.breadcrumbs,
                content_type=ContentType.FIGURE,
            )
        ]

    # Long descriptions: fall back to text splitter
    return _split_text_children(parent, max_chars, overlap_chars)
