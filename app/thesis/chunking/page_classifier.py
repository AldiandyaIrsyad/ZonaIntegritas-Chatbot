"""Per-page classification for the hybrid PDF processing pipeline.

Each page is independently classified into a :class:`PageType` based on
the ratio of visual-garbage elements detected by the Unstructured parser.
This enables the :class:`IngestWorker` to route visual-heavy pages (SOP
flowcharts, diagrams) to VLM full-page extraction while keeping
text-rich and table-rich pages in the standard Unstructured pipeline.

Design rationale:
- Classification is page-scoped, not document-scoped, so mixed documents
  (e.g. SOPs with formal-text cover pages AND flowchart body pages) are
  handled correctly without any per-document label.
- ``VISUAL`` classification replaces the Unstructured output entirely.
  ``TABLE_RICH`` only transforms table elements in-place. A page can be
  both ``TABLE_RICH`` and ``TEXT_RICH`` simultaneously.
- Garbage detection uses only "text length ≤ 3 chars" — simpler and more
  robust than pattern-matching on repeated characters, which has too many
  edge cases in OCR output.

This module is pure Python (no infra imports), compliant with the
``thesis/`` purity rule.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional

from app.thesis.chunking.models import ContentType, ParsedElement


class PageType(str, Enum):
    """The dominant structural type of a PDF page.

    Attributes:
        TEXT_RICH: Page is primarily narrative text, titles, list items.
            Unstructured output is used as-is.
        TABLE_RICH: Page contains one or more well-formed HTML tables.
            Tables are converted to Markdown; other elements kept as-is.
        VISUAL: Page is dominated by images/figures with garbage or empty
            OCR output (flowcharts, SOP diagrams, scanned images).
            Unstructured output is discarded; VLM renders the full page.
        MIXED: Page has both substantive text and visual elements but does
            not qualify as VISUAL. Treated as TEXT_RICH with table conversion.
    """

    TEXT_RICH = "text_rich"
    TABLE_RICH = "table_rich"
    VISUAL = "visual"
    MIXED = "mixed"


@dataclass(frozen=True)
class PageClassification:
    """Classification result for a single PDF page.

    Attributes:
        page_number: 1-indexed page number.
        page_type: The dominant structural type of this page.
        element_count: Total number of Unstructured elements on this page.
        image_count: Number of Image/Figure elements on this page.
        table_count: Number of Table elements on this page.
        text_element_count: Number of text-type elements (Title, Narrative, etc.).
        garbage_image_count: Number of image elements with empty or ≤3-char text.
        image_ratio: image_count / element_count (0.0–1.0).
        garbage_ratio: garbage_image_count / image_count (0.0–1.0), 0 if no images.
    """

    page_number: int
    page_type: PageType
    element_count: int
    image_count: int
    table_count: int
    text_element_count: int
    garbage_image_count: int
    image_ratio: float
    garbage_ratio: float


# Element types that count as "text" for classification
_TEXT_ELEMENT_TYPES = frozenset({
    "Title", "NarrativeText", "ListItem", "Header",
    "Footer", "UncategorizedText", "Address", "FigureCaption",
    "EmailAddress", "Formula",
})

# Element types that count as "image/visual" for classification
_VISUAL_ELEMENT_TYPES = frozenset({"Image", "Figure"})

# Element types that count as "table" for classification
_TABLE_ELEMENT_TYPES = frozenset({"Table"})

# Maximum text length for an image element to be considered "garbage" OCR
# output purely on length (empty strings, single-char noise like "L"/"6",
# 2-3 char OCR fragments like "qp").
_GARBAGE_TEXT_MAX_LEN = 3

# For longer-but-still-short OCR text, also flag as garbage when it's mostly
# non-alphanumeric (e.g. "~   ~if ~!11" from a misread letterhead/seal) —
# catches noise up to this length that _GARBAGE_TEXT_MAX_LEN alone misses,
# without misclassifying short-but-legitimate OCR text (page numbers, short
# captions), which is predominantly alphanumeric.
_GARBAGE_DENSITY_MAX_LEN = 20
_GARBAGE_MIN_ALNUM_RATIO = 0.5


def _is_garbage_ocr_text(text: str) -> bool:
    """Heuristic garbage-OCR detector for a single image/figure element's text.

    Combines a pure length check (very short text is always garbage) with a
    length-and-density check (longer-but-still-short text that's mostly
    non-alphanumeric, e.g. OCR noise from a misread letterhead).
    """
    stripped = text.strip()
    if len(stripped) <= _GARBAGE_TEXT_MAX_LEN:
        return True
    if len(stripped) <= _GARBAGE_DENSITY_MAX_LEN:
        alnum_count = sum(1 for ch in stripped if ch.isalnum())
        alnum_ratio = alnum_count / len(stripped)
        if alnum_ratio < _GARBAGE_MIN_ALNUM_RATIO:
            return True
    return False

# Default thresholds for VISUAL classification
DEFAULT_IMAGE_RATIO_THRESHOLD = 0.5
DEFAULT_GARBAGE_RATIO_THRESHOLD = 0.7

# Prompt for VLM full-page extraction (Indonesian output)
VLM_PAGE_EXTRACTION_PROMPT = (
    "Ekstrak SEMUA konten dari halaman dokumen ini dalam format Markdown yang bersih. "
    "Untuk tabel, gunakan sintaks tabel Markdown. "
    "Untuk bagan alir (flowchart) atau diagram proses, deskripsikan setiap langkah "
    "secara berurutan menggunakan daftar bernomor, sertakan aktor yang terlibat, "
    "keputusan (decision points), dan urutan kejadian. "
    "Untuk teks biasa, pertahankan heading dan paragraf. "
    "Untuk prosedur atau SOP, jelaskan setiap langkah dengan jelas. "
    "Keluarkan HANYA konten Markdown, tanpa komentar tambahan. "
    "Gunakan Bahasa Indonesia."
)


def classify_page(
    elements: List[ParsedElement],
    page_number: int,
    image_ratio_threshold: float = DEFAULT_IMAGE_RATIO_THRESHOLD,
    garbage_ratio_threshold: float = DEFAULT_GARBAGE_RATIO_THRESHOLD,
) -> PageClassification:
    """Classify a single PDF page based on its Unstructured element composition.

    The classification drives downstream routing in the ingestion pipeline:
    - ``VISUAL`` → discard Unstructured output, run VLM full-page extraction
    - ``TABLE_RICH`` → convert HTML tables to Markdown, keep text elements
    - ``TEXT_RICH`` / ``MIXED`` → keep Unstructured output as-is (with table conversion)

    Args:
        elements: All ParsedElement objects whose ``page_number`` metadata
            matches this page. Must be non-empty.
        page_number: The 1-indexed page number (for the result dataclass only).
        image_ratio_threshold: Minimum fraction of elements that must be images
            for a page to be considered potentially VISUAL. Default 0.5.
        garbage_ratio_threshold: Minimum fraction of image elements that must
            have garbage text (≤ 3 chars) for the page to be classified VISUAL.
            Default 0.7.

    Returns:
        :class:`PageClassification` describing this page's type and statistics.
    """
    if not elements:
        return PageClassification(
            page_number=page_number,
            page_type=PageType.TEXT_RICH,
            element_count=0,
            image_count=0,
            table_count=0,
            text_element_count=0,
            garbage_image_count=0,
            image_ratio=0.0,
            garbage_ratio=0.0,
        )

    total = len(elements)
    image_count = 0
    table_count = 0
    text_element_count = 0
    garbage_image_count = 0

    for el in elements:
        etype = el.element_type
        if etype in _VISUAL_ELEMENT_TYPES:
            image_count += 1
            if _is_garbage_ocr_text(el.text):
                garbage_image_count += 1
        elif etype in _TABLE_ELEMENT_TYPES:
            table_count += 1
        elif etype in _TEXT_ELEMENT_TYPES:
            text_element_count += 1
        # Unknown types not counted in any category

    image_ratio = image_count / total
    garbage_ratio = garbage_image_count / image_count if image_count > 0 else 0.0

    # --- Classification logic ---
    # VISUAL: more than half the elements are images AND most image OCR is garbage.
    # This signals a flowchart/diagram page where Unstructured is unreliable.
    if image_ratio >= image_ratio_threshold and garbage_ratio >= garbage_ratio_threshold:
        page_type = PageType.VISUAL
    elif table_count > 0:
        # Any substantive tables present → TABLE_RICH (may also have text)
        page_type = PageType.TABLE_RICH
    elif image_count > 0 and text_element_count > 0:
        # Some images but mostly text — don't go VLM, treat as mixed
        page_type = PageType.MIXED
    else:
        page_type = PageType.TEXT_RICH

    return PageClassification(
        page_number=page_number,
        page_type=page_type,
        element_count=total,
        image_count=image_count,
        table_count=table_count,
        text_element_count=text_element_count,
        garbage_image_count=garbage_image_count,
        image_ratio=image_ratio,
        garbage_ratio=garbage_ratio,
    )


def group_elements_by_page(
    elements: List[ParsedElement],
) -> Dict[Optional[int], List[ParsedElement]]:
    """Group parsed elements by their page_number metadata.

    Elements without a page_number are grouped under the key ``None``
    and treated as TEXT_RICH during classification.

    Args:
        elements: Flat list of parsed elements from the document parser.

    Returns:
        Dictionary mapping page_number (int | None) → list of elements.
    """
    groups: Dict[Optional[int], List[ParsedElement]] = {}
    for el in elements:
        page = el.metadata.get("page_number")
        groups.setdefault(page, []).append(el)
    return groups


def classify_all_pages(
    elements: List[ParsedElement],
    image_ratio_threshold: float = DEFAULT_IMAGE_RATIO_THRESHOLD,
    garbage_ratio_threshold: float = DEFAULT_GARBAGE_RATIO_THRESHOLD,
) -> Dict[Optional[int], PageClassification]:
    """Classify all pages in a document from a flat element list.

    Convenience wrapper around :func:`group_elements_by_page` and
    :func:`classify_page`.

    Args:
        elements: All parsed elements from the document parser.
        image_ratio_threshold: See :func:`classify_page`.
        garbage_ratio_threshold: See :func:`classify_page`.

    Returns:
        Dictionary mapping page_number → :class:`PageClassification`.
    """
    groups = group_elements_by_page(elements)
    return {
        page: classify_page(
            page_elements,
            page_number=page or 0,
            image_ratio_threshold=image_ratio_threshold,
            garbage_ratio_threshold=garbage_ratio_threshold,
        )
        for page, page_elements in groups.items()
    }
