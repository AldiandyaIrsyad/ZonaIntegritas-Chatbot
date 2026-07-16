"""Content-type router for the chunking pipeline.

This module provides a pure function that classifies each
:class:`ParsedElement` into a :class:`ContentType` so the chunker can
route it to the appropriate splitting strategy.

Classification rules:
- ``element_type == "Table"`` → :attr:`ContentType.TABLE`
- ``element_type`` in image/figure types → :attr:`ContentType.FIGURE`
- Default → :attr:`ContentType.TEXT`

This module is pure Python (no infra imports), respecting the
``thesis/`` purity rule.
"""

from __future__ import annotations

from typing import List

from app.thesis.chunking.models import ContentType, ParsedElement


# Element types from unstructured.io that indicate a table
TABLE_ELEMENT_TYPES = {"Table"}

# Element types from unstructured.io that indicate a visual/figure element
# "Image" is the standard unstructured type; "Figure" is a common variant
FIGURE_ELEMENT_TYPES = {"Image", "Figure"}

# Element types that indicate section boundaries (used by the chunker)
SECTION_BOUNDARY_TYPES = {"Title"}

# Element types to ignore (noise, page numbers, repeating headers/footers)
IGNORE_ELEMENT_TYPES = {"Header", "Footer", "PageNumber"}


def classify_element(element: ParsedElement) -> ContentType:
    """Classify a parsed element into a content type for routing.

    Args:
        element: The parsed element to classify.

    Returns:
        The content type: TABLE, FIGURE, or TEXT.
    """
    if element.element_type in TABLE_ELEMENT_TYPES:
        return ContentType.TABLE
    if element.element_type in FIGURE_ELEMENT_TYPES:
        return ContentType.FIGURE
    return ContentType.TEXT


def classify_elements(elements: List[ParsedElement]) -> List[ParsedElement]:
    """Classify all elements in-place, returning the same list.

    Sets ``element.content_type`` for each element based on
    :func:`classify_element`.

    Args:
        elements: List of parsed elements to classify.

    Returns:
        The same list (modified in-place) for convenience.
    """
    for element in elements:
        element.content_type = classify_element(element)
    return elements
