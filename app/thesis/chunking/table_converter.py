"""HTML-to-Markdown table converter for the chunking pipeline.

Converts raw HTML table strings (produced by the Unstructured parser's
``text_as_html`` field) into clean Markdown table syntax. Markdown tables
are substantially more compact than HTML and produce much better dense
vector embeddings (BGE-M3) because the model can focus on the cell content
rather than parsing HTML tags.

Design decisions:
- Uses BeautifulSoup for robust HTML parsing (handles malformed HTML from OCR).
- Merges multi-line cell text into single lines for Markdown compatibility.
- Skips ``colspan`` / ``rowspan`` cells gracefully (outputs a ``[merged]``
  placeholder to signal merged cells without corrupting the table structure).
- Falls back to the original HTML string if parsing fails, so the pipeline
  never loses data.
- Does NOT attempt to render pixel-perfect tables — the goal is semantic
  fidelity for embedding, not visual reproduction.

This module is pure Python with one allowed external dependency:
``beautifulsoup4`` (already installed in the project).
"""

from __future__ import annotations

import structlog
from typing import List, Optional

from bs4 import BeautifulSoup, Tag

logger = structlog.get_logger(__name__)

# Placeholder rendered for visually merged cells (colspan/rowspan > 1)
_MERGED_CELL_PLACEHOLDER = "[merged]"

# Maximum character width of a single cell before truncation.
# Very long cells (e.g. full paragraphs in a table cell) are truncated
# to keep the Markdown table readable. The full text is preserved in
# the parent chunk HTML; this is only used for the child chunk embedding.
_MAX_CELL_CHARS = 300


def html_table_to_markdown(html: str) -> str:
    """Convert an HTML table string to a Markdown table.

    Parses the HTML using BeautifulSoup, extracts rows from ``<thead>``
    and ``<tbody>`` (falling back to all ``<tr>`` elements), and renders
    them as a GFM Markdown table with a separator row after the header.

    If no header row (``<th>`` elements) is found, the first data row is
    treated as the header.

    Args:
        html: Raw HTML string containing a ``<table>`` element. May be
            malformed (e.g. from OCR-based table detection).

    Returns:
        Markdown table string, or the original ``html`` string if conversion
        fails or the table contains no data rows.
    """
    if not html or not html.strip():
        return html

    try:
        return _convert(html)
    except Exception as exc:
        logger.warning(
            "table_converter.failed",
            error=str(exc),
            html_snippet=html[:100],
        )
        return html


def _convert(html: str) -> str:
    """Internal conversion logic (raises on error for caller to catch).

    Args:
        html: HTML string to convert.

    Returns:
        Markdown table string.
    """
    soup = BeautifulSoup(html, "html.parser")
    table = soup.find("table")

    if not table or not isinstance(table, Tag):
        # No <table> tag found — return as-is
        return html

    # --- Extract header rows ---
    header_rows: List[List[str]] = []
    thead = table.find("thead")
    if thead and isinstance(thead, Tag):
        header_rows = [
            _extract_cells(row)
            for row in thead.find_all("tr")
            if isinstance(row, Tag)
        ]

    # --- Extract body rows ---
    body_rows: List[List[str]] = []
    tbody = table.find("tbody")
    if tbody and isinstance(tbody, Tag):
        body_rows = [
            _extract_cells(row)
            for row in tbody.find_all("tr")
            if isinstance(row, Tag)
        ]

    # Fallback: no explicit thead/tbody, grab all <tr> elements
    if not header_rows and not body_rows:
        all_rows = [
            _extract_cells(row)
            for row in table.find_all("tr")
            if isinstance(row, Tag)
        ]
        if not all_rows:
            return html
        header_rows = [all_rows[0]]
        body_rows = all_rows[1:]

    # If we only have body rows and no header, promote first body row
    if not header_rows and body_rows:
        header_rows = [body_rows[0]]
        body_rows = body_rows[1:]

    if not header_rows:
        return html

    # --- Determine column count (max across all rows) ---
    col_count = max(
        (len(row) for row in header_rows + body_rows if row),
        default=0,
    )
    if col_count == 0:
        return html

    # --- Render Markdown ---
    lines: List[str] = []

    # Header rows (all <thead> rows)
    for row in header_rows:
        padded = _pad_row(row, col_count)
        lines.append(_render_row(padded))

    # Separator row (GFM requirement)
    lines.append(_render_row(["---"] * col_count))

    # Body rows
    for row in body_rows:
        if not any(cell.strip() for cell in row):
            continue  # Skip entirely empty rows
        padded = _pad_row(row, col_count)
        lines.append(_render_row(padded))

    if not lines:
        return html

    return "\n".join(lines)


def _extract_cells(row: Tag) -> List[str]:
    """Extract cell text values from a ``<tr>`` tag.

    Handles both ``<th>`` and ``<td>`` tags. Cells with ``colspan`` or
    ``rowspan`` > 1 are rendered with a placeholder to signal the merge
    without corrupting the column count.

    Args:
        row: A BeautifulSoup ``<tr>`` tag.

    Returns:
        List of cell text strings for this row.
    """
    cells: List[str] = []
    for cell in row.find_all(["th", "td"]):
        if not isinstance(cell, Tag):
            continue

        # Detect merged cells
        try:
            colspan = int(cell.get("colspan", 1) or 1)  # type: ignore[arg-type]
            rowspan = int(cell.get("rowspan", 1) or 1)  # type: ignore[arg-type]
        except (ValueError, TypeError):
            colspan, rowspan = 1, 1

        text = _clean_cell_text(cell.get_text(separator=" ", strip=True))

        # Primary cell
        cells.append(text)

        # Fill additional columns for colspan
        if colspan > 1:
            for _ in range(colspan - 1):
                cells.append(_MERGED_CELL_PLACEHOLDER)

    return cells


def _clean_cell_text(text: str) -> str:
    """Normalise cell text for Markdown rendering.

    - Collapses whitespace and newlines to single spaces.
    - Escapes pipe characters (``|``) that would break Markdown table syntax.
    - Truncates at :const:`_MAX_CELL_CHARS`.

    Args:
        text: Raw cell text from BeautifulSoup.

    Returns:
        Cleaned cell text safe for embedding inside a Markdown table cell.
    """
    # Collapse whitespace
    cleaned = " ".join(text.split())
    # Escape pipe characters inside cells
    cleaned = cleaned.replace("|", "\\|")
    # Truncate
    if len(cleaned) > _MAX_CELL_CHARS:
        cleaned = cleaned[:_MAX_CELL_CHARS] + "…"
    return cleaned


def _pad_row(row: List[str], col_count: int) -> List[str]:
    """Pad or trim a row to exactly ``col_count`` cells.

    Args:
        row: List of cell strings for one table row.
        col_count: Target number of columns.

    Returns:
        Row list of exactly ``col_count`` elements.
    """
    padded = list(row)
    if len(padded) < col_count:
        padded.extend([""] * (col_count - len(padded)))
    elif len(padded) > col_count:
        padded = padded[:col_count]
    return padded


def _render_row(cells: List[str]) -> str:
    """Render a list of cell strings as one Markdown table row.

    Args:
        cells: Cell text strings (already padded to column count).

    Returns:
        Markdown table row string, e.g. ``"| Col A | Col B |"``.
    """
    return "| " + " | ".join(cells) + " |"
