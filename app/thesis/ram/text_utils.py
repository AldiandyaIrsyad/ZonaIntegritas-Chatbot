"""Shared text-splitting helpers for the Response Assessment Module.

Used both to decide what generated text to run NLI on
(``ChatService._split_propositions``) and to window the retrieved KB text
for reverse-mapping the exact NLI premise (``RAMService.assess_sentence``).

Pure Python (stdlib ``re`` + the sibling ``thesis.chunking.table_converter``
module) — no infra imports, per the ``thesis/`` purity rule (see
``docs/02-arsitektur.md`` §2.2).
"""
import re
from typing import List, Tuple

from app.thesis.chunking.table_converter import is_markdown_table, split_markdown_table_lines

# A digit-preceded ".", "?" or "!" is almost always a markdown list marker
# (e.g. "1.", "2.") rather than a sentence end, so the negative lookbehind
# excludes it. Newlines already delimit markdown list items/paragraphs
# unambiguously, so they're treated as boundaries in their own right instead
# of being flattened to spaces first. Wrapped in one capturing group so
# split() also returns the exact separator text (e.g. "\n\n" for a
# paragraph break vs. " " for a plain sentence gap).
_SENTENCE_BOUNDARY = re.compile(r'((?:(?<!\d[.?!])(?<=[.?!])\s+)|(?:\n+))')


def split_sentences_with_seps(text: str) -> List[Tuple[str, str]]:
    """Split text into (sentence, trailing_separator) pairs.

    The trailing separator is the exact whitespace that followed the
    sentence in the source text (e.g. "\\n\\n" for a paragraph break),
    so callers can reconstruct the original formatting instead of always
    joining fragments with a single space.

    Args:
        text: Input text (may contain markdown list markers and newlines).

    Returns:
        Non-empty, stripped sentence fragments paired with the separator
        that followed them ("" for the trailing, not-yet-terminated
        fragment, if any).
    """
    raw = _SENTENCE_BOUNDARY.split(text)
    pairs: List[Tuple[str, str]] = []
    for i in range(0, len(raw), 2):
        frag = raw[i]
        sep = raw[i + 1] if i + 1 < len(raw) else ""
        if frag and frag.strip():
            pairs.append((frag.strip(), sep))
    return pairs


def split_sentences(text: str) -> List[str]:
    """Split text into sentence-like units.

    Args:
        text: Input text (may contain markdown list markers and newlines).

    Returns:
        Non-empty, stripped sentence fragments.
    """
    return [s for s, _ in split_sentences_with_seps(text)]


def split_table_windows(
    text: str,
    rows_per_window: int = 3,
    row_step: int = 2,
) -> List[str]:
    """Split a Markdown table into overlapping row-group windows.

    Unlike ``split_sentences``, which treats every newline as a boundary
    (shredding a table into headerless row fragments after the first
    window), this keeps the header + separator row prepended to *every*
    window, mirroring the header-repetition strategy already used for
    child-chunk embedding
    (``app.thesis.chunking.table_converter``/``logic._split_markdown_table_rows``)
    — but sized for reranker/NLI windows, not embedding-sized chunks.

    Args:
        text: Raw context text (e.g. ``RetrievedContext.text`` for a
            ``content_type == "table"`` context).
        rows_per_window: Number of data rows grouped per window.
        row_step: Sliding step between window start rows. A step smaller
            than ``rows_per_window`` produces overlapping windows.

    Returns:
        List of ``"<header>\\n<separator>\\n<row>...\\n<row>"`` window
        strings. No synthetic trailing period is appended (unlike
        ``split_sentences`` windows) — appending punctuation after a
        row's closing ``|`` would corrupt the row. Returns ``[]`` if
        ``text`` isn't a parseable Markdown table (e.g. raw HTML that
        failed conversion at ingest time) — callers should fall back to
        ``split_sentences`` in that case.
    """
    if not is_markdown_table(text):
        return []
    parsed = split_markdown_table_lines(text)
    if parsed is None:
        return []
    header, separator, data_rows = parsed
    if not data_rows:
        return []

    windows: List[str] = []
    for i in range(0, max(1, len(data_rows)), row_step):
        row_group = data_rows[i:i + rows_per_window]
        if not row_group:
            continue
        windows.append("\n".join([header, separator] + row_group))
    return windows
