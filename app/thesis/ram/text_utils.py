"""Shared text-splitting helpers for the Response Assessment Module.

Used both to decide what generated text to run NLI on
(``ChatService._split_propositions``) and to window the retrieved KB text
for reverse-mapping the exact NLI premise (``RAMService.assess_sentence``).
"""
import re
from typing import List, Tuple

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
