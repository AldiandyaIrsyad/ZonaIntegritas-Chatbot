"""AI and ML infrastructure interface contracts.

Defines frozen result dataclasses and structural :class:`typing.Protocol` types
for all AI/ML adapters used in the infra layer.  Consumers (services, tests,
analysis) depend on these abstractions — never on concrete infra implementations.

Example::

    from app.core.interfaces.ai import IReranker, RankedResult

    def build_pipeline(reranker: IReranker) -> ...:
        ...
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Protocol, runtime_checkable


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PromptGuardResult:
    """Outcome of a single prompt injection classification request.

    Attributes:
        is_safe: ``True`` if the input was classified as safe.
        message: Human-readable explanation of the classification result
                 (e.g. ``"Safe"`` or ``"Policy violation: MALICIOUS (0.92)"``).
    """

    is_safe: bool
    message: str


@dataclass(frozen=True)
class NLIResult:
    """Outcome of a single Natural Language Inference call.

    Attributes:
        label: Canonical verdict — one of ``"entailment"``, ``"neutral"``,
               or ``"contradiction"``.
        entailment_score: Confidence that the hypothesis is entailed by the
                          premise (0.0–1.0).
        contradiction_score: Confidence that the hypothesis contradicts the
                              premise (0.0–1.0).
        neutral_score: Confidence that the hypothesis is neutral to the
                       premise (0.0–1.0).
        source_title: Display name of the PDF that provided the best-matching
                      context chunk.
        page: Page number of the best-matching context chunk, if available.
    """

    label: str
    entailment_score: float = 0.0
    contradiction_score: float = 0.0
    neutral_score: float = 0.0
    source_title: str = ""
    page: Optional[int] = None


@dataclass(frozen=True)
class EmbeddingResult:
    """Dense and sparse embedding vectors for a single text input.

    Attributes:
        dense: Float vector produced by the embedding model (e.g. 1024 dims
               for BAAI/bge-m3).
        sparse_indices: Token indices for the BM25 sparse representation.
        sparse_values: Corresponding weights for each sparse index.
    """

    dense: list[float]
    sparse_indices: list[int]
    sparse_values: list[float]


@dataclass(frozen=True)
class RankedResult:
    """A document chunk scored by a cross-encoder reranker.

    Attributes:
        index: Original position of this document in the input list.
        text: The document text that was scored.
        score: Relevance score (higher is more relevant; unbounded float).
    """

    index: int
    text: str
    score: float


# ---------------------------------------------------------------------------
# Protocols
# ---------------------------------------------------------------------------


@runtime_checkable
class IPromptGuard(Protocol):
    """Structural contract for prompt injection detection adapters.

    Any object that provides :meth:`check_prompt` and :meth:`close` with the
    correct signatures satisfies this Protocol — no subclassing required.
    """

    async def check_prompt(self, text: str) -> PromptGuardResult:
        """Classify a user input for prompt injection or jailbreak attempts.

        Args:
            text: Raw user input to classify.

        Returns:
            :class:`PromptGuardResult` indicating whether the input is safe.
        """
        ...

    async def close(self) -> None:
        """Release any underlying connections or resources."""
        ...


@runtime_checkable
class INLIProvider(Protocol):
    """Structural contract for Natural Language Inference adapters.

    Any object that provides :meth:`check` and :meth:`close` with the correct
    signatures satisfies this Protocol — no subclassing required.
    """

    async def check(self, premise: str, hypothesis: str) -> NLIResult:
        """Compare a hypothesis against a reference premise using NLI.

        Args:
            premise: Reference context (e.g. a KB parent chunk).
            hypothesis: Statement to verify against the premise (e.g. an LLM
                        output sentence).

        Returns:
            :class:`NLIResult` with the canonical label and per-class scores.
        """
        ...

    async def close(self) -> None:
        """Release any underlying connections or resources."""
        ...


@runtime_checkable
class IEmbeddingProvider(Protocol):
    """Structural contract for text embedding adapters.

    Any object that provides :meth:`embed_texts` and :meth:`close` with the
    correct signatures satisfies this Protocol — no subclassing required.
    """

    async def embed_texts(self, texts: list[str]) -> list[EmbeddingResult]:
        """Generate dense and sparse embeddings for a list of texts.

        Args:
            texts: Input strings to embed.  Order is preserved in the output.

        Returns:
            One :class:`EmbeddingResult` per input text, in the same order.
        """
        ...

    async def close(self) -> None:
        """Release any underlying connections or resources."""
        ...


@runtime_checkable
class IReranker(Protocol):
    """Structural contract for cross-encoder document reranking adapters.

    Any object that provides :meth:`rerank` and :meth:`close` with the correct
    signatures satisfies this Protocol — no subclassing required.
    """

    async def rerank(
        self,
        query: str,
        documents: list[str],
        top_k: int = 3,
    ) -> list[RankedResult]:
        """Score and rank candidate documents by relevance to the query.

        Args:
            query: The user's search query.
            documents: Candidate document texts to score.
            top_k: Maximum number of results to return.  Must be ``> 0``.

        Returns:
            Up to ``top_k`` :class:`RankedResult` items sorted by descending
            relevance score.
        """
        ...

    async def close(self) -> None:
        """Release any underlying connections or resources."""
        ...
