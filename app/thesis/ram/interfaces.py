"""Ports (Protocol interfaces) for the Response Assessment Module (RAM).

The RAM validates generated LLM sentences against retrieved KB context
using Natural Language Inference (NLI) to detect hallucinations per
sentence. It is part of the pure ``thesis`` research core and defines its
own ports so it never imports chat/kb infra directly (Dependency Inversion).

Ports → adapters map:
    - :class:`INLIModel`       → ``app/chat/infra/nli_client.py::NLIClient``
                                (Infinity-hosted NLI model; wired in
                                ``app/chat/dependency.py::get_nli_client``)
    - :class:`IRerankerModel`  → ``app/kb/infra/infinity_reranker.py::InfinityReranker``
                                (the same adapter fulfills the KB
                                ``IReranker`` port; this is a narrower view
                                for the RAM's reverse-mapping step)
"""

from dataclasses import dataclass
from typing import List, Optional, Protocol


@dataclass(frozen=True)
class NLIResult:
    """Outcome of a single Natural Language Inference call.

    Attributes:
        label: Canonical verdict ("entailment", "neutral", or "contradiction").
        entailment_score: Confidence that the hypothesis is entailed by the premise (0.0–1.0).
        contradiction_score: Confidence that the hypothesis contradicts the premise (0.0–1.0).
        neutral_score: Confidence that the hypothesis is neutral to the premise (0.0–1.0).
        source_title: Display name of the PDF that provided the best-matching context chunk.
        page: Page number of the best-matching context chunk, if available.
        doc_id: ID of the source PDF document, for downloading the original file.
        evidence_snippet: The exact reranker-matched sub-passage of the
            context that the NLI check was run against, sanitized and
            truncated for display. Lets a user see which specific part of
            the retrieved context backed a given sentence, without needing
            to re-run generation.
    """
    label: str
    entailment_score: float = 0.0
    contradiction_score: float = 0.0
    neutral_score: float = 0.0
    source_title: str = ""
    page: Optional[int] = None
    doc_id: str = ""
    evidence_snippet: str = ""


@dataclass(frozen=True)
class RetrievedContext:
    """A retrieved context block from the knowledge base.

    Attributes:
        text: The full parent chunk text.
        source_title: Display name of the source document.
        page: Page number of the chunk, if available.
        breadcrumbs: Hierarchical section path (e.g. ["BAB I", "Pasal 5"]).
        content_type: Structural type ("text", "table", "figure").
        chunk_id: UUID of the child chunk that matched the query.
        path: ltree-style dot path of the parent chunk.
        doc_id: ID of the source PDF document, for downloading the original file.
    """
    text: str
    source_title: str
    page: Optional[int] = None
    breadcrumbs: List[str] = ()  # type: ignore[assignment]
    content_type: str = "text"
    chunk_id: str = ""
    path: str = ""
    doc_id: str = ""


class INLIModel(Protocol):
    """Port for Natural Language Inference adapters in the research core.

    Implemented by: ``app/chat/infra/nli_client.py::NLIClient``
    (an Infinity-hosted NLI model; wired in
    ``app/chat/dependency.py::get_nli_client``).
    """

    async def check(self, premise: str, hypothesis: str) -> NLIResult:
        """Compare a hypothesis against a reference premise using NLI.

        Args:
            premise: Reference context (e.g. a KB parent chunk).
            hypothesis: Statement to verify against the premise.

        Returns:
            NLIResult with the canonical label and per-class scores.
        """
        ...


@dataclass(frozen=True)
class RerankResult:
    """Result of a reranking operation.

    Attributes:
        index: Original position of the document in the input list.
        score: Relevance score assigned by the reranker (higher = more relevant).
    """

    index: int
    score: float


class IRerankerModel(Protocol):
    """Port for reranker adapters in the research core.

    Implemented by: ``app/kb/infra/infinity_reranker.py::InfinityReranker``
    (the same adapter fulfills the KB ``IReranker`` port; wired in
    ``app/kb/dependency.py::get_reranker``).
    """

    async def rerank(
        self,
        query: str,
        documents: List[str],
        top_k: Optional[int] = None,
    ) -> List[RerankResult]:
        """Rerank documents by relevance to the query.

        Args:
            query: The search query (e.g., hypothesis).
            documents: List of document texts (e.g., context windows).
            top_k: Optional limit on results.

        Returns:
            List of RerankResult ordered by descending relevance.
        """
        ...