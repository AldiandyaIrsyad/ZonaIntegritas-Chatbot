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
    """Structural contract for Natural Language Inference adapters in the research core."""

    async def check(self, premise: str, hypothesis: str) -> NLIResult:
        """Compare a hypothesis against a reference premise using NLI.

        Args:
            premise: Reference context (e.g. a KB parent chunk).
            hypothesis: Statement to verify against the premise.

        Returns:
            NLIResult with the canonical label and per-class scores.
        """
        ...


class IEmbeddingModel(Protocol):
    """Structural contract for text embedding adapters in the research core."""

    async def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """Generate dense embeddings for a list of texts.

        Args:
            texts: Input strings to embed. Order is preserved.

        Returns:
            A list of dense embedding vectors (List of floats).
        """
        ...


@dataclass(frozen=True)
class RerankResult:
    """Result of a reranking operation."""
    index: int
    score: float


class IRerankerModel(Protocol):
    """Structural contract for reranker adapters in the research core."""

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