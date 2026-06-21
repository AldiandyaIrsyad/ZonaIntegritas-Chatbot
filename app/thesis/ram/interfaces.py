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
    """
    label: str
    entailment_score: float = 0.0
    contradiction_score: float = 0.0
    neutral_score: float = 0.0
    source_title: str = ""
    page: Optional[int] = None


@dataclass(frozen=True)
class RetrievedContext:
    """A retrieved context block from the knowledge base."""
    text: str
    source_title: str
    page: Optional[int] = None


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