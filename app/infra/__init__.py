"""Infrastructure adapters package.

Re-exports all concrete infra implementations so consumers can import from a
single, stable namespace::

    from app.infra import (
        DocumentParser,
        EmbeddingProvider,
        LLMConnection,
        LocalStorageProvider,
        NLIProvider,
        PromptGuardProvider,
        QdrantStore,
        Reranker,
        ThumbnailContext,
    )
"""

from app.infra.document_parser import DocumentParser
from app.infra.embedding_provider import EmbeddingProvider
from app.infra.llm_connection import LLMConnection
from app.infra.nli import NLIProvider
from app.infra.prompt_guard import PromptGuardProvider
from app.infra.reranker import Reranker
from app.infra.storage import LocalStorageProvider
from app.infra.thumbnail import (
    DefaultThumbnailStrategy,
    ImageThumbnailStrategy,
    PDFThumbnailStrategy,
    ThumbnailContext,
)
from app.infra.vector_store import QdrantStore

__all__ = [
    "DocumentParser",
    "EmbeddingProvider",
    "LLMConnection",
    "LocalStorageProvider",
    "NLIProvider",
    "PromptGuardProvider",
    "QdrantStore",
    "Reranker",
    "DefaultThumbnailStrategy",
    "ImageThumbnailStrategy",
    "PDFThumbnailStrategy",
    "ThumbnailContext",
]
