"""Core interface contracts for the application.

All Protocols and result dataclasses are re-exported from this package so
consumers can import from a single, stable namespace::

    from app.core.interfaces import (
        IReranker, RankedResult,
        IVectorStore, ChunkVector, SearchResult,
    )
"""

from app.core.interfaces.ai import (
    EmbeddingResult,
    IEmbeddingProvider,
    INLIProvider,
    IPromptGuard,
    IReranker,
    NLIResult,
    PromptGuardResult,
    RankedResult,
)
from app.core.interfaces.infra import (
    ChunkVector,
    IDocumentParser,
    ILLMConnection,
    IStorageProvider,
    IThumbnailStrategy,
    IVectorStore,
    ParsedElement,
    SearchResult,
)
from app.core.interfaces.rag import (
    IIngestionService,
    IRAGRepository,
    IRetrievalService,
    RetrievedContext,
)
from app.core.interfaces.llm import ILLMService
from app.core.interfaces.ivm import IIVMService


__all__ = [
    # AI result types
    "EmbeddingResult",
    "NLIResult",
    "PromptGuardResult",
    "RankedResult",
    # AI protocols
    "IEmbeddingProvider",
    "INLIProvider",
    "IPromptGuard",
    "IReranker",
    # Infra result types
    "ChunkVector",
    "ParsedElement",
    "SearchResult",
    # Infra protocols
    "IDocumentParser",
    "ILLMConnection",
    "IStorageProvider",
    "IThumbnailStrategy",
    "IVectorStore",
    # RAG services/types
    "IIngestionService",
    "IRAGRepository",
    "IRetrievalService",
    "RetrievedContext",
    # LLM services
    "ILLMService",
    # IVM
    "IIVMService",
]
