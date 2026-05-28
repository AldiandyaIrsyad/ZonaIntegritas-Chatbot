"""
Hybrid retrieval and reranking service.

Implements the read path of the RAG pipeline:
Query → Embed → Hybrid Search (Qdrant) → Parent Lookup (Postgres) → Rerank → Context
"""
import logging
from dataclasses import dataclass
from typing import List

from sqlalchemy.ext.asyncio import AsyncSession

from src.infra import EmbeddingProvider, QdrantStore, Reranker

from .repository import RAGRepository

logger = logging.getLogger(__name__)


@dataclass
class RetrievedContext:
    """A single context block ready to be injected into the LLM prompt."""
    text: str
    doc_id: str
    score: float
    source_title: str


class RetrievalService:
    """
    Orchestrates the hybrid retrieval and reranking pipeline.

    Pipeline:
    1. Embed the user query → dense + sparse vectors
    2. Hybrid search in Qdrant (dense + BM25, is_active filter)
    3. Extract unique parent_chunk_ids from top-K child results
    4. Fetch full parent texts from PostgreSQL
    5. Rerank parent texts against the query via cross-encoder
    6. Return top reranked contexts with source metadata
    """

    def __init__(
        self,
        db: AsyncSession,
        embedding_provider: EmbeddingProvider,
        vector_store: QdrantStore,
        reranker: Reranker,
    ):
        self.db = db
        self.embedding_provider = embedding_provider
        self.vector_store = vector_store
        self.reranker = reranker
        self.rag_repo = RAGRepository(db)

    async def retrieve_context(
        self,
        query: str,
        top_k_search: int = 15,
        top_k_rerank: int = 3,
    ) -> List[RetrievedContext]:
        """Execute the full retrieval pipeline for a user query.

        Args:
            query: The user's question or message.
            top_k_search: Number of child chunks to retrieve from Qdrant.
            top_k_rerank: Number of parent contexts to return after reranking.

        Returns:
            List of RetrievedContext sorted by relevance, ready for LLM prompt.
            Returns empty list if no relevant documents are found.
        """
        if not query.strip():
            return []

        # Step 1: Embed the query
        query_embeddings = await self.embedding_provider.embed_texts([query])
        if not query_embeddings:
            logger.warning("Failed to embed query")
            return []

        query_emb = query_embeddings[0]

        # Step 2: Hybrid search in Qdrant
        search_results = await self.vector_store.hybrid_search(
            dense_vector=query_emb.dense,
            sparse_indices=query_emb.sparse_indices,
            sparse_values=query_emb.sparse_values,
            top_k=top_k_search,
        )

        if not search_results:
            logger.info("No search results found for query")
            return []

        # Step 3: Extract unique parent_chunk_ids (preserving order)
        seen_parent_ids: set[str] = set()
        unique_parent_ids: list[str] = []
        for result in search_results:
            if result.parent_chunk_id not in seen_parent_ids:
                seen_parent_ids.add(result.parent_chunk_id)
                unique_parent_ids.append(result.parent_chunk_id)

        # Step 4: Fetch parent chunks from PostgreSQL
        parent_chunks = await self.rag_repo.get_parent_chunks_by_ids(
            unique_parent_ids
        )

        if not parent_chunks:
            logger.warning("No parent chunks found in PostgreSQL")
            return []

        # Build a map for quick lookup
        parent_map = {pc.id: pc for pc in parent_chunks}

        # Prepare texts for reranking (maintain order)
        parent_texts = []
        parent_ids_ordered = []
        for pid in unique_parent_ids:
            if pid in parent_map:
                parent_texts.append(parent_map[pid].text)
                parent_ids_ordered.append(pid)

        if not parent_texts:
            return []

        # Step 5: Rerank parent texts against the query
        ranked_results = await self.reranker.rerank(
            query=query,
            documents=parent_texts,
            top_k=top_k_rerank,
        )

        # Step 6: Build RetrievedContext with source metadata
        contexts: List[RetrievedContext] = []
        for ranked in ranked_results:
            parent_id = parent_ids_ordered[ranked.index]
            parent_chunk = parent_map[parent_id]

            source_title = parent_chunk.document.title

            contexts.append(
                RetrievedContext(
                    text=ranked.text,
                    doc_id=parent_chunk.doc_id,
                    score=ranked.score,
                    source_title=source_title,
                )
            )

        logger.info(
            "Retrieved %d contexts for query (from %d search results, "
            "%d unique parents)",
            len(contexts),
            len(search_results),
            len(unique_parent_ids),
        )
        return contexts

    async def _get_document_title(self, doc_id: str) -> str:
        """Fetch the title of a PDFDocument for source attribution."""
        from sqlalchemy.future import select

        from src.knowledge_base import PDFDocument

        result = await self.db.execute(
            select(PDFDocument.title).where(PDFDocument.id == doc_id)
        )
        row = result.first()
        return row[0] if row else "Unknown Document"
