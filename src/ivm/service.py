import logging
from typing import Optional

from fastapi import HTTPException

from src.infra import EmbeddingProvider, PromptGuardProvider, QdrantStore

logger = logging.getLogger(__name__)

class IVMService:
    """
    Input Validation Module (IVM)
    Responsible for validating prompts before they enter the RAG/LLM pipeline.
    """

    def __init__(
        self,
        prompt_guard: PromptGuardProvider,
        security_threshold: float,
        similarity_threshold: float,
        embedding_provider: EmbeddingProvider,
        vector_store: QdrantStore,
    ):
        self.prompt_guard = prompt_guard
        self.security_threshold = security_threshold
        self.similarity_threshold = similarity_threshold
        self.embedding_provider = embedding_provider
        self.vector_store = vector_store

    async def validate_prompt(self, query: str) -> None:
        """
        Validates the user's prompt. Raises HTTPException if validation fails.
        """
        if not query.strip():
            return
            
        await self._check_malicious(query)
        await self._check_relevance(query)

    async def _check_malicious(self, query: str) -> None:
        """
        Validates the query against Prompt Guard.
        """
        is_safe, message = await self.prompt_guard.check_prompt(query)
        if not is_safe:
            logger.warning(f"Malicious prompt detected by Prompt Guard: {message}")
            raise HTTPException(
                status_code=400,
                detail="Malicious prompt detected."
            )

    async def _check_relevance(self, query: str) -> None:
        """
        Validates that the query is relevant to the knowledge base using BGE-M3 pre-RAG check.
        """
        try:
            query_embeddings = await self.embedding_provider.embed_texts([query])
            if not query_embeddings:
                return

            query_emb = query_embeddings[0]
            
            search_results = await self.vector_store.hybrid_search(
                dense_vector=query_emb.dense,
                sparse_indices=query_emb.sparse_indices,
                sparse_values=query_emb.sparse_values,
                top_k=1,
            )

            if not search_results:
                # If KB is empty, maybe don't block? We assume KB has at least 1 PDF.
                # If no search results because KB is empty, let it pass.
                return

            best_score = search_results[0].score

            logger.info(f"Query: {query}, Best score: {best_score}, threshold: {self.similarity_threshold}")
            
            # Since hybrid_search uses RRF or cosine depending on sparse vectors, 
            # the threshold might need tuning. 
            if best_score < self.similarity_threshold:
                logger.warning(f"Irrelevant query detected. Max score {best_score} < threshold {self.similarity_threshold}")
                raise HTTPException(
                    status_code=400,
                    detail="Query is not relevant to the knowledge base."
                )

        except HTTPException:
            raise
        except Exception as e:
            logger.warning(f"Failed to check relevance: {e}", exc_info=True)

    async def validate_document_relevance(self, embeddings: list) -> None:
        """
        Validates that an uploaded document is relevant to the knowledge base
        by checking a random sample of its chunk embeddings.
        """
        if not embeddings:
            return

        import random
        # Sample up to 5 chunks to keep the check fast
        sample_size = min(5, len(embeddings))
        sampled_embeddings = random.sample(embeddings, sample_size)

        best_overall_score = 0.0

        try:
            for emb in sampled_embeddings:
                search_results = await self.vector_store.hybrid_search(
                    dense_vector=emb.dense,
                    sparse_indices=emb.sparse_indices,
                    sparse_values=emb.sparse_values,
                    top_k=1,
                )

                if not search_results:
                    # If KB is empty, we allow the upload
                    return

                best_score = search_results[0].score
                best_overall_score = max(best_overall_score, best_score)

                if best_overall_score >= self.similarity_threshold:
                    # Found at least one relevant chunk, document is allowed
                    logger.info(f"Document relevant. Best chunk score: {best_score} >= {self.similarity_threshold}")
                    return

            # If we get here, none of the sampled chunks were relevant
            logger.warning(f"Irrelevant document detected. Max score across samples: {best_overall_score} < {self.similarity_threshold}")
            raise HTTPException(
                status_code=400,
                detail="Uploaded document is not relevant to the knowledge base."
            )
        except HTTPException:
            raise
        except Exception as e:
            logger.warning(f"Failed to check document relevance: {e}", exc_info=True)
