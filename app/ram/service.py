"""
RAM Service — Response Assessment Module

Validates LLM sentences against KB RAG context using NLI.

Usage:
    # Build premise once per request (expensive concatenation done once)
    premise = ram_service.build_premise(contexts)

    # Call per sentence inside the streaming generator
    result = await ram_service.assess_sentence(sentence, premise)
    if result.label == "contradiction":
        sentence += " *(contradictive)*"
"""
import dataclasses
import math
from typing import List, Optional

import structlog

from app.core.interfaces.ai import IEmbeddingProvider, INLIProvider, NLIResult, EmbeddingResult
from app.core.interfaces.rag import RetrievedContext
from app.core.interfaces.ram import IRAMService

logger = structlog.get_logger(__name__)

LABEL_NEUTRAL = "neutral"
LABEL_ENTAILMENT = "entailment"
LABEL_CONTRADICTION = "contradiction"

# How many contexts to include in the premise (prevents exceeding NLI max_length)
MAX_PREMISE_CONTEXTS = 5


class RAMService(IRAMService):
    """
    Response Assessment Module Service.

    Builds a premise from retrieved KB contexts and validates LLM-generated
    sentences against it via NLI. Designed to be called per-sentence inside
    the streaming generator in ChatService.

    When NLI is disabled (nli_enabled=False), assess_sentence returns a
    neutral result immediately — zero overhead, zero model calls.

    Args:
        nli (INLIProvider): The NLI inference client.
        embedding_provider (IEmbeddingProvider): The embedding client.
        enabled (bool, optional): Whether NLI assessment is enabled. Defaults to True.
    """

    def __init__(self, nli: INLIProvider, embedding_provider: IEmbeddingProvider, enabled: bool = True):
        self.nli = nli
        self.embedding_provider = embedding_provider
        self.enabled = enabled

    def build_premise(self, contexts: List[RetrievedContext]) -> str:
        """Concatenate KB parent chunk texts into a single NLI premise.

        Called once per generate() invocation, not per sentence. The premise
        is then passed to every assess_sentence() call for that request.

        Args:
            contexts (List[RetrievedContext]): Retrieved knowledge-base contexts (from RetrievalService).
                      Only KB contexts should be passed — session PDFs are excluded.

        Returns:
            str: A single string combining the top-N context texts, separated by
                 double newlines. Returns "" if contexts is empty.
        """
        if not contexts:
            return ""

        # Limit to top-N to stay within model token limits
        top_contexts = contexts[:MAX_PREMISE_CONTEXTS]
        premise = "\n\n".join(ctx.text for ctx in top_contexts)

        logger.debug(
            "Built NLI premise from %d contexts (%d chars)",
            len(top_contexts),
            len(premise),
        )
        return premise

    async def assess_sentence(
        self,
        sentence: str,
        premise: str,
        contexts: List[RetrievedContext],
        context_embs: Optional[List[EmbeddingResult]] = None,
    ) -> NLIResult:
        """Run NLI on a single sentence against the most relevant KB premise chunk.
        Uses embeddings to find the most relevant chunk first, then runs NLI against it.

        Args:
            sentence (str): A complete sentence from the LLM's response (hypothesis).
            premise (str): The concatenated KB context string (legacy, not used for NLI).
            contexts (List[RetrievedContext]): The retrieved contexts to reverse map against.
            context_embs (Optional[List[EmbeddingResult]]): Precomputed embeddings for contexts.

        Returns:
            NLIResult: NLIResult with canonical label and confidence scores.
                Returns neutral default if:
                - NLI is disabled via config kill-switch
                - contexts is empty
                - sentence is blank
        """
        if not self.enabled:
            return NLIResult(
                label=LABEL_NEUTRAL,
                entailment_score=1.0,
                contradiction_score=0.0,
            )

        if not contexts or not sentence.strip():
            return NLIResult(
                label=LABEL_NEUTRAL,
                entailment_score=0.5,
                contradiction_score=0.0,
            )

        best_ctx = None

        try:
            # Reverse mapping using embeddings to find the best context FIRST
            sentence_embs = await self.embedding_provider.embed_texts([sentence])
            if not sentence_embs:
                return NLIResult(label=LABEL_NEUTRAL, entailment_score=0.5, contradiction_score=0.0)
                
            sentence_dense = sentence_embs[0].dense
            
            top_contexts = contexts[:MAX_PREMISE_CONTEXTS]
            
            if context_embs is None:
                context_texts = [ctx.text for ctx in top_contexts]
                context_embs = await self.embedding_provider.embed_texts(context_texts)
            
            if not context_embs or len(context_embs) != len(top_contexts):
                return NLIResult(label=LABEL_NEUTRAL, entailment_score=0.5, contradiction_score=0.0)

            best_idx = 0
            best_score = -1.0
            
            for i, ctx_emb in enumerate(context_embs):
                c_dense = ctx_emb.dense
                
                # Cosine similarity
                dot_product = sum(a * b for a, b in zip(sentence_dense, c_dense))
                norm_a = math.sqrt(sum(a * a for a in sentence_dense))
                norm_b = math.sqrt(sum(b * b for b in c_dense))
                
                if norm_a == 0 or norm_b == 0:
                    score = 0.0
                else:
                    score = dot_product / (norm_a * norm_b)
                    
                if score > best_score:
                    best_score = score
                    best_idx = i
                    
            best_ctx = top_contexts[best_idx]
            
        except Exception as e:
            logger.warning("Failed to reverse map citation: %s", str(e), exc_info=True)
            return NLIResult(label=LABEL_NEUTRAL, entailment_score=0.5, contradiction_score=0.0)

        # Now run NLI using ONLY the best context text as the premise
        best_premise = best_ctx.text

        import difflib
        if len(best_premise) > 800:
            s = difflib.SequenceMatcher(None, best_premise, sentence)
            match = s.find_longest_match(0, len(best_premise), 0, len(sentence))
            
            window_size = 600
            start_idx = max(0, match.a - window_size // 2)
            end_idx = min(len(best_premise), match.a + match.size + window_size // 2)
            best_premise = best_premise[start_idx:end_idx]

        logger.debug(
            "Assessing sentence (%d chars) against best context premise (%d chars)",
            len(sentence),
            len(best_premise),
        )

        try:
            result = await self.nli.check(premise=best_premise, hypothesis=sentence)
            
            # NLIResult is frozen, use dataclasses.replace to attach metadata
            return dataclasses.replace(
                result, 
                source_title=best_ctx.source_title, 
                page=best_ctx.page
            )
        except Exception as e:
            logger.warning("NLI check failed: %s", str(e), exc_info=True)
            return NLIResult(label=LABEL_NEUTRAL, entailment_score=0.5, contradiction_score=0.0)
