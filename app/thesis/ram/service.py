"""
RAM Service — Response Assessment Module

Validates LLM sentences against KB RAG context using NLI.

Usage:
    # Build premise once per request (expensive concatenation done once)
    premise = ram_service.build_premise(contexts)

    # Call per sentence inside the streaming generator
    result = await ram_service.assess_sentence(sentence, premise, contexts)
    if result.label == "contradiction":
        sentence += " *(contradictive)*"
"""
import dataclasses
import difflib
import math
from typing import List, Optional

import structlog

from .interfaces import IRerankerModel, INLIModel, NLIResult, RetrievedContext
from .text_utils import split_sentences

logger = structlog.get_logger(__name__)

LABEL_NEUTRAL = "neutral"
LABEL_ENTAILMENT = "entailment"
LABEL_CONTRADICTION = "contradiction"

# How many contexts to include in the premise (prevents exceeding NLI max_length)
MAX_PREMISE_CONTEXTS = 5

# Max length of the evidence snippet surfaced in citation tooltips.
EVIDENCE_SNIPPET_MAX_CHARS = 140


def _sanitize_snippet(text: str, max_len: int = EVIDENCE_SNIPPET_MAX_CHARS) -> str:
    """Collapse whitespace and strip characters that would break the
    ``*(STATUS: SCORE; ...; Evidence:"...")*`` citation marker grammar,
    then truncate for display.
    """
    cleaned = " ".join(text.split())
    cleaned = cleaned.replace('"', "'")
    cleaned = cleaned.translate(str.maketrans("", "", ";)*"))
    if len(cleaned) > max_len:
        cleaned = cleaned[: max_len - 1].rstrip() + "…"
    return cleaned


class RAMService:
    """
    Response Assessment Module Service.

    Builds a premise from retrieved KB contexts and validates LLM-generated
    sentences against it via NLI. Designed to be called per-sentence inside
    the streaming generator in ChatService.

    When NLI is disabled (nli_enabled=False), assess_sentence returns a
    neutral result immediately — zero overhead, zero model calls.

    Args:
        nli_model (INLIModel): The NLI inference client.
        reranker_model (IRerankerModel): The reranker client for exact premise extraction.
        enabled (bool, optional): Whether NLI assessment is enabled. Defaults to True.
    """

    def __init__(self, nli_model: INLIModel, reranker_model: IRerankerModel, enabled: bool = True):
        self.nli_model = nli_model
        self.reranker_model = reranker_model
        self.enabled = enabled

    def build_premise(self, contexts: List[RetrievedContext]) -> str:
        """Concatenate KB parent chunk texts into a single NLI premise.

        Called once per generate() invocation, not per sentence. The premise
        is then passed to every assess_sentence() call for that request.

        Args:
            contexts (List[RetrievedContext]): Retrieved knowledge-base contexts.

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
    ) -> NLIResult:
        """Run NLI on a single sentence (proposition) against the most relevant KB premise chunk.
        Uses a reranker with sliding windows to find the exact sub-chunk first, then runs NLI against it.

        Args:
            sentence (str): A complete sentence from the LLM's response (hypothesis).
            premise (str): The concatenated KB context string (legacy, not used for NLI).
            contexts (List[RetrievedContext]): The retrieved contexts to reverse map against.

        Returns:
            NLIResult: NLIResult with canonical label and confidence scores.
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

        best_ctx = contexts[0]
        best_premise = best_ctx.text[:800] # Fallback
        
        try:
            top_contexts = contexts[:MAX_PREMISE_CONTEXTS]
            
            # Create sliding windows of ~2-3 sentences for precise reranking
            windows: List[str] = []
            window_to_ctx: List[RetrievedContext] = []
            
            for ctx in top_contexts:
                # Split into sentence-like units for windowing (guards
                # against markdown list markers being mistaken for
                # sentence ends; see text_utils.split_sentences).
                sentences = [
                    s if s.endswith((".", "?", "!")) else s + "."
                    for s in split_sentences(ctx.text)
                ]
                # Group into windows of 3 sentences with 1 sentence overlap
                if not sentences:
                    windows.append(ctx.text)
                    window_to_ctx.append(ctx)
                    continue
                    
                window_size = 3
                step = 2
                for i in range(0, max(1, len(sentences)), step):
                    window = " ".join(sentences[i:i + window_size])
                    if len(window) > 20: # skip tiny fragments
                        windows.append(window)
                        window_to_ctx.append(ctx)
            
            if not windows:
                return NLIResult(label=LABEL_NEUTRAL, entailment_score=0.5, contradiction_score=0.0)

            # Rerank windows against the hypothesis sentence
            rerank_results = await self.reranker_model.rerank(
                query=sentence,
                documents=windows,
                top_k=1
            )
            
            if rerank_results:
                top_result = rerank_results[0]
                best_idx = top_result.index
                if best_idx < len(windows):
                    best_premise = windows[best_idx]
                    best_ctx = window_to_ctx[best_idx]
                    
        except Exception as e:
            logger.warning("Failed to reverse map citation with reranker: %s", str(e), exc_info=True)
            return NLIResult(label=LABEL_NEUTRAL, entailment_score=0.5, contradiction_score=0.0)

        logger.debug(
            "Assessing sentence (%d chars) against best context premise (%d chars)",
            len(sentence),
            len(best_premise),
        )

        try:
            result = await self.nli_model.check(premise=best_premise, hypothesis=sentence)
            
            # NLIResult is frozen, use dataclasses.replace to attach metadata
            return dataclasses.replace(
                result,
                source_title=best_ctx.source_title,
                page=best_ctx.page,
                doc_id=best_ctx.doc_id,
                evidence_snippet=_sanitize_snippet(best_premise),
            )
        except Exception as e:
            logger.warning("NLI check failed: %s", str(e), exc_info=True)
            return NLIResult(label=LABEL_NEUTRAL, entailment_score=0.5, contradiction_score=0.0)