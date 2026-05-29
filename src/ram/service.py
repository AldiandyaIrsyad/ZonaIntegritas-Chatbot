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
import logging
from typing import List

from src.infra.nli import NLIProvider, NLIResult, LABEL_NEUTRAL
from src.rag.retrieval import RetrievedContext

logger = logging.getLogger(__name__)

# How many contexts to include in the premise (prevents exceeding NLI max_length)
MAX_PREMISE_CONTEXTS = 5


class RAMService:
    """
    Response Assessment Module Service.

    Builds a premise from retrieved KB contexts and validates LLM-generated
    sentences against it via NLI. Designed to be called per-sentence inside
    the streaming generator in ChatService.

    When NLI is disabled (nli_enabled=False), assess_sentence returns a
    neutral result immediately — zero overhead, zero model calls.
    """

    def __init__(self, nli: NLIProvider, enabled: bool = True):
        self.nli = nli
        self.enabled = enabled

    def build_premise(self, contexts: List[RetrievedContext]) -> str:
        """Concatenate KB parent chunk texts into a single NLI premise.

        Called once per generate() invocation, not per sentence. The premise
        is then passed to every assess_sentence() call for that request.

        Args:
            contexts: Retrieved knowledge-base contexts (from RetrievalService).
                      Only KB contexts should be passed — session PDFs are excluded.

        Returns:
            A single string combining the top-N context texts, separated by
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
    ) -> NLIResult:
        """Run NLI on a single sentence against the pre-built KB premise.

        Args:
            sentence: A complete sentence from the LLM's response (hypothesis).
            premise: The concatenated KB context string (built via build_premise).

        Returns:
            NLIResult with canonical label and confidence scores.
            Returns neutral default if:
            - NLI is disabled via config kill-switch
            - premise is empty (no KB context was retrieved)
            - sentence is blank
        """
        if not self.enabled:
            return NLIResult(
                label=LABEL_NEUTRAL,
                entailment_score=1.0,
                contradiction_score=0.0,
            )

        if not premise or not sentence.strip():
            return NLIResult(
                label=LABEL_NEUTRAL,
                entailment_score=0.5,
                contradiction_score=0.0,
            )

        logger.debug(
            "Assessing sentence (%d chars) against premise (%d chars)",
            len(sentence),
            len(premise),
        )

        return await self.nli.check(premise=premise, hypothesis=sentence)
