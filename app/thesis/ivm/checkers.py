"""
Concrete IRelevanceChecker implementations.

- LLMJudgeRelevanceChecker: wraps the existing LLM-as-judge (IJudge), one
  call per check, fails closed on error.
- SimilarityThresholdRelevanceChecker: thresholds retrieval scores only,
  no model call (kNN-OOD framing, Sun et al. ICML 2022).
- NliEntailmentRelevanceChecker: thresholds NLI entailment_score between
  query and joined context (Yin, Hay & Roth, EMNLP 2019).
"""
from typing import List

import structlog

from .interfaces import IJudge, IRelevanceChecker
from app.thesis.ram.interfaces import INLIModel

logger = structlog.get_logger(__name__)


class LLMJudgeRelevanceChecker(IRelevanceChecker):
    """Relevance backend using an LLM-as-judge (today's default method)."""

    def __init__(self, judge: IJudge):
        self.judge = judge

    async def check_query(
        self, query: str, context_chunks: List[str], context_scores: List[float]
    ) -> bool:
        combined_context = "\n".join(context_chunks)
        return await self.judge.evaluate_relevance(query, combined_context)


class SimilarityThresholdRelevanceChecker(IRelevanceChecker):
    """Relevance backend using only the retrieval scores already computed
    upstream — no LLM call, no re-embedding, no new dependencies.

    Treats the top (max) retrieval score as a nearest-neighbor OOD signal,
    following the kNN-OOD framing of Sun et al., "Out-of-Distribution
    Detection with Deep Nearest Neighbors", ICML 2022: distance/similarity
    to the nearest in-distribution neighbor(s) in embedding space is used
    directly as the OOD score, without any extra classifier or LLM call.

    IMPORTANT: ``context_scores`` are Qdrant RRF-fusion scores (see
    app/kb/infra/qdrant_store.py::hybrid_search, models.Fusion.RRF), NOT a
    bounded cosine similarity in [0, 1]. RRF scores are a sum of
    1/(rank + k) terms across the fused ranked lists, so their scale is
    small and depends on how many ranked lists a candidate appears in.
    ``threshold`` must be calibrated empirically against this KB's actual
    score distribution — do not assume a default tuned elsewhere carries
    over.
    """

    def __init__(self, threshold: float):
        self.threshold = threshold

    async def check_query(
        self, query: str, context_chunks: List[str], context_scores: List[float]
    ) -> bool:
        if not context_scores:
            logger.warning("similarity_threshold_checker.no_scores", query=query[:100])
            return False

        top_score = max(context_scores)
        is_relevant = top_score >= self.threshold
        logger.info(
            "similarity_threshold_checker.result",
            query=query[:100],
            top_score=top_score,
            threshold=self.threshold,
            is_relevant=is_relevant,
        )
        return is_relevant


class NliEntailmentRelevanceChecker(IRelevanceChecker):
    """Relevance backend using an NLI model's entailment score between the
    joined KB context (premise) and the user query (hypothesis).

    Grounded in Yin, Hay & Roth, "Benchmarking Zero-shot Text
    Classification: Datasets, Evaluations and Entailment Approach", EMNLP
    2019 — using the entailment probability from a general-purpose NLI
    model as an off-the-shelf relevance/classification score, rather than
    training a task-specific classifier.

    Reuses the Indonesian NLI model already wired up for RAM
    (StevenLimcorn/indo-roberta-indonli via Infinity) through the
    INLIModel Protocol (app/thesis/ram/interfaces.py). This module depends
    on the Protocol only, not app/chat/infra's concrete NLIClient — the
    concrete instance is injected at the composition root
    (app/chat/dependency.py) to preserve dependency inversion.

    Design note: thresholds the continuous entailment_score rather than
    checking ``label == "entailment"``. For genuinely off-topic queries
    the NLI model frequently outputs low-confidence "neutral" rather than
    a clean unrelated signal, so a discrete-label check would under-reject
    off-topic queries; thresholding the continuous score is more robust
    (mirrors RAMService's own NLI_CONFIDENCE_THRESHOLD approach in
    app/thesis/ram/service.py).

    KNOWN LIMITATION: NLIClient.check() fails open on infra errors — it
    catches exceptions internally and returns a neutral
    NLIResult(entailment_score=0.5, ...) rather than raising (see
    app/chat/infra/nli_client.py). At the default threshold of 0.5 this
    means an Infinity outage is treated as relevant rather than rejected,
    unlike LLMJudgeRelevanceChecker's fail-closed behavior on exception.
    If fail-closed semantics matter for your deployment, set
    ``ood_nli_entailment_threshold`` above 0.5 (e.g. 0.55) as a mitigation.
    """

    def __init__(self, nli_model: INLIModel, threshold: float):
        self.nli_model = nli_model
        self.threshold = threshold

    async def check_query(
        self, query: str, context_chunks: List[str], context_scores: List[float]
    ) -> bool:
        if not context_chunks:
            logger.warning("nli_entailment_checker.no_context", query=query[:100])
            return False

        premise = "\n".join(context_chunks)
        result = await self.nli_model.check(premise=premise, hypothesis=query)
        is_relevant = result.entailment_score >= self.threshold
        logger.info(
            "nli_entailment_checker.result",
            query=query[:100],
            label=result.label,
            entailment_score=result.entailment_score,
            threshold=self.threshold,
            is_relevant=is_relevant,
        )
        return is_relevant
