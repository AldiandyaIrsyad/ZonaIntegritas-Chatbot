"""
Concrete IRelevanceChecker implementations.

- LLMJudgeRelevanceChecker: wraps the existing LLM-as-judge (IJudge), one
  call per check, fails closed on error.
- SimilarityThresholdRelevanceChecker: thresholds retrieval scores only,
  no model call (kNN-OOD framing, Sun et al. ICML 2022).
- NliEntailmentRelevanceChecker: thresholds NLI entailment_score between
  query and joined context (Yin, Hay & Roth, EMNLP 2019).

Fulfills: ``app/thesis/ivm/interfaces.py::IRelevanceChecker``.
Wired in: ``app/chat/dependency.py::get_relevance_checker``, which selects
one of the three classes below at runtime based on ``ChatConfig.ood_method``
("llm_judge" / "similarity_threshold" / "nli_entailment").

Part of the pure ``thesis`` research core: no infra imports (see
``docs/02-arsitektur.md`` §2.2). ``NliEntailmentRelevanceChecker`` depends
only on the ``INLIModel`` Protocol from ``app/thesis/ram/interfaces.py``,
not on any concrete HTTP client.
"""
from typing import List

import structlog

from .interfaces import IJudge, IRelevanceChecker
from app.thesis.ram.interfaces import INLIModel

logger = structlog.get_logger(__name__)


class LLMJudgeRelevanceChecker(IRelevanceChecker):
    """Relevance backend using an LLM-as-judge (today's default method).

    Delegates the actual relevance decision to an injected :class:`IJudge`
    (see ``app/thesis/ivm/judge.py::LLMJudge``) — one LLM call per check.
    Fails closed: any exception from the judge propagates to the caller
    (``RelevanceService.check_relevance``) rather than being swallowed here.

    Fulfills: ``app/thesis/ivm/interfaces.py::IRelevanceChecker``.
    Wired in: ``app/chat/dependency.py::get_relevance_checker`` (selected
    when ``ChatConfig.ood_method == "llm_judge"``, the default).
    """

    def __init__(self, judge: IJudge):
        """Args:
            judge: The LLM-as-judge backend to delegate relevance calls to.
        """
        self.judge = judge

    async def check_query(
        self, query: str, context_chunks: List[str], context_scores: List[float]
    ) -> bool:
        """Ask the judge whether ``query`` is on-topic for the joined context.

        ``context_scores`` is accepted for :class:`IRelevanceChecker`
        interface compatibility but unused — this backend only reasons
        over the context text, not the retrieval scores.

        Args:
            query: The user's raw query text.
            context_chunks: Text of the top retrieved KB contexts.
            context_scores: Unused by this implementation.

        Returns:
            bool: True if the judge considers the query in-domain.
        """
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

    Fulfills: ``app/thesis/ivm/interfaces.py::IRelevanceChecker``.
    Wired in: ``app/chat/dependency.py::get_relevance_checker`` (selected
    when ``ChatConfig.ood_method == "similarity_threshold"``, via
    ``config.ood_similarity_threshold``).
    """

    def __init__(self, threshold: float):
        """Args:
            threshold: Minimum top retrieval score for a query to be
                considered in-domain. Must be calibrated against this KB's
                own RRF score distribution (see class docstring).
        """
        self.threshold = threshold

    async def check_query(
        self, query: str, context_chunks: List[str], context_scores: List[float]
    ) -> bool:
        """Decide relevance from the max retrieval score alone.

        Args:
            query: The user's raw query text (used only for logging here).
            context_chunks: Unused by this implementation.
            context_scores: Similarity/RRF scores already computed by
                retrieval, in the same order as ``context_chunks``.

        Returns:
            bool: True if the top score meets or exceeds ``self.threshold``.
        """
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

    Fulfills: ``app/thesis/ivm/interfaces.py::IRelevanceChecker``.
    Wired in: ``app/chat/dependency.py::get_relevance_checker`` (selected
    when ``ChatConfig.ood_method == "nli_entailment"``, sharing the same
    ``NLIClient`` instance wired for RAM via
    ``app/chat/dependency.py::get_nli_client``).
    """

    def __init__(self, nli_model: INLIModel, threshold: float):
        """Args:
            nli_model: NLI backend (``INLIModel`` Protocol) used to score
                entailment between context and query.
            threshold: Minimum entailment_score for a query to be considered
                in-domain (see KNOWN LIMITATION above re: fail-open default).
        """
        self.nli_model = nli_model
        self.threshold = threshold

    async def check_query(
        self, query: str, context_chunks: List[str], context_scores: List[float]
    ) -> bool:
        """Run NLI with the joined context as premise and the query as hypothesis.

        Args:
            query: The user's raw query text — the NLI hypothesis.
            context_chunks: Text of the top retrieved KB contexts, joined
                into a single NLI premise.
            context_scores: Unused by this implementation.

        Returns:
            bool: True if ``entailment_score >= self.threshold``.
        """
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
