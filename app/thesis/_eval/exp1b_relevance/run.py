"""Experiment 1b — IVM Relevance Evaluation.

Evaluates all three production IVM relevance/OOD backends
(``app/thesis/ivm/checkers.py``) against Subset C (boundary relevance
queries), plus a keyword-overlap baseline — a 4-way comparison instead of
the earlier 2-way (baseline vs LLM-judge) table.

Skripsi §3.3.2, Tabel 3.10.

Backends under test (all reused directly from production, not
reimplemented — see writing/weekend_fixes_plan.md M13):
    1. ``llm_judge``: LLMJudgeRelevanceChecker wrapping the production
       LLMJudge (app/thesis/ivm/judge.py) — same prompt, same
       max_tokens=400, same last-word parse the production system uses.
       The previous version of this script used its own ad hoc prompt,
       10-token budget, and first-substring parse, which (per §4.6.3 of
       the thesis) a reasoning model's chain-of-thought preamble defeats —
       so it was not actually testing the component the system ships.
    2. ``similarity_threshold``: SimilarityThresholdRelevanceChecker,
       thresholding the top retrieval score — no LLM call.
    3. ``nli_entailment``: NliEntailmentRelevanceChecker, thresholding NLI
       entailment_score between query and joined context.
    4. keyword-overlap baseline (JDIH_LEXICON, unchanged).

Both non-LLM backends' default thresholds are documented placeholders in
production (``ood_similarity_threshold=0.02``, ``ood_nli_entailment_threshold
=0.5`` — see app/chat/config.py) that were never calibrated against this
KB's actual score distribution; ``--similarity-threshold``/``--nli-threshold``
let a calibration sweep be run instead of trusting the placeholder.

Metrics (§3.4): Accuracy, Precision, Recall, F1, FPR + bootstrap CI.
Reported overall and per subtype.

Usage:
    python -m app.thesis._eval.exp1b_relevance.run \\
        --dataset data/subset_c.csv \\
        --api-url http://localhost:8000 \\
        --infinity-url http://localhost:7997
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Set, Tuple

import httpx

from app.thesis._eval._shared.clients import EvalNLIClient, get_llm_client_from_env
from app.thesis._eval._shared.csv_export import write_results_csv
from app.thesis._eval._shared.dataset import load_subset_c, SubsetCRow
from app.thesis._eval._shared.metrics import (
    BinaryMetrics,
    CI,
    bootstrap_binary_ci,
    compute_binary_metrics,
)
from app.thesis.ivm.checkers import (
    LLMJudgeRelevanceChecker,
    NliEntailmentRelevanceChecker,
    SimilarityThresholdRelevanceChecker,
)
from app.thesis.ivm.judge import LLMJudge

DEFAULT_OUTPUT_CSV = "data/results/exp1b_relevance.csv"


# JDIH lexicon — keywords that indicate in-domain queries about UPI's internal
# legal/regulatory documents. In production, this would be built from KB
# document titles and known terms.
JDIH_LEXICON: Set[str] = {
    "upi", "universitas pendidikan indonesia", "jdih", "statuta",
    "peraturan rektor", "sk rektor", "keputusan rektor", "rektor",
    "senat akademik", "majelis wali amanat", "mwa", "dewan pertimbangan",
    "peraturan mwa", "peraturan senat", "pedoman", "peraturan akademik",
    "ukt", "uang kuliah tunggal", "cuti akademik", "wisuda", "yudisium",
    "fakultas", "dekan", "dosen", "kepegawaian", "statuta upi",
    "organisasi dan tata kerja", "ortaker", "renstra", "peraturan universitas",
}

@dataclass
class SubtypeResult:
    """Metrics for a single subtype.

    Attributes:
        subtype: Subtype name.
        metrics: Binary classification metrics.
        accuracy_ci: Bootstrap CI for accuracy.
    """

    subtype: str
    metrics: BinaryMetrics
    accuracy_ci: CI


async def retrieve_contexts_detailed(
    api_url: str,
    query: str,
    top_k: int = 8,
) -> Tuple[List[str], List[float]]:
    """Retrieve top-k contexts and their retrieval scores from the KB.

    Returns per-chunk text and score (not just concatenated text) so all
    three IRelevanceChecker backends can be driven identically to
    production's ``RelevanceService`` — SimilarityThresholdRelevanceChecker
    needs ``context_scores``, which the earlier version of this function
    discarded (M20 in writing/weekend_fixes_plan.md).

    ``top_k=8`` matches production's ``RERANK_TOP_K``
    (app/kb/application/search_service.py) — the number of chunks the
    reranker actually hands to IVM/generation in the live pipeline,
    regardless of what a caller's own ``top_k`` requests further upstream.

    Args:
        api_url: Base URL of the running application.
        query: Search query.
        top_k: Number of contexts to retrieve.

    Returns:
        Tuple of (chunk texts, chunk scores), same order, same length.
    """
    async with httpx.AsyncClient(base_url=api_url, timeout=30.0) as client:
        response = await client.get(
            "/api/kb/search",
            params={"q": query, "top_k": top_k},
        )
        if response.status_code != 200:
            return [], []
        results = response.json()
        chunks = [r.get("text", "") for r in results]
        scores = [float(r.get("score", 0.0)) for r in results]
        return chunks, scores


async def run_llm_judge(
    checker: LLMJudgeRelevanceChecker,
    api_url: str,
    dataset: List[SubsetCRow],
    top_k: int = 8,
) -> List[bool]:
    """Run the production LLMJudgeRelevanceChecker over Subset C.

    Reuses ``app/thesis/ivm/judge.py``'s ``LLMJudge`` directly (via the
    production ``LLMJudgeRelevanceChecker`` wrapper) instead of a
    duplicated ad hoc prompt/parser, so this experiment tests the actual
    component the system ships — including its ``max_tokens=400`` budget
    and last-word parse, both load-bearing for reasoning models (§4.6.3;
    M13 in writing/weekend_fixes_plan.md).

    Args:
        checker: Production LLMJudgeRelevanceChecker instance.
        api_url: Base URL for KB search.
        dataset: List of Subset C rows.
        top_k: Number of contexts to retrieve per query.

    Returns:
        List of predictions (True = relevant/in-domain).
    """
    predictions: List[bool] = []
    for row in dataset:
        chunks, scores = await retrieve_contexts_detailed(api_url, row.query, top_k)
        try:
            predictions.append(await checker.check_query(row.query, chunks, scores))
        except Exception:
            # Fail-closed: treat errors as irrelevant (matches LLMJudge's
            # own fail-closed behavior on exception).
            predictions.append(False)
    return predictions


async def run_similarity_threshold(
    checker: SimilarityThresholdRelevanceChecker,
    api_url: str,
    dataset: List[SubsetCRow],
    top_k: int = 8,
) -> List[bool]:
    """Run the production SimilarityThresholdRelevanceChecker over Subset C.

    No LLM call — thresholds the top retrieval (RRF-fusion) score already
    returned by /api/kb/search. See the checker's own docstring
    (app/thesis/ivm/checkers.py) for why its default threshold is a
    placeholder that needs empirical calibration, which is exactly what
    this experiment (run with --similarity-threshold swept) provides.

    Args:
        checker: Production SimilarityThresholdRelevanceChecker instance.
        api_url: Base URL for KB search.
        dataset: List of Subset C rows.
        top_k: Number of contexts to retrieve per query.

    Returns:
        List of predictions (True = relevant/in-domain).
    """
    predictions: List[bool] = []
    for row in dataset:
        chunks, scores = await retrieve_contexts_detailed(api_url, row.query, top_k)
        predictions.append(await checker.check_query(row.query, chunks, scores))
    return predictions


async def run_nli_entailment(
    checker: NliEntailmentRelevanceChecker,
    api_url: str,
    dataset: List[SubsetCRow],
    top_k: int = 8,
) -> List[bool]:
    """Run the production NliEntailmentRelevanceChecker over Subset C.

    Thresholds the Indonesian NLI model's entailment_score between the
    query (hypothesis) and joined retrieved context (premise), reusing the
    same EvalNLIClient Exp3/Exp4 use.

    Args:
        checker: Production NliEntailmentRelevanceChecker instance.
        api_url: Base URL for KB search.
        dataset: List of Subset C rows.
        top_k: Number of contexts to retrieve per query.

    Returns:
        List of predictions (True = relevant/in-domain).
    """
    predictions: List[bool] = []
    for row in dataset:
        chunks, scores = await retrieve_contexts_detailed(api_url, row.query, top_k)
        predictions.append(await checker.check_query(row.query, chunks, scores))
    return predictions


def run_keyword_overlap_baseline(
    dataset: List[SubsetCRow],
    threshold: int = 1,
) -> List[bool]:
    """Run keyword-overlap baseline.

    A query is predicted as relevant if it contains at least ``threshold``
    terms from the JDIH lexicon.

    Args:
        dataset: List of Subset C rows.
        threshold: Minimum number of lexicon matches for relevance.

    Returns:
        List of predictions (True = relevant/in-domain).
    """
    predictions: List[bool] = []
    for row in dataset:
        query_lower = row.query.lower()
        matches = sum(1 for term in JDIH_LEXICON if term in query_lower)
        predictions.append(matches >= threshold)
    return predictions


def compute_per_subtype(
    predictions: List[bool],
    dataset: List[SubsetCRow],
) -> List[SubtypeResult]:
    """Compute metrics per subtype.

    Args:
        predictions: Model predictions (True = in-domain).
        dataset: Subset C rows.

    Returns:
        List of SubtypeResult, one per subtype.
    """
    by_subtype: Dict[str, List[Tuple[bool, bool]]] = defaultdict(list)
    for pred, row in zip(predictions, dataset):
        gt = row.label == "in_domain"
        by_subtype[row.subtype].append((pred, gt))

    results: List[SubtypeResult] = []
    for subtype, pairs in sorted(by_subtype.items()):
        preds = [p for p, _ in pairs]
        gts = [g for _, g in pairs]
        metrics = compute_binary_metrics(preds, gts)
        ci = bootstrap_binary_ci(preds, gts, metric="accuracy")
        results.append(SubtypeResult(subtype=subtype, metrics=metrics, accuracy_ci=ci))

    return results


def print_report(
    system_name: str,
    overall: BinaryMetrics,
    overall_ci: CI,
    per_subtype: List[SubtypeResult],
) -> None:
    """Print a formatted evaluation report.

    Args:
        system_name: Name of the system being evaluated.
        overall: Overall binary metrics.
        overall_ci: Bootstrap CI for overall accuracy.
        per_subtype: Per-subtype results.
    """
    print(f"\n{'=' * 70}")
    print(f"  {system_name}")
    print(f"{'=' * 70}")
    print(f"  Samples: {overall.total}")
    print()
    print(f"  {'Metric':<20} {'Value':>10} {'95% CI':>20}")
    print(f"  {'-' * 52}")
    print(f"  {'Accuracy':<20} {overall.accuracy:>10.4f} [{overall_ci.lower:.4f}, {overall_ci.upper:.4f}]")
    print(f"  {'Precision':<20} {overall.precision:>10.4f}")
    print(f"  {'Recall':<20} {overall.recall:>10.4f}")
    print(f"  {'F1-Score':<20} {overall.f1:>10.4f}")
    print(f"  {'FPR':<20} {overall.fpr:>10.4f}")

    if per_subtype:
        # Only Accuracy and FPR (not Precision/Recall/F1, which are
        # mathematically undefined for Subset C's single-class subtypes,
        # e.g. off_topic is 100% out_of_domain — see M15 in
        # writing/weekend_fixes_plan.md). n is shown explicitly since some
        # subtypes are small (near_miss_government is n=6) and a subtype
        # accuracy shouldn't be read as a stable rate at that size.
        print(f"\n  Per Subtype:")
        print(f"  {'Subtype':<25} {'n':>5} {'Acc':>8} {'FPR':>8}")
        print(f"  {'-' * 49}")
        for r in per_subtype:
            m = r.metrics
            print(f"  {r.subtype:<25} {m.total:>5} {m.accuracy:>8.4f} {m.fpr:>8.4f}")
    print()


async def async_main(args: argparse.Namespace) -> None:
    """Async entry point for Experiment 1b.

    Args:
        args: Parsed command-line arguments.
    """
    try:
        dataset = load_subset_c(args.dataset)
    except FileNotFoundError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)

    print(f"Loaded {len(dataset)} samples from Subset C")

    ground_truths = [row.label == "in_domain" for row in dataset]
    preds: Dict[str, List[bool]] = {}

    def write_csv() -> None:
        rows = []
        for i, row in enumerate(dataset):
            r: Dict[str, object] = {
                "query": row.query,
                "subtype": row.subtype,
                "true_label": row.label,
            }
            for name, p in preds.items():
                r[f"{name}_pred"] = "in_domain" if p[i] else "out_of_domain"
                r[f"{name}_correct"] = p[i] == ground_truths[i]
            rows.append(r)
        write_results_csv(args.output_csv, rows)

    # --- 1. Keyword overlap baseline ---
    print("\nRunning keyword-overlap baseline...")
    preds["baseline"] = run_keyword_overlap_baseline(dataset, threshold=args.keyword_threshold)
    print_report(
        "Baseline (Keyword Overlap)",
        compute_binary_metrics(preds["baseline"], ground_truths),
        bootstrap_binary_ci(preds["baseline"], ground_truths, metric="accuracy"),
        compute_per_subtype(preds["baseline"], dataset),
    )

    # --- 2. LLM-as-Judge (production LLMJudge, via LLMJudgeRelevanceChecker) ---
    if args.skip_llm_judge:
        print("\nSkipping LLM-as-Judge (--skip-llm-judge)")
    else:
        try:
            # A single session_id lets OpenRouter route every judge call for
            # this run to the same upstream provider, so the shared judge
            # system prompt (identical on all 56 calls) can be provider-
            # cached instead of re-billed in full each time.
            llm_client = get_llm_client_from_env(session_id="exp1b-llm-judge")
        except ValueError as e:
            print(f"\nSkipping LLM-as-Judge: {e}", file=sys.stderr)
        else:
            try:
                judge = LLMJudge(llm_connection=llm_client, model=llm_client.model)
                checker = LLMJudgeRelevanceChecker(judge)
                print("\nRunning LLM-as-Judge (production LLMJudge)...")
                preds["llm_judge"] = await run_llm_judge(checker, args.api_url, dataset, args.top_k)
            finally:
                await llm_client.aclose()
            print_report(
                "LLM-as-Judge (production)",
                compute_binary_metrics(preds["llm_judge"], ground_truths),
                bootstrap_binary_ci(preds["llm_judge"], ground_truths, metric="accuracy"),
                compute_per_subtype(preds["llm_judge"], dataset),
            )

    # --- 3. similarity_threshold (no LLM call) ---
    if args.skip_similarity:
        print("\nSkipping similarity_threshold (--skip-similarity)")
    else:
        sim_checker = SimilarityThresholdRelevanceChecker(threshold=args.similarity_threshold)
        print(f"\nRunning similarity_threshold (threshold={args.similarity_threshold})...")
        preds["similarity_threshold"] = await run_similarity_threshold(sim_checker, args.api_url, dataset, args.top_k)
        print_report(
            f"Similarity Threshold ({args.similarity_threshold})",
            compute_binary_metrics(preds["similarity_threshold"], ground_truths),
            bootstrap_binary_ci(preds["similarity_threshold"], ground_truths, metric="accuracy"),
            compute_per_subtype(preds["similarity_threshold"], dataset),
        )

    # --- 4. nli_entailment ---
    if args.skip_nli:
        print("\nSkipping nli_entailment (--skip-nli)")
    else:
        nli_client = EvalNLIClient(base_url=args.infinity_url, model=args.nli_model)
        try:
            nli_checker = NliEntailmentRelevanceChecker(nli_model=nli_client, threshold=args.nli_threshold)
            print(f"\nRunning nli_entailment (threshold={args.nli_threshold})...")
            preds["nli_entailment"] = await run_nli_entailment(nli_checker, args.api_url, dataset, args.top_k)
        finally:
            await nli_client.aclose()
        print_report(
            f"NLI Entailment ({args.nli_threshold})",
            compute_binary_metrics(preds["nli_entailment"], ground_truths),
            bootstrap_binary_ci(preds["nli_entailment"], ground_truths, metric="accuracy"),
            compute_per_subtype(preds["nli_entailment"], dataset),
        )

    write_csv()


def main() -> None:
    """Entry point for Experiment 1b."""
    parser = argparse.ArgumentParser(
        description=(
            "Experiment 1b: IVM Relevance Evaluation — keyword-overlap baseline vs "
            "the three production OOD backends (llm_judge, similarity_threshold, "
            "nli_entailment)."
        )
    )
    parser.add_argument("--dataset", required=True, help="Path to Subset C CSV")
    parser.add_argument(
        "--api-url",
        default="http://localhost:8000",
        help="Base URL of the running application (for KB search)",
    )
    parser.add_argument(
        "--infinity-url",
        default="http://localhost:7997",
        help="Infinity server base URL (for nli_entailment)",
    )
    parser.add_argument(
        "--nli-model",
        default="StevenLimcorn/indo-roberta-indonli",
        help="NLI model identifier (must match a model loaded on the Infinity server)",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=8,
        help="Number of contexts to retrieve per query (default: 8, matching "
        "production's RERANK_TOP_K)",
    )
    parser.add_argument(
        "--keyword-threshold",
        type=int,
        default=1,
        help="Minimum lexicon matches for keyword baseline (default: 1)",
    )
    parser.add_argument(
        "--similarity-threshold",
        type=float,
        default=0.02,
        help="Threshold for similarity_threshold backend (default: 0.02, "
        "matching production's placeholder default — calibrate empirically, "
        "these are RRF-fusion scores, not bounded cosine similarity)",
    )
    parser.add_argument(
        "--nli-threshold",
        type=float,
        default=0.5,
        help="Minimum entailment_score for nli_entailment backend (default: 0.5, "
        "matching production's default)",
    )
    parser.add_argument(
        "--skip-llm-judge",
        action="store_true",
        help="Skip the LLM-as-Judge backend",
    )
    parser.add_argument(
        "--skip-similarity",
        action="store_true",
        help="Skip the similarity_threshold backend",
    )
    parser.add_argument(
        "--skip-nli",
        action="store_true",
        help="Skip the nli_entailment backend",
    )
    parser.add_argument(
        "--output-csv",
        default=DEFAULT_OUTPUT_CSV,
        help="Path to write raw per-row results CSV",
    )
    args = parser.parse_args()
    asyncio.run(async_main(args))


if __name__ == "__main__":
    main()
