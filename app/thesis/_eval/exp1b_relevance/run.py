"""Experiment 1b — IVM Relevance (LLM-as-Judge) Evaluation.

Evaluates the LLM-as-Judge relevance assessment against Subset C (boundary
relevance queries). Compares against a keyword-overlap baseline.

Skripsi §3.3.2, Tabel 3.10.

The LLM-as-Judge pipeline:
    1. Retrieve top-3 contexts from KB via /api/kb/search
    2. LLM-as-Judge gives binary verdict (relevant/irrelevant)
    3. Compare to ground truth

Baseline: keyword overlap — query is relevant if it contains ZI lexicon terms.

Metrics (§3.4): Accuracy, Precision, Recall, F1, FPR + bootstrap CI.
Reported overall and per subtype.

Usage:
    python -m app.thesis._eval.exp1b_relevance.run \\
        --dataset data/subset_c.csv \\
        --api-url http://localhost:8000
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Set, Tuple

import httpx

from app.thesis._eval._shared.clients import EvalLLMClient, get_llm_client_from_env
from app.thesis._eval._shared.dataset import load_subset_c, SubsetCRow
from app.thesis._eval._shared.metrics import (
    BinaryMetrics,
    CI,
    bootstrap_binary_ci,
    compute_binary_metrics,
)


# ZI lexicon — keywords that indicate in-domain ZI queries.
# In production, this would be built from KB document titles and known terms.
ZI_LEXICON: Set[str] = {
    "zona integritas", "zi", "wbk", "wbbm", "wilayah birokrasi bersih",
    "wilayah bebas korupsi", "lke", "laporan kinerja", "area perubahan",
    "manajemen perubahan", "akuntabilitas", "transparansi", "anti korupsi",
    "pemberantasan korupsi", "integritas", "birokrasi", "reformasi birokrasi",
    "panrb", "aparatur negara", "permenpan", "kepatuhan", "etika pemerintahan",
    "pelayanan publik", "mal administrasi", "whistleblowing", "gratifikasi",
    "sistem manajemen mutu", "smm", "good governance", "tata kelola",
}

JUDGE_SYSTEM_PROMPT = """\
You are a relevance judge. Given a user query and retrieved context from a \
knowledge base about Indonesian bureaucratic reform (Zona Integritas), \
determine if the query can be answered using ONLY the provided context.

Respond with ONLY "RELEVANT" or "IRRELEVANT".
- RELEVANT: The context contains information to answer the query.
- IRRELEVANT: The context does not contain relevant information, or the query \
is outside the knowledge base domain.
"""


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


async def retrieve_contexts(
    api_url: str,
    query: str,
    top_k: int = 3,
) -> str:
    """Retrieve top-k contexts from the KB search endpoint.

    Args:
        api_url: Base URL of the running application.
        query: Search query.
        top_k: Number of contexts to retrieve.

    Returns:
        Concatenated context text.
    """
    async with httpx.AsyncClient(base_url=api_url, timeout=30.0) as client:
        response = await client.get(
            "/api/kb/search",
            params={"q": query, "top_k": top_k},
        )
        if response.status_code != 200:
            return ""
        results = response.json()
        return "\n\n".join(r.get("text", "") for r in results)


async def run_llm_judge(
    llm_client: EvalLLMClient,
    api_url: str,
    dataset: List[SubsetCRow],
    top_k: int = 3,
) -> List[bool]:
    """Run LLM-as-Judge relevance evaluation.

    Args:
        llm_client: Configured LLM client for the judge.
        api_url: Base URL for KB search.
        dataset: List of Subset C rows.
        top_k: Number of contexts to retrieve per query.

    Returns:
        List of predictions (True = relevant/in-domain).
    """
    predictions: List[bool] = []
    for row in dataset:
        context = await retrieve_contexts(api_url, row.query, top_k)
        messages = [
            {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
            {"role": "user", "content": f"Query: {row.query}\n\nContext:\n{context}"},
        ]
        try:
            response = await llm_client.chat(messages, temperature=0.0, max_tokens=10)
            is_relevant = "RELEVANT" in response.upper() and "IRRELEVANT" not in response.upper()
            predictions.append(is_relevant)
        except Exception:
            # Fail-closed: treat errors as irrelevant
            predictions.append(False)
    return predictions


def run_keyword_overlap_baseline(
    dataset: List[SubsetCRow],
    threshold: int = 1,
) -> List[bool]:
    """Run keyword-overlap baseline.

    A query is predicted as relevant if it contains at least ``threshold``
    terms from the ZI lexicon.

    Args:
        dataset: List of Subset C rows.
        threshold: Minimum number of lexicon matches for relevance.

    Returns:
        List of predictions (True = relevant/in-domain).
    """
    predictions: List[bool] = []
    for row in dataset:
        query_lower = row.query.lower()
        matches = sum(1 for term in ZI_LEXICON if term in query_lower)
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
        print(f"\n  Per Subtype:")
        print(f"  {'Subtype':<25} {'Acc':>8} {'Prec':>8} {'Rec':>8} {'F1':>8} {'FPR':>8}")
        print(f"  {'-' * 69}")
        for r in per_subtype:
            m = r.metrics
            print(
                f"  {r.subtype:<25} {m.accuracy:>8.4f} {m.precision:>8.4f} "
                f"{m.recall:>8.4f} {m.f1:>8.4f} {m.fpr:>8.4f}"
            )
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

    # --- Keyword overlap baseline ---
    print("\nRunning keyword-overlap baseline...")
    baseline_preds = run_keyword_overlap_baseline(dataset, threshold=args.keyword_threshold)
    baseline_metrics = compute_binary_metrics(baseline_preds, ground_truths)
    baseline_ci = bootstrap_binary_ci(baseline_preds, ground_truths, metric="accuracy")
    baseline_subtype = compute_per_subtype(baseline_preds, dataset)
    print_report("Baseline (Keyword Overlap)", baseline_metrics, baseline_ci, baseline_subtype)

    if args.no_judge:
        return

    # --- LLM-as-Judge ---
    try:
        llm_client = get_llm_client_from_env()
    except ValueError as e:
        print(f"\nSkipping LLM-as-Judge: {e}", file=sys.stderr)
        return

    try:
        print("\nRunning LLM-as-Judge...")
        judge_preds = await run_llm_judge(llm_client, args.api_url, dataset, args.top_k)
    finally:
        await llm_client.aclose()

    judge_metrics = compute_binary_metrics(judge_preds, ground_truths)
    judge_ci = bootstrap_binary_ci(judge_preds, ground_truths, metric="accuracy")
    judge_subtype = compute_per_subtype(judge_preds, dataset)
    print_report("LLM-as-Judge", judge_metrics, judge_ci, judge_subtype)


def main() -> None:
    """Entry point for Experiment 1b."""
    parser = argparse.ArgumentParser(
        description="Experiment 1b: IVM Relevance (LLM-as-Judge) Evaluation vs keyword-overlap baseline."
    )
    parser.add_argument("--dataset", required=True, help="Path to Subset C CSV")
    parser.add_argument(
        "--api-url",
        default="http://localhost:8000",
        help="Base URL of the running application (for KB search)",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=3,
        help="Number of contexts to retrieve per query (default: 3)",
    )
    parser.add_argument(
        "--keyword-threshold",
        type=int,
        default=1,
        help="Minimum lexicon matches for keyword baseline (default: 1)",
    )
    parser.add_argument(
        "--no-judge",
        action="store_true",
        help="Skip LLM-as-Judge (run baseline only)",
    )
    args = parser.parse_args()
    asyncio.run(async_main(args))


if __name__ == "__main__":
    main()
