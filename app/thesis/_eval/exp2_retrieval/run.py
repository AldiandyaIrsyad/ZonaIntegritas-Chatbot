"""Experiment 2 — Retrieval Quality Evaluation.

Evaluates hybrid retrieval (dense + sparse with RRF fusion) against
dense-only and sparse-only baselines using Subset A (RAG QA triplets).

Skripsi §3.3.3, Tabel 3.10.

Metrics (§3.4): Hit Rate@k (k=1,3,5) and MRR, reported per category
and overall.

Usage:
    python -m app.thesis._eval.exp2_retrieval.run \\
        --dataset data/subset_a.csv \\
        --api-url http://localhost:8000
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Tuple

import httpx

from app.thesis._eval._shared.dataset import load_subset_a, SubsetARow
from app.thesis._eval._shared.metrics import RetrievalMetrics, compute_retrieval_metrics


@dataclass
class CategoryResult:
    """Retrieval metrics for a single category.

    Attributes:
        category: Category name.
        metrics: Aggregated retrieval metrics.
        sample_count: Number of queries in this category.
    """

    category: str
    metrics: RetrievalMetrics
    sample_count: int


async def retrieve(
    api_url: str,
    query: str,
    top_k: int,
    mode: str,
) -> List[str]:
    """Retrieve document IDs from the KB search endpoint.

    Args:
        api_url: Base URL of the running application.
        query: Search query.
        top_k: Number of results to retrieve.
        mode: Retrieval mode ("hybrid", "dense", or "sparse").

    Returns:
        List of retrieved doc_ids in ranked order.
    """
    async with httpx.AsyncClient(base_url=api_url, timeout=30.0) as client:
        response = await client.get(
            "/api/kb/search",
            params={"q": query, "top_k": top_k, "mode": mode},
        )
        if response.status_code != 200:
            return []
        results = response.json()
        return [r.get("doc_id", "") for r in results]


async def evaluate_mode(
    api_url: str,
    dataset: List[SubsetARow],
    mode: str,
    top_k: int = 5,
) -> RetrievalMetrics:
    """Evaluate a single retrieval mode over the dataset.

    Collects (retrieved_doc_ids, [relevant_doc_id]) tuples per query and
    delegates to ``compute_retrieval_metrics`` for Hit Rate@k and MRR@k
    aggregation (k ∈ {1, 3, 5}).

    Args:
        api_url: Base URL of the running application.
        dataset: List of Subset A rows.
        mode: Retrieval mode ("hybrid", "dense", or "sparse").
        top_k: Number of results to retrieve per query.

    Returns:
        Aggregated RetrievalMetrics across all queries.
    """
    results_per_query: List[Tuple[List[str], List[str]]] = []

    for i, row in enumerate(dataset, 1):
        retrieved_ids = await retrieve(api_url, row.question, top_k, mode)
        relevant_id = row.source_doc_id
        results_per_query.append((retrieved_ids, [relevant_id] if relevant_id else []))

        if i % 10 == 0:
            print(f"  [{mode}] Processed {i}/{len(dataset)} queries...")

    return compute_retrieval_metrics(results_per_query)


async def evaluate_mode_per_category(
    api_url: str,
    dataset: List[SubsetARow],
    mode: str,
    top_k: int = 5,
) -> List[CategoryResult]:
    """Evaluate retrieval mode per category.

    Args:
        api_url: Base URL of the running application.
        dataset: List of Subset A rows.
        mode: Retrieval mode.
        top_k: Maximum k for Hit Rate@k.

    Returns:
        List of CategoryResult, one per category.
    """
    by_category: Dict[str, List[SubsetARow]] = defaultdict(list)
    for row in dataset:
        by_category[row.category].append(row)

    results: List[CategoryResult] = []
    for category, rows in sorted(by_category.items()):
        metrics = await evaluate_mode(api_url, rows, mode, top_k)
        results.append(CategoryResult(category=category, metrics=metrics, sample_count=len(rows)))

    return results


def print_report(
    mode_name: str,
    overall: RetrievalMetrics,
    per_category: List[CategoryResult],
) -> None:
    """Print a formatted retrieval evaluation report.

    Args:
        mode_name: Name of the retrieval mode.
        overall: Overall retrieval metrics.
        per_category: Per-category results.
    """
    print(f"\n{'=' * 70}")
    print(f"  Retrieval Mode: {mode_name}")
    print(f"{'=' * 70}")
    print(f"  Total Queries: {overall.query_count}")
    print()
    print(f"  {'Metric':<20} {'Value':>10}")
    print(f"  {'-' * 32}")
    print(f"  {'Hit Rate@1':<20} {overall.hit_rate_at_1:>10.4f}")
    print(f"  {'Hit Rate@3':<20} {overall.hit_rate_at_3:>10.4f}")
    print(f"  {'Hit Rate@5':<20} {overall.hit_rate_at_5:>10.4f}")
    print(f"  {'MRR@1':<20} {overall.mrr_at_1:>10.4f}")
    print(f"  {'MRR@3':<20} {overall.mrr_at_3:>10.4f}")
    print(f"  {'MRR@5':<20} {overall.mrr_at_5:>10.4f}")

    if per_category:
        print(f"\n  Per Category:")
        print(f"  {'Category':<25} {'N':>5} {'HR@1':>8} {'HR@3':>8} {'HR@5':>8} {'MRR@5':>8}")
        print(f"  {'-' * 70}")
        for r in per_category:
            m = r.metrics
            print(
                f"  {r.category:<25} {r.sample_count:>5} "
                f"{m.hit_rate_at_1:>8.4f} {m.hit_rate_at_3:>8.4f} "
                f"{m.hit_rate_at_5:>8.4f} {m.mrr_at_5:>8.4f}"
            )
    print()


async def async_main(args: argparse.Namespace) -> None:
    """Async entry point for Experiment 2.

    Args:
        args: Parsed command-line arguments.
    """
    try:
        dataset = load_subset_a(args.dataset)
    except FileNotFoundError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)

    # Filter out out-of-domain rows: they have source_doc_id="NONE" and
    # always score 0 on retrieval, which would pollute the metrics without
    # providing meaningful signal (Issue 7).
    in_domain_dataset = [r for r in dataset if r.category != "out-of-domain"]
    skipped = len(dataset) - len(in_domain_dataset)
    print(f"Loaded {len(dataset)} samples from Subset A ({skipped} out-of-domain rows filtered)")
    dataset = in_domain_dataset

    modes_to_eval: List[str] = []
    if args.mode == "all":
        modes_to_eval = ["hybrid", "dense", "sparse"]
    else:
        modes_to_eval = [args.mode]

    for mode in modes_to_eval:
        print(f"\nEvaluating mode: {mode}")
        overall = await evaluate_mode(args.api_url, dataset, mode, args.top_k)
        per_category = await evaluate_mode_per_category(args.api_url, dataset, mode, args.top_k)
        print_report(mode.upper(), overall, per_category)


def main() -> None:
    """Entry point for Experiment 2."""
    parser = argparse.ArgumentParser(
        description="Experiment 2: Retrieval Quality Evaluation (hybrid vs dense vs sparse)."
    )
    parser.add_argument("--dataset", required=True, help="Path to Subset A CSV")
    parser.add_argument(
        "--api-url",
        default="http://localhost:8000",
        help="Base URL of the running application (for KB search)",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Maximum k for Hit Rate@k (default: 5)",
    )
    parser.add_argument(
        "--mode",
        default="all",
        choices=["all", "hybrid", "dense", "sparse"],
        help="Retrieval mode to evaluate (default: all)",
    )
    args = parser.parse_args()
    asyncio.run(async_main(args))


if __name__ == "__main__":
    main()
