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
from typing import Dict, List

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

    Args:
        api_url: Base URL of the running application.
        dataset: List of Subset A rows.
        mode: Retrieval mode ("hybrid", "dense", or "sparse").
        top_k: Maximum k for Hit Rate@k computation.

    Returns:
        Aggregated RetrievalMetrics across all queries.
    """
    all_hit_1: List[bool] = []
    all_hit_3: List[bool] = []
    all_hit_5: List[bool] = []
    all_rr: List[float] = []

    for i, row in enumerate(dataset, 1):
        retrieved_ids = await retrieve(api_url, row.question, top_k, mode)
        relevant_id = row.source_doc_id

        hit_1 = relevant_id in retrieved_ids[:1]
        hit_3 = relevant_id in retrieved_ids[:3]
        hit_5 = relevant_id in retrieved_ids[:5]

        rr = 0.0
        for rank, rid in enumerate(retrieved_ids, 1):
            if rid == relevant_id:
                rr = 1.0 / rank
                break

        all_hit_1.append(hit_1)
        all_hit_3.append(hit_3)
        all_hit_5.append(hit_5)
        all_rr.append(rr)

        if i % 10 == 0:
            print(f"  [{mode}] Processed {i}/{len(dataset)} queries...")

    return RetrievalMetrics(
        hit_rate_at_1=sum(all_hit_1) / len(all_hit_1) if all_hit_1 else 0.0,
        hit_rate_at_3=sum(all_hit_3) / len(all_hit_3) if all_hit_3 else 0.0,
        hit_rate_at_5=sum(all_hit_5) / len(all_hit_5) if all_hit_5 else 0.0,
        mrr=sum(all_rr) / len(all_rr) if all_rr else 0.0,
        total=len(dataset),
    )


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
    print(f"  Total Queries: {overall.total}")
    print()
    print(f"  {'Metric':<20} {'Value':>10}")
    print(f"  {'-' * 32}")
    print(f"  {'Hit Rate@1':<20} {overall.hit_rate_at_1:>10.4f}")
    print(f"  {'Hit Rate@3':<20} {overall.hit_rate_at_3:>10.4f}")
    print(f"  {'Hit Rate@5':<20} {overall.hit_rate_at_5:>10.4f}")
    print(f"  {'MRR':<20} {overall.mrr:>10.4f}")

    if per_category:
        print(f"\n  Per Category:")
        print(f"  {'Category':<25} {'N':>5} {'HR@1':>8} {'HR@3':>8} {'HR@5':>8} {'MRR':>8}")
        print(f"  {'-' * 65}")
        for r in per_category:
            m = r.metrics
            print(
                f"  {r.category:<25} {r.sample_count:>5} "
                f"{m.hit_rate_at_1:>8.4f} {m.hit_rate_at_3:>8.4f} "
                f"{m.hit_rate_at_5:>8.4f} {m.mrr:>8.4f}"
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

    print(f"Loaded {len(dataset)} samples from Subset A")

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
