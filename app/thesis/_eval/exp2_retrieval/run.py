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

from app.thesis._eval._shared.csv_export import write_results_csv
from app.thesis._eval._shared.dataset import load_subset_a, SubsetARow
from app.thesis._eval._shared.metrics import (
    RetrievalMetrics,
    compute_retrieval_metrics,
    hit_rate_at_k,
    reciprocal_rank_at_k,
)

DEFAULT_OUTPUT_CSV = "data/results/exp2_retrieval.csv"


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

    ``/api/kb/search`` returns CHUNK-level hits, each carrying the doc_id it
    came from — several of the top-k chunks routinely belong to the same
    document (measured: top-5 averages only ~3.8 unique documents across
    all three modes, see writing/weekend_fixes_plan.md M9). This function
    returns the raw chunk-level list; callers computing Hit Rate@k/MRR@k
    should deduplicate first (see dedup_doc_ids) so "@k" means "top-k
    distinct documents", matching how the metric is defined in §3.5.6/7,
    not "top-k chunks that may repeat the same 2-3 documents".

    Args:
        api_url: Base URL of the running application.
        query: Search query.
        top_k: Number of results to retrieve.
        mode: Retrieval mode ("hybrid", "dense", or "sparse").

    Returns:
        List of retrieved doc_ids in ranked (chunk-level) order.
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


def dedup_doc_ids(doc_ids: List[str]) -> List[str]:
    """Collapse a chunk-level doc_id list to unique documents, rank-preserved.

    Keeps the first (best-ranked) occurrence of each doc_id and drops
    later repeats, so a document's rank in the deduped list is the best
    rank any of its chunks achieved.

    Args:
        doc_ids: Chunk-level doc_ids in ranked order (may repeat).

    Returns:
        Unique doc_ids in first-occurrence (best-rank) order.
    """
    seen: set = set()
    unique: List[str] = []
    for doc_id in doc_ids:
        if doc_id and doc_id not in seen:
            seen.add(doc_id)
            unique.append(doc_id)
    return unique


async def evaluate_mode(
    api_url: str,
    dataset: List[SubsetARow],
    mode: str,
    top_k: int = 5,
) -> Tuple[RetrievalMetrics, List[Tuple[SubsetARow, List[str], List[str]]]]:
    """Evaluate a single retrieval mode over the dataset.

    Deduplicates each query's chunk-level results to unique documents (M9
    in writing/weekend_fixes_plan.md) before computing Hit Rate@k/MRR@k, so
    "@k" is measured over document rank, not chunk rank.

    Args:
        api_url: Base URL of the running application.
        dataset: List of Subset A rows.
        mode: Retrieval mode ("hybrid", "dense", or "sparse").
        top_k: Number of results to retrieve per query.

    Returns:
        Tuple of (aggregated RetrievalMetrics computed on deduped doc
        rankings, per-query raw results as (row, raw_chunk_doc_ids,
        deduped_doc_ids) triples — deduped_doc_ids is what metrics were
        computed from; raw_chunk_doc_ids is kept for CSV transparency and
        for compute_per_category to regroup without re-querying (M12)).
    """
    results_per_query: List[Tuple[List[str], List[str]]] = []
    raw_per_query: List[Tuple[SubsetARow, List[str], List[str]]] = []

    for i, row in enumerate(dataset, 1):
        retrieved_ids = await retrieve(api_url, row.question, top_k, mode)
        deduped_ids = dedup_doc_ids(retrieved_ids)
        relevant_id = row.source_doc_id
        results_per_query.append((deduped_ids, [relevant_id] if relevant_id else []))
        raw_per_query.append((row, retrieved_ids, deduped_ids))

        if i % 10 == 0:
            print(f"  [{mode}] Processed {i}/{len(dataset)} queries...")

    return compute_retrieval_metrics(results_per_query), raw_per_query


def compute_per_category(
    raw_per_query: List[Tuple[SubsetARow, List[str], List[str]]],
) -> List[CategoryResult]:
    """Regroup an already-computed evaluate_mode() pass by category.

    Previously this re-ran every query through the API a second time
    (evaluate_mode_per_category), so overall and per-category figures came
    from two independent passes with no guarantee they'd reconcile, and
    Exp2's API/HyDE cost was doubled for no gain (M12 in
    writing/weekend_fixes_plan.md). This instead regroups the single pass's
    raw per-query results, so per-category numbers are guaranteed
    consistent with the overall figure they're a breakdown of.

    Args:
        raw_per_query: The (row, raw_chunk_doc_ids, deduped_doc_ids)
            triples returned by evaluate_mode.

    Returns:
        List of CategoryResult, one per category.
    """
    by_category: Dict[str, List[Tuple[List[str], List[str]]]] = defaultdict(list)
    for row, _raw_ids, deduped_ids in raw_per_query:
        relevant = [row.source_doc_id] if row.source_doc_id else []
        by_category[row.category].append((deduped_ids, relevant))

    results: List[CategoryResult] = []
    for category, pairs in sorted(by_category.items()):
        results.append(
            CategoryResult(
                category=category,
                metrics=compute_retrieval_metrics(pairs),
                sample_count=len(pairs),
            )
        )
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

    csv_rows = []
    for mode in modes_to_eval:
        print(f"\nEvaluating mode: {mode}")
        overall, raw_per_query = await evaluate_mode(args.api_url, dataset, mode, args.top_k)
        per_category = compute_per_category(raw_per_query)
        print_report(mode.upper(), overall, per_category)

        for row, retrieved_ids, deduped_ids in raw_per_query:
            relevant = [row.source_doc_id] if row.source_doc_id else []
            csv_rows.append({
                "question": row.question,
                "category": row.category,
                "mode": mode,
                "source_doc_id": row.source_doc_id,
                "retrieved_doc_ids": "|".join(retrieved_ids),
                "retrieved_doc_ids_dedup": "|".join(deduped_ids),
                "hit_at_1": hit_rate_at_k(deduped_ids, relevant, 1),
                "hit_at_3": hit_rate_at_k(deduped_ids, relevant, 3),
                "hit_at_5": hit_rate_at_k(deduped_ids, relevant, 5),
                "reciprocal_rank": reciprocal_rank_at_k(deduped_ids, relevant, args.top_k),
            })

    write_results_csv(args.output_csv, csv_rows)


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
    parser.add_argument(
        "--output-csv",
        default=DEFAULT_OUTPUT_CSV,
        help="Path to write raw per-row results CSV",
    )
    args = parser.parse_args()
    asyncio.run(async_main(args))


if __name__ == "__main__":
    main()
