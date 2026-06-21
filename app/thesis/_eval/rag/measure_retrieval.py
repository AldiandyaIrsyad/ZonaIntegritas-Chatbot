"""RAG Retrieval Quality Evaluation Script.

Measures retrieval quality for the KB search pipeline by computing
standard IR metrics (Recall@K, MRR@K) against a labelled query set.

This script calls the live SearchService via HTTP, so the application
must be running when you execute it.

Usage:
    python -m app.thesis._eval.rag.measure_retrieval \\
        --dataset path/to/rag_queries.csv \\
        --api-url http://localhost:8000 \\
        --top-k 15

Dataset CSV format:
    query,relevant_doc_ids
    "what is X","[\"uuid1\", \"uuid2\"]"
    
    relevant_doc_ids: JSON-encoded list of PDFDocument IDs that should appear
                      in the top-K results for this query.
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import httpx


@dataclass
class RetrievalMetrics:
    """Aggregated retrieval metrics over all evaluated queries.

    Attributes:
        recall_at_k: Fraction of relevant documents found in top-K.
        mrr_at_k: Mean Reciprocal Rank at K.
        query_count: Total number of queries evaluated.
    """

    recall_at_k: float
    mrr_at_k: float
    query_count: int


def load_dataset(path: str) -> List[Tuple[str, List[str]]]:
    """Load the RAG evaluation dataset.

    Args:
        path: Path to CSV with columns: query, relevant_doc_ids.

    Returns:
        List of (query, relevant_doc_ids) tuples.

    Raises:
        FileNotFoundError: If the dataset file does not exist.
    """
    csv_path = Path(path)
    if not csv_path.is_file():
        raise FileNotFoundError(f"Dataset not found: {csv_path}")

    rows: List[Tuple[str, List[str]]] = []
    with csv_path.open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            query = row["query"].strip()
            relevant_ids: List[str] = json.loads(row["relevant_doc_ids"])
            rows.append((query, relevant_ids))
    return rows


def compute_recall_at_k(retrieved_doc_ids: List[str], relevant_ids: List[str]) -> float:
    """Compute Recall@K for a single query.

    Args:
        retrieved_doc_ids: Ordered list of retrieved document IDs (top-K).
        relevant_ids: Ground-truth relevant document IDs.

    Returns:
        float: Recall@K in [0, 1].
    """
    if not relevant_ids:
        return 1.0
    retrieved_set = set(retrieved_doc_ids)
    hits = sum(1 for rid in relevant_ids if rid in retrieved_set)
    return hits / len(relevant_ids)


def compute_rr_at_k(retrieved_doc_ids: List[str], relevant_ids: List[str]) -> float:
    """Compute Reciprocal Rank for a single query.

    Args:
        retrieved_doc_ids: Ordered list of retrieved document IDs (top-K).
        relevant_ids: Ground-truth relevant document IDs.

    Returns:
        float: Reciprocal rank (1/rank of first hit), or 0.0 if no hit.
    """
    relevant_set = set(relevant_ids)
    for rank, doc_id in enumerate(retrieved_doc_ids, start=1):
        if doc_id in relevant_set:
            return 1.0 / rank
    return 0.0


async def run_evaluation(
    api_url: str,
    dataset: List[Tuple[str, List[str]]],
    top_k: int,
    session_id: str,
) -> RetrievalMetrics:
    """Call the SearchService via HTTP and compute metrics.

    Args:
        api_url: Base URL of the running application.
        dataset: List of (query, relevant_doc_ids) tuples.
        top_k: Number of results to retrieve per query.
        session_id: Chat session ID for context filtering.

    Returns:
        RetrievalMetrics with aggregated scores.
    """
    recall_sum = 0.0
    mrr_sum = 0.0

    async with httpx.AsyncClient(base_url=api_url, timeout=30.0) as client:
        for query, relevant_ids in dataset:
            # POST to the chat stream endpoint to trigger a search internally
            # This requires a dedicated /api/kb/search endpoint in production;
            # for now we call the internal search route if exposed, or adapt
            # to whatever endpoint exposes search results.
            response = await client.get(
                "/api/kb/search",
                params={"q": query, "top_k": top_k, "session_id": session_id},
            )
            if response.status_code != 200:
                print(
                    f"WARN: Search for '{query[:40]}...' returned {response.status_code}",
                    file=sys.stderr,
                )
                continue

            results = response.json()
            retrieved_doc_ids = [r["doc_id"] for r in results]

            recall_sum += compute_recall_at_k(retrieved_doc_ids, relevant_ids)
            mrr_sum += compute_rr_at_k(retrieved_doc_ids, relevant_ids)

    n = len(dataset)
    return RetrievalMetrics(
        recall_at_k=recall_sum / n if n else 0.0,
        mrr_at_k=mrr_sum / n if n else 0.0,
        query_count=n,
    )


async def async_main(args: argparse.Namespace) -> None:
    """Async entry point for the RAG evaluation script.

    Args:
        args: Parsed command-line arguments.
    """
    try:
        dataset = load_dataset(args.dataset)
    except FileNotFoundError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)

    print(f"Loaded {len(dataset)} queries | top_k={args.top_k}\n")

    metrics = await run_evaluation(
        api_url=args.api_url,
        dataset=dataset,
        top_k=args.top_k,
        session_id=args.session_id,
    )

    print(f"Queries evaluated : {metrics.query_count}")
    print(f"Recall@{args.top_k}         : {metrics.recall_at_k:.4f}")
    print(f"MRR@{args.top_k}            : {metrics.mrr_at_k:.4f}")


def main() -> None:
    """Entry point for the RAG retrieval quality script."""
    parser = argparse.ArgumentParser(
        description="Measure RAG retrieval quality (Recall@K, MRR@K)."
    )
    parser.add_argument("--dataset", required=True, help="Path to the CSV evaluation dataset")
    parser.add_argument(
        "--api-url",
        default="http://localhost:8000",
        help="Base URL of the running application",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=15,
        help="Number of results to retrieve per query (default: 15)",
    )
    parser.add_argument(
        "--session-id",
        default="eval-session",
        help="Session ID for KB search context filtering",
    )
    args = parser.parse_args()
    asyncio.run(async_main(args))


if __name__ == "__main__":
    main()
