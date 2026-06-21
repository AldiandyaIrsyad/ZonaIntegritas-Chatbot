"""
IVM Strategy Comparison Evaluation Script.

Benchmarks all IRelevanceStrategy implementations against a labelled query
dataset. Outputs precision, recall, F1, and accuracy per strategy so results
can be tabulated in the thesis.

Usage:
    python -m app.thesis._eval.ivm.compare_strategies \\
        --dataset path/to/ivm_queries.csv \\
        --threshold 0.4

Dataset CSV format:
    query,label,scores
    "what is X",relevant,"[0.85, 0.72, 0.61]"
    "random query",irrelevant,"[0.21, 0.18, 0.10]"

    label: "relevant" | "irrelevant"
    scores: JSON-encoded list of float similarity scores (top-K)
"""

from __future__ import annotations

import argparse
import ast
import csv
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Tuple

from app.thesis.ivm.strategies import (
    SilhouetteKNNStrategy,
    StrictRelevanceStrategy,
    TopOneStrategy,
)
from app.thesis.ivm.interfaces import IRelevanceStrategy


@dataclass
class EvalResult:
    """Evaluation metrics for a single strategy run.

    Attributes:
        strategy_name: Human-readable strategy class name.
        tp: True positives (correct relevant predictions).
        fp: False positives (irrelevant labelled as relevant).
        tn: True negatives (correct irrelevant predictions).
        fn: False negatives (relevant labelled as irrelevant).
    """

    strategy_name: str
    tp: int = 0
    fp: int = 0
    tn: int = 0
    fn: int = 0

    @property
    def precision(self) -> float:
        """Proportion of relevant predictions that are correct."""
        denom = self.tp + self.fp
        return self.tp / denom if denom else 0.0

    @property
    def recall(self) -> float:
        """Proportion of actual relevant items that were found."""
        denom = self.tp + self.fn
        return self.tp / denom if denom else 0.0

    @property
    def f1(self) -> float:
        """Harmonic mean of precision and recall."""
        p, r = self.precision, self.recall
        denom = p + r
        return 2 * p * r / denom if denom else 0.0

    @property
    def accuracy(self) -> float:
        """Overall fraction of correct predictions."""
        total = self.tp + self.fp + self.tn + self.fn
        return (self.tp + self.tn) / total if total else 0.0


def load_dataset(path: str) -> List[Tuple[List[float], bool]]:
    """Load the evaluation dataset from a CSV file.

    Args:
        path: Path to the CSV file with columns: query, label, scores.

    Returns:
        List of (scores, is_relevant) tuples.

    Raises:
        FileNotFoundError: If the CSV file does not exist.
        ValueError: If a row has an invalid label or malformed scores.
    """
    csv_path = Path(path)
    if not csv_path.is_file():
        raise FileNotFoundError(f"Dataset not found: {csv_path}")

    rows: List[Tuple[List[float], bool]] = []
    with csv_path.open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for line_num, row in enumerate(reader, start=2):
            label = row["label"].strip().lower()
            if label not in ("relevant", "irrelevant"):
                raise ValueError(
                    f"Line {line_num}: invalid label {row['label']!r}. "
                    "Expected 'relevant' or 'irrelevant'."
                )

            try:
                scores: List[float] = json.loads(row["scores"])
            except json.JSONDecodeError:
                # Fallback: try Python literal eval for "[0.5, 0.3]" format
                scores = ast.literal_eval(row["scores"])

            rows.append((scores, label == "relevant"))

    return rows


def evaluate_strategy(
    strategy: IRelevanceStrategy,
    dataset: List[Tuple[List[float], bool]],
    threshold: float,
) -> EvalResult:
    """Run a strategy against the full dataset and return metrics.

    Args:
        strategy: The IRelevanceStrategy instance to benchmark.
        dataset: List of (scores, is_relevant) tuples.
        threshold: Similarity threshold to pass to the strategy.

    Returns:
        EvalResult with tp/fp/tn/fn counts and derived metrics.
    """
    result = EvalResult(strategy_name=type(strategy).__name__)

    for scores, ground_truth in dataset:
        predicted = strategy.evaluate(scores=scores, similarity_threshold=threshold)

        if predicted and ground_truth:
            result.tp += 1
        elif predicted and not ground_truth:
            result.fp += 1
        elif not predicted and ground_truth:
            result.fn += 1
        else:
            result.tn += 1

    return result


def print_table(results: List[EvalResult]) -> None:
    """Print a formatted comparison table to stdout.

    Args:
        results: List of EvalResult objects to display.
    """
    header = f"{'Strategy':<30} {'Precision':>10} {'Recall':>10} {'F1':>10} {'Accuracy':>10}"
    print(header)
    print("-" * len(header))
    for r in results:
        print(
            f"{r.strategy_name:<30} "
            f"{r.precision:>10.4f} "
            f"{r.recall:>10.4f} "
            f"{r.f1:>10.4f} "
            f"{r.accuracy:>10.4f}"
        )


def main() -> None:
    """Entry point for the IVM strategy comparison script."""
    parser = argparse.ArgumentParser(
        description="Compare IVM relevance strategies on a labelled dataset."
    )
    parser.add_argument("--dataset", required=True, help="Path to the CSV evaluation dataset")
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.4,
        help="Similarity threshold (default: 0.4)",
    )
    args = parser.parse_args()

    try:
        dataset = load_dataset(args.dataset)
    except (FileNotFoundError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)

    print(f"Loaded {len(dataset)} samples | threshold={args.threshold}\n")

    strategies: List[IRelevanceStrategy] = [
        TopOneStrategy(),
        StrictRelevanceStrategy(),
        SilhouetteKNNStrategy(),
    ]

    results = [
        evaluate_strategy(strategy, dataset, threshold=args.threshold)
        for strategy in strategies
    ]

    print_table(results)


if __name__ == "__main__":
    main()
