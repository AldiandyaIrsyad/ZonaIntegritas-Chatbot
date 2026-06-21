"""RAM NLI Model Comparison Evaluation Script.

Benchmarks NLI model accuracy for the Response Assessment Module by
comparing predicted labels against a human-annotated ground truth dataset.

Usage:
    python -m app.thesis._eval.ram.compare_nli_models \\
        --dataset path/to/ram_pairs.csv \\
        --model StevenLimcorn/indo-roberta-indonli \\
        --infinity-url http://localhost:7997

Dataset CSV format:
    sentence,context,label
    "X adalah Y","X adalah Y karena...",entailment
    "X adalah Z","X adalah Y karena...",contradiction
    "Cuaca cerah","X adalah Y karena...",neutral

    label: "entailment" | "neutral" | "contradiction"
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple

import httpx

from app.thesis.ram.interfaces import INLIModel, NLIResult


# ---------------------------------------------------------------------------
# Inline adapter so we can call Infinity without the full chat infra
# ---------------------------------------------------------------------------

class _InfinityNLIAdapter(INLIModel):
    """Minimal NLI adapter for evaluation — connects to an Infinity server.

    Args:
        base_url: Infinity server base URL.
        model: NLI model identifier.
    """

    def __init__(self, base_url: str, model: str) -> None:
        self._model = model
        self._sep = " </s></s> " if "roberta" in model.lower() else " [SEP] "
        self._client = httpx.AsyncClient(
            base_url=base_url.rstrip("/"),
            timeout=httpx.Timeout(30.0, connect=10.0),
        )

    async def check(self, premise: str, hypothesis: str) -> NLIResult:
        """Run NLI inference for a single premise/hypothesis pair.

        Args:
            premise: Reference context string.
            hypothesis: Statement to evaluate against the premise.

        Returns:
            NLIResult with canonical label and per-class confidence scores.
        """
        text = f"{premise}{self._sep}{hypothesis}"
        response = await self._client.post(
            "/classify",
            json={"model": self._model, "input": [text], "raw_scores": True},
        )
        response.raise_for_status()
        data = response.json()

        items = data.get("data", [])
        if not items:
            return NLIResult(label="neutral", entailment_score=0.5, contradiction_score=0.0)

        predictions = items[0]
        score_dict: Dict[str, float] = {}
        if isinstance(predictions, list):
            for p in predictions:
                score_dict[str(p.get("label", "")).lower()] = float(p.get("score", 0.0))
        elif isinstance(predictions, dict) and "score" in predictions:
            raw = predictions["score"]
            if isinstance(raw, dict):
                score_dict = {k.lower(): float(v) for k, v in raw.items()}

        label_map = {
            "label_0": "entailment", "label_1": "neutral", "label_2": "contradiction",
            "entailment": "entailment", "neutral": "neutral", "contradiction": "contradiction",
        }
        scores: Dict[str, float] = {
            label_map.get(k, "neutral"): v for k, v in score_dict.items()
        }

        best = max(scores.items(), key=lambda x: x[1], default=("neutral", 0.0))
        return NLIResult(
            label=best[0],
            entailment_score=scores.get("entailment", 0.0),
            neutral_score=scores.get("neutral", 0.0),
            contradiction_score=scores.get("contradiction", 0.0),
        )

    async def aclose(self) -> None:
        """Close the underlying HTTP client."""
        await self._client.aclose()


# ---------------------------------------------------------------------------
# Evaluation logic
# ---------------------------------------------------------------------------

@dataclass
class ConfusionMatrix:
    """Per-class counts for a 3-class classification problem.

    Attributes:
        label: The NLI class this matrix belongs to.
        tp: True positives.
        fp: False positives.
        fn: False negatives.
    """

    label: str
    tp: int = 0
    fp: int = 0
    fn: int = 0

    @property
    def precision(self) -> float:
        """Precision for this class."""
        denom = self.tp + self.fp
        return self.tp / denom if denom else 0.0

    @property
    def recall(self) -> float:
        """Recall for this class."""
        denom = self.tp + self.fn
        return self.tp / denom if denom else 0.0

    @property
    def f1(self) -> float:
        """F1 score for this class."""
        p, r = self.precision, self.recall
        denom = p + r
        return 2 * p * r / denom if denom else 0.0


def load_dataset(path: str) -> List[Tuple[str, str, str]]:
    """Load the RAM evaluation dataset.

    Args:
        path: Path to CSV with columns: sentence, context, label.

    Returns:
        List of (sentence, context, label) tuples.

    Raises:
        FileNotFoundError: If the CSV file does not exist.
        ValueError: If a row has an invalid label.
    """
    csv_path = Path(path)
    if not csv_path.is_file():
        raise FileNotFoundError(f"Dataset not found: {csv_path}")

    valid_labels = {"entailment", "neutral", "contradiction"}
    rows: List[Tuple[str, str, str]] = []
    with csv_path.open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for line_num, row in enumerate(reader, start=2):
            label = row["label"].strip().lower()
            if label not in valid_labels:
                raise ValueError(
                    f"Line {line_num}: invalid label {row['label']!r}. "
                    f"Expected one of {valid_labels}."
                )
            rows.append((row["sentence"], row["context"], label))
    return rows


async def run_evaluation(
    nli: _InfinityNLIAdapter,
    dataset: List[Tuple[str, str, str]],
) -> Tuple[float, Dict[str, ConfusionMatrix]]:
    """Run NLI inference on all pairs and compute metrics.

    Args:
        nli: Configured NLI adapter.
        dataset: List of (sentence, context, ground_truth_label).

    Returns:
        Tuple of (overall_accuracy, per_class_confusion_matrices).
    """
    labels = ["entailment", "neutral", "contradiction"]
    matrices: Dict[str, ConfusionMatrix] = {lbl: ConfusionMatrix(label=lbl) for lbl in labels}
    correct = 0

    for sentence, context, ground_truth in dataset:
        result = await nli.check(premise=context, hypothesis=sentence)
        predicted = result.label

        if predicted == ground_truth:
            correct += 1
            matrices[ground_truth].tp += 1
        else:
            matrices.get(predicted, matrices["neutral"]).fp += 1
            matrices[ground_truth].fn += 1

    accuracy = correct / len(dataset) if dataset else 0.0
    return accuracy, matrices


def print_report(model: str, accuracy: float, matrices: Dict[str, ConfusionMatrix]) -> None:
    """Print a formatted per-class report and macro averages.

    Args:
        model: NLI model identifier string.
        accuracy: Overall classification accuracy.
        matrices: Per-class confusion matrices.
    """
    print(f"\nModel: {model}")
    print(f"Overall Accuracy: {accuracy:.4f}\n")
    header = f"{'Class':<20} {'Precision':>10} {'Recall':>10} {'F1':>10}"
    print(header)
    print("-" * len(header))
    for cm in matrices.values():
        print(f"{cm.label:<20} {cm.precision:>10.4f} {cm.recall:>10.4f} {cm.f1:>10.4f}")

    macro_p = sum(cm.precision for cm in matrices.values()) / len(matrices)
    macro_r = sum(cm.recall for cm in matrices.values()) / len(matrices)
    macro_f1 = sum(cm.f1 for cm in matrices.values()) / len(matrices)
    print(f"\n{'Macro avg':<20} {macro_p:>10.4f} {macro_r:>10.4f} {macro_f1:>10.4f}")


async def async_main(args: argparse.Namespace) -> None:
    """Async entry point for the RAM evaluation script.

    Args:
        args: Parsed command-line arguments.
    """
    try:
        dataset = load_dataset(args.dataset)
    except (FileNotFoundError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)

    print(f"Loaded {len(dataset)} sentence pairs")

    nli = _InfinityNLIAdapter(base_url=args.infinity_url, model=args.model)
    try:
        accuracy, matrices = await run_evaluation(nli, dataset)
        print_report(args.model, accuracy, matrices)
    finally:
        await nli.aclose()


def main() -> None:
    """Entry point for the RAM NLI model comparison script."""
    parser = argparse.ArgumentParser(
        description="Compare NLI model accuracy for the Response Assessment Module."
    )
    parser.add_argument("--dataset", required=True, help="Path to the CSV evaluation dataset")
    parser.add_argument(
        "--model",
        default="StevenLimcorn/indo-roberta-indonli",
        help="NLI model identifier on the Infinity server",
    )
    parser.add_argument(
        "--infinity-url",
        default="http://localhost:7997",
        help="Infinity server base URL",
    )
    args = parser.parse_args()
    asyncio.run(async_main(args))


if __name__ == "__main__":
    main()
