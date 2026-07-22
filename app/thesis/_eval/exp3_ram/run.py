"""Experiment 3 — RAM (NLI-based Hallucination Detection) Evaluation.

Evaluates the NLI-based hallucination detection (indo-roberta) against
Subset D (sentence-level annotations). Compares against a token-containment
similarity baseline.

Skripsi §3.3.4, Tabel 3.10.

The NLI pipeline:
    1. For each (sentence, retrieved_context) pair, run NLI classification
    2. Map 4-label annotations → 3-class NLI labels:
       - supported → entailment
       - not_supported → contradiction
       - partially_supported → neutral
       - no_source_needed → EXCLUDED
    3. Compute 3×3 confusion matrix + per-class P/R/F1 + Cohen's Kappa

Baseline: token-containment similarity with threshold (high containment →
entailment, low → contradiction, mid → neutral) — see
run_containment_baseline for why this replaced an earlier token-Jaccard
baseline (M6 in writing/weekend_fixes_plan.md: Jaccard's union-denominator
made every score on this dataset land below any reachable threshold).

Metrics (§3.4): Accuracy, Precision, Recall, F1 (macro), Cohen's Kappa + CI.

Usage:
    python -m app.thesis._eval.exp3_ram.run \\
        --dataset data/subset_d.csv \\
        --infinity-url http://localhost:7997 \\
        --nli-model StevenLimcorn/indo-roberta-indonli
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

from app.thesis._eval._shared.clients import EvalNLIClient
from app.thesis._eval._shared.csv_export import write_results_csv
from app.thesis._eval._shared.dataset import load_subset_d, SubsetDRow
from app.thesis._eval._shared.metrics import (
    CI,
    MultiClassMetrics,
    bootstrap_ci,
    compute_multiclass_metrics,
    token_containment_similarity,
)


# Ground-truth label mapping: 4-label → 3-class NLI
LABEL_MAP: Dict[str, str] = {
    "supported": "entailment",
    "not_supported": "contradiction",
    "partially_supported": "neutral",
    # 'no_source_needed' is excluded
}

NLI_LABELS: List[str] = ["entailment", "neutral", "contradiction"]


@dataclass
class EvalResult:
    """Results of a single evaluation run.

    Attributes:
        system_name: Name of the system being evaluated.
        metrics: Multi-class metrics.
        kappa_ci: Bootstrap CI for Cohen's Kappa.
    """

    system_name: str
    metrics: MultiClassMetrics
    kappa_ci: CI


def filter_dataset(dataset: List[SubsetDRow]) -> Tuple[List[SubsetDRow], List[str]]:
    """Filter out 'no_source_needed' rows and map labels to NLI classes.

    Args:
        dataset: Full Subset D rows.

    Returns:
        Tuple of (filtered_rows, mapped_ground_truth_labels).
    """
    filtered: List[SubsetDRow] = []
    ground_truths: List[str] = []
    for row in dataset:
        mapped = LABEL_MAP.get(row.label)
        if mapped is None:
            continue  # Exclude no_source_needed
        filtered.append(row)
        ground_truths.append(mapped)
    return filtered, ground_truths


async def run_nli_classification(
    client: EvalNLIClient,
    dataset: List[SubsetDRow],
) -> List[str]:
    """Run NLI classification on all (sentence, context) pairs.

    Args:
        client: Configured NLI model client.
        dataset: Filtered Subset D rows.

    Returns:
        List of predicted NLI labels.
    """
    predictions: List[str] = []
    for i, row in enumerate(dataset, 1):
        result = await client.check(
            premise=row.retrieved_context,
            hypothesis=row.sentence_text,
        )
        predictions.append(result.label)
        if i % 20 == 0:
            print(f"  [NLI] Processed {i}/{len(dataset)} sentences...")
    return predictions


def run_containment_baseline(
    dataset: List[SubsetDRow],
    entail_threshold: float = 0.7,
    contradiction_threshold: float = 0.3,
) -> List[str]:
    """Run token-containment similarity baseline.

    Replaces the earlier token-Jaccard baseline (M6 in
    writing/weekend_fixes_plan.md): Jaccard's union-of-both-texts
    denominator is dominated by the 4,000-22,000-token retrieved context
    almost regardless of a ~20-token sentence's overlap with it, so every
    Jaccard score on this dataset landed in roughly [0, 0.1] — below the
    0.15 contradiction threshold the old defaults used, making the
    "baseline" a constant classifier (measured: it predicted
    ``contradiction`` for 83/83 Subset D rows). Containment
    (``|sentence ∩ context| / |sentence|``) instead measures how much of
    the sentence's own vocabulary appears in the context, which stays
    meaningful regardless of the length mismatch (measured mean ~0.88,
    genuine 0.0-1.0 spread on the same data).

    The thresholds below are picked to be reachable given that measured
    distribution, not tuned against Subset D's ground truth (whose
    contradiction/partially_supported classes have only 2-4 examples each —
    see M5 in writing/weekend_fixes_plan.md — too few to calibrate a
    threshold against without overfitting to noise). Recalibrate once
    Subset D is rebuilt with a less degenerate label distribution.

    Classification rules:
        - containment ≥ entail_threshold → entailment
        - containment < contradiction_threshold → contradiction
        - otherwise → neutral

    Args:
        dataset: Filtered Subset D rows.
        entail_threshold: Threshold for entailment classification.
        contradiction_threshold: Threshold for contradiction classification.

    Returns:
        List of predicted NLI labels.
    """
    predictions: List[str] = []
    for row in dataset:
        sim = token_containment_similarity(row.sentence_text, row.retrieved_context)
        if sim >= entail_threshold:
            predictions.append("entailment")
        elif sim < contradiction_threshold:
            predictions.append("contradiction")
        else:
            predictions.append("neutral")
    return predictions


def print_confusion_matrix(metrics: MultiClassMetrics) -> None:
    """Print a formatted confusion matrix.

    Args:
        metrics: Multi-class metrics with confusion matrix.
    """
    labels = metrics.labels
    col_width = 14
    actual_vs_pred = "Actual \\ Pred"
    header = f"  {actual_vs_pred:<15}"
    for lbl in labels:
        header += f"{lbl:>{col_width}}"
    print(header)
    print(f"  {'-' * (15 + col_width * len(labels))}")
    for i, lbl in enumerate(labels):
        row_str = f"  {lbl:<15}"
        for j in range(len(labels)):
            row_str += f"{metrics.confusion[i][j]:>{col_width}}"
        print(row_str)


def print_report(result: EvalResult) -> None:
    """Print a formatted evaluation report.

    Args:
        result: Evaluation result.
    """
    m = result.metrics
    print(f"\n{'=' * 70}")
    print(f"  {result.system_name}")
    print(f"{'=' * 70}")
    print(f"  Samples: {sum(sum(row) for row in m.confusion)}")
    print()
    print(f"  Overall Accuracy: {m.accuracy:.4f}")
    print(f"  Macro Precision:  {m.macro_precision:.4f}")
    print(f"  Macro Recall:     {m.macro_recall:.4f}")
    print(f"  Macro F1:         {m.macro_f1:.4f}")
    print(f"  Cohen's Kappa:    {m.cohen_kappa:.4f} [{result.kappa_ci.lower:.4f}, {result.kappa_ci.upper:.4f}]")

    print(f"\n  Per-Class:")
    print(f"  {'Class':<20} {'Precision':>10} {'Recall':>10} {'F1':>10}")
    print(f"  {'-' * 52}")
    for lbl in m.labels:
        p, r, f1 = m.per_class.get(lbl, (0.0, 0.0, 0.0))
        print(f"  {lbl:<20} {p:>10.4f} {r:>10.4f} {f1:>10.4f}")

    print(f"\n  Confusion Matrix:")
    print_confusion_matrix(m)
    print()


async def async_main(args: argparse.Namespace) -> None:
    """Async entry point for Experiment 3.

    Args:
        args: Parsed command-line arguments.
    """
    try:
        dataset = load_subset_d(args.dataset)
    except FileNotFoundError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)

    print(f"Loaded {len(dataset)} samples from Subset D")

    # Filter and map labels
    filtered, ground_truths = filter_dataset(dataset)
    print(f"After filtering 'no_source_needed': {len(filtered)} samples")

    if not filtered:
        print("ERROR: No valid samples after filtering.", file=sys.stderr)
        sys.exit(1)

    # --- Token-containment baseline ---
    print("\nRunning token-containment baseline...")
    baseline_preds = run_containment_baseline(
        filtered,
        entail_threshold=args.entail_threshold,
        contradiction_threshold=args.contradiction_threshold,
    )
    baseline_metrics = compute_multiclass_metrics(baseline_preds, ground_truths, NLI_LABELS)
    # Bootstrap CI for Cohen's Kappa
    kappa_samples: List[float] = []
    # Simple bootstrap: resample predictions and recompute kappa
    import random
    rng = random.Random(42)
    n = len(baseline_preds)
    for _ in range(1000):
        indices = [rng.randrange(n) for _ in range(n)]
        preds_sample = [baseline_preds[i] for i in indices]
        gts_sample = [ground_truths[i] for i in indices]
        m = compute_multiclass_metrics(preds_sample, gts_sample, NLI_LABELS)
        kappa_samples.append(m.cohen_kappa)
    kappa_samples.sort()
    baseline_kappa_ci = CI(
        point=baseline_metrics.cohen_kappa,
        lower=kappa_samples[25],
        upper=kappa_samples[974],
    )
    print_report(EvalResult(
        system_name="Baseline (Token-Containment Similarity)",
        metrics=baseline_metrics,
        kappa_ci=baseline_kappa_ci,
    ))

    output_csv = args.output_csv or f"data/results/exp3_ram_{Path(args.dataset).stem}.csv"
    nli_preds: List[str] = []

    def write_csv() -> None:
        rows = []
        for i, row in enumerate(filtered):
            r = {
                "question_id": row.question_id,
                "sentence_id": row.sentence_id,
                "sentence_text": row.sentence_text,
                "true_label": ground_truths[i],
                "containment_pred": baseline_preds[i],
            }
            if nli_preds:
                r["nli_pred"] = nli_preds[i]
            rows.append(r)
        write_results_csv(output_csv, rows)

    if args.no_nli:
        write_csv()
        return

    # --- NLI Model ---
    nli_client = EvalNLIClient(
        base_url=args.infinity_url,
        model=args.nli_model,
    )
    try:
        print("\nRunning NLI classification...")
        nli_preds = await run_nli_classification(nli_client, filtered)
    finally:
        await nli_client.aclose()

    nli_metrics = compute_multiclass_metrics(nli_preds, ground_truths, NLI_LABELS)
    # Bootstrap CI for Cohen's Kappa
    kappa_samples_nli: List[float] = []
    rng2 = random.Random(42)
    for _ in range(1000):
        indices = [rng2.randrange(n) for _ in range(n)]
        preds_sample = [nli_preds[i] for i in indices]
        gts_sample = [ground_truths[i] for i in indices]
        m = compute_multiclass_metrics(preds_sample, gts_sample, NLI_LABELS)
        kappa_samples_nli.append(m.cohen_kappa)
    kappa_samples_nli.sort()
    nli_kappa_ci = CI(
        point=nli_metrics.cohen_kappa,
        lower=kappa_samples_nli[25],
        upper=kappa_samples_nli[974],
    )
    print_report(EvalResult(
        system_name="NLI Model (indo-roberta)",
        metrics=nli_metrics,
        kappa_ci=nli_kappa_ci,
    ))
    write_csv()


def main() -> None:
    """Entry point for Experiment 3."""
    parser = argparse.ArgumentParser(
        description="Experiment 3: RAM (NLI-based Hallucination Detection) Evaluation vs token-containment baseline."
    )
    parser.add_argument("--dataset", required=True, help="Path to Subset D CSV")
    parser.add_argument(
        "--infinity-url",
        default="http://localhost:7997",
        help="Infinity server base URL",
    )
    parser.add_argument(
        "--nli-model",
        default="StevenLimcorn/indo-roberta-indonli",
        help="NLI model identifier (must match a model loaded on the Infinity server)",
    )
    parser.add_argument(
        "--entail-threshold",
        type=float,
        default=0.7,
        help="Containment threshold for entailment (baseline; default 0.7, "
        "reachable given the observed containment distribution — unlike "
        "the old Jaccard default, see run_containment_baseline)",
    )
    parser.add_argument(
        "--contradiction-threshold",
        type=float,
        default=0.3,
        help="Containment threshold for contradiction (baseline; default 0.3)",
    )
    parser.add_argument(
        "--no-nli",
        action="store_true",
        help="Skip NLI model evaluation (run baseline only)",
    )
    parser.add_argument(
        "--output-csv",
        default=None,
        help="Path to write raw per-row results CSV "
        "(default: data/results/exp3_ram_<dataset filename stem>.csv)",
    )
    args = parser.parse_args()
    asyncio.run(async_main(args))


if __name__ == "__main__":
    main()
