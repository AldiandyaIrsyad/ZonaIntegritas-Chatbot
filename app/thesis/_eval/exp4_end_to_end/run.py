"""Experiment 4 — End-to-End RAG Pipeline Evaluation.

Evaluates the full RAG pipeline (IVM → retrieval → generation → RAM) against
Subset A (RAG QA triplets). Compares against a no-guardrail baseline
(skip IVM + RAM, same retrieval + generation).

Skripsi §3.3.5, Tabel 3.10.

Metrics (§3.4):
    - BERTScore F1 (§3.4.8) — answer quality vs ground truth
    - Faithfulness (§3.4.9) — proportion of supported sentences
    - Abstention Accuracy (§3.4.10) — correct refusal of out-of-domain queries

Usage:
    python -m app.thesis._eval.exp4_end_to_end.run \\
        --dataset data/subset_a.csv \\
        --api-url http://localhost:8000
"""

from __future__ import annotations

import argparse
import asyncio
import json
import re
import sys
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import httpx

from app.thesis._eval._shared.dataset import load_subset_a, SubsetARow
from app.thesis._eval._shared.metrics import (
    CI,
    abstention_accuracy,
    bert_score_f1,
    bootstrap_ci,
    faithfulness,
)


# Citation format: *(STATUS: SCORE; SOURCE; Page N)*
CITATION_PATTERN = re.compile(
    r"\*?\s*\("
    r"(?P<status>Supported|Contradiction|Neutral)"
    r":\s*(?P<score>[\d.]+)"
    r";\s*(?P<source>[^;]+)"
    r"(?:;\s*Page\s+(?P<page>\d+))?"
    r"\)\s*\*?"
)


@dataclass
class PipelineResult:
    """Result of running the full pipeline on a single query.

    Attributes:
        question: The input question.
        response: Full system response text.
        citations: List of parsed citation tuples (status, score, source, page).
        abstained: Whether the system abstained (refused/warned).
        category: Question category.
        ground_truth: Ground-truth answer.
    """

    question: str
    response: str
    citations: List[Tuple[str, float, str, Optional[int]]] = field(default_factory=list)
    abstained: bool = False
    category: str = ""
    ground_truth: str = ""


@dataclass
class E2EMetrics:
    """Aggregated end-to-end metrics.

    Attributes:
        bertscore_f1: BERTScore F1 (answer quality).
        bertscore_f1_ci: Bootstrap CI for BERTScore F1.
        faithfulness_score: Mean faithfulness across responses.
        faithfulness_ci: Bootstrap CI for faithfulness.
        abstention_acc: Abstention accuracy for out-of-domain queries.
        abstention_ci: Bootstrap CI for abstention accuracy.
        total_queries: Total number of queries evaluated.
        out_of_domain_count: Number of out-of-domain queries.
    """

    bertscore_f1: float = 0.0
    bertscore_f1_ci: CI = field(default_factory=lambda: CI(0.0, 0.0, 0.0))
    faithfulness_score: float = 0.0
    faithfulness_ci: CI = field(default_factory=lambda: CI(0.0, 0.0, 0.0))
    abstention_acc: float = 0.0
    abstention_ci: CI = field(default_factory=lambda: CI(0.0, 0.0, 0.0))
    total_queries: int = 0
    out_of_domain_count: int = 0


def parse_citations(response: str) -> List[Tuple[str, float, str, Optional[int]]]:
    """Parse citation markers from a system response.

    Extracts all citations matching the format:
        *(STATUS: SCORE; SOURCE; Page N)*

    Args:
        response: Full system response text.

    Returns:
        List of (status, score, source, page) tuples.
    """
    citations: List[Tuple[str, float, str, Optional[int]]] = []
    for match in CITATION_PATTERN.finditer(response):
        status = match.group("status")
        try:
            score = float(match.group("score"))
        except ValueError:
            score = 0.0
        source = match.group("source").strip()
        page_str = match.group("page")
        page = int(page_str) if page_str else None
        citations.append((status, score, source, page))
    return citations


def split_sentences(text: str) -> List[str]:
    """Split text into sentences.

    Args:
        text: Input text.

    Returns:
        List of sentence strings.
    """
    # Simple sentence splitter — handles common Indonesian/English punctuation
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    return [s.strip() for s in sentences if s.strip()]


def extract_sentence_labels(citations: List[Tuple[str, float, str, Optional[int]]]) -> List[str]:
    """Map citation statuses to faithfulness labels.

    Args:
        citations: Parsed citation tuples.

    Returns:
        List of labels for faithfulness computation.
    """
    label_map = {
        "Supported": "supported",
        "Contradiction": "not_supported",
        "Neutral": "partially_supported",
    }
    return [label_map.get(c[0], "partially_supported") for c in citations]


async def run_pipeline(
    api_url: str,
    row: SubsetARow,
    session_id: Optional[str] = None,
    skip_guardrails: bool = False,
) -> PipelineResult:
    """Run the full RAG pipeline on a single query via the chat API.

    Args:
        api_url: Base URL of the running application.
        row: Subset A row with question and ground truth.
        session_id: Optional session ID for the chat.
        skip_guardrails: If True, append ``?skip_guardrails=true`` to bypass
            IVM + RAM (baseline mode for Experiment 4).

    Returns:
        PipelineResult with response and parsed citations.
    """
    async with httpx.AsyncClient(base_url=api_url, timeout=120.0) as client:
        # Create a session
        if session_id is None:
            try:
                resp = await client.post("/api/chat/sessions")
                if resp.status_code == 200:
                    session_id = resp.json().get("id")
            except Exception:
                session_id = None

        # Send the message to the streaming endpoint
        stream_url = f"/api/chat/sessions/{session_id}/stream"
        if skip_guardrails:
            stream_url += "?skip_guardrails=true"
        try:
            resp = await client.post(
                stream_url,
                json={"message": row.question},
                timeout=120.0,
            )
            if resp.status_code != 200:
                return PipelineResult(
                    question=row.question,
                    response="",
                    abstained=True,
                    category=row.category,
                    ground_truth=row.ground_truth_answer,
                )

            # The response is an NDJSON stream
            response_text = ""
            retrieved_context = ""
            content_type = resp.headers.get("content-type", "")
            if "application/x-ndjson" in content_type or "text/plain" in content_type:
                for line in resp.text.strip().split("\n"):
                    if not line:
                        continue
                    try:
                        chunk = json.loads(line)
                        chunk_type = chunk.get("type")
                        if chunk_type == "chunk":
                            response_text += chunk.get("content", "")
                        elif chunk_type == "context":
                            retrieved_context = chunk.get("content", "")
                        elif chunk_type == "error":
                            # IVM block or pipeline rejection → abstained
                            return PipelineResult(
                                question=row.question,
                                response=chunk.get("content", ""),
                                abstained=True,
                                category=row.category,
                                ground_truth=row.ground_truth_answer,
                            )
                        elif chunk_type == "done":
                            break
                    except json.JSONDecodeError:
                        response_text += line
            else:
                data = resp.json()
                response_text = data.get("content", data.get("response", ""))
        except httpx.TimeoutException:
            return PipelineResult(
                question=row.question,
                response="",
                abstained=True,
                category=row.category,
                ground_truth=row.ground_truth_answer,
            )

    citations = parse_citations(response_text)
    return PipelineResult(
        question=row.question,
        response=response_text,
        citations=citations,
        abstained=False,
        category=row.category,
        ground_truth=row.ground_truth_answer,
    )


async def run_no_guardrail_pipeline(
    api_url: str,
    row: SubsetARow,
    session_id: Optional[str] = None,
) -> PipelineResult:
    """Run the pipeline without guardrails (baseline).

    Calls the chat endpoint with ``?skip_guardrails=true`` to bypass IVM
    (safety + relevance) and RAM (per-sentence assessment). Retrieval still
    runs so the LLM has context.

    Args:
        api_url: Base URL of the running application.
        row: Subset A row.
        session_id: Optional session ID.

    Returns:
        PipelineResult.
    """
    return await run_pipeline(api_url, row, session_id, skip_guardrails=True)


def compute_e2e_metrics(
    results: List[PipelineResult],
) -> E2EMetrics:
    """Compute end-to-end metrics from pipeline results.

    Args:
        results: List of pipeline results.

    Returns:
        E2EMetrics with BERTScore, faithfulness, and abstention accuracy.
    """
    metrics = E2EMetrics(total_queries=len(results))

    # Separate in-domain and out-of-domain
    in_domain: List[PipelineResult] = []
    out_of_domain: List[PipelineResult] = []
    for r in results:
        if r.category == "out-of-domain" or r.category == "out_of_domain":
            out_of_domain.append(r)
        else:
            in_domain.append(r)
    metrics.out_of_domain_count = len(out_of_domain)

    # --- BERTScore F1 (in-domain only) ---
    if in_domain:
        candidates = [r.response for r in in_domain if r.response]
        references = [r.ground_truth for r in in_domain if r.response]
        if candidates and references:
            _, _, f1 = bert_score_f1(candidates, references)
            metrics.bertscore_f1 = f1
            # Bootstrap CI
            f1_samples: List[float] = []
            import random
            rng = random.Random(42)
            n = len(candidates)
            for _ in range(1000):
                indices = [rng.randrange(n) for _ in range(n)]
                c_sample = [candidates[i] for i in indices]
                r_sample = [references[i] for i in indices]
                _, _, f1_s = bert_score_f1(c_sample, r_sample)
                f1_samples.append(f1_s)
            f1_samples.sort()
            metrics.bertscore_f1_ci = CI(
                point=f1,
                lower=f1_samples[25],
                upper=f1_samples[974],
            )

    # --- Faithfulness (in-domain only) ---
    if in_domain:
        faithfulness_scores: List[float] = []
        for r in in_domain:
            if not r.citations:
                # No citations → assume all sentences are "no_source_needed"
                sentences = split_sentences(r.response)
                labels = ["no_source_needed"] * len(sentences)
            else:
                labels = extract_sentence_labels(r.citations)
            score = faithfulness(labels)
            faithfulness_scores.append(score)
        if faithfulness_scores:
            metrics.faithfulness_score = sum(faithfulness_scores) / len(faithfulness_scores)
            metrics.faithfulness_ci = bootstrap_ci(faithfulness_scores, statistic="mean")

    # --- Abstention Accuracy (out-of-domain only) ---
    if out_of_domain:
        abstained_flags = [r.abstained for r in out_of_domain]
        metrics.abstention_acc = abstention_accuracy(abstained_flags, len(out_of_domain))
        # Wilson interval for abstention accuracy
        from app.thesis._eval._shared.metrics import wilson_interval
        successes = sum(1 for a in abstained_flags if a)
        metrics.abstention_ci = wilson_interval(successes, len(out_of_domain))

    return metrics


def print_report(
    system_name: str,
    metrics: E2EMetrics,
) -> None:
    """Print a formatted end-to-end evaluation report.

    Args:
        system_name: Name of the system being evaluated.
        metrics: Computed end-to-end metrics.
    """
    print(f"\n{'=' * 70}")
    print(f"  {system_name}")
    print(f"{'=' * 70}")
    print(f"  Total Queries:      {metrics.total_queries}")
    print(f"  In-Domain:          {metrics.total_queries - metrics.out_of_domain_count}")
    print(f"  Out-of-Domain:      {metrics.out_of_domain_count}")
    print()
    print(f"  {'Metric':<25} {'Value':>10} {'95% CI':>25}")
    print(f"  {'-' * 62}")
    print(f"  {'BERTScore F1':<25} {metrics.bertscore_f1:>10.4f} "
          f"[{metrics.bertscore_f1_ci.lower:.4f}, {metrics.bertscore_f1_ci.upper:.4f}]")
    print(f"  {'Faithfulness':<25} {metrics.faithfulness_score:>10.4f} "
          f"[{metrics.faithfulness_ci.lower:.4f}, {metrics.faithfulness_ci.upper:.4f}]")
    print(f"  {'Abstention Accuracy':<25} {metrics.abstention_acc:>10.4f} "
          f"[{metrics.abstention_ci.lower:.4f}, {metrics.abstention_ci.upper:.4f}]")
    print()


async def async_main(args: argparse.Namespace) -> None:
    """Async entry point for Experiment 4.

    Args:
        args: Parsed command-line arguments.
    """
    try:
        dataset = load_subset_a(args.dataset)
    except FileNotFoundError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)

    print(f"Loaded {len(dataset)} samples from Subset A")

    # --- Full pipeline (with guardrails) ---
    print("\nRunning full RAG pipeline (IVM → retrieval → generation → RAM)...")
    full_results: List[PipelineResult] = []
    for i, row in enumerate(dataset, 1):
        result = await run_pipeline(args.api_url, row)
        full_results.append(result)
        if i % 10 == 0:
            print(f"  Processed {i}/{len(dataset)} queries...")

    full_metrics = compute_e2e_metrics(full_results)
    print_report("Full Pipeline (with IVM + RAM guardrails)", full_metrics)

    if args.no_baseline:
        return

    # --- No-guardrail baseline ---
    print("\nRunning no-guardrail baseline pipeline...")
    baseline_results: List[PipelineResult] = []
    for i, row in enumerate(dataset, 1):
        result = await run_no_guardrail_pipeline(args.api_url, row)
        baseline_results.append(result)
        if i % 10 == 0:
            print(f"  Processed {i}/{len(dataset)} queries...")

    baseline_metrics = compute_e2e_metrics(baseline_results)
    print_report("Baseline (no guardrails)", baseline_metrics)


def main() -> None:
    """Entry point for Experiment 4."""
    parser = argparse.ArgumentParser(
        description="Experiment 4: End-to-End RAG Pipeline Evaluation (with guardrails vs no-guardrail baseline)."
    )
    parser.add_argument("--dataset", required=True, help="Path to Subset A CSV")
    parser.add_argument(
        "--api-url",
        default="http://localhost:8000",
        help="Base URL of the running application",
    )
    parser.add_argument(
        "--no-baseline",
        action="store_true",
        help="Skip the no-guardrail baseline",
    )
    args = parser.parse_args()
    asyncio.run(async_main(args))


if __name__ == "__main__":
    main()
