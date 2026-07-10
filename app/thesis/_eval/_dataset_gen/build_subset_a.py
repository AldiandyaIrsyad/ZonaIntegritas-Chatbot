"""Build Subset A — RAG QA Triplets.

Generates RAG QA triplets (question, category, ground_truth_answer,
source_doc_id, source_context) from the knowledge base using the
Generator-Evaluator architecture.

Skripsi §3.2.1, Tabel 3.3.

Pipeline:
    1. Load KB documents from the running application
    2. Generator (DeepSeek V4) produces draft QA triplets from each document
    3. Panel (5 models) validates: is the question answerable from the context?
       Is the ground_truth_answer hallucination-free?
    4. Accept if ≥4/5 panel members agree
    5. Output CSV matching Tabel 3.3 schema

Usage:
    python -m app.thesis._eval._dataset_gen.build_subset_a \\
        --api-url http://localhost:8000 \\
        --output data/subset_a.csv \\
        --count 100
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import sys
from pathlib import Path
from typing import Any, Dict, List

import httpx
import structlog

from app.thesis._eval._dataset_gen.config import DatasetGenSettings, get_dataset_gen_settings
from app.thesis._eval._dataset_gen.concordance import BlindInjectionTracker
from app.thesis._eval._dataset_gen.generator import DatasetGenerator
from app.thesis._eval._dataset_gen.panel import EvaluatorPanel

logger = structlog.get_logger(__name__)

CATEGORIES = ["factual", "procedural", "multi-hop", "out-of-domain"]

GENERATOR_SYSTEM_PROMPT = """\
You are a dataset generator for a RAG evaluation benchmark about Indonesian \
bureaucratic reform (Zona Integritas / ZI). Generate high-quality QA triplets \
in Indonesian.

For each item, output a JSON object on its own line (JSONL format) with these fields:
- "question": A question in Indonesian
- "category": One of: factual, procedural, multi-hop, out-of-domain
- "ground_truth_answer": The correct answer based on the context (or "NONE" for out-of-domain)
- "source_context": The verbatim paragraph from the source document (or "NONE" for out-of-domain)

Do not include markdown code fences. Output one JSON object per line.
"""

VALIDATION_PROMPT = """\
You are evaluating a generated QA triplet for a RAG benchmark.

Question: {question}
Category: {category}
Ground Truth Answer: {ground_truth_answer}
Source Context: {source_context}

Evaluate:
1. Is the question clear and answerable?
2. Is the ground_truth_answer correct based on the source_context?
3. For out-of-domain items, is the question truly outside the ZI domain?

Answer with ONLY 'YES' or 'NO'.
"""


async def fetch_kb_documents(api_url: str) -> List[Dict[str, Any]]:
    """Fetch all active PDF documents from the KB API.

    Args:
        api_url: Base URL of the running application.

    Returns:
        List of document metadata dicts.
    """
    async with httpx.AsyncClient(base_url=api_url, timeout=30.0) as client:
        response = await client.get("/api/admin/pdfs")
        response.raise_for_status()
        docs = response.json()
        return [d for d in docs if d.get("active", True)]


async def fetch_document_text(api_url: str, doc_id: str, doc_title: str) -> str:
    """Fetch text content for a document by searching its chunks.

    Uses the document's own title as the search query (domain-agnostic) and
    post-filters by ``doc_id`` to ensure only this document's chunks are
    joined. ``top_k`` is set to the API maximum (100) to maximise coverage.

    Args:
        api_url: Base URL of the running application.
        doc_id: Document ID (used as a post-filter).
        doc_title: Document title (used as the search query).

    Returns:
        Concatenated text from the document's chunks, or empty string on
        failure.
    """
    if not doc_title:
        return ""
    async with httpx.AsyncClient(base_url=api_url, timeout=30.0) as client:
        response = await client.get(
            "/api/kb/search",
            params={"q": doc_title, "top_k": 100},
        )
        if response.status_code != 200:
            return ""
        results = response.json()
        return "\n\n".join(
            r.get("text", "") for r in results if r.get("doc_id") == doc_id
        )


async def build_subset_a(
    settings: DatasetGenSettings,
    api_url: str,
    output_path: str,
    count: int,
) -> None:
    """Build Subset A (RAG QA triplets) and save to CSV.

    Args:
        settings: Dataset generation settings.
        api_url: Base URL of the running application.
        output_path: Path to output CSV file.
        count: Target number of accepted items.
    """
    if not settings.openrouter_api_key:
        logger.error("datagen.subset_a.missing_api_key")
        sys.exit(1)

    # 1. Fetch KB documents
    logger.info("datagen.subset_a.fetching_docs")
    docs = await fetch_kb_documents(api_url)
    if not docs:
        logger.error("datagen.subset_a.no_active_docs")
        sys.exit(1)
    logger.info("datagen.subset_a.docs_found", count=len(docs))

    # 2. Initialize generator and panel
    generator = DatasetGenerator(settings)
    panel = EvaluatorPanel(settings)
    blind_tracker = BlindInjectionTracker()

    accepted_items: List[Dict[str, str]] = []
    total_generated = 0
    total_rejected = 0

    # Per-category targets matching skripsi Tabel 3.3 (total 85, within 80-100 range).
    # Scaled proportionally when --count differs from 85.
    CATEGORY_TARGETS = {"factual": 30, "procedural": 25, "multi-hop": 20, "out-of-domain": 10}
    scale = count / sum(CATEGORY_TARGETS.values()) if count != 85 else 1.0
    category_targets = {cat: max(1, int(t * scale)) for cat, t in CATEGORY_TARGETS.items()}

    # Track accepted counts per category
    accepted_per_category: Dict[str, int] = {cat: 0 for cat in CATEGORIES}

    try:
        # Iterate documents round-robin, generating per-category items until targets met
        doc_cycle = 0
        max_cycles = 5  # safety bound to avoid infinite loops if acceptance rate is low
        while (
            any(accepted_per_category[c] < category_targets[c] for c in CATEGORIES)
            and len(accepted_items) < count
            and doc_cycle < max_cycles
        ):
            for doc in docs:
                if all(accepted_per_category[c] >= category_targets[c] for c in CATEGORIES):
                    break

                doc_id = doc.get("id", "")
                doc_title = doc.get("title", "Unknown")
                logger.info(
                    "datagen.subset_a.processing_doc",
                    cycle=doc_cycle + 1,
                    doc_id=doc_id,
                    doc_title=doc_title,
                )

                # Fetch document text
                doc_text = await fetch_document_text(api_url, doc_id, doc_title)
                if not doc_text:
                    logger.warning(
                        "datagen.subset_a.no_text_for_doc",
                        doc_id=doc_id,
                        doc_title=doc_title,
                    )
                    continue

                # Generate drafts for each category that still needs items
                for category in CATEGORIES:
                    remaining = category_targets[category] - accepted_per_category[category]
                    if remaining <= 0:
                        continue

                    items_per_category = min(remaining, 5)  # generate in small batches
                    logger.info(
                        "datagen.subset_a.generating",
                        category=category,
                        batch_size=items_per_category,
                        target=category_targets[category],
                        have=accepted_per_category[category],
                    )

                    seed_prompt = (
                        f"Based on this document about Zona Integritas:\n\n{doc_text[:3000]}\n\n"
                        f"Generate {items_per_category} {category} questions in Indonesian. "
                        f"Category '{category}' means: "
                    )
                    if category == "factual":
                        seed_prompt += "questions answerable from a single paragraph."
                    elif category == "procedural":
                        seed_prompt += "questions about steps, processes, or procedures."
                    elif category == "multi-hop":
                        seed_prompt += "questions requiring synthesis of multiple paragraphs."
                    else:  # out-of-domain
                        seed_prompt += "questions OUTSIDE the ZI domain (not about bureaucratic reform)."

                    try:
                        drafts = await generator.generate(
                            seed_prompt=seed_prompt,
                            count=items_per_category,
                            system_prompt=GENERATOR_SYSTEM_PROMPT,
                        )
                    except Exception as e:
                        logger.error("datagen.subset_a.generator_error", error=str(e), exc_info=True)
                        continue

                    total_generated += len(drafts)

                    # Validate each draft
                    for draft in drafts:
                        if accepted_per_category[category] >= category_targets[category]:
                            break
                        if len(accepted_items) >= count:
                            break

                        if not isinstance(draft.parsed, dict):
                            continue

                        item = draft.parsed
                        question = item.get("question", "").strip()
                        if not question:
                            continue

                        # Validate with panel
                        validation_context = VALIDATION_PROMPT.format(
                            question=question,
                            category=item.get("category", category),
                            ground_truth_answer=item.get("ground_truth_answer", ""),
                            source_context=item.get("source_context", ""),
                        )

                        try:
                            verdict = await panel.evaluate(
                                prompt="Is this QA triplet valid and high-quality?",
                                context=validation_context,
                            )
                        except Exception as e:
                            logger.error("datagen.subset_a.panel_error", error=str(e), exc_info=True)
                            continue

                        if verdict.accepted:
                            row = {
                                "question": question,
                                "category": item.get("category", category),
                                "ground_truth_answer": item.get("ground_truth_answer", ""),
                                "source_doc_id": doc_id if category != "out-of-domain" else "NONE",
                                "source_context": item.get("source_context", ""),
                            }
                            accepted_items.append(row)
                            accepted_per_category[category] += 1
                            # Track 5/5-unanimous items for blind injection
                            if verdict.yes_count == len(verdict.votes):
                                blind_tracker.add_candidate({**row, "_panel_yes": verdict.yes_count})
                            logger.info(
                                "datagen.subset_a.accepted",
                                category=category,
                                accepted=accepted_per_category[category],
                                target=category_targets[category],
                            )
                        else:
                            total_rejected += 1
                            logger.info(
                                "datagen.subset_a.rejected",
                                yes=verdict.yes_count,
                                total=verdict.no_count + verdict.yes_count,
                            )
            doc_cycle += 1

    finally:
        await generator.aclose()
        await panel.aclose()

    # Write blind-injection sidecar
    blind_tracker.write_sidecar(
        output_path.replace(".csv", "_blind_injection.csv"),
        fieldnames=["question", "category", "ground_truth_answer", "source_doc_id", "source_context"],
    )

    # 3. Write CSV
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    with output.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["question", "category", "ground_truth_answer", "source_doc_id", "source_context"],
        )
        writer.writeheader()
        writer.writerows(accepted_items)

    logger.info(
        "datagen.subset_a.complete",
        generated=total_generated,
        accepted=len(accepted_items),
        rejected=total_rejected,
        output=output_path,
    )


def main() -> None:
    """Entry point for Subset A generation."""
    parser = argparse.ArgumentParser(
        description="Build Subset A (RAG QA Triplets) using Generator-Evaluator architecture."
    )
    parser.add_argument(
        "--api-url",
        default="http://localhost:8000",
        help="Base URL of the running application",
    )
    parser.add_argument(
        "--output",
        default="data/subset_a.csv",
        help="Output CSV path",
    )
    parser.add_argument(
        "--count",
        type=int,
        default=100,
        help="Target number of accepted items",
    )
    args = parser.parse_args()

    settings = get_dataset_gen_settings()
    asyncio.run(build_subset_a(settings, args.api_url, args.output, args.count))


if __name__ == "__main__":
    main()
