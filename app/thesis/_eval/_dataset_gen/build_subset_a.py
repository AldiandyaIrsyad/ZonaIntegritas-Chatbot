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


async def fetch_document_text(api_url: str, doc_id: str) -> str:
    """Fetch text content for a document by searching its chunks.

    Args:
        api_url: Base URL of the running application.
        doc_id: Document ID.

    Returns:
        Concatenated text from the document's parent chunks.
    """
    async with httpx.AsyncClient(base_url=api_url, timeout=30.0) as client:
        # Use a broad search to retrieve chunks from this document
        response = await client.get(
            "/api/kb/search",
            params={"q": "zona integritas", "top_k": 50},
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
        print("ERROR: DATAGEN_OPENROUTER_API_KEY not set.", file=sys.stderr)
        sys.exit(1)

    # 1. Fetch KB documents
    print("Fetching KB documents...")
    docs = await fetch_kb_documents(api_url)
    if not docs:
        print("ERROR: No active documents found in KB.", file=sys.stderr)
        sys.exit(1)
    print(f"Found {len(docs)} active documents")

    # 2. Initialize generator and panel
    generator = DatasetGenerator(settings)
    panel = EvaluatorPanel(settings)

    accepted_items: List[Dict[str, str]] = []
    total_generated = 0
    total_rejected = 0

    try:
        for doc in docs:
            if len(accepted_items) >= count:
                break

            doc_id = doc.get("id", "")
            doc_title = doc.get("title", "Unknown")
            print(f"\nProcessing document: {doc_title} ({doc_id})")

            # Fetch document text
            doc_text = await fetch_document_text(api_url, doc_id)
            if not doc_text:
                print(f"  No text found for document {doc_id}, skipping")
                continue

            # Generate drafts for each category
            for category in CATEGORIES:
                if len(accepted_items) >= count:
                    break

                items_per_category = max(1, count // (len(docs) * len(CATEGORIES)))
                print(f"  Generating {items_per_category} {category} questions...")

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
                    print(f"  Generator error: {e}", file=sys.stderr)
                    continue

                total_generated += len(drafts)

                # Validate each draft
                for draft in drafts:
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
                        print(f"  Panel error: {e}", file=sys.stderr)
                        continue

                    if verdict.accepted:
                        accepted_items.append({
                            "question": question,
                            "category": item.get("category", category),
                            "ground_truth_answer": item.get("ground_truth_answer", ""),
                            "source_doc_id": doc_id if category != "out-of-domain" else "NONE",
                            "source_context": item.get("source_context", ""),
                        })
                        print(f"  ✓ Accepted ({len(accepted_items)}/{count})")
                    else:
                        total_rejected += 1
                        print(f"  ✗ Rejected ({verdict.yes_count}/{verdict.no_count + verdict.yes_count})")

    finally:
        await generator.aclose()
        await panel.aclose()

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

    print(f"\n{'=' * 60}")
    print(f"Subset A generation complete:")
    print(f"  Generated: {total_generated}")
    print(f"  Accepted:  {len(accepted_items)}")
    print(f"  Rejected:  {total_rejected}")
    print(f"  Output:    {output_path}")
    print(f"{'=' * 60}")


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
