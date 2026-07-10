"""Build Subset D — RAM Ground Truth.

Generates sentence-level NLI annotations by running questions through the
full RAG pipeline, decomposing responses into sentences, and having the
panel label each sentence.

Skripsi §3.2.1, Tabel 3.6.

Pipeline:
    1. Load Subset A questions (stratified sample)
    2. Run each question through the full chat pipeline via API
    3. Decompose the response into sentences
    4. Panel (5 models) labels each sentence: supported, partially_supported,
       not_supported, no_source_needed
    5. Accept if ≥4/5 panel members agree on the label
    6. Output CSV matching Tabel 3.6 schema

Usage:
    python -m app.thesis._eval._dataset_gen.build_subset_d \\
        --api-url http://localhost:8000 \\
        --subset-a data/subset_a.csv \\
        --output data/subset_d.csv \\
        --count 30
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import httpx
import structlog

from app.thesis._eval._dataset_gen.config import DatasetGenSettings, get_dataset_gen_settings
from app.thesis._eval._dataset_gen.concordance import BlindInjectionTracker
from app.thesis._eval._dataset_gen.panel import EvaluatorPanel

logger = structlog.get_logger(__name__)

# Sentence splitting regex: splits on . ! ? followed by space or end
SENTENCE_PATTERN = re.compile(r"(?<=[.!?])\s+")

LABELS = ["supported", "partially_supported", "not_supported", "no_source_needed"]

VALIDATION_PROMPT = """\
You are evaluating a sentence from an LLM response for a hallucination \
detection benchmark.

Question: {question}
Full Response: {full_response}
Sentence (ID {sentence_id}): {sentence_text}
Retrieved Context: {retrieved_context}

Label the sentence as one of:
- "supported": The sentence is directly entailed by the retrieved context
- "partially_supported": Some claims in the sentence are supported, others are not
- "not_supported": The sentence contradicts or is not supported by the context
- "no_source_needed": The sentence doesn't need verification (greetings, transitions, etc.)
"""


def split_sentences(text: str) -> List[str]:
    """Split text into sentences.

    Args:
        text: The response text.

    Returns:
        List of sentence strings.
    """
    if not text or not text.strip():
        return []
    sentences = SENTENCE_PATTERN.split(text.strip())
    return [s.strip() for s in sentences if s.strip()]


async def run_pipeline(
    api_url: str,
    session_id: str,
    question: str,
) -> Optional[Dict[str, Any]]:
    """Run a question through the full chat pipeline.

    Calls the streaming chat endpoint (``/api/chat/sessions/{id}/stream``)
    and consumes the NDJSON stream, accumulating ``chunk`` events into the
    response text and capturing the ``context`` event emitted before
    streaming begins.

    Args:
        api_url: Base URL of the running application.
        session_id: Chat session ID.
        question: The question to ask.

    Returns:
        Dict with response text and retrieved context, or None on error.
    """
    async with httpx.AsyncClient(base_url=api_url, timeout=180.0) as client:
        # Send message and collect NDJSON stream.
        # The chat API expects {"message": ...} (see app/chat/api.py ChatRequest).
        response = await client.post(
            f"/api/chat/sessions/{session_id}/stream",
            json={"message": question},
            headers={"Accept": "application/x-ndjson"},
        )
        if response.status_code != 200:
            return None

        full_response = ""
        retrieved_context = ""

        for line in response.text.strip().split("\n"):
            line = line.strip()
            if not line:
                continue
            try:
                chunk = json.loads(line)
                chunk_type = chunk.get("type", "")
                if chunk_type == "chunk":
                    full_response += chunk.get("content", "")
                elif chunk_type == "context":
                    retrieved_context = chunk.get("content", "")
                elif chunk_type == "error":
                    # Pipeline rejected the query (e.g. IVM block); stop early
                    return {"response": chunk.get("content", ""), "context": retrieved_context}
                elif chunk_type == "done":
                    break
            except Exception:
                continue

        return {
            "response": full_response,
            "context": retrieved_context,
        }


async def create_session(api_url: str) -> str:
    """Create a new chat session.

    Args:
        api_url: Base URL of the running application.

    Returns:
        Session ID string.
    """
    async with httpx.AsyncClient(base_url=api_url, timeout=30.0) as client:
        response = await client.post("/api/chat/sessions")
        response.raise_for_status()
        return response.json()["id"]


def load_subset_a(path: str, count: int) -> List[Dict[str, str]]:
    """Load questions from Subset A CSV.

    Args:
        path: Path to Subset A CSV.
        count: Maximum number of questions to load.

    Returns:
        List of question dicts.
    """
    input_path = Path(path)
    if not input_path.exists():
        logger.error("datagen.subset_d.subset_a_not_found", path=path)
        sys.exit(1)

    with input_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        questions = list(reader)

    # Stratified sample: try to get diverse categories
    by_category: Dict[str, List[Dict[str, str]]] = {}
    for q in questions:
        cat = q.get("category", "unknown")
        by_category.setdefault(cat, []).append(q)

    # Round-robin sample across categories
    sampled: List[Dict[str, str]] = []
    idx = 0
    while len(sampled) < count:
        added = False
        for cat in by_category:
            if idx < len(by_category[cat]):
                sampled.append(by_category[cat][idx])
                added = True
                if len(sampled) >= count:
                    break
        if not added:
            break
        idx += 1

    return sampled


async def build_subset_d(
    settings: DatasetGenSettings,
    api_url: str,
    subset_a_path: str,
    output_path: str,
    count: int,
) -> None:
    """Build Subset D (RAM ground truth) and save to CSV.

    Args:
        settings: Dataset generation settings.
        api_url: Base URL of the running application.
        subset_a_path: Path to Subset A CSV.
        output_path: Path to output CSV file.
        count: Number of questions to process.
    """
    if not settings.openrouter_api_key:
        logger.error("datagen.subset_d.missing_api_key")
        sys.exit(1)

    # 1. Load Subset A questions
    logger.info("datagen.subset_d.loading_subset_a")
    questions = load_subset_a(subset_a_path, count)
    if not questions:
        logger.error("datagen.subset_d.no_questions")
        sys.exit(1)
    logger.info("datagen.subset_d.loaded_questions", count=len(questions))

    panel = EvaluatorPanel(settings)
    blind_tracker = BlindInjectionTracker()

    accepted_items: List[Dict[str, Any]] = []
    total_sentences = 0
    total_rejected = 0

    try:
        for i, q in enumerate(questions):
            question = q.get("question", "").strip()
            question_id = f"q-{i + 1:03d}"
            logger.info("datagen.subset_d.processing", index=i + 1, total=len(questions), question=question[:60])

            # 2. Create session and run pipeline
            try:
                session_id = await create_session(api_url)
            except Exception as e:
                logger.error("datagen.subset_d.session_create_failed", error=str(e), exc_info=True)
                continue

            try:
                result = await run_pipeline(api_url, session_id, question)
            except Exception as e:
                logger.error("datagen.subset_d.pipeline_error", error=str(e), exc_info=True)
                continue

            if not result or not result.get("response"):
                logger.warning("datagen.subset_d.no_response", question_id=question_id)
                continue

            full_response = result["response"]
            retrieved_context = result.get("context", "")

            # 3. Decompose into sentences
            sentences = split_sentences(full_response)
            logger.info("datagen.subset_d.decomposed", question_id=question_id, sentence_count=len(sentences))

            # 4. Panel labels each sentence via majority label vote
            for sent_idx, sentence in enumerate(sentences):
                validation_context = VALIDATION_PROMPT.format(
                    question=question,
                    full_response=full_response,
                    sentence_id=sent_idx,
                    sentence_text=sentence,
                    retrieved_context=retrieved_context,
                )

                try:
                    verdict = await panel.evaluate_label(
                        prompt="Assign the correct label to this sentence.",
                        context=validation_context,
                        valid_labels=LABELS,
                    )
                except Exception as e:
                    logger.error("datagen.subset_d.panel_error", sentence_id=sent_idx, error=str(e), exc_info=True)
                    continue

                if verdict.accepted and verdict.accepted_label:
                    label = verdict.accepted_label
                    vote_summary = ", ".join(
                        f"{v.label or 'none'}" for v in verdict.votes
                    )
                    row = {
                        "question_id": question_id,
                        "question": question,
                        "full_response": full_response,
                        "sentence_id": sent_idx,
                        "sentence_text": sentence,
                        "retrieved_context": retrieved_context,
                        "label": label,
                        "verifier_note": f"Panel majority ({verdict.label_counts}; votes: {vote_summary})",
                    }
                    accepted_items.append(row)
                    total_sentences += 1
                    # Track 5/5-unanimous label items for blind injection
                    top_count = max(verdict.label_counts.values()) if verdict.label_counts else 0
                    if top_count == len(verdict.votes):
                        blind_tracker.add_candidate({**row, "_panel_label": label})
                    logger.info("datagen.subset_d.sentence_accepted", sentence_id=sent_idx, label=label)
                else:
                    total_rejected += 1
                    logger.info("datagen.subset_d.sentence_rejected", sentence_id=sent_idx, label_counts=dict(verdict.label_counts))

    finally:
        await panel.aclose()

    # Write blind-injection sidecar
    blind_tracker.write_sidecar(
        output_path.replace(".csv", "_blind_injection.csv"),
        fieldnames=[
            "question_id", "question", "full_response", "sentence_id",
            "sentence_text", "retrieved_context", "label", "verifier_note",
        ],
    )

    # 5. Write CSV
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    with output.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "question_id",
                "question",
                "full_response",
                "sentence_id",
                "sentence_text",
                "retrieved_context",
                "label",
                "verifier_note",
            ],
        )
        writer.writeheader()
        writer.writerows(accepted_items)

    logger.info(
        "datagen.subset_d.complete",
        questions_processed=len(questions),
        sentences_accepted=total_sentences,
        sentences_rejected=total_rejected,
        output=output_path,
    )


def main() -> None:
    """Entry point for Subset D generation."""
    parser = argparse.ArgumentParser(
        description="Build Subset D (RAM Ground Truth) using Generator-Evaluator architecture."
    )
    parser.add_argument(
        "--api-url",
        default="http://localhost:8000",
        help="Base URL of the running application",
    )
    parser.add_argument(
        "--subset-a",
        default="data/subset_a.csv",
        help="Path to Subset A CSV",
    )
    parser.add_argument(
        "--output",
        default="data/subset_d.csv",
        help="Output CSV path",
    )
    parser.add_argument(
        "--count",
        type=int,
        default=30,
        help="Number of questions to process",
    )
    args = parser.parse_args()

    settings = get_dataset_gen_settings()
    asyncio.run(build_subset_d(settings, args.api_url, args.subset_a, args.output, args.count))


if __name__ == "__main__":
    main()
