"""Build Subset D — RAM Ground Truth.

Generates sentence-level NLI annotations by running questions through the
full RAG pipeline, decomposing responses into sentences, and having the
panel label each sentence.

Skripsi §3.2.1, Tabel 3.6.

Pipeline:
    1. Load Subset A questions (stratified sample)
    2. Run each question through the full chat pipeline via API
    3. Decompose the response into sentences, dropping fragments too short
       to be a real proposition (MIN_SENTENCE_TOKENS)
    4. Panel labels each sentence: supported, partially_supported,
       not_supported, no_source_needed — using only the chunk(s) actually
       relevant to that sentence (select_relevant_chunks), not the full
       retrieved context, to keep panel cost and noise down
    5. Accept the majority label if it reaches acceptance_threshold; if not
       but at least 2 panel members still agree on some label, accept that
       as a "disputed" plurality rather than dropping the sentence outright
       (dropping every disagreement was found to silently zero out the
       partially_supported class — see writing/weekend_fixes_plan.md M5)
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
from typing import Any, Dict, List, Optional, Tuple

import httpx
import structlog

from app.thesis._eval._dataset_gen.config import DatasetGenSettings, get_dataset_gen_settings
from app.thesis._eval._dataset_gen.concordance import BlindInjectionTracker
from app.thesis._eval._dataset_gen.panel import EvaluatorPanel
from app.thesis.ram.text_utils import split_sentences as _split_sentences_production

logger = structlog.get_logger(__name__)

# Inline citation marker format emitted by ChatService._format_citation(),
# e.g. *(Supported: 0.66; <doc title>; Page 3; DocID:<uuid>; Evidence:"...")*
# Must be stripped BEFORE sentence splitting: abbreviated titles/names inside
# the marker (e.g. "M.Ag.", "Dr.", "S.Pd.") contain periods that the naive
# sentence splitter otherwise mistakes for sentence boundaries, corrupting
# sentence_text with citation-marker fragments instead of real sentences.
CITATION_MARKER_PATTERN = re.compile(
    r"\*?\s*\("
    r"(?:Supported|Contradiction|Neutral)"
    r":\s*[\d.]+"
    r";\s*[^;]+"
    r"(?:;\s*Page\s+\d+)?"
    r"(?:;\s*DocID:[\w-]+)?"
    r'(?:;\s*Evidence:"[^"]*")?'
    r"\)\s*\*?"
)

LABELS = ["supported", "partially_supported", "not_supported", "no_source_needed"]

# Question and Full Response are identical across every sentence of one
# question; Retrieved Context (narrowed per sentence by
# select_relevant_chunks, below) is usually identical too since most
# sentences of one response draw on the same chunk(s). Sentence is the one
# field that varies per call. Ordering the prompt so the varying field comes
# LAST maximizes the shared prefix across a question's per-sentence panel
# calls, which is what OpenRouter's provider-side prompt caching (see
# EvaluatorPanel.evaluate_label's session_id param) actually caches against
# — see writing/weekend_fixes_plan.md §2.4.
VALIDATION_PROMPT = """\
You are evaluating a sentence from an LLM response for a hallucination \
detection benchmark.

Question: {question}
Full Response: {full_response}
Retrieved Context: {retrieved_context}
Sentence (ID {sentence_id}): {sentence_text}

Label the sentence as one of:
- "supported": The sentence is directly entailed by the retrieved context
- "partially_supported": Some claims in the sentence are supported, others are not
- "not_supported": The sentence contradicts or is not supported by the context
- "no_source_needed": The sentence doesn't need verification (greetings, transitions, etc.)
"""

# Minimum token count for a sentence to be considered a real proposition
# rather than a fragment (e.g. a bare name split out of a markdown list:
# "Arief Johari, S.T., M.Ds."). Fragments have no defensible entailment
# label and their presence in the earlier version of this dataset (M8 in
# writing/weekend_fixes_plan.md) diluted label quality. 5 is a pragmatic
# floor, not a linguistic parse — a real Indonesian clause is rarely
# shorter than this once citation markers are already stripped.
MIN_SENTENCE_TOKENS = 5

# How many of the response's retrieved chunks to keep per sentence when
# building the panel prompt (see select_relevant_chunks). Narrowing from
# "all ~15 chunks" (15-22k chars observed, per the audit) to the few chunks
# actually relevant to this sentence is both a quality fix (the panel isn't
# drowned in irrelevant text) and the single largest cost lever available —
# larger than any model swap (writing/weekend_fixes_plan.md §2.4).
MAX_CONTEXT_CHUNKS_FOR_PANEL = 3


def split_sentences(text: str) -> List[str]:
    """Split text into sentences.

    Strips inline citation markers first (see ``CITATION_MARKER_PATTERN``),
    since their embedded periods (e.g. in abbreviated titles) otherwise get
    mistaken for sentence boundaries, corrupting the result with citation
    fragments instead of real sentences. Delegates the actual splitting to
    the production splitter (``app.thesis.ram.text_utils.split_sentences``),
    which already handles markdown list markers ("1.", "2.") correctly —
    production only ever splits pre-citation text, so this eval-only
    citation-stripping step has no equivalent there.

    Args:
        text: The response text.

    Returns:
        List of sentence strings.
    """
    if not text or not text.strip():
        return []
    cleaned = CITATION_MARKER_PATTERN.sub("", text)
    return _split_sentences_production(cleaned)


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
        Dict with response text, flat retrieved context, and the structured
        per-chunk list (``chunks``, each a dict with ``title``/``page``/
        ``breadcrumbs``/``text`` — see chat_service.py's context payload),
        or None on error. ``chunks`` lets callers narrow context per
        sentence (select_relevant_chunks) instead of guessing chunk
        boundaries out of the flat joined string.
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
        retrieved_chunks: List[Dict[str, Any]] = []

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
                    retrieved_chunks = chunk.get("chunks", []) or []
                elif chunk_type == "error":
                    # Pipeline rejected the query (e.g. IVM block); stop early
                    return {
                        "response": chunk.get("content", ""),
                        "context": retrieved_context,
                        "chunks": retrieved_chunks,
                    }
                elif chunk_type == "done":
                    break
            except Exception:
                continue

        return {
            "response": full_response,
            "context": retrieved_context,
            "chunks": retrieved_chunks,
        }


def select_relevant_chunks(
    sentence: str,
    chunks: List[Dict[str, Any]],
    max_chunks: int = MAX_CONTEXT_CHUNKS_FOR_PANEL,
) -> str:
    """Narrow the retrieved chunks to the few most relevant to ``sentence``.

    Ranks chunks by token containment of ``sentence`` (``|sent ∩ chunk| /
    |sent|``) and returns the top ``max_chunks``, joined in their original
    retrieval order (not by score) so the excerpt still reads coherently.
    Falls back to joining everything if there are no real chunk boundaries
    to work with (e.g. an older API response without the ``chunks`` field).

    This exists for two reasons at once: it keeps the panel from having to
    read all ~15 retrieved chunks (15-22k chars observed) to label a single
    20-token sentence — the single largest cost lever available for this
    pipeline (writing/weekend_fixes_plan.md §2.4) — and it improves label
    quality, since a human annotator would naturally focus on the passage
    that actually supports the sentence rather than being handed the whole
    context indiscriminately.

    Args:
        sentence: The sentence being labeled.
        chunks: Structured per-chunk dicts from the ``context`` NDJSON
            event (``title``/``page``/``breadcrumbs``/``text``).
        max_chunks: Maximum number of chunks to keep.

    Returns:
        The narrowed context text (chunk texts joined with blank lines).
    """
    if not chunks:
        return ""
    sent_tokens = set(re.findall(r"\w+", sentence.lower()))
    if not sent_tokens:
        return "\n\n".join(c.get("text", "") for c in chunks[:max_chunks])

    def _containment(chunk: Dict[str, Any]) -> float:
        chunk_tokens = set(re.findall(r"\w+", chunk.get("text", "").lower()))
        if not chunk_tokens:
            return 0.0
        return len(sent_tokens & chunk_tokens) / len(sent_tokens)

    ranked = sorted(range(len(chunks)), key=lambda i: _containment(chunks[i]), reverse=True)
    keep = sorted(ranked[:max_chunks])  # restore original retrieval order
    return "\n\n".join(chunks[i].get("text", "") for i in keep)


def is_fragment_sentence(sentence: str, min_tokens: int = MIN_SENTENCE_TOKENS) -> bool:
    """Check whether ``sentence`` is too short to be a real proposition.

    Args:
        sentence: Candidate sentence (citation markers already stripped).
        min_tokens: Minimum whitespace-separated token count.

    Returns:
        True if the sentence should be skipped (M8 in
        writing/weekend_fixes_plan.md — fragments like bare names have no
        defensible entailment label).
    """
    return len(sentence.split()) < min_tokens


async def label_sentence(
    panel: EvaluatorPanel,
    question: str,
    question_id: str,
    full_response: str,
    sentence: str,
    sent_idx: int,
    retrieved_context: str,
    chunks: List[Dict[str, Any]],
    session_id: Optional[str] = None,
) -> Tuple[Optional[Dict[str, Any]], str, bool]:
    """Label one sentence via the panel, with plurality fallback (M5 fix).

    Shared by build_subset_d.py and build_subset_d_hard.py so both scripts
    get the same fragment filter, chunk narrowing, and accept/dispute/drop
    logic without duplicating it.

    The stored ``retrieved_context`` in the returned row is always the
    FULL flat context (not the panel's narrowed view) — the narrowing in
    select_relevant_chunks is a labeling aid to keep the panel's prompt
    small and focused, not a claim about what "the" retrieved context was;
    downstream consumers (Exp3, Exp4) should still see everything the
    system actually retrieved.

    Args:
        panel: Evaluator panel for sentence-level labeling.
        question: The question this sentence's response answers.
        question_id: ID to tag the output row with.
        full_response: The full LLM response this sentence was split from.
        sentence: The sentence text (citation markers already stripped).
        sent_idx: Index of this sentence within the response.
        retrieved_context: Full flat retrieved context for the question.
        chunks: Structured per-chunk list for this question (may be empty
            for responses captured before this field existed).
        session_id: Optional OpenRouter sticky-routing session id — pass
            the same id for every sentence of one question.

    Returns:
        Tuple of (row dict or None, status, unanimous). status is one of
        "accepted" (clean majority), "disputed" (plurality fallback, M5),
        "fragment" (skipped before the panel call, M8), "no_signal"
        (panel ran but no label reached even a 2-vote plurality), or
        "error" (the panel call itself failed). ``unanimous`` is True only
        when every panel member independently agreed — the criterion the
        blind-injection tracker uses for candidate selection.
    """
    if is_fragment_sentence(sentence):
        return None, "fragment", False

    narrowed_context = select_relevant_chunks(sentence, chunks) or retrieved_context
    validation_context = VALIDATION_PROMPT.format(
        question=question,
        full_response=full_response,
        sentence_id=sent_idx,
        sentence_text=sentence,
        retrieved_context=narrowed_context,
    )

    try:
        verdict = await panel.evaluate_label(
            prompt="Assign the correct label to this sentence.",
            context=validation_context,
            valid_labels=LABELS,
            session_id=session_id,
        )
    except Exception as e:
        logger.error(
            "datagen.subset_d.panel_error",
            question_id=question_id,
            sentence_id=sent_idx,
            error=str(e),
            exc_info=True,
        )
        return None, "error", False

    label: Optional[str] = None
    status = "accepted"
    top_count = max(verdict.label_counts.values()) if verdict.label_counts else 0

    if verdict.accepted and verdict.accepted_label:
        label = verdict.accepted_label
    elif verdict.label_counts:
        # M5 fix: a verdict that misses the acceptance threshold used to be
        # dropped outright — which systematically discards every sentence
        # the panel actually disagreed about, and disproportionately
        # destroys "partially_supported" specifically, since it's the label
        # most likely to draw a split vote (Subset D shipped with 0/83
        # partially_supported rows before this fix). Accept the plurality
        # instead, as long as at least 2 models independently agree — a
        # genuine 3-way split (e.g. 1-1-1 on a 3-model panel) still has no
        # usable signal and is dropped.
        top_label, top_count = max(verdict.label_counts.items(), key=lambda kv: kv[1])
        if top_count >= 2:
            label, status = top_label, "disputed"

    if label is None:
        return None, "no_signal", False

    vote_summary = ", ".join(f"{v.label or 'none'}" for v in verdict.votes)
    note_prefix = "Panel majority" if status == "accepted" else "Panel plurality, below acceptance threshold"
    row: Dict[str, Any] = {
        "question_id": question_id,
        "question": question,
        "full_response": full_response,
        "sentence_id": sent_idx,
        "sentence_text": sentence,
        "retrieved_context": retrieved_context,
        "label": label,
        "verifier_note": f"{note_prefix} ({verdict.label_counts}; votes: {vote_summary})",
    }
    unanimous = top_count == len(verdict.votes)
    return row, status, unanimous


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
    status_counts: Dict[str, int] = {"accepted": 0, "disputed": 0, "fragment": 0, "no_signal": 0, "error": 0}

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
            chunks = result.get("chunks", [])

            # 3. Decompose into sentences
            sentences = split_sentences(full_response)
            logger.info("datagen.subset_d.decomposed", question_id=question_id, sentence_count=len(sentences))

            # 4-5. Panel labels each sentence (majority, with plurality
            # fallback — see label_sentence). All sentences of one question
            # share a session_id so the shared Question/Full-Response/
            # Context prefix can be provider-cached across the panel calls.
            panel_session_id = f"subset-d-{question_id}"
            for sent_idx, sentence in enumerate(sentences):
                row, status, unanimous = await label_sentence(
                    panel=panel,
                    question=question,
                    question_id=question_id,
                    full_response=full_response,
                    sentence=sentence,
                    sent_idx=sent_idx,
                    retrieved_context=retrieved_context,
                    chunks=chunks,
                    session_id=panel_session_id,
                )
                status_counts[status] = status_counts.get(status, 0) + 1

                if row is not None:
                    accepted_items.append(row)
                    if unanimous:
                        blind_tracker.add_candidate({**row, "_panel_label": row["label"]})
                    logger.info(
                        "datagen.subset_d.sentence_labeled",
                        sentence_id=sent_idx,
                        label=row["label"],
                        status=status,
                    )
                else:
                    logger.info("datagen.subset_d.sentence_skipped", sentence_id=sent_idx, status=status)

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
        sentences_written=len(accepted_items),
        **status_counts,
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
