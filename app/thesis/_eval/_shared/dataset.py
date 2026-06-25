"""Dataset loaders for the thesis evaluation subsets.

Loads CSV files matching the column schemas defined in skripsi §3.2.1
(Tabel 3.3–3.6).

Subset A — RAG QA triplets (Tabel 3.3):
    question, category, ground_truth_answer, source_doc_id, source_context

Subset B — Adversarial inputs (Tabel 3.4):
    query, label, attack_type

Subset C — Boundary relevance (Tabel 3.5):
    query, label, subtype

Subset D — RAM ground truth (Tabel 3.6):
    question_id, question, full_response, sentence_id, sentence_text,
    retrieved_context, label, verifier_note
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional


@dataclass(frozen=True)
class SubsetARow:
    """A single row from Subset A (RAG QA triplets).

    Attributes:
        question: User query.
        category: Question category (factual, procedural, multi-hop, out-of-domain).
        ground_truth_answer: Reference answer from source documents.
        source_doc_id: ID of the source document.
        source_context: Verbatim paragraph from the source document (NONE for out-of-domain).
    """

    question: str
    category: str
    ground_truth_answer: str
    source_doc_id: str
    source_context: str


@dataclass(frozen=True)
class SubsetBRow:
    """A single row from Subset B (adversarial inputs).

    Attributes:
        query: Input text.
        label: 'safe' or 'malicious'.
        attack_type: Subtype (jailbreak, dan_attempt, hidden_instruction, safe_normal, safe_complex).
    """

    query: str
    label: str
    attack_type: str


@dataclass(frozen=True)
class SubsetCRow:
    """A single row from Subset C (boundary relevance).

    Attributes:
        query: User query.
        label: 'in_domain' or 'out_of_domain'.
        subtype: Subtype for error analysis.
    """

    query: str
    label: str
    subtype: str


@dataclass(frozen=True)
class SubsetDRow:
    """A single row from Subset D (RAM ground truth).

    Attributes:
        question_id: ID linking back to the Subset A question.
        question: The original question.
        full_response: Full system response.
        sentence_id: Index of the sentence within the response.
        sentence_text: Individual sentence text.
        retrieved_context: Context retrieved by the system.
        label: Annotation label (supported, partially_supported, not_supported, no_source_needed).
        verifier_note: Researcher verification note.
    """

    question_id: str
    question: str
    full_response: str
    sentence_id: int
    sentence_text: str
    retrieved_context: str
    label: str
    verifier_note: str


def _open_csv(path: str) -> csv.DictReader:
    """Open a CSV file and return a DictReader.

    Args:
        path: Path to the CSV file.

    Returns:
        csv.DictReader for the file.

    Raises:
        FileNotFoundError: If the file does not exist.
    """
    csv_path = Path(path)
    if not csv_path.is_file():
        raise FileNotFoundError(f"Dataset not found: {csv_path}")
    fh = csv_path.open(newline="", encoding="utf-8")
    return csv.DictReader(fh)


def load_subset_a(path: str) -> List[SubsetARow]:
    """Load Subset A (RAG QA triplets) from CSV.

    Args:
        path: Path to the CSV file.

    Returns:
        List of SubsetARow records.
    """
    reader = _open_csv(path)
    rows: List[SubsetARow] = []
    for row in reader:
        rows.append(
            SubsetARow(
                question=row["question"].strip(),
                category=row["category"].strip().lower(),
                ground_truth_answer=row.get("ground_truth_answer", "").strip(),
                source_doc_id=row.get("source_doc_id", "").strip(),
                source_context=row.get("source_context", "").strip(),
            )
        )
    return rows


def load_subset_b(path: str) -> List[SubsetBRow]:
    """Load Subset B (adversarial inputs) from CSV.

    Args:
        path: Path to the CSV file.

    Returns:
        List of SubsetBRow records.
    """
    reader = _open_csv(path)
    rows: List[SubsetBRow] = []
    for row in reader:
        rows.append(
            SubsetBRow(
                query=row["query"].strip(),
                label=row["label"].strip().lower(),
                attack_type=row.get("attack_type", "").strip().lower(),
            )
        )
    return rows


def load_subset_c(path: str) -> List[SubsetCRow]:
    """Load Subset C (boundary relevance) from CSV.

    Args:
        path: Path to the CSV file.

    Returns:
        List of SubsetCRow records.
    """
    reader = _open_csv(path)
    rows: List[SubsetCRow] = []
    for row in reader:
        rows.append(
            SubsetCRow(
                query=row["query"].strip(),
                label=row["label"].strip().lower(),
                subtype=row.get("subtype", "").strip().lower(),
            )
        )
    return rows


def load_subset_d(path: str) -> List[SubsetDRow]:
    """Load Subset D (RAM ground truth) from CSV.

    Args:
        path: Path to the CSV file.

    Returns:
        List of SubsetDRow records.
    """
    reader = _open_csv(path)
    rows: List[SubsetDRow] = []
    for row in reader:
        rows.append(
            SubsetDRow(
                question_id=row.get("question_id", "").strip(),
                question=row.get("question", "").strip(),
                full_response=row.get("full_response", "").strip(),
                sentence_id=int(row.get("sentence_id", 0) or 0),
                sentence_text=row["sentence_text"].strip(),
                retrieved_context=row.get("retrieved_context", "").strip(),
                label=row["label"].strip().lower(),
                verifier_note=row.get("verifier_note", "").strip(),
            )
        )
    return rows
