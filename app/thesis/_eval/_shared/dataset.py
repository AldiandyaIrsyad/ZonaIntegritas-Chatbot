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
        source_doc_id: ID of the source document. May hold several
            pipe-separated IDs — see ``gold_doc_ids``.
        source_context: Verbatim paragraph from the source document (NONE for out-of-domain).
    """

    question: str
    category: str
    ground_truth_answer: str
    source_doc_id: str
    source_context: str

    @property
    def gold_doc_ids(self) -> List[str]:
        """Return the relevant document IDs for retrieval scoring.

        A multi-hop question synthesises several paragraphs and can legitimately
        have more than one correct source document, but the column holds a
        single ID, so such a question is scored as a miss whenever retrieval
        surfaces one of its *other* valid sources (M11 in
        writing/overhaul.md). Pipe-separated IDs are therefore
        accepted here; single-ID rows are unaffected.

        Returns:
            List of gold document IDs — empty for out-of-domain rows, which
            carry the sentinel "NONE".
        """
        raw = (self.source_doc_id or "").strip()
        if not raw or raw.upper() == "NONE":
            return []
        return [part.strip() for part in raw.split("|") if part.strip()]


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
        panel_yes: How many panel members voted to accept this row's label.
            Defaults to 0 for CSVs generated before this column existed.
        panel_size: Panel size at generation time. 0 when unrecorded.
    """

    query: str
    label: str
    subtype: str
    panel_yes: int = 0
    panel_size: int = 0

    def is_contested(self, strict_threshold: int = 4) -> bool:
        """Whether the panel fell short of the strict rule on this row.

        Subset C's boundary subtypes — ``near_miss_government`` above all —
        are defined as sitting near the domain edge, so a split panel is
        information about the boundary rather than evidence of a bad item.
        Rows admitted one vote below the strict threshold are kept and marked
        here, so Exp1b can report the strict and contested slices separately
        instead of pooling them or silently discarding the hard cases.

        Args:
            strict_threshold: The strict acceptance threshold to compare
                against (the generation-time ``acceptance_threshold``).

        Returns:
            True if this row was admitted below ``strict_threshold``. Rows
            with no recorded vote count (older CSVs) are treated as not
            contested, since they could only have been accepted strictly.
        """
        return 0 < self.panel_yes < strict_threshold


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
        construction: How the row was made — ``natural`` (real pipeline output) or
            ``perturbed`` (a verified counterfactual edit). Empty for legacy files.
        perturbation_family: The edit strategy for a ``perturbed`` row (e.g.
            ``factual_flip``); empty otherwise.
        intended_label: The label a ``perturbed`` row was designed to carry, which
            the panel then confirmed equals ``label``; empty otherwise.
        perturbation_of: The parent ``question_id:sentence_id`` a perturbed row was
            derived from; empty otherwise.
        difficulty_band: ``easy``/``medium``/``hard`` from how the production NLI
            verifier fared on the row (a reported slice, not an inclusion gate).
        split: ``core`` (balanced diagnostic) or ``natural`` (realistic base rate).
        edit_note: Short description of the perturbation, for auditing.
    """

    question_id: str
    question: str
    full_response: str
    sentence_id: int
    sentence_text: str
    retrieved_context: str
    label: str
    verifier_note: str
    # Optional slice metadata from the hard rebuild. Absent in legacy 2026-07-19
    # files, so all default to empty and load via ``.get()`` — Exp3/Exp4 keep
    # reading the old schema unchanged.
    construction: str = ""
    perturbation_family: str = ""
    intended_label: str = ""
    perturbation_of: str = ""
    difficulty_band: str = ""
    split: str = ""
    edit_note: str = ""


def _int_or_zero(value: Optional[str]) -> int:
    """Parse an optional CSV cell as an int, defaulting to 0.

    Columns added after a dataset was generated are absent (None) in the older
    file, and may be blank in a hand-edited one; neither should crash a load.

    Args:
        value: Raw cell value, or None if the column is absent.

    Returns:
        The parsed integer, or 0 when missing or unparseable.
    """
    try:
        return int(str(value).strip())
    except (TypeError, ValueError):
        return 0


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
                panel_yes=_int_or_zero(row.get("panel_yes")),
                panel_size=_int_or_zero(row.get("panel_size")),
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
                construction=row.get("construction", "").strip().lower(),
                perturbation_family=row.get("perturbation_family", "").strip().lower(),
                intended_label=row.get("intended_label", "").strip().lower(),
                perturbation_of=row.get("perturbation_of", "").strip(),
                difficulty_band=row.get("difficulty_band", "").strip().lower(),
                split=row.get("split", "").strip().lower(),
                edit_note=row.get("edit_note", "").strip(),
            )
        )
    return rows
