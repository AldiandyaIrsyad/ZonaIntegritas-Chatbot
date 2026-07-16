"""Raw data exporter for the combined multi-PDF visualization snapshot.

Serializes a :class:`CombinedSnapshot` into machine-readable JSON and CSV
files so that the visualization can be rebuilt or re-rendered later
without re-running the (expensive) real pipeline.

Output layout (all written into ``output_dir``)::

    combined_01_documents.json     # DocIngestionSummary[] (per-doc stats)
    combined_02_queries.json       # QueryRetrievalSummary[] (per-query stats)
    combined_03_all_retrieved.json # All retrieved parents across all queries
    combined_04_score_matrix.json  # Query × document score matrix
    combined_summary.json         # Top-level metadata

    combined_documents.csv         # Tabular doc summaries
    combined_queries.csv           # Tabular query × doc pairs
    combined_retrieved.csv         # All retrieved parents across queries
"""

from __future__ import annotations

import csv
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List

import structlog

from .capture import CombinedSnapshot

logger = structlog.get_logger(__name__)


def serialize_combined_snapshot(
    snapshot: CombinedSnapshot,
    output_dir: Path,
) -> List[Path]:
    """Serialize a combined snapshot to JSON + CSV files.

    Args:
        snapshot: The combined multi-PDF snapshot.
        output_dir: Directory to write artifacts into (created if missing).

    Returns:
        List of paths to all written files, in order.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    written: List[Path] = []

    # ── JSON: one file per data view ────────────────────────────────

    # 01 — Documents
    written.append(_write_json(
        output_dir / "combined_01_documents.json",
        {
            "view": "documents",
            "doc_count": len(snapshot.doc_summaries),
            "documents": [asdict(ds) for ds in snapshot.doc_summaries],
        },
    ))

    # 02 — Queries
    written.append(_write_json(
        output_dir / "combined_02_queries.json",
        {
            "view": "queries",
            "query_count": len(snapshot.query_summaries),
            "queries": [asdict(qs) for qs in snapshot.query_summaries],
        },
    ))

    # 03 — All retrieved parents (full text, across all queries)
    all_retrieved: List[Dict[str, Any]] = []
    for ret in snapshot.retrievals:
        for rp in ret.retrieved_parents:
            entry = asdict(rp)
            entry["query"] = ret.query
            all_retrieved.append(entry)
    written.append(_write_json(
        output_dir / "combined_03_all_retrieved.json",
        {
            "view": "all_retrieved",
            "total_results": len(all_retrieved),
            "results": all_retrieved,
        },
    ))

    # 04 — Score matrix (query × doc)
    doc_titles = [ds.doc_title for ds in snapshot.doc_summaries]
    matrix_data: List[Dict[str, Any]] = []
    for qs in snapshot.query_summaries:
        row: Dict[str, Any] = {"query": qs.query}
        for dt in doc_titles:
            score = qs.per_doc_best_score.get(dt, 0.0)
            count = qs.per_doc_counts.get(dt, 0)
            row[dt] = {"best_score": score, "result_count": count}
        matrix_data.append(row)
    written.append(_write_json(
        output_dir / "combined_04_score_matrix.json",
        {
            "view": "score_matrix",
            "queries": [qs.query for qs in snapshot.query_summaries],
            "documents": doc_titles,
            "matrix": matrix_data,
        },
    ))

    # Summary
    total_elements = sum(ds.element_count for ds in snapshot.doc_summaries)
    total_parents = sum(ds.parent_count for ds in snapshot.doc_summaries)
    total_children = sum(ds.child_count for ds in snapshot.doc_summaries)
    total_points = sum(ds.qdrant_point_count for ds in snapshot.doc_summaries)
    total_results = sum(qs.total_results for qs in snapshot.query_summaries)
    written.append(_write_json(
        output_dir / "combined_summary.json",
        {
            "timestamp": snapshot.timestamp,
            "qdrant_collection": snapshot.qdrant_collection,
            "sqlite_path": snapshot.sqlite_path,
            "doc_count": len(snapshot.doc_summaries),
            "query_count": len(snapshot.queries),
            "queries": list(snapshot.queries),
            "counts": {
                "total_elements": total_elements,
                "total_parents": total_parents,
                "total_children": total_children,
                "total_points": total_points,
                "total_results": total_results,
            },
        },
    ))

    # ── CSV: tabular views ───────────────────────────────────────────

    # Documents CSV
    written.append(_write_csv(
        output_dir / "combined_documents.csv",
        headers=[
            "doc_id", "doc_title", "pdf_path",
            "element_count", "parent_count", "child_count",
            "qdrant_point_count", "total_chars", "content_type_counts",
        ],
        rows=[
            [
                ds.doc_id,
                ds.doc_title,
                ds.pdf_path,
                ds.element_count,
                ds.parent_count,
                ds.child_count,
                ds.qdrant_point_count,
                ds.total_chars,
                json.dumps(ds.content_type_counts, ensure_ascii=False),
            ]
            for ds in snapshot.doc_summaries
        ],
    ))

    # Queries CSV (one row per query × doc pair)
    query_rows: List[List[Any]] = []
    for qs in snapshot.query_summaries:
        for dt in doc_titles:
            query_rows.append([
                qs.query,
                dt,
                qs.per_doc_counts.get(dt, 0),
                qs.per_doc_best_score.get(dt, 0.0),
                qs.total_results,
                qs.top_doc_title,
                qs.top_doc_score,
            ])
    written.append(_write_csv(
        output_dir / "combined_queries.csv",
        headers=[
            "query", "doc_title", "result_count", "best_score",
            "total_results", "top_doc_title", "top_doc_score",
        ],
        rows=query_rows,
    ))

    # Retrieved CSV (all retrieved parents across all queries)
    retrieved_rows: List[List[Any]] = []
    for ret in snapshot.retrievals:
        for rp in ret.retrieved_parents:
            retrieved_rows.append([
                ret.query,
                rp.rank,
                rp.parent_chunk_id,
                rp.rrf_score,
                rp.dense_score if rp.dense_score is not None else "",
                rp.sparse_score if rp.sparse_score is not None else "",
                rp.source_title,
                rp.page if rp.page is not None else "",
                " > ".join(rp.breadcrumbs),
                rp.content_type,
            ])
    written.append(_write_csv(
        output_dir / "combined_retrieved.csv",
        headers=[
            "query", "rank", "parent_chunk_id", "rrf_score",
            "dense_score", "sparse_score", "source_title",
            "page", "breadcrumbs", "content_type",
        ],
        rows=retrieved_rows,
    ))

    logger.info(
        "viz.combined.json_export.complete",
        output_dir=str(output_dir),
        files_written=len(written),
    )
    return written


# ── Helpers ───────────────────────────────────────────────────────────


def _write_json(path: Path, data: Dict[str, Any]) -> Path:
    """Write a JSON file with UTF-8 encoding and 2-space indentation.

    Args:
        path: Destination file path.
        data: Dictionary to serialize.

    Returns:
        The path that was written to.
    """
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def _write_csv(path: Path, headers: List[str], rows: List[List[Any]]) -> Path:
    """Write a CSV file with a header row.

    Args:
        path: Destination file path.
        headers: Column header names.
        rows: List of row lists (each row's length must match headers).

    Returns:
        The path that was written to.
    """
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        writer.writerows(rows)
    return path
