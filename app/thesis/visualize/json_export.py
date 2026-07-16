"""Raw data exporter for pipeline visualization snapshots.

Serializes a :class:`PipelineSnapshot` into machine-readable JSON and CSV
files so that the visualization can be rebuilt or re-rendered later
without re-running the (expensive) real pipeline.

Output layout (all written into ``output_dir``)::

    01_parsed_elements.json      # ParsedElementSnapshot[] + counts
    02_parent_chunks.json        # ParentChunkSnapshot[] + counts (full text)
    03_child_chunks.json         # ChildChunkSnapshot[] (full text)
    04_embeddings.json           # EmbeddingSnapshot[] (dense preview + sparse top5)
    05_qdrant_points.json        # QdrantPointSnapshot[] + point count
    06_query_embedding.json      # query text + dense preview + sparse nnz
    07_search_results.json       # dense/sparse/hybrid SearchResultSnapshot[]
    08_retrieved_parents.json    # RetrievedParentSnapshot[] (full text)
    pipeline_summary.json        # top-level counts + metadata

    parsed_elements.csv          # tabular view of stage 1
    parent_chunks.csv            # tabular view of stage 3
    child_chunks.csv             # tabular view of stage 5
    embeddings.csv               # tabular view of stage 6
    search_results.csv          # tabular view of stage 7 (all 3 modes)
    retrieved_parents.csv        # tabular view of stage 8

Limitation:
    ``EmbeddingSnapshot`` stores only an 8-float dense preview and the
    top-5 sparse tokens (not the full 1024-dim vectors) to keep the JSON
    size manageable. Full vectors remain in the Qdrant collection / SQLite
    DB if needed for downstream analysis.
"""

from __future__ import annotations

import csv
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List

import structlog

from .capture import PipelineSnapshot

logger = structlog.get_logger(__name__)


def serialize_snapshot(snapshot: PipelineSnapshot, output_dir: Path) -> List[Path]:
    """Serialize a pipeline snapshot to JSON + CSV files.

    Args:
        snapshot: The combined ingestion + retrieval snapshot.
        output_dir: Directory to write artifacts into (created if missing).

    Returns:
        List of paths to all written files, in order.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    written: List[Path] = []

    ing = snapshot.ingestion
    ret = snapshot.retrieval

    # ── JSON: one file per stage ─────────────────────────────────────
    written.append(_write_json(
        output_dir / "01_parsed_elements.json",
        {
            "stage": "1_pdf_parsing",
            "doc_id": ing.doc_id,
            "doc_title": ing.doc_title,
            "pdf_path": ing.pdf_path,
            "element_type_counts": ing.element_type_counts,
            "total_element_chars": ing.total_element_chars,
            "elements": [asdict(e) for e in ing.elements],
        },
    ))

    written.append(_write_json(
        output_dir / "02_parent_chunks.json",
        {
            "stage": "3_parent_chunking",
            "doc_id": ing.doc_id,
            "content_type_counts": ing.content_type_counts,
            "total_parent_chars": ing.total_parent_chars,
            "parents": [asdict(p) for p in ing.parents],
        },
    ))

    written.append(_write_json(
        output_dir / "03_child_chunks.json",
        {
            "stage": "5_child_chunking",
            "doc_id": ing.doc_id,
            "total_child_chars": ing.total_child_chars,
            "children": [asdict(c) for c in ing.children],
        },
    ))

    written.append(_write_json(
        output_dir / "04_embeddings.json",
        {
            "stage": "6_hybrid_embedding",
            "doc_id": ing.doc_id,
            "model": "BAAI/bge-m3",
            "note": "dense_preview is 8 floats only; full 1024-dim vectors are in Qdrant/SQLite",
            "embeddings": [asdict(e) for e in ing.embeddings],
        },
    ))

    written.append(_write_json(
        output_dir / "05_qdrant_points.json",
        {
            "stage": "7_vector_storage",
            "doc_id": ing.doc_id,
            "qdrant_collection": ing.qdrant_collection,
            "qdrant_point_count": ing.qdrant_point_count,
            "points": [asdict(p) for p in ing.qdrant_points],
        },
    ))

    written.append(_write_json(
        output_dir / "06_query_embedding.json",
        {
            "stage": "query_embedding",
            "query": ret.query,
            "query_dense_preview": ret.query_dense_preview,
            "query_sparse_nnz": ret.query_sparse_nnz,
            "note": "dense_preview is 8 floats only",
        },
    ))

    written.append(_write_json(
        output_dir / "07_search_results.json",
        {
            "stage": "3_hybrid_vector_search",
            "query": ret.query,
            "dense_results": [asdict(r) for r in ret.dense_results],
            "sparse_results": [asdict(r) for r in ret.sparse_results],
            "hybrid_results": [asdict(r) for r in ret.hybrid_results],
        },
    ))

    written.append(_write_json(
        output_dir / "08_retrieved_parents.json",
        {
            "stage": "4_small_to_big_retrieval",
            "query": ret.query,
            "retrieved_parents": [asdict(p) for p in ret.retrieved_parents],
        },
    ))

    written.append(_write_json(
        output_dir / "pipeline_summary.json",
        {
            "timestamp": snapshot.timestamp,
            "doc_id": ing.doc_id,
            "doc_title": ing.doc_title,
            "pdf_path": ing.pdf_path,
            "sqlite_path": ing.sqlite_path,
            "qdrant_collection": ing.qdrant_collection,
            "query": ret.query,
            "counts": {
                "parsed_elements": len(ing.elements),
                "parent_chunks": len(ing.parents),
                "child_chunks": len(ing.children),
                "embeddings": len(ing.embeddings),
                "qdrant_points": ing.qdrant_point_count,
                "dense_results": len(ret.dense_results),
                "sparse_results": len(ret.sparse_results),
                "hybrid_results": len(ret.hybrid_results),
                "retrieved_parents": len(ret.retrieved_parents),
            },
            "element_type_counts": ing.element_type_counts,
            "content_type_counts": ing.content_type_counts,
            "total_chars": {
                "elements": ing.total_element_chars,
                "parents": ing.total_parent_chars,
                "children": ing.total_child_chars,
            },
        },
    ))

    # ── CSV: tabular views ───────────────────────────────────────────
    written.append(_write_csv(
        output_dir / "parsed_elements.csv",
        headers=["index", "element_type", "text_preview", "text_length", "page", "metadata_keys"],
        rows=[
            [e.index, e.element_type, e.text_preview, e.text_length, e.page, "|".join(e.metadata_keys)]
            for e in ing.elements
        ],
    ))

    written.append(_write_csv(
        output_dir / "parent_chunks.csv",
        headers=["id", "chunk_index", "page", "breadcrumbs", "content_type", "text_length"],
        rows=[
            [p.id, p.chunk_index, p.page, " > ".join(p.breadcrumbs), p.content_type, p.text_length]
            for p in ing.parents
        ],
    ))

    written.append(_write_csv(
        output_dir / "child_chunks.csv",
        headers=["id", "parent_chunk_id", "parent_index", "text_length", "content_type"],
        rows=[
            [c.id, c.parent_chunk_id, c.parent_index, c.text_length, c.content_type]
            for c in ing.children
        ],
    ))

    written.append(_write_csv(
        output_dir / "embeddings.csv",
        headers=["child_id", "dense_dim", "sparse_nnz", "dense_preview", "sparse_top5"],
        rows=[
            [
                e.child_id,
                e.dense_dim,
                e.sparse_nnz,
                json.dumps(e.dense_preview),
                json.dumps(e.sparse_top5),
            ]
            for e in ing.embeddings
        ],
    ))

    search_rows: List[List[Any]] = []
    for mode_name, results in [
        ("dense", ret.dense_results),
        ("sparse", ret.sparse_results),
        ("hybrid", ret.hybrid_results),
    ]:
        for r in results:
            search_rows.append([mode_name, r.rank, r.chunk_id, r.parent_chunk_id, r.doc_id, r.score])
    written.append(_write_csv(
        output_dir / "search_results.csv",
        headers=["mode", "rank", "chunk_id", "parent_chunk_id", "doc_id", "score"],
        rows=search_rows,
    ))

    written.append(_write_csv(
        output_dir / "retrieved_parents.csv",
        headers=["rank", "parent_chunk_id", "rrf_score", "dense_score", "sparse_score", "source_title", "page", "breadcrumbs", "content_type"],
        rows=[
            [
                p.rank,
                p.parent_chunk_id,
                p.rrf_score,
                p.dense_score if p.dense_score is not None else "",
                p.sparse_score if p.sparse_score is not None else "",
                p.source_title,
                p.page if p.page is not None else "",
                " > ".join(p.breadcrumbs),
                p.content_type,
            ]
            for p in ret.retrieved_parents
        ],
    ))

    logger.info(
        "viz.json_export.complete",
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
