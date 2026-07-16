"""CLI entry point for the RAG pipeline visualization tool.

Runs the full ingestion + retrieval pipeline against a temporary SQLite
database and an ephemeral Qdrant collection, then generates a
self-contained HTML report.

Usage::

    .venv/bin/python -m app.thesis.visualize.run \\
        --pdf-path datasets/permenpanrb-no-5-tahun-2024.pdf

    # With a custom query
    .venv/bin/python -m app.thesis.visualize.run \\
        --pdf-path datasets/permenpanrb-no-5-tahun-2024.pdf \\
        --query "Apa itu zona integritas?"

Requires the real infrastructure services to be running:
    - Unstructured API  (port 8001)
    - Infinity          (port 7997)
    - Qdrant           (port 6333)
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

import httpx
import structlog

from app.thesis.vlm.interfaces import IVLMEnricher
from app.thesis.chunking.models import ContentType

from .capture import ParentChunkSnapshot, PipelineSnapshot
from .html_report import write_html
from .ingestion_viz import run_ingestion
from .json_export import serialize_snapshot
from .retrieval_viz import run_retrieval
from .temp_db import create_temp_engine, init_temp_db

logger = structlog.get_logger(__name__)

_DEFAULT_OUTPUT_DIR = "viz_output"
_DEFAULT_TOP_K = 10


def main() -> None:
    """Parse CLI arguments and run the visualization pipeline."""
    parser = argparse.ArgumentParser(
        description="Visualize the RAG ingestion + retrieval pipelines.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--pdf-path",
        required=True,
        help="Absolute or relative path to the PDF file to ingest.",
    )
    parser.add_argument(
        "--pdf-title",
        default=None,
        help="Human-readable title for the document (defaults to filename).",
    )
    parser.add_argument(
        "--query",
        default=None,
        help="Search query for retrieval (auto-extracted if omitted).",
    )
    parser.add_argument(
        "--unstructured-url",
        default="http://localhost:8001",
        help="Base URL of the Unstructured API (default: http://localhost:8001).",
    )
    parser.add_argument(
        "--unstructured-api-key",
        default="",
        help="API key for Unstructured Cloud (Bearer token). Empty for local self-hosted.",
    )
    parser.add_argument(
        "--infinity-url",
        default="http://localhost:7997",
        help="Base URL of the Infinity server (default: http://localhost:7997).",
    )
    parser.add_argument(
        "--qdrant-host",
        default="127.0.0.1",
        help="Qdrant host (default: 127.0.0.1).",
    )
    parser.add_argument(
        "--qdrant-port",
        type=int,
        default=6333,
        help="Qdrant HTTP port (default: 6333).",
    )
    parser.add_argument(
        "--embedding-model",
        default="BAAI/bge-m3",
        help="Embedding model name for Infinity (default: BAAI/bge-m3).",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=_DEFAULT_TOP_K,
        help=f"Number of search results per mode (default: {_DEFAULT_TOP_K}).",
    )
    parser.add_argument(
        "--output-dir",
        default=_DEFAULT_OUTPUT_DIR,
        help=f"Output directory for HTML + SQLite files (default: {_DEFAULT_OUTPUT_DIR}).",
    )
    parser.add_argument(
        "--cleanup-qdrant",
        action="store_true",
        help="Delete the ephemeral Qdrant collection after the run.",
    )
    parser.add_argument(
        "--export-json",
        dest="export_json",
        action="store_true",
        default=True,
        help="Export raw JSON + CSV artifacts per pipeline stage (default: True).",
    )
    parser.add_argument(
        "--no-export-json",
        dest="export_json",
        action="store_false",
        help="Skip exporting raw JSON + CSV artifacts.",
    )
    parser.add_argument(
        "--vlm",
        choices=["cloud", "local", "fallback", "none"],
        default="none",
        help="VLM mode for figure enrichment: 'cloud' (OpenRouter), 'local' (Ollama), 'fallback' (heuristic), or 'none' (skip, default).",
    )
    args = parser.parse_args()

    asyncio.run(async_main(args))


async def async_main(args: argparse.Namespace) -> None:
    """Run the full visualization pipeline asynchronously.

    Args:
        args: Parsed CLI arguments.
    """
    # Resolve paths
    pdf_path = os.path.realpath(args.pdf_path)
    if not os.path.isfile(pdf_path):
        print(f"Error: PDF not found: {pdf_path}", file=sys.stderr)
        sys.exit(1)

    pdf_title = args.pdf_title or os.path.basename(pdf_path)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    run_id = uuid.uuid4().hex[:8]
    sqlite_path = str(output_dir / f"viz_{timestamp}_{run_id}.sqlite")
    html_path = str(output_dir / f"viz_{timestamp}_{run_id}.html")
    qdrant_collection = f"viz_{run_id}"

    # Pre-flight: check services
    await _preflight_check(args)

    print(f"\n{'=' * 60}")
    print(f"  RAG Pipeline Visualization")
    print(f"{'=' * 60}")
    print(f"  PDF:          {pdf_title}")
    print(f"  SQLite:       {sqlite_path}")
    print(f"  Qdrant:       {qdrant_collection}")
    print(f"  Output HTML:  {html_path}")
    print(f"{'=' * 60}\n")

    # Initialize temp SQLite DB
    engine = create_temp_engine(sqlite_path)
    session_maker = await init_temp_db(engine)

    # Create VLM enricher if requested
    vlm_enricher = _create_vlm_enricher(args.vlm, pdf_path)

    try:
        async with session_maker() as session:
            # ── Ingestion ──────────────────────────────────────────
            print("📥 Running ingestion pipeline...")
            ingestion = await run_ingestion(
                pdf_path=pdf_path,
                pdf_title=pdf_title,
                session=session,
                sqlite_path=sqlite_path,
                qdrant_collection=qdrant_collection,
                unstructured_url=args.unstructured_url,
                infinity_url=args.infinity_url,
                qdrant_host=args.qdrant_host,
                qdrant_port=args.qdrant_port,
                embedding_model=args.embedding_model,
                unstructured_api_key=args.unstructured_api_key,
                vlm_enricher=vlm_enricher,
            )
            print(
                f"   ✓ {len(ingestion.elements)} elements → "
                f"{len(ingestion.parents)} parents → "
                f"{len(ingestion.children)} children → "
                f"{ingestion.qdrant_point_count} Qdrant points"
            )

            # ── Determine query ────────────────────────────────────
            query = args.query
            if not query:
                query = _extract_default_query(ingestion.parents)
                print(f"   ℹ Auto-extracted query: \"{query}\"")

            # ── Retrieval ──────────────────────────────────────────
            print("🔎 Running retrieval pipeline...")
            retrieval = await run_retrieval(
                query=query,
                ingestion=ingestion,
                session=session,
                infinity_url=args.infinity_url,
                qdrant_host=args.qdrant_host,
                qdrant_port=args.qdrant_port,
                embedding_model=args.embedding_model,
                top_k=args.top_k,
            )
            print(
                f"   ✓ Dense: {len(retrieval.dense_results)} | "
                f"Sparse: {len(retrieval.sparse_results)} | "
                f"Hybrid: {len(retrieval.hybrid_results)} results"
            )

        # ── Build HTML report ─────────────────────────────────────
        print("📊 Generating HTML report...")
        snapshot = PipelineSnapshot(
            timestamp=timestamp,
            ingestion=ingestion,
            retrieval=retrieval,
        )
        write_html(snapshot, html_path)

        # ── Export raw JSON + CSV artifacts ──────────────────────────
        json_files: List[Path] = []
        if args.export_json:
            print("📄 Exporting raw JSON + CSV artifacts...")
            json_files = serialize_snapshot(snapshot, output_dir)
            print(f"   ✓ Wrote {len(json_files)} files to {output_dir}")

        print(f"\n{'=' * 60}")
        print(f"  ✅ Done!")
        print(f"{'=' * 60}")
        print(f"  HTML report:  {html_path}")
        print(f"  SQLite DB:    {sqlite_path}")
        print(f"  Raw data:     {output_dir} ({len(json_files)} JSON+CSV files)")
        print(f"  Qdrant:       {qdrant_collection} "
              f"({'kept' if not args.cleanup_qdrant else 'will be deleted'})")
        print(f"{'=' * 60}\n")

    finally:
        # Cleanup Qdrant collection if requested
        if args.cleanup_qdrant:
            await _cleanup_qdrant(args.qdrant_host, args.qdrant_port, qdrant_collection)
        await engine.dispose()


def _extract_default_query(parents: List[ParentChunkSnapshot]) -> str:
    """Extract a default query from the first TEXT parent chunk.

    Takes the first sentence (>20 chars) from the first parent chunk
    whose content_type is TEXT, to ensure the query returns results.

    Args:
        parents: List of ParentChunkSnapshot objects.

    Returns:
        A query string, or a fallback if no suitable text is found.
    """
    for pc in parents:
        if pc.content_type != ContentType.TEXT.value:
            continue
        # Strip the [Context: ...] prefix if present
        text = pc.text
        if text.startswith("[Context:"):
            newline_idx = text.find("\n\n")
            if newline_idx != -1:
                text = text[newline_idx + 2 :]

        # Find first sentence
        for sep in [". ", "? ", "! "]:
            idx = text.find(sep)
            if idx != -1:
                sentence = str(text[: idx + 1].strip())
                if len(sentence) > 20:
                    return sentence

        # No sentence boundary found — take first 100 chars
        if len(text) > 20:
            return str(text[:100].strip())

    return "What is this document about?"


def _create_vlm_enricher(
    vlm_mode: str,
    pdf_path: str,
) -> Optional[IVLMEnricher]:
    """Create a VLM enricher based on the CLI --vlm flag.

    Args:
        vlm_mode: One of 'cloud', 'local', 'fallback', 'none'.
        pdf_path: Path to the source PDF (for fallback mode).

    Returns:
        An IVLMEnricher instance, or None if VLM is disabled.
    """
    if vlm_mode == "none":
        return None

    from app.kb.config import get_vlm_settings
    from app.thesis.vlm import FallbackVLMClient, OllamaVLMClient, OpenRouterVLMClient

    settings = get_vlm_settings()

    if vlm_mode == "cloud" and settings.cloud_api_key:
        enricher: Optional[IVLMEnricher] = OpenRouterVLMClient(
            api_key=settings.cloud_api_key,
            model=settings.cloud_model,
            base_url=settings.cloud_base_url,
            timeout=settings.timeout,
        )
    elif vlm_mode == "local":
        enricher = OllamaVLMClient(
            base_url=settings.local_base_url,
            model=settings.local_model,
            timeout=settings.timeout,
        )
    else:
        enricher = FallbackVLMClient(pdf_path=pdf_path)

    if enricher is not None:
        print(f"   ℹ VLM enricher: {vlm_mode} mode")
    return enricher


async def _preflight_check(args: argparse.Namespace) -> None:
    """Check that all required services are reachable.

    When ``--unstructured-api-key`` is set, the Unstructured Cloud API
    is used and the preflight ping is skipped (the cloud endpoint does
    not respond to a bare GET).

    Args:
        args: Parsed CLI arguments with service URLs.
    """
    services = []
    if not args.unstructured_api_key:
        services.append(("Unstructured API", args.unstructured_url))
    services.append(("Infinity", args.infinity_url))
    services.append(("Qdrant", f"http://{args.qdrant_host}:{args.qdrant_port}"))
    all_ok = True
    for name, url in services:
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                resp = await client.get(url)
                # Qdrant returns 200 for root, others may return 404 — just check reachable
                if resp.status_code < 500:
                    print(f"  ✓ {name} reachable at {url}")
                else:
                    print(f"  ✗ {name} returned {resp.status_code} at {url}")
                    all_ok = False
        except Exception as e:
            print(f"  ✗ {name} unreachable at {url}: {e}")
            all_ok = False

    if not all_ok:
        print(
            "\nError: Some services are not reachable. "
            "Start them with: docker compose --profile ingestion --profile chat up -d",
            file=sys.stderr,
        )
        sys.exit(1)


async def _cleanup_qdrant(host: str, port: int, collection: str) -> None:
    """Delete the ephemeral Qdrant collection.

    Args:
        host: Qdrant host.
        port: Qdrant HTTP port.
        collection: Collection name to delete.
    """
    from qdrant_client import AsyncQdrantClient

    client = AsyncQdrantClient(host=host, port=port)
    try:
        await client.delete_collection(collection_name=collection)
        print(f"   🗑 Deleted Qdrant collection: {collection}")
    except Exception as e:
        logger.warning("viz.cleanup.qdrant_failed", error=str(e))
    finally:
        await client.close()


if __name__ == "__main__":
    main()
