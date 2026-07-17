"""CLI entry point for the combined multi-PDF visualization tool.

Ingests multiple PDFs into a single shared Qdrant collection, then runs
multiple queries against that collection to show how documents interact
during hybrid retrieval (dense + sparse + RRF fusion).

Usage::

    .venv/bin/python -m tools.visualize.run_combined \\
        --pdf-dir datasets/

    # With explicit PDFs and queries
    .venv/bin/python -m tools.visualize.run_combined \\
        --pdf-paths datasets/doc1.pdf datasets/doc2.pdf \\
        --queries "Apa itu zona integritas?" "Bagaimana evaluasi WBK?"

Requires the real infrastructure services to be running:
    - Unstructured API  (port 8001, or cloud with --unstructured-api-key)
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

from .capture import CombinedSnapshot
from .combined_html import write_combined_html
from .combined_json_export import serialize_combined_snapshot
from .combined_viz import run_combined
from .temp_db import create_temp_engine, init_temp_db

logger = structlog.get_logger(__name__)

_DEFAULT_OUTPUT_DIR = "viz_output"
_DEFAULT_TOP_K = 10

# Default ZI-themed queries (Bahasa Indonesia)
_DEFAULT_QUERIES: List[str] = [
    "Apa itu zona integritas?",
    "Apa syarat pengusulan zona integritas?",
    "Bagaimana evaluasi WBK/WBBM?",
    "Apa saja program kerja zona integritas?",
    "Siapa yang bertanggung jawab atas zona integritas?",
]


def main() -> None:
    """Parse CLI arguments and run the combined visualization pipeline."""
    parser = argparse.ArgumentParser(
        description="Visualize multi-PDF cross-document retrieval.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--pdf-dir",
        default=None,
        help="Directory to scan for *.pdf files (e.g. datasets/).",
    )
    parser.add_argument(
        "--pdf-paths",
        nargs="+",
        default=None,
        help="Explicit list of PDF paths (overrides --pdf-dir).",
    )
    parser.add_argument(
        "--queries",
        nargs="+",
        default=None,
        help=f"List of queries (default: {len(_DEFAULT_QUERIES)} ZI-themed queries).",
    )
    parser.add_argument(
        "--unstructured-url",
        default="http://localhost:8001",
        help="Base URL of the Unstructured API (default: http://localhost:8001).",
    )
    parser.add_argument(
        "--unstructured-api-key",
        default="",
        help="API key for Unstructured Cloud. Empty for local self-hosted.",
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
        help="Export raw JSON + CSV artifacts (default: True).",
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
    """Run the combined visualization pipeline asynchronously.

    Args:
        args: Parsed CLI arguments.
    """
    # Resolve PDF paths
    pdf_paths = _resolve_pdf_paths(args)
    if not pdf_paths:
        print("Error: No PDF files found.", file=sys.stderr)
        sys.exit(1)

    # Derive titles from filenames
    pdf_titles = [_derive_title(p) for p in pdf_paths]

    # Resolve queries
    queries = args.queries if args.queries else list(_DEFAULT_QUERIES)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    run_id = uuid.uuid4().hex[:8]
    sqlite_path = str(output_dir / f"viz_combined_{timestamp}_{run_id}.sqlite")
    html_path = str(output_dir / f"viz_combined_{timestamp}_{run_id}.html")
    qdrant_collection = f"vizcomb_{run_id}"

    # Pre-flight: check services
    await _preflight_check(args)

    print(f"\n{'=' * 60}")
    print(f"  Combined Multi-PDF Retrieval Visualization")
    print(f"{'=' * 60}")
    print(f"  Documents:    {len(pdf_paths)} PDFs")
    print(f"  Queries:      {len(queries)} queries")
    print(f"  SQLite:       {sqlite_path}")
    print(f"  Qdrant:       {qdrant_collection}")
    print(f"  Output HTML:  {html_path}")
    print(f"{'=' * 60}\n")

    # Initialize temp SQLite DB
    engine = create_temp_engine(sqlite_path)
    session_maker = await init_temp_db(engine)

    # Create VLM enricher if requested (use first PDF for fallback mode)
    vlm_enricher = _create_vlm_enricher(args.vlm, pdf_paths[0] if pdf_paths else "")

    try:
        async with session_maker() as session:
            # ── Run combined pipeline ────────────────────────────────
            print("📥 Ingesting all PDFs into shared collection...")
            print("🔎 Running all queries against shared collection...")
            snapshot = await run_combined(
                pdf_paths=pdf_paths,
                pdf_titles=pdf_titles,
                queries=queries,
                session=session,
                sqlite_path=sqlite_path,
                qdrant_collection=qdrant_collection,
                unstructured_url=args.unstructured_url,
                unstructured_api_key=args.unstructured_api_key,
                infinity_url=args.infinity_url,
                qdrant_host=args.qdrant_host,
                qdrant_port=args.qdrant_port,
                embedding_model=args.embedding_model,
                top_k=args.top_k,
                vlm_enricher=vlm_enricher,
            )

        # Set the real timestamp (run_combined uses a placeholder)
        snapshot = CombinedSnapshot(
            timestamp=timestamp,
            qdrant_collection=snapshot.qdrant_collection,
            sqlite_path=snapshot.sqlite_path,
            ingestions=snapshot.ingestions,
            retrievals=snapshot.retrievals,
            doc_summaries=snapshot.doc_summaries,
            query_summaries=snapshot.query_summaries,
            queries=snapshot.queries,
        )

        # ── Build HTML report ─────────────────────────────────────
        print("\n📊 Generating combined HTML report...")
        write_combined_html(snapshot, html_path)

        # ── Export raw JSON + CSV artifacts ──────────────────────────
        json_files: List[Path] = []
        if args.export_json:
            print("📄 Exporting raw JSON + CSV artifacts...")
            json_files = serialize_combined_snapshot(snapshot, output_dir)
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


def _resolve_pdf_paths(args: argparse.Namespace) -> List[str]:
    """Resolve the list of PDF paths from CLI args.

    Args:
        args: Parsed CLI arguments.

    Returns:
        List of absolute PDF file paths.
    """
    if args.pdf_paths:
        paths = [os.path.realpath(p) for p in args.pdf_paths]
    elif args.pdf_dir:
        pdf_dir = os.path.realpath(args.pdf_dir)
        if not os.path.isdir(pdf_dir):
            print(f"Error: PDF directory not found: {pdf_dir}", file=sys.stderr)
            sys.exit(1)
        paths = sorted(
            os.path.join(pdf_dir, f)
            for f in os.listdir(pdf_dir)
            if f.lower().endswith(".pdf")
        )
        paths = [os.path.realpath(p) for p in paths]
    else:
        print(
            "Error: Either --pdf-dir or --pdf-paths must be specified.",
            file=sys.stderr,
        )
        sys.exit(1)

    # Validate all paths exist
    valid: List[str] = []
    for p in paths:
        if os.path.isfile(p):
            valid.append(p)
        else:
            print(f"Warning: PDF not found, skipping: {p}", file=sys.stderr)

    return valid


def _derive_title(pdf_path: str) -> str:
    """Derive a human-readable title from a PDF filename.

    Strips the extension and replaces underscores/dashes with spaces.

    Args:
        pdf_path: Path to the PDF file.

    Returns:
        A cleaned-up title string.
    """
    name = os.path.basename(pdf_path)
    # Remove .pdf extension
    if name.lower().endswith(".pdf"):
        name = name[:-4]
    # Replace underscores and dashes with spaces
    name = name.replace("_", " ").replace("-", " ")
    # Collapse multiple spaces
    name = " ".join(name.split())
    return name


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
    is used and the preflight ping is skipped.

    Args:
        args: Parsed CLI arguments with service URLs.
    """
    services: List[tuple[str, str]] = []
    if not args.unstructured_api_key:
        services.append(("Unstructured API", args.unstructured_url))
    services.append(("Infinity", args.infinity_url))
    services.append(("Qdrant", f"http://{args.qdrant_host}:{args.qdrant_port}"))

    all_ok = True
    for name, url in services:
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                resp = await client.get(url)
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
        logger.warning("viz.combined.cleanup.qdrant_failed", error=str(e))
    finally:
        await client.close()


if __name__ == "__main__":
    main()
